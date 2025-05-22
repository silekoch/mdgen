import torch, tqdm, math
from torch.utils.checkpoint import checkpoint

from .standalone_hyena import HyenaOperator
from ..transport.transport import t_to_alpha
from .mha import MultiheadAttention
import numpy as np
import torch.nn as nn
from .layers import GaussianFourierProjection, TimestepEmbedder, FinalLayer
from .layers import gelu, modulate
from .ipa import InvariantPointAttention
from ..utils import DirichletConditionalFlow, simplex_proj, get_offsets

from schnetpack import properties
from schnetpack.nn import GaussianRBF, CosineCutoff
from schnetpack.representation import SchNet

def grad_checkpoint(func, args, checkpointing=False):
    if checkpointing:
        return checkpoint(func, *args, use_reentrant=False)
    else:
        return func(*args)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LatentMDGenModel(nn.Module):
    def __init__(self, args, latent_dim):
        super().__init__()
        self.args = args
        if self.args.design:
            assert self.args.prepend_ipa

        self.latent_to_emb = nn.Linear(latent_dim, args.embed_dim)
        if self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            self.latent_to_emb_f = nn.Linear(7, args.embed_dim)
            self.latent_to_emb_r = nn.Linear(7, args.embed_dim)

        cond_dim = latent_dim
        if self.args.design: cond_dim -= 20
        self.cond_to_emb = nn.Linear(cond_dim, args.embed_dim)
        self.mask_to_emb = nn.Embedding(2, args.embed_dim)
        if self.args.design:
            self.x_d_to_emb = nn.Linear(20, args.embed_dim)

        ipa_args = {
            'c_s': args.embed_dim,
            'c_z': 0,
            'c_hidden': args.ipa_head_dim,
            'no_heads': args.ipa_heads,
            'no_qk_points': args.ipa_qk,
            'no_v_points': args.ipa_v,
            'dropout': args.dropout,
        }

        if args.prepend_ipa:
            if not self.args.no_aa_emb:
                self.aatype_to_emb = nn.Embedding(21, args.embed_dim)
            self.ipa_layers = nn.ModuleList(
                [
                    IPALayer(
                        embed_dim=args.embed_dim,
                        ffn_embed_dim=4 * args.embed_dim,
                        mha_heads=args.mha_heads,
                        dropout=args.dropout,
                        use_rotary_embeddings=not args.no_rope,
                        ipa_args=ipa_args
                    )
                    for _ in range(args.num_layers)
                ]
            )

        if args.prepend_schnet:
            if not self.args.no_aa_emb:
                self.aatype_to_emb = nn.Embedding(21, args.embed_dim)
            else:
                raise NotImplementedError("Non-AA embedding ablation not implemented for SchNet")

            radial_basis = GaussianRBF(n_rbf=self.args.schnet_n_rbf, cutoff=self.args.schnet_cutoff)
            cutoff_fn = CosineCutoff(self.args.schnet_cutoff)

            self.schnet = SchNet(
                n_atom_basis=args.embed_dim,
                n_interactions=args.num_layers,
                radial_basis=radial_basis,
                cutoff_fn=cutoff_fn,
                n_filters=self.args.schnet_filters,
                nuclear_embedding=self.aatype_to_emb,
            )

        self.layers = nn.ModuleList(
            [
                LatentMDGenLayer(
                    embed_dim=args.embed_dim,
                    ffn_embed_dim=4 * args.embed_dim,
                    mha_heads=args.mha_heads,
                    dropout=args.dropout,
                    hyena=args.hyena,
                    bottleneck_attention=args.bottleneck_attention,
                    num_frames=args.num_frames,
                    use_rotary_embeddings=not args.no_rope,
                    deactivate_pos_rope=args.no_rope_in_pos,
                    use_time_attention=True,
                    ipa_args=ipa_args if args.interleave_ipa else None,
                )
                for _ in range(args.num_layers)
            ]
        )

        if not (self.args.dynamic_mpnn or self.args.mpnn):
            self.emb_to_latent = FinalLayer(args.embed_dim, latent_dim)
        if args.design:
            self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
            self.fc2 = nn.Linear(args.embed_dim, args.embed_dim)
            self.fc3 = nn.Linear(args.embed_dim, args.embed_dim)
            self.emb_to_logits = nn.Linear(args.embed_dim, 20)

        self.t_embedder = TimestepEmbedder(args.embed_dim)
        if args.abs_pos_emb:
            self.register_buffer('pos_embed',
                                 nn.Parameter(torch.zeros(1, args.crop, args.embed_dim), requires_grad=False))

        if args.abs_time_emb:
            self.register_buffer('time_embed',
                                 nn.Parameter(torch.zeros(1, args.num_frames, args.embed_dim), requires_grad=False))

        self.args = args
        if self.args.design:
            self.condflow = DirichletConditionalFlow(K=20, alpha_spacing=0.001,
                                                     alpha_max=args.alpha_max)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.args.interleave_ipa:
            for block in self.layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.args.abs_pos_emb:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.args.crop))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.args.abs_time_emb:
            time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], np.arange(self.args.num_frames))
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if not (self.args.dynamic_mpnn or self.args.mpnn):
            # Zero-out output layers:
            nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.emb_to_latent.linear.weight, 0)
            nn.init.constant_(self.emb_to_latent.linear.bias, 0)

    def run_ipa(
            self,
            t,
            mask,
            start_frames,
            end_frames,
            aatype,
            x_d=None
    ):
        if self.args.sim_condition or self.args.mpnn:
            B, L = mask.shape
            x = torch.zeros(B, L, self.args.embed_dim, device=mask.device)
            if aatype is not None and not self.args.no_aa_emb:
                x = x + self.aatype_to_emb(aatype)
            if self.args.design:
                x = x + self.x_d_to_emb(x_d)  # pass in only the simplex data
            for layer in self.ipa_layers:
                x = layer(x, t, mask, frames=start_frames)
        elif self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            x_f = start_frames.invert().compose(end_frames).to_tensor_7()
            x_r = end_frames.invert().compose(start_frames).to_tensor_7()
            x_f = self.latent_to_emb_f(x_f)
            x_r = self.latent_to_emb_r(x_r)
            if aatype is not None and not self.args.no_aa_emb:
                x_f = x_f + self.aatype_to_emb(aatype)
                x_r = x_r + self.aatype_to_emb(aatype)
            if self.args.design:
                x_f = x_f + self.x_d_to_emb(x_d)
                x_r = x_r + self.x_d_to_emb(x_d)
            for layer in self.ipa_layers:
                x_r = layer(x_r, t, mask, frames=start_frames)
                x_f = layer(x_f, t, mask, frames=end_frames)
            x = (x_r + x_f)

        # x = x[:, None] + x_latent
        return x

    def run_schnet(self, start_frames, aatype, device):
        B, L = aatype.shape
        x = torch.zeros(B, L, self.args.embed_dim, device=device)
        
        translations = start_frames.get_trans()  # (B, L, 3)

        Rij_all_pairs = translations[:, :, None, :] - translations[:, None, :, :]  # (B, L, L, 3)
        # Set diagonal to infinity (excludes self interactions)
        idx = torch.arange(L, device=Rij_all_pairs.device)
        Rij_all_pairs[:, idx, idx, :] = torch.inf

        # Get indices of neighbors with distance < cutoff
        cutoff = self.args.schnet_cutoff

        # Compare squared distances to avoid sqrt
        dist_sq = torch.sum(Rij_all_pairs**2, dim=-1)  # (B, L, L)
        mask = dist_sq < (cutoff**2)  # (B, L, L)

        # SchNet has no notion of batch size, the batches are handled implicitly
        # by the index pairs that are allowed to interact. Thus, we need to 
        # flatten the batch into this format. 

        # Get the indices (batch, center AA, neighbor AA) of the valid pairs
        b_indices, i_indices, j_indices = torch.nonzero(mask, as_tuple=True)

        # Extract the offset vectors Rij for the valid pairs
        valid_Rij = Rij_all_pairs[b_indices, i_indices, j_indices]  # (N_valid, 3)

        # Compute the global indices for idx_i and idx_j
        # These indices refer to the flattened list of residues (of size B*L).
        flattened_idx_i = b_indices * L + i_indices  # (B*L)
        flattened_idx_j = b_indices * L + j_indices  # (B*L)

        schnet_input = {
            properties.Z: aatype.reshape((B*L)),  # Residue types
            properties.Rij: valid_Rij,  # Neighbor offsets
            properties.idx_i: flattened_idx_i,  # Indices of center residues
            properties.idx_j: flattened_idx_j,  # Indices of neighboring residues
        }

        schnet_output = self.schnet(schnet_input)
        schnet_embedding = schnet_output['scalar_representation']

        x = x + schnet_embedding.reshape(B, L, self.args.embed_dim)

        return x

    def forward(self, x, t, mask,
                start_frames=None, end_frames=None,
                x_cond=None, x_cond_mask=None,
                aatype=None
                ):
        if self.args.dynamic_mpnn:
            x = x[:, [0, -1]]
            x_cond = x_cond[:, [0, -1]]
            x_cond_mask = x_cond_mask[:, [0, -1]]
            mask = mask[:, [0, -1]]
        if self.args.mpnn:
            x = x[:, :1]
            x_cond = x_cond[:, :1]
            x_cond_mask = x_cond_mask[:, :1]
            mask = mask[:, :1]

        if self.args.design:
            x_d = x[..., -20:].mean(1)
        else:
            x_d = None

        x = self.latent_to_emb(x)  # 384 dim token
        if self.args.abs_pos_emb:
            x = x + self.pos_embed

        if self.args.abs_time_emb:
            x = x + self.time_embed[:, :, None]

        if x_cond is not None:
            x = x + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)  # token has cond g, tau

        t = self.t_embedder(t * self.args.time_multiplier)[:, None]

        if self.args.prepend_ipa:  # IPA doesn't need checkpointing
            x = x + self.run_ipa(t[:, 0], mask[:, 0], start_frames, end_frames, aatype, x_d=x_d)[:, None]

        if self.args.prepend_schnet:
            x = x + self.run_schnet(start_frames, aatype, device=mask.device)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            x = grad_checkpoint(layer, (x, t, mask, start_frames), self.args.grad_checkpointing)

        if not (self.args.dynamic_mpnn or self.args.mpnn):
            latent = self.emb_to_latent(x, t)
        if self.args.design:  ### this is also kind of weird
            x_l = self.fc2(gelu(self.fc1(x)))
            x_l = x_l.mean(1)
            logits = self.emb_to_logits(gelu(self.fc3(x_l)))
            if self.args.dynamic_mpnn or self.args.mpnn:
                return logits[:, None, :]
            latent[:, :, :, -20:] = latent[:, :, :, -20:] + logits[:, None, :, :]
        return latent

    # x, t, mask, start_frames=None, end_frames=None, x_cond=None, x_cond_mask=None, aatype=None
    def forward_inference(self, x, t, mask,
                          start_frames=None, end_frames=None,
                          x_cond=None, x_cond_mask=None,
                          aatype=None
                          ):
        if not self.args.design or self.args.dynamic_mpnn or self.args.mpnn:
            return self.forward(x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype)
        else:
            x_discrete = x[:, :, :, -20:]
            B, T, L, _ = x_discrete.shape
            if not torch.allclose(x_discrete.sum(3), torch.ones((B, T, L), device=x.device), atol=1e-4) or not (
                    x_discrete >= 0).all():
                print(
                    f'WARNING: xt.min(): {x_discrete.min()}. Some values of xt do not lie on the simplex. There are '
                    f'{(x_discrete < 0).sum()} negative values in xt of shape {x_discrete.shape} that are negative. '
                    f'We are projecting '
                    f'them onto the simplex.')

                # x_discrete = simplex_proj(x_discrete)
            latent = self.forward(x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype)
            latent_continuous = latent[:, :, :, :-20]
            logits = latent[:, :, :, -20:]

            flow_probs = torch.nn.functional.softmax(logits / self.args.dirichlet_flow_temp, -1)
            if not torch.allclose(flow_probs.sum(3), torch.ones((B, T, L), device=x.device), atol=1e-4) or not (
                    flow_probs >= 0).all():
                print(
                    f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the '
                    f'simplex. There are we are {(flow_probs < 0).sum()} negative values in flow_probs of shape '
                    f'{flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs)

            alpha, dalpha_dt = t_to_alpha(t[0], self.args);
            alpha = alpha.item()

            if alpha > self.args.alpha_max:
                alpha = self.args.alpha_max - self.condflow.alpha_spacing
            c_factor = self.condflow.c_factor(x_discrete.cpu().numpy(), alpha)
            c_factor = torch.from_numpy(c_factor).to(x_discrete)
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {x_discrete.min()}, flow_probs.min(): {flow_probs.min()}')

                if self.args.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                else:
                    raise RuntimeError(
                        f'NAN cfactor after: xt.min(): {x_discrete.min()}, flow_probs.min(): {flow_probs.min()}')

            if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')
            eye = torch.eye(20).to(x_discrete)
            cond_flows = (eye - x_discrete.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1) * dalpha_dt

            return torch.cat([latent_continuous, flow], -1)


class Attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = MultiheadAttention(*args, **kwargs)

    def forward(self, x, mask):
        x = x.transpose(0, 1)
        key_padding_mask = 1 - mask if mask is not None else None
        x, _ = self.attn(query=x, key=x, value=x, key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)
        return x


class CrossAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = MultiheadAttention(*args, **kwargs)

    def forward(self, x, y, mask):
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        key_padding_mask = 1 - mask if mask is not None else None
        x, _ = self.attn(query=x, key=y, value=y, key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)
        return x


class BottleneckAttention(nn.Module):
    def __init__(self, num_latents, embed_dim, ffn_embed_dim, mha_heads, elementwise_affine=False, *args, **kwargs):
        super().__init__()
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.mha_heads = mha_heads
        self.ffn_embed_dim = ffn_embed_dim

        self.latent_vectors = nn.Parameter(torch.empty(num_latents, self.embed_dim))
        nn.init.xavier_uniform_(self.latent_vectors)

        # Collect:
        # The latents attend to the input sequence
        # Latents are Q, Sequence are K, V
        self.in_norm_lat = nn.LayerNorm(self.embed_dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.norm_seq = nn.LayerNorm(self.embed_dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.in_attn = CrossAttention(self.embed_dim, self.mha_heads, *args, **kwargs)

        # The latents interact among themselves
        self.bottleneck_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.bottleneck = Attention(self.embed_dim, self.mha_heads, *args, **kwargs)

        self.ffn_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        # Scatter: 
        # The input sequence attends to the latents
        # Sequence are Q, Latents are K, V
        self.out_norm_lat = nn.LayerNorm(self.embed_dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.out_attn = CrossAttention(self.embed_dim, self.mha_heads, *args, **kwargs)

    def forward(self, x, mask):
        # x: (B * T, L, C)
        # mask: (B * T, L)
        BT, L, C = x.shape

        latents = self.latent_vectors.unsqueeze(0).repeat(BT, 1, 1)
        # latents: (B * T, num_latents, C)

        # latents attend to x
        residual = latents
        latents = self.in_norm_lat(latents)
        x = self.norm_seq(x)
        latents = self.in_attn(latents, x, mask=mask)
        latents = residual + latents

        # latents attend to latents
        residual = latents
        latents = self.bottleneck_norm(latents)
        latents = self.bottleneck(latents, mask=None)  # No masking here, as all latents are valid as keys.
        latents = residual + latents

        # latents processed by FFN
        residual = latents
        latents = self.ffn_norm(latents)
        latents = self.fc2(gelu(self.fc1(latents)))
        latents = residual + latents

        # x attend to latents
        latents = self.out_norm_lat(latents)
        x = self.out_attn(x, latents, mask=None)  # No masking here, as all latents are valid as keys.

        return x


class IPALayer(nn.Module):
    """Transformer layer block."""

    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, dropout=0.0,
                 use_rotary_embeddings=False, ipa_args=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6 * self.embed_dim, bias=True)
        )

        self.cond_norm = nn.LayerNorm(7)
        self.cond_to_emb = nn.Linear(7, self.embed_dim)

        self.mha_l = Attention(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None):
        shift_msa_l, scale_msa_l, gate_msa_l, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
        
        frames = frames.to_tensor_7()
        x = x + self.cond_to_emb(self.cond_norm(frames))

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(x, mask=mask)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x


class LatentMDGenLayer(nn.Module):
    """Transformer layer block."""

    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, dropout=0.0, num_frames=50, hyena=False,
                 bottleneck_attention=False, use_rotary_embeddings=False, deactivate_pos_rope=False,
                 use_time_attention=True, ipa_args=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.hyena = hyena
        self.bottleneck_attention = bottleneck_attention
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_time_attention = use_time_attention
        self.use_rotary_embeddings = use_rotary_embeddings
        self.deactivate_pos_rope = deactivate_pos_rope
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 9 * self.embed_dim, bias=True)
        )

        if ipa_args is not None:
            self.ipa_norm = nn.LayerNorm(self.embed_dim)
            self.ipa = InvariantPointAttention(**ipa_args)

        if self.hyena:
            self.mha_t = HyenaOperator(
                d_model=self.embed_dim,
                l_max=self.num_frames,
                order=2,
                filter_order=64,
            )

        else:
            self.mha_t = Attention(
                self.embed_dim,
                self.mha_heads,
                add_bias_kv=add_bias_kv,
                dropout=dropout,
                use_rotary_embeddings=self.use_rotary_embeddings,
            )

        if self.bottleneck_attention:
            assert not self.use_rotary_embeddings or self.deactivate_pos_rope, "RoPE not supported in BottleneckAttention"
            self.mha_l = BottleneckAttention(
                num_latents=2,
                embed_dim=self.embed_dim,
                ffn_embed_dim=self.ffn_embed_dim,
                mha_heads=self.mha_heads,
                elementwise_affine=False,
                add_bias_kv=add_bias_kv,
                dropout=dropout,
            )
        else: 
            self.mha_l = Attention(
                self.embed_dim,
                self.mha_heads,
                add_bias_kv=add_bias_kv,
                dropout=dropout,
                use_rotary_embeddings=self.use_rotary_embeddings and not self.deactivate_pos_rope,
            )

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None):
        B, T, L, C = x.shape

        shift_msa_l, scale_msa_l, gate_msa_l, \
            shift_msa_t, scale_msa_t, gate_msa_t, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(9, dim=-1)

        if hasattr(self, 'ipa'):
            x = x + self.ipa(self.ipa_norm(x), frames[:, None], frame_mask=mask)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(
            x.reshape(B * T, L, C),
            mask=mask.reshape(B * T, L),  # [:,None].expand(-1, T, -1).reshape(B * T, L)
        ).reshape(B, T, L, C)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_t, scale_msa_t)
        if self.hyena:
            assert (mask - 1).sum() == 0
            x = self.mha_t(
                x.transpose(1, 2).reshape(B * L, T, C)
            ).reshape(B, L, T, C).transpose(1, 2)
        else:
            x = self.mha_t(
                x.transpose(1, 2).reshape(B * L, T, C),
                mask=mask.transpose(1, 2).reshape(B * L, T)
            ).reshape(B, L, T, C).transpose(1, 2)
        x = residual + gate_msa_t.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x
