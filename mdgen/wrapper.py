from .ema import ExponentialMovingAverage
from .logger import get_logger
from .residue_constants import aatype_to_str_sequence

logger = get_logger(__name__)

import pytorch_lightning as pl
import torch, time, os, wandb
import numpy as np
import pandas as pd
from .rigid_utils import Rigid, Rotation
from collections import defaultdict
from functools import partial

from .model.latent_model import LatentMDGenModel
from .transport.transport import create_transport, Sampler
from .utils import get_offsets, atom14_to_pdb
from .tensor_utils import tensor_tree_map
from .geometry import frames_torsions_to_atom14, atom37_to_atom14


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except:
            pass
    return out


DESIGN_IDX = [1, 2]
COND_IDX = [0, 3]
DESIGN_MAP_TO_COND = [0, 0, 3, 3]


class Wrapper(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = 0

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.mean().item()
        log = self._log
        if self.stage == 'train' or self.args.validate:
            log["iter_" + key].append(data)
        log[self.stage + "_" + key].append(data)

    def load_ema_weights(self):
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling 
        # load_state_dict().
        logger.info('Loading EMA weights')
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        logger.info('Restoring cached weights')
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        if self.args.ema:
            self.ema.update(self.model)

    def training_step(self, batch, batch_idx):
        if self.args.ema:
            if (self.ema.device != self.device):
                self.ema.to(self.device)
        return self.general_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        if self.args.ema:
            if (self.ema.device != self.device):
                self.ema.to(self.device)
            if (self.cached_weights is None):
                self.load_ema_weights()

        self.general_step(batch, stage='val')
        self.validation_step_extra(batch, batch_idx)
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()

    def validation_step_extra(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self.print_log(prefix='train', save=False)

    def on_validation_epoch_end(self):
        if self.args.ema:
            self.restore_cached_weights()
        self.print_log(prefix='val', save=False)

    def on_before_optimizer_step(self, optimizer):
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            self.print_log()

        if self.args.check_grad:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    logger.warning(f"Param {name} has no grad")

    def on_load_checkpoint(self, checkpoint):
        logger.info('Loading EMA state dict')
        if self.args.ema:
            ema = checkpoint["ema"]
            self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        if self.args.ema:
            if self.cached_weights is not None:
                self.restore_cached_weights()
            checkpoint["ema"] = self.ema.state_dict()

    def print_log(self, prefix='iter', save=False, extra_logs=None):
        log = self._log
        log = {key: log[key] for key in log if f"{prefix}_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update({
            'epoch': self.trainer.current_epoch,
            'trainer_step': self.trainer.global_step + int(prefix == 'iter'),
            'iter_step': self.iter_step,
            f'{prefix}_count': len(log[next(iter(log))]),

        })
        if extra_logs:
            mean_log.update(extra_logs)
        try:
            for param_group in self.optimizers().optimizer.param_groups:
                mean_log['lr'] = param_group['lr']
        except:
            pass

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.args.wandb:
                wandb.log(mean_log)
            if save:
                path = os.path.join(
                    os.environ["MODEL_DIR"],
                    f"{prefix}_{self.trainer.current_epoch}.csv"
                )
                pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if f"{prefix}_" in key:
                del self._log[key]

    def configure_optimizers(self):
        cls = torch.optim.AdamW if self.args.adamW else torch.optim.Adam
        optimizer = cls(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
        )
        return optimizer


class NewMDGenWrapper(Wrapper):
    def __init__(self, args, guide_by_known=False):
        args.guide_by_known = guide_by_known  # Kinda hacky, but we want to inject this only at inference time and keep the model flexible at training time. 
        super().__init__(args)
        for key in [
            'inpainting',
            'no_torsion',
            'hyena',
            'no_aa_emb',
            'supervise_all_torsions',
            'supervise_no_torsions',
            'design_key_frames',
            'no_design_torsion',
            'cond_interval',
            'mpnn',
            'dynamic_mpnn',
            'no_offsets',
            'no_frames',
            'translations_only',
            'c_alpha_only',
            'prepend_aatype_to_emb',
            'overwrite_ipa_with_linear',
            'prepend_schnet',
            'bottleneck_attention',
            'no_rope_in_pos',
            'supervise_no_rotations',
        ]:
            if not hasattr(args, key):
                setattr(args, key, False)
        # args.latent_dim = 7 if not self.args.tps_condition else 14
        latent_dim = 21 if not (self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn) else 28
        if args.design:
            latent_dim += 20
        if args.no_frames:
            latent_dim = 111
        if args.translations_only:
            latent_dim = 3 if not (self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn) else 6
        if args.c_alpha_only:
            latent_dim = 3
        
        self.latent_dim = latent_dim
        self.model = LatentMDGenModel(args, latent_dim)

        self.transport = create_transport(
            args,
            args.path_type,
            args.prediction,
            None,  # args.loss_weight,
            # args.train_eps,
            # args.sample_eps,
        )  # default: velocity; 
        self.transport_sampler = Sampler(self.transport)

        if not hasattr(args, 'ema'):
            args.ema = False
        if args.ema:
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=args.ema_decay
            )
            self.cached_weights = None

    def prep_hyena_batch(self, batch):
        B, T, L, _ = batch['latents'].shape
        rigids = Rigid(trans=batch['trans'], rots=Rotation(rot_mats=batch['rots']))

        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=self.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.tps_condition:
            cond_mask[:, 0] = cond_mask[:, -1] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        if self.args.inpainting or self.args.dynamic_mpnn or self.args.mpnn:
            cond_mask[:, :, COND_IDX] = 1

        aatype_mask = torch.ones_like(batch['seqres'])
        if self.args.design:
            aatype_mask[:, DESIGN_IDX] = 0
        ######## 
        return {
            'latents': batch['latents'].float(),
            'loss_mask': batch['loss_mask'].unsqueeze(1).expand(-1, T, -1, -1),
            'model_kwargs': {
                'start_frames': rigids,
                'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1),
                'aatype': torch.where(aatype_mask.bool(), batch['seqres'], 20),
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), batch['latents'].float(), 0.0),
                'x_cond_mask': cond_mask,
            }
        }

    def prep_batch_no_frames(self, batch):
        
        B, T, L, _, _ = batch['atom37'].shape
        
        latents = batch['atom37'].reshape(B, T, L, 111)
        mask = batch['mask'][:,None,:,1].expand(-1, T, -1)

        loss_mask = batch['mask'][:,None,:,:,None].expand(-1, T, -1, -1, 3)
        loss_mask = loss_mask.reshape(B, T, L, 111)
        
        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=mask.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
            
        aatype_mask = torch.ones_like(batch['seqres'])

        return {
            'latents': latents,
            'loss_mask': loss_mask,
            'model_kwargs': {
                'mask': mask,
                'aatype': torch.where(aatype_mask.bool(), batch['seqres'], 20),
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                'x_cond_mask': cond_mask,
            }
        }

    def prep_batch_translations_only(self, batch):
        # Assume batch['trans'] has shape (B, T, L, 3) - absolute translations from dataset.py
        # Assume batch['mask'] has shape (B, L) - residue mask
        # Assume batch['seqres'] has shape (B, L) - amino acid types

        B, T, L, _ = batch['trans'].shape
        latent_dim = 3 # Base dimension for translation

        # 1. Calculate Target Relative Translations ('latents')
        start_trans = batch['trans'][:, 0] # Key frame(s) for offset calculation
        latents = batch['trans'] - start_trans.unsqueeze(1) # Relative offset from start frame (B, T, L, 3)

        # --- Adapt for different conditioning if needed ---
        # Example for TPS/Inpainting (latent_dim would be 6)
        if self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            raise NotImplementedError
        #    latent_dim = 6
        #    end_trans = batch['trans'][:, -1:]
        #    rel_start_trans = batch['trans'] - start_trans
        #    rel_end_trans = batch['trans'] - end_trans
        #    latents = torch.cat([rel_start_trans, rel_end_trans], -1) # Shape (B, T, L, 6)
        # else: # Default sim_condition
        #    latents = batch['trans'] - start_trans # Shape (B, T, L, 3)
        # ----------------------------------------------------

        # 2. Create Loss Mask
        # Mask should be 1 only where residues exist, matching latent dimensions
        loss_mask = batch['mask'].unsqueeze(1).unsqueeze(-1).expand(B, T, L, latent_dim)

        # 3. Prepare Conditioning Information for model_kwargs
        # Conditioning mask (indicates known time points/residues)
        cond_mask = torch.zeros(B, T, L, dtype=torch.int, device=latents.device)
        if self.args.sim_condition: # Example condition
            cond_mask[:, 0] = 1    # Condition on the first frame
        # Add other conditioning logic for TPS, inpainting, intervals etc. based on self.args

        # Conditional latents (the known relative translations based on cond_mask)
        x_cond = torch.where(cond_mask.unsqueeze(-1).bool(), latents, torch.zeros_like(latents))

        # Model's view of the residue mask
        model_residue_mask = batch['mask'].unsqueeze(1).expand(B, T, L)

        # Create dummy identity rotations matching the shape of translations
        dummy_start_rots = Rotation.identity(start_trans.shape[:-1], device=start_trans.device) # Shape (B, L)

        # Create dummy Rigid objects
        dummy_start_frames = Rigid(rots=dummy_start_rots, trans=start_trans) # Shape (B, L) Rigid object

        return_dict = {
            'latents': latents,      # Target relative translations (B, T, L, latent_dim)
            'loss_mask': loss_mask,  # Mask for loss calculation (B, T, L, latent_dim)
            'model_kwargs': {
                'mask': model_residue_mask,       # Residue existence mask (B, T, L)
                'aatype': batch['seqres'],           # Amino acid sequence types (B, L)
                'x_cond': x_cond,                 # Known relative translations (B, T, L, latent_dim)
                'x_cond_mask': cond_mask,         # Mask indicating known values in x_cond (B, T, L)

                # IPA requires start frames, as a workaround we provide dummy identity rotations
                'start_frames': dummy_start_frames, # Dummy start frames (B, L) Rigid object
            }
        }

        return return_dict


    def prep_batch_c_alpha_only(self, batch):
        ca_coordinates = batch['ca_coordinates']  # Shape (B, T, L, 3)
        B, T, L, _ = ca_coordinates.shape

        key_frames = Rigid(
            trans=batch['key_frame_trans'],
            rots=Rotation(rot_mats=batch['key_frame_rots'])
        )  # B, K, L
        
        if self.args.no_offsets:
            latents = ca_coordinates
            initial_rigids = None
        else:
            latents = key_frames.invert_apply(ca_coordinates)
            initial_rigids = key_frames[:, 0]

        loss_mask = batch['mask'].unsqueeze(1).unsqueeze(-1).expand(B, T, L, 3)

        cond_mask = torch.zeros(B, T, L, dtype=int, device=latents.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
            aatype_mask = torch.ones_like(batch['seqres'])
        else:
            raise NotImplementedError()

        if self.args.prepend_ipa:
            # In this case initial_rigids == start_frames, but they are not the same. 
            # initial_rigids is used for reconstruction at inference time. 
            # start_frames is given to the model as conditiong.
            start_frames = key_frames[:, 0]  # Backbone frames for IPA
        elif self.args.prepend_schnet:
            start_frames = ca_coordinates[:, 0]  # Absolute positions for SchNet
        else:
            start_frames = None  # No conditioning for everything else

        return {
            'latents': latents,
            'loss_mask': loss_mask,
            'initial_rigids': initial_rigids,
            'model_kwargs': {
                'start_frames': start_frames,
                'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1),
                'aatype': torch.where(aatype_mask.bool(), batch['seqres'], 20),
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                'x_cond_mask': cond_mask,
            }
        }

    def prep_batch(self, batch):

        if self.args.no_frames:
            return self.prep_batch_no_frames(batch)

        if self.args.c_alpha_only:
            return self.prep_batch_c_alpha_only(batch)

        # if self.args.hyena:
        if 'latents' in batch:
            return self.prep_hyena_batch(batch)
        
        if self.args.translations_only:
            return self.prep_batch_translations_only(batch)

        rigids = Rigid(
            trans=batch['trans'],
            rots=Rotation(rot_mats=batch['rots'])
        )  # B, T, L
        B, T, L = rigids.shape
        if self.args.design_key_frames:
            rigids = Rigid.cat([
                rigids[:, :1, DESIGN_MAP_TO_COND],  # replace designed rototranslations in the key frames
                rigids[:, 1:-1],
                rigids[:, -1:, DESIGN_MAP_TO_COND]
            ], 1)

        if self.args.no_offsets:
            offsets = rigids.to_tensor_7()
        else:
            offsets = get_offsets(rigids[:, 0:1], rigids)
        #### make sure the quaternions have real part
        offsets[..., :4] *= torch.where(offsets[:, :, :, 0:1] < 0, -1, 1)

        frame_loss_mask = batch['mask'].unsqueeze(-1).expand(-1, -1, 7)  # B, L, 7
        torsion_loss_mask = batch['torsion_mask'].unsqueeze(-1).expand(-1, -1, -1, 2).reshape(B, L, 14)

        if self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            offsets_r = get_offsets(rigids[:, -1:], rigids)
            offsets_r[..., :4] *= torch.where(offsets_r[:, :, :, 0:1] < 0, -1, 1)
            offsets = torch.cat([offsets, offsets_r], -1)
            frame_loss_mask = torch.cat([frame_loss_mask, frame_loss_mask], -1)

        if self.args.no_torsion:
            latents = torch.cat([offsets, torch.zeros_like(batch['torsions'].view(B, T, L, 14))], -1)
        elif self.args.no_design_torsion:
            torsions_ = batch['torsions'].clone()
            torsions_[:, :, DESIGN_IDX] = 0
            latents = torch.cat([offsets, torsions_.view(B, T, L, 14)], -1)
        else:
            latents = torch.cat([offsets, batch['torsions'].view(B, T, L, 14)], -1)

        if self.args.supervise_all_torsions:
            torsion_loss_mask = torch.ones_like(torsion_loss_mask)
        elif self.args.supervise_no_torsions:
            torsion_loss_mask = torch.zeros_like(torsion_loss_mask)

        loss_mask = torch.cat([frame_loss_mask, torsion_loss_mask], -1)
        loss_mask = loss_mask.unsqueeze(1).expand(-1, T, -1, -1)
        
        if self.args.supervise_no_rotations:
            # Indices for rotation quaternions in the 7D frame representation
            rotation_indices = [0, 1, 2, 3]

            # If using TPS/inpainting (14D frame offset = 7D from start + 7D from end):
            if self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
                rotation_indices = [0, 1, 2, 3, 7, 8, 9, 10]
            
            # Zero out the mask for the rotation components
            loss_mask[..., rotation_indices] = 0.0

        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=offsets.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.tps_condition:
            cond_mask[:, 0] = cond_mask[:, -1] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        if self.args.inpainting or self.args.dynamic_mpnn or self.args.mpnn:
            cond_mask[:, :, COND_IDX] = 1

        aatype_mask = torch.ones_like(batch['seqres'])
        if self.args.design:
            aatype_mask[:, DESIGN_IDX] = 0
        ######## 

        return {
            'rigids': rigids,
            'latents': latents,
            'loss_mask': loss_mask,
            'model_kwargs': {
                'start_frames': rigids[:, 0],
                'end_frames': rigids[:, -1],
                'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1),
                'aatype': torch.where(aatype_mask.bool(), batch['seqres'], 20),
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                'x_cond_mask': cond_mask,
            }
        }

    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        start1 = time.time()

        prep = self.prep_batch(batch)

        start = time.time()
        out_dict = self.transport.training_losses(
            model=self.model,
            x1=prep['latents'],
            aatype1=batch['seqres'] if self.args.design else None,
            mask=prep['loss_mask'],
            model_kwargs=prep['model_kwargs']
        )
        self.log('model_dur', time.time() - start)
        loss = out_dict['loss']
        self.log('loss', loss)

        if self.args.design:
            aa_out = torch.argmax(out_dict['logits'], dim=-1)
            aa_recovery = aa_out == batch['seqres'][:, None, :].expand(-1, aa_out.shape[1], -1)

            self.log('category_pred_design_aa_recovery', aa_recovery[:, :, 1:-1].float().mean().item())
            cond_aa_recovery = torch.cat([aa_recovery[:, :, 0:1], aa_recovery[:, :, -1:]], 2)
            self.log('category_pred_cond_aa_recovery', cond_aa_recovery.float().mean().item())

            self.log('loss_continuous', out_dict['loss_continuous'].mean())
            self.log('loss_discrete', out_dict['loss_discrete'])

        self.log('time', out_dict['t'])
        self.log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.log('name', ','.join(batch['name']))
        self.log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()
        return loss.mean()

    def inference(self, batch):

        prep = self.prep_batch(batch)

        latents = prep['latents']
        B, T, L, _ = latents.shape
        if not self.args.no_frames and not self.args.c_alpha_only:
            rigids = prep['rigids']

        ### oracle
        # if self.args.oracle:
        #     assert self.args.sim_condition  # only works with that
        #     offsets = get_offsets(rigids[:, 0:1], rigids)
        #     torsions = batch['torsions'].view(B, T, L, 14)
        # else:
        if self.args.dynamic_mpnn or self.args.mpnn:
            x1 = prep['latents']
            x_d = torch.zeros(x1.shape[0], x1.shape[1], x1.shape[2], 20, device=self.device)
            xt = torch.cat([x1, x_d], dim=-1)
            logits = self.model.forward_inference(xt, torch.ones(B, device=self.device),
                                                  **prep['model_kwargs'])
            aa_out = torch.argmax(logits, -1)
            atom14 = frames_torsions_to_atom14(rigids, batch['torsions'],
                                               batch['seqres'][:, None].expand(B, T, L))
            return atom14, aa_out

        if self.args.design:
            zs_continuous = torch.randn(B, T, L, self.latent_dim - 20, device=latents.device)
            zs_discrete = torch.distributions.Dirichlet(torch.ones(B, L, 20, device=latents.device)).sample()
            zs_discrete = zs_discrete[:, None].expand(-1, T, -1, -1)
            zs = torch.cat([zs_continuous, zs_discrete], -1)
        else:
            zs = torch.randn(B, T, L, self.latent_dim, device=self.device)

        sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method)
        # num_steps=self.args.inference_steps)  # default to ode

        if self.args.guide_by_known:
            samples = sample_fn(
                zs,
                self.model.forward_inference,
                x1_known = prep['model_kwargs']['x_cond'],
                x1_known_mask = prep['model_kwargs']['x_cond_mask'],
                **prep['model_kwargs'],
            )[-1]
        else:
            samples = sample_fn(
                zs,
                self.model.forward_inference,
                **prep['model_kwargs'],
            )[-1]

        if self.args.c_alpha_only:
            if not self.args.sim_condition: 
                raise NotImplementedError("See code below for design handling")
            aa_out = batch['seqres'][:, None].expand(B, T, L)

            if self.args.no_offsets:
                return samples, aa_out
            else:
                initial_rigids = prep['initial_rigids']
                return initial_rigids.apply(samples), aa_out
                
        if self.args.no_frames:
            atom14 = atom37_to_atom14(
                samples.cpu().numpy().reshape(B, T, L, 37, 3),
                batch['seqres'][0].cpu().numpy()
            )
            return torch.from_numpy(atom14).float(), batch['seqres'][:, None].expand(B, T, L)
            
        offsets = samples[..., :7]
        
        if self.args.tps_condition or self.args.inpainting:
            torsions = samples[..., 14:28]
            logits = samples[..., -20:]
        else:
            torsions = samples[..., 7:21]
            logits = samples[..., -20:]

        
        if self.args.no_offsets:
            frames = Rigid.from_tensor_7(offsets, normalize_quats=True)
        else:
            frames = rigids[:, 0:1].compose(Rigid.from_tensor_7(offsets, normalize_quats=True))
        if self.args.design:
            trans = frames.get_trans()
            rots = frames.get_rots().get_rot_mats()
            frames = Rigid(trans=trans, rots=Rotation(rot_mats=rots))
        torsions = torsions.reshape(B, T, L, 7, 2)
        if not self.args.oracle:
            torsions = torsions / torch.linalg.norm(torsions, dim=-1, keepdims=True)
        atom14 = frames_torsions_to_atom14(frames, torsions.view(B, T, L, 7, 2),
                                           batch['seqres'][:, None].expand(B, T, L))

        if self.args.design:
            aa_out = torch.argmax(logits, -1)
        else:
            aa_out = batch['seqres'][:, None].expand(B, T, L)
        return atom14, aa_out

    def validation_step_extra(self, batch, batch_idx):

        do_designability = batch_idx < self.args.inference_batches and (
                (self.current_epoch + 1) % self.args.designability_freq == 0 or \
                self.args.validate) and self.trainer.is_global_zero
        if do_designability:
            atom14, aa_out = self.inference(batch)
            aa_recovery = aa_out == batch['seqres'][:, None, :].expand(-1, aa_out.shape[1], -1)
            self.log('design_aa_recovery', aa_recovery[:, :, 1:-1].float().mean().item())
            cond_aa_recovery = torch.cat([aa_recovery[:, :, 0:1], aa_recovery[:, :, -1:]], 2)
            self.log('cond_aa_recovery', cond_aa_recovery.float().mean().item())
            self.log('seq_pred', ','.join([aatype_to_str_sequence(aa) for aa in aa_out[:, 0]]))
            self.log('seq_true', ','.join([aatype_to_str_sequence(aa) for aa in batch['seqres']]))
            prot_name = batch['name'][0]
            path = os.path.join(os.environ["MODEL_DIR"], f'epoch{self.current_epoch}_{prot_name}.pdb')

            atom14_to_pdb(atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)
        else:
            self.log('design_aa_recovery', np.nan)
            self.log('cond_aa_recovery', np.nan)
            self.log('seq_pred', 'nan')
            self.log('seq_true', 'nan')
