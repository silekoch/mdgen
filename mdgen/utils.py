import itertools
import numpy as np
import scipy
import torch
from . import protein
from . import residue_constants as rc
from .geometry import atom14_to_atom37

def get_offsets(ref_frame, rigids):
    B, T, L = rigids.shape
    if T > 500000:
        offsets1 = ref_frame.invert().compose(rigids[:, : 500000]).to_tensor_7()
        offsets2 = ref_frame.invert().compose(rigids[:, 500000:]).to_tensor_7()
        return torch.cat([offsets1, offsets2], 1)
    else:
        return ref_frame.invert().compose(rigids).to_tensor_7()

def simplex_proj(seq):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)

def compute_distance_matrix(coords):
    """
    Compute pairwise distance matrix for coordinates with arbitrary leading dimensions.
    
    Args:
        coords: tensor of shape (..., L, 3) where ... represents any number of leading dimensions
    
    Returns:
        distance_matrix: tensor of shape (..., L, L)
    """
    # Get the shape - last two dimensions are (L, 3)
    *_, L, _ = coords.shape
    
    # Expand dimensions for broadcasting
    coords_i = coords.unsqueeze(-2)  # (..., L, 1, 3)
    coords_j = coords.unsqueeze(-3)  # (..., 1, L, 3)
    
    # Compute squared differences and distances
    diff = coords_i - coords_j  # (..., L, L, 3)
    distances = torch.linalg.vector_norm(diff, dim=-1)  # (..., L, L)
    
    return distances

def compute_crop_start(
    sequence_length: int,
    crop_length: int,
    anchor_index: int,
    center: bool = False,
    random_bias: bool = True,
) -> int:
    """
    Compute the start index for cropping a sequence.
    
    Args:
        sequence_length: Total length of the original sequence
        crop_length: Desired length of the crop
        anchor_index: Index that must be included in the crop
        center: If True, center the anchor in the crop. If False, random position.
        random_bias: If crop_length is even and center is True randomly 
            choose one of the two valid start positions. 
        
    Returns:
        Start index for the crop
    """
    assert crop_length > 0 and sequence_length > 0 and anchor_index >= 0
    assert crop_length < sequence_length
    assert anchor_index < sequence_length

    # Calculate valid range for start index
    min_start = max(0, anchor_index - crop_length + 1)
    max_start = min(sequence_length - crop_length, anchor_index)
    
    if center:
        # Center the anchor in the crop
        ideal_start = anchor_index - crop_length // 2
        
        # Handle even crop length with random left/right bias
        if random_bias and crop_length % 2 == 0 and np.random.random() < 0.5:
            ideal_start += 1
        
        # Clamp to valid range
        return max(min_start, min(max_start, ideal_start))
    else:
        # Random position within valid range
        return np.random.randint(min_start, max_start + 1)

class DirichletConditionalFlow:
    def __init__(self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.alpha_spacing = alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        # if the bs is close to the edge of the simplex in one of its entries, then we want the c factor to be 0 for high alphas.
        # That is the rationale for why we return 0s in the case of an overflow.

        beta = scipy.special.beta(alpha, self.K - 1) # betafunction(alpha, K-1)
        beta_div = np.where(bs < 1, beta / ((1 - bs) ** (self.K - 1)), 0)
        beta_div_full = np.where((bs ** (alpha - 1)) > 0, beta_div / (bs ** (alpha - 1)), 0)

        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)

        final = interp * beta_div_full
        return final

def atom14_to_pdb(atom14, aatype, path):
    prots = []
    for i, pos in enumerate(atom14):
        pos = atom14_to_atom37(pos, aatype)
        prots.append(create_full_prot(pos, aatype=aatype))
    with open(path, 'w') as f:
        f.write(prots_to_pdb(prots))


def create_full_prot(
        atom37: np.ndarray,
        aatype=None,
        b_factors=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    atom37_mask = np.sum(np.abs(atom37), axis=-1) > 1e-7
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    chain_index = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index
    )


def prots_to_pdb(prots):
    ss = ''
    for i, prot in enumerate(prots):
        ss += f'MODEL {i}\n'
        prot = protein.to_pdb(prot)
        ss += '\n'.join(prot.split('\n')[2:-3])
        ss += '\nENDMDL\n'
    return ss

def atom1_to_pdb(atom1, aatype, path):
    """Save a coarse-grained MD trajectory with only C-alpha atoms.
    
    Args:
        atom1: Array of shape (num_frames, num_residues, 3) containing C-alpha coordinates
        aatype: Array of amino acid types for each residue
        path: Output path for PDB file
    """
    prots = []
    for i, ca_coords in enumerate(atom1):
        prots.append(create_ca_only_prot(ca_coords, aatype=aatype))
    with open(path, 'w') as f:
        f.write(prots_to_pdb(prots))


def create_ca_only_prot(
        ca_coords: np.ndarray,
        aatype=None,
        b_factors=None,
    ):
    """Create a Protein object with only C-alpha atoms.
    
    Args:
        ca_coords: Array of shape (num_residues, 3) containing C-alpha coordinates
        aatype: Array of amino acid types for each residue
        b_factors: B-factors for each residue (optional)
    
    Returns:
        Protein object with only CA atoms
    """
    assert ca_coords.ndim == 2
    assert ca_coords.shape[-1] == 3
    n = ca_coords.shape[0]
    
    # Create atom37 representation with only CA atoms
    atom37 = np.zeros([n, 37, 3], dtype=np.float32)
    atom37_mask = np.zeros([n, 37], dtype=np.float32)

    ca_index = rc.atom_order['CA']
    atom37[:, ca_index, :] = ca_coords
    atom37_mask[:, ca_index] = 1.0
    
    residue_index = np.arange(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    chain_index = np.zeros(n, dtype=int)
    
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index
    )

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch