import os
import torch

from mdgen.utils import compute_crop_start
from .rigid_utils import Rigid
from .residue_constants import restype_order
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames, \
    atom14_to_ca, ca_to_frames
       
class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1):
        super().__init__()
        self.args = args
        self.repeat = repeat

        # Load the dataset and filter missing files
        df = pd.read_csv(split, index_col='name')
        self.valid_names = []

        for name in df.index:
            full_name = f"{name}_R1" if args.atlas else name  # For ATLAS we only check one replica and assume the rest are there
            file_path = f"{args.data_dir}/{full_name}{args.suffix}.npy"
            if os.path.exists(file_path):
                self.valid_names.append(name)

        self.df = df.loc[self.valid_names]

    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx, env_idx=None):
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres = self.df.seqres[name]
        else:
            name = self.args.overfit_peptide
            seqres = name

        if self.args.atlas:
            i = np.random.randint(1, 4) if self.args.atlas_replica is None else self.args.atlas_replica
            full_name = f"{name}_R{i}"
        else:
            full_name = name
        arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')  # (T, L, 14, 3), in angstrom
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]
        
        frame_start = np.random.choice(np.arange(arr.shape[0] - self.args.num_frames))
        if self.args.overfit_frame:
            frame_start = 0
        end = frame_start + self.args.num_frames
        arr = np.copy(arr[frame_start:end]).astype(np.float32)
        if self.args.copy_frames:
            arr[1:] = arr[0]
        
        seqres = np.array([restype_order[c] for c in seqres])
        ca_coordinates = atom14_to_ca(torch.from_numpy(arr))  # (T, L, 3)
        
        if self.args.sample_radius:
            env_idx = np.random.choice(np.arange(len(seqres))) if env_idx is None else env_idx
            if self.args.overfit_local_env:
                env_idx = 0

            distances = torch.linalg.vector_norm(ca_coordinates[:, env_idx:env_idx+1] - ca_coordinates, dim=-1)  # (T, L)

            cutoff_mask = torch.le(distances, self.args.sample_radius)  # (T, L)
            traj_cutoff_mask = torch.any(cutoff_mask, dim=0)  # (L)

            # Create mapping from old to new index
            old_to_new_idx = torch.cumsum(traj_cutoff_mask, dim=0) - 1  # cumsum increases each time it encounters a True in traj_cutoff_mask
            old_to_new_idx[~traj_cutoff_mask] = -1  # Mark removed positions with -1
            env_idx = old_to_new_idx[env_idx]
            assert env_idx >= 0, "Local environment anchor was removed!"

            arr = arr[:,traj_cutoff_mask]
            ca_coordinates = ca_coordinates[:,traj_cutoff_mask]
            seqres = seqres[traj_cutoff_mask]

        # arr should be in ANGSTROMS
        key_frames = ca_to_frames(ca_coordinates[0:1])
        frames = atom14_to_frames(torch.from_numpy(arr))
        aatype = torch.from_numpy(seqres)[None].expand(self.args.num_frames, -1)
        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()
        
        L = arr.shape[1]
        mask = np.ones(L, dtype=np.float32)

        if self.args.translations_only:
            return {
                'name': full_name,
                'frame_start': frame_start,
                'trans': frames._trans,
                'seqres': seqres,
                'mask': mask, # (L,)
            }

        if self.args.no_frames:
            return {
                'name': full_name,
                'frame_start': frame_start,
                'atom37': atom37,
                'seqres': seqres,
                'mask': restype_atom37_mask[seqres], # (L,)
            }
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        
        torsion_mask = torsion_mask[0]

        if L > self.args.crop:
            if self.args.sample_radius:
                start = compute_crop_start(L, self.args.crop, env_idx, self.args.center_crop)
                if self.args.overfit_crop:
                    start = compute_crop_start(L, self.args.crop, env_idx, True, False)
            else:
                start = np.random.randint(0, L - self.args.crop + 1) if not self.args.overfit_crop else 0
            seqres = seqres[start:start+self.args.crop]
            mask = mask[start:start+self.args.crop]
            ca_coordinates = ca_coordinates[:,start:start+self.args.crop]
            key_frames = key_frames[:,start:start+self.args.crop]
            frames = frames[:,start:start+self.args.crop]
            torsions = torsions[:,start:start+self.args.crop]
            torsion_mask = torsion_mask[start:start+self.args.crop]
        elif L < self.args.crop:
            pad = self.args.crop - L
            seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
            mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
            ca_coordinates = torch.cat([ca_coordinates, torch.zeros((ca_coordinates.shape[0], pad, 3), dtype=torch.float32)], dim=1)
            key_frames = Rigid.cat([
                key_frames, 
                Rigid.identity((key_frames.shape[0], pad), requires_grad=False, fmt='rot_mat')
            ], dim=1)
            frames = Rigid.cat([
                frames, 
                Rigid.identity((self.args.num_frames, pad), requires_grad=False, fmt='rot_mat')
            ], dim=1)
            torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)], dim=1)
            torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])

        item = {
                'name': full_name,
                'frame_start': frame_start,
                'seqres': seqres,
                'mask': mask, # (L,)
        }
        if self.args.c_alpha_only or self.args.ca_cond or self.args.attn_mask_radius:
            item |= {
                'ca_coordinates': ca_coordinates,
            }
        if self.args.c_alpha_only or self.args.ca_cond: 
            item |= {
                'key_frame_rots': key_frames._rots._rot_mats,
                'key_frame_trans': key_frames._trans,
            }
        if not self.args.c_alpha_only:
            item |= {
                'rots': frames._rots._rot_mats,
                'trans': frames._trans,
                'torsions': torsions,
                'torsion_mask': torsion_mask,
            }
        
        return item

