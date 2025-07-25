import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sim_ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default=None, required=True)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--num_frames', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=100)
parser.add_argument('--no_frames', action='store_true')
parser.add_argument('--tps', action='store_true')
parser.add_argument('--xtc', action='store_true')
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
parser.add_argument('--c_alpha_only', action='store_true')
parser.add_argument('--guide_by_known', action='store_true')
parser.add_argument('--ca_cond', action='store_true')
args = parser.parse_args()

import os, torch, mdtraj, tqdm, time
import numpy as np
from mdgen.geometry import atom14_to_ca, atom14_to_frames, atom14_to_atom37, atom37_to_torsions, ca_to_frames
from mdgen.residue_constants import restype_order, restype_atom37_mask
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper
from mdgen.utils import atom14_to_pdb, atom1_to_pdb
from mdgen import residue_constants as rc
import pandas as pd

os.makedirs(args.out_dir, exist_ok=True)


def get_batch_inpaint(name, seqres, num_frames):
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r')

    arr = np.copy(arr[:num_frames]).astype(np.float32)
    ca_coordinates = atom14_to_ca(torch.from_numpy(arr))  # (T, L, 3)
    key_frames = ca_to_frames(ca_coordinates[0:1])
    
    arr = np.copy(arr[0:1]).astype(np.float32)

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)

    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    return {
        'ca_coordinates': ca_coordinates,
        'key_frame_rots': key_frames._rots._rot_mats,
        'key_frame_trans': key_frames._trans,
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'mask': mask, # (L,)
    }

def rollout_inpaint(model, batch):
    expanded_batch = {
        'ca_coordinates': batch['ca_coordinates'],
        'key_frame_rots': batch['key_frame_rots'],
        'key_frame_trans': batch['key_frame_trans'],
        'torsions': batch['torsions'].expand(-1, args.num_frames, -1, -1, -1),
        'torsion_mask': batch['torsion_mask'],
        'trans': batch['trans'].expand(-1, args.num_frames, -1, -1),
        'rots': batch['rots'].expand(-1, args.num_frames, -1, -1, -1),
        'seqres': batch['seqres'],
        'mask': batch['mask'],
    }

    return model.inference(expanded_batch)

def get_batch(name, seqres, num_frames):
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r')

    arr = np.copy(arr[0:1]).astype(np.float32)

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)

    if args.c_alpha_only:
        ca_coordinates = atom14_to_ca(torch.from_numpy(arr))
        key_frames = ca_to_frames(ca_coordinates[0:1])
        return {
            'ca_coordinates': ca_coordinates,
            'key_frame_rots': key_frames._rots._rot_mats,
            'key_frame_trans': key_frames._trans,
            'seqres': seqres,
            'mask': mask,
        }
    
    if args.no_frames:
        return {
            'atom37': atom37,
            'seqres': seqres,
            'mask': restype_atom37_mask[seqres],
        }
        
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'mask': mask, # (L,)
    }

def rollout(model, batch):
    # Blow up batch consisting of a single frame to a full trajectory, 
    # by reusing the frame along the time axis.
    if args.c_alpha_only:
        expanded_batch = {
            'ca_coordinates': batch['ca_coordinates'].expand(-1, args.num_frames, -1, -1),
            'key_frame_rots': batch['key_frame_rots'].expand(-1, args.num_frames, -1, -1, -1),
            'key_frame_trans': batch['key_frame_trans'].expand(-1, args.num_frames, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    elif args.no_frames:
        expanded_batch = {
            'atom37': batch['atom37'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    else:    
        expanded_batch = {
            'torsions': batch['torsions'].expand(-1, args.num_frames, -1, -1, -1),
            'torsion_mask': batch['torsion_mask'],
            'trans': batch['trans'].expand(-1, args.num_frames, -1, -1),
            'rots': batch['rots'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }

    return model.inference(expanded_batch)
    
    
def do(model, name, seqres):
    if args.ca_cond:
        item = get_batch_inpaint(name, seqres, num_frames = model.args.num_frames)
    else:
        item = get_batch(name, seqres, num_frames = model.args.num_frames)
    batch = next(iter(torch.utils.data.DataLoader([item])))

    batch = tensor_tree_map(lambda x: x.cuda(), batch)
    
    all_trajs = []
    start = time.time()
    for _ in tqdm.trange(args.num_rollouts):
        if args.ca_cond:
            traj_out, _ = rollout_inpaint(model, batch)
        else:
            traj_out, _ = rollout(model, batch)
        all_trajs.append(traj_out[0])

    print(time.time() - start)
    all_trajs = torch.cat(all_trajs).cpu().numpy()
    aatype = np.array([restype_order[c] for c in seqres])
    
    path = os.path.join(args.out_dir, f'{name}.pdb')
    if args.c_alpha_only:
        atom1_to_pdb(all_trajs, aatype, path)
    else:
        atom14_to_pdb(all_trajs, aatype, path)

    if args.xtc:
        traj = mdtraj.load(path)
        traj.superpose(traj)
        traj.save(os.path.join(args.out_dir, f'{name}.xtc'))
        traj[0].save(os.path.join(args.out_dir, f'{name}.pdb'))

@torch.no_grad()
def main():
    model = NewMDGenWrapper.load_from_checkpoint(args.sim_ckpt, guide_by_known=args.guide_by_known)
    model.eval().to('cuda')

    df = pd.read_csv(args.split, index_col='name')
    names = np.array(df.index)

    jobs = []
    n_expected = len(names) if not args.pdb_id else len(args.pdb_id)
    for name in names:
        if args.pdb_id and name not in args.pdb_id:
            continue
        if not os.path.exists(f'{args.data_dir}/{name}{args.suffix}.npy'):
            continue
        if len(df.seqres[name]) >= 256:
            print(f'Skipping protein "{name}" with length {len(df.seqres[name])}')
            continue
        jobs.append(name)
    n_not_found = n_expected - len(jobs)
    if n_not_found:
        print(f'Did not find {n_not_found}/{n_expected} molecules '
              f'specified in the split. Skipping those ...')

    for name in jobs:
        do(model, name, df.seqres[name])

main()