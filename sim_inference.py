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
args = parser.parse_args()

import os, torch, mdtraj, tqdm, time
import numpy as np
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order, restype_atom37_mask
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper
from mdgen.utils import atom14_to_pdb, atom1_to_pdb
from mdgen import residue_constants as rc
import pandas as pd




os.makedirs(args.out_dir, exist_ok=True)



def get_batch(name, seqres, num_frames):
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r')

    if not args.tps: # else keep all frames
        arr = np.copy(arr[0:1]).astype(np.float32)

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)

    if args.c_alpha_only:
        return {
            'c_alpha_positions': atom37[:, :, rc.atom_order['CA']],
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
    if args.c_alpha_only:
        expanded_batch = {
            'trans': batch['c_alpha_positions'].expand(-1, args.num_frames, -1, -1),
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
    if args.c_alpha_only:
        atom1, _ = model.inference(expanded_batch)
    else:
        atom14, _ = model.inference(expanded_batch)
    new_batch = {**batch}

    if args.c_alpha_only:
        new_batch['trans'] = atom1
        return atom1, new_batch
    elif args.no_frames:
        new_batch['atom37'] = torch.from_numpy(
            atom14_to_atom37(atom14[:,-1].cpu(), batch['seqres'][0].cpu())
        ).cuda()[:,None].float()
    else:
        frames = atom14_to_frames(atom14[:,-1])
        new_batch['trans'] = frames._trans[None]
        new_batch['rots'] = frames._rots._rot_mats[None]
        atom37 = atom14_to_atom37(atom14[0,-1].cpu(), batch['seqres'][0].cpu())
        torsions, _ = atom37_to_torsions(atom37, batch['seqres'][0].cpu())
        new_batch['torsions'] = torsions[None, None].cuda()

    return atom14, new_batch
    
    
def do(model, name, seqres):
    item = get_batch(name, seqres, num_frames = model.args.num_frames)
    batch = next(iter(torch.utils.data.DataLoader([item])))

    batch = tensor_tree_map(lambda x: x.cuda(), batch)  
    
    all_atom14 = []
    start = time.time()
    for _ in tqdm.trange(args.num_rollouts):
        # If c_alpha_only, this is actually atom1 not atom14.
        atom14, batch = rollout(model, batch)
        all_atom14.append(atom14)

    print(time.time() - start)
    all_atom14 = torch.cat(all_atom14, 1)
    
    path = os.path.join(args.out_dir, f'{name}.pdb')
    if args.c_alpha_only:
        atom1_to_pdb(all_atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)
    else:
        atom14_to_pdb(all_atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

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
        jobs.append(name)
    n_not_found = n_expected - len(jobs)
    if n_not_found:
        print(f'Did not find {n_not_found}/{n_expected} molecules '
              f'specified in the split. Skipping those ...')

    for name in jobs:
        do(model, name, df.seqres[name])

main()