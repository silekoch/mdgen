import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mddir', type=str, default='share/4AA_sims')
parser.add_argument('--pdbdir', type=str, required=True) 
parser.add_argument('--save', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--save_name', type=str, default='out.pkl')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--no_msm', action='store_true')
parser.add_argument('--no_decorr', action='store_true')
parser.add_argument('--no_traj_msm', action='store_true')
parser.add_argument('--truncate', type=int, default=None)
parser.add_argument('--truncate_ref', type=int, default=None)
parser.add_argument('--stride_ref', type=int, default=1)
parser.add_argument('--msm_lag', type=int, default=10)
parser.add_argument('--ito', action='store_true')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--c_alpha_only', action='store_true')
parser.add_argument('--max_plot_features', type=int, default=6)

args = parser.parse_args()

import mdgen.analysis_deeptime
import mdgen.plots1d
import mdgen.utils
import deeptime
import tqdm, os, pickle
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf, acf
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_jsd_by_feature_type(jsd_dict, ax):
    """Plot JSD values grouped by feature type"""
    # Separate features by type
    phi_features = {k: v for k, v in jsd_dict.items() if 'PHI' in k and '|' not in k}
    psi_features = {k: v for k, v in jsd_dict.items() if 'PSI' in k and '|' not in k}
    chi_features = {k: v for k, v in jsd_dict.items() if 'CHI' in k}
    ca_dihedral_features = {k: v for k, v in jsd_dict.items() if 'CA_DIHEDRAL' in k}
    ca_bond_features = {k: v for k, v in jsd_dict.items() if 'CA_BOND' in k}
    ca_angle_features = {k: v for k, v in jsd_dict.items() if 'CA_ANGLE' in k}
    tica_features = {k: v for k, v in jsd_dict.items() if 'TICA' in k}
    rama_features = {k: v for k, v in jsd_dict.items() if '|' in k}
    
    feature_types = []
    mean_jsds = []
    
    for name, features in [('PHI', phi_features), ('PSI', psi_features), 
                          ('CHI', chi_features), ('CA Dih', ca_dihedral_features),
                          ('CA Bond', ca_bond_features), ('CA Angle', ca_angle_features),
                          ('TICA', tica_features), ('Rama', rama_features)]:
        if features:
            feature_types.append(name)
            mean_jsds.append(np.mean(list(features.values())))
    
    if feature_types:
        bars = ax.bar(feature_types, mean_jsds, color=colors[:len(feature_types)])
        ax.set_ylabel('Mean Jensen-Shannon Divergence')
        ax.set_title('JSD by Feature Type')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, mean_jsds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)

def sample_features_for_plot(ref_data, traj_data, max_features=None):
    """Sample same random subset of features from both datasets for plotting"""
    if ref_data.shape[1] <= max_features:
        return ref_data, traj_data, None
    indices = np.random.choice(ref_data.shape[1], max_features, replace=False)
    return ref_data[:, indices], traj_data[:, indices], indices

def main(name):
    out = {}
    np.random.seed(137)
    fig, axs = plt.subplots(6, 4, figsize=(20, 30))

    ### BACKBONE torsion marginals PLOT ONLY
    if args.plot and not args.c_alpha_only:
        feats, traj = mdgen.analysis_deeptime.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=False, cossin=False)
        if args.truncate: traj = traj[:args.truncate]
        feats, ref = mdgen.analysis_deeptime.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=False, cossin=False)
        if args.truncate_ref: ref = ref[:args.truncate_ref:args.stride_ref]
        feats = [feat.split(' ', 2)[0] + ' ' + feat.split(' ', 2)[-1] for feat in feats]  # Cut out chain ID
        ref_plot, traj_plot, indices = sample_features_for_plot(ref, traj, args.max_plot_features)
        plot_feats = [feats[i] for i in indices] if indices is not None else feats
        mdgen.plots1d.plot_feature_histograms(ref_plot, ax=axs[0,0], color=colors[0])
        mdgen.plots1d.plot_feature_histograms(traj_plot, feature_labels=plot_feats, ax=axs[0,0], color=colors[1])
        axs[0,0].set_title('BB torsions')
        axs[0,0].set_xlabel('Angle (rad)')

        # Omega plotted separately as sanity check
        omega_feats, omega_traj = mdgen.analysis_deeptime.get_featurized_omega_traj(f'{args.pdbdir}/{name}', cossin=False)
        if args.truncate: omega_traj = omega_traj[:args.truncate]
        omega_feats, omega_ref = mdgen.analysis_deeptime.get_featurized_omega_traj(f'{args.mddir}/{name}/{name}', cossin=False)
        if args.truncate_ref: omega_ref = omega_ref[:args.truncate_ref:args.stride_ref]
        omega_ref_plot, omega_traj_plot, omega_indices = sample_features_for_plot(omega_ref, omega_traj, args.max_plot_features)
        omega_feats = [feat.split(' ', 2)[-1] for feat in omega_feats]  # Cut out angle name and chain ID
        plot_omega_feats = [omega_feats[i] for i in omega_indices] if omega_indices is not None else omega_feats
        mdgen.plots1d.plot_feature_histograms(omega_ref_plot, ax=axs[1,0], color=colors[0])
        mdgen.plots1d.plot_feature_histograms(omega_traj_plot, feature_labels=plot_omega_feats, ax=axs[1,0], color=colors[1])
        axs[1,0].set_title('OMEGA torsions')
        axs[1,0].set_xlabel('Angle (rad)')

    if args.plot:
        # CA dihedrals
        ca_torsion_feats, ca_torsion_traj = mdgen.analysis_deeptime.get_featurized_ca_traj(f'{args.pdbdir}/{name}', cossin=False)
        if args.truncate: ca_torsion_traj = ca_torsion_traj[:args.truncate]
        ca_torsion_feats, ca_torsion_ref = mdgen.analysis_deeptime.get_featurized_ca_traj(f'{args.mddir}/{name}/{name}', cossin=False)
        if args.truncate_ref: ca_torsion_ref = ca_torsion_ref[:args.truncate_ref:args.stride_ref]
        ca_torsion_ref_plot, ca_torsion_traj_plot, _ = sample_features_for_plot(ca_torsion_ref, ca_torsion_traj, args.max_plot_features)
        mdgen.plots1d.plot_feature_histograms(ca_torsion_ref_plot, ax=axs[3,0], color=colors[0])
        mdgen.plots1d.plot_feature_histograms(ca_torsion_traj_plot, ax=axs[3,0], color=colors[1])
        axs[3,0].set_title('CA dihedrals')
        axs[3,0].set_xlabel('Angle (rad)')

        # CA bond lengths
        ca_bond_feats, ca_bond_traj = mdgen.analysis_deeptime.get_featurized_ca_bonds_traj(f'{args.pdbdir}/{name}')
        if args.truncate: ca_bond_traj = ca_bond_traj[:args.truncate]
        _, ca_bond_ref = mdgen.analysis_deeptime.get_featurized_ca_bonds_traj(f'{args.mddir}/{name}/{name}')
        if args.truncate_ref: ca_bond_ref = ca_bond_ref[:args.truncate_ref:args.stride_ref]
        
        ca_bond_ref_plot, ca_bond_traj_plot, _ = sample_features_for_plot(ca_bond_ref, ca_bond_traj, args.max_plot_features)
        mdgen.plots1d.plot_feature_histograms(ca_bond_ref_plot, ax=axs[3,1], color=colors[0])
        mdgen.plots1d.plot_feature_histograms(ca_bond_traj_plot, ax=axs[3,1], color=colors[1])
        axs[3,1].set_title('CA Bond Lengths')
        axs[3,1].set_xlabel('Distance (nm)')
        
        # CA bond angles
        ca_angle_feats, ca_angle_traj = mdgen.analysis_deeptime.get_featurized_ca_angles_traj(f'{args.pdbdir}/{name}', cossin=False)
        if args.truncate: ca_angle_traj = ca_angle_traj[:args.truncate]
        _, ca_angle_ref = mdgen.analysis_deeptime.get_featurized_ca_angles_traj(f'{args.mddir}/{name}/{name}', cossin=False)
        if args.truncate_ref: ca_angle_ref = ca_angle_ref[:args.truncate_ref:args.stride_ref]
        
        ca_angle_ref_plot, ca_angle_traj_plot, _ = sample_features_for_plot(ca_angle_ref, ca_angle_traj, args.max_plot_features)
        mdgen.plots1d.plot_feature_histograms(ca_angle_ref_plot, ax=axs[3,2], color=colors[0])
        mdgen.plots1d.plot_feature_histograms(ca_angle_traj_plot, ax=axs[3,2], color=colors[1])
        axs[3,2].set_title('CA Bond Angles')
        axs[3,2].set_xlabel('Angle (rad)')
    
    ### JENSEN SHANNON DISTANCES ON ALL TORSIONS
    out['JSD'] = {}
    if not args.c_alpha_only:
        feats, traj = mdgen.analysis_deeptime.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=True, cossin=False)
        if args.truncate: traj = traj[:args.truncate]
        feats, ref = mdgen.analysis_deeptime.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True, cossin=False)
        if args.truncate_ref: ref = ref[:args.truncate_ref:args.stride_ref]

        out['features'] = feats

        for i, feat in enumerate(feats):
            ref_p = np.histogram(ref[:,i], range=(-np.pi, np.pi), bins=100)[0]
            traj_p = np.histogram(traj[:,i], range=(-np.pi, np.pi), bins=100)[0]
            out['JSD'][feat] = jensenshannon(ref_p, traj_p)

        #### JENSEN SHANNON DISTANCE ON RAMACHANDRAN PLOTS
        feats_indexed = enumerate(feats)

        # Sort by residue index first, then angle name.
        feats_sorted = sorted(feats_indexed, key=lambda x: (int(x[1].split()[-1]), x[1].split()[0]))  # TODO have to do this only for backbone angles (phi and psi, not chi)

        # Filter out chi angles and other stuff that might creeped in 
        feats_filtered = list(filter(lambda x: x[1].split()[0] in ['PHI', 'PSI'], feats_sorted))

        # Cut off first and last entry, it's a lonely PSI angle in front and a lonely PHI at the end
        feats_trimmed = feats_filtered[1:-1]

        # Now the feats_trimmed should be alternating PHI and PSI grouped by residue index,
        # e.g.: [(0, 'PHI 0 ALA 1'), (4, 'PSI 0 ALA 1'), (1, 'PHI 0 ASP 2'), (5, 'PSI 0 ASP 2')]
        feat_pairs = mdgen.utils.batched(feats_trimmed, n=2)

        for (i, i_name), (j, j_name) in feat_pairs:
            ref_p = np.histogram2d(*ref[:,[i, j]].T, range=((-np.pi, np.pi),(-np.pi,np.pi)), bins=50)[0]
            traj_p = np.histogram2d(*traj[:,[i, j]].T, range=((-np.pi, np.pi),(-np.pi,np.pi)), bins=50)[0]
            out['JSD']['|'.join([i_name, j_name])] = jensenshannon(ref_p.flatten(), traj_p.flatten())

    ### JENSEN SHANNON DISTANCE ON C-ALPHA
    out['features_ca_torsion'] = ca_torsion_feats
    out['features_ca_bonds'] = ca_bond_feats
    out['features_ca_angles'] = ca_angle_feats

    for i, ca_feat in enumerate(ca_torsion_feats):
        ref_p = np.histogram(ca_torsion_ref[:,i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(ca_torsion_traj[:,i], range=(-np.pi, np.pi), bins=100)[0]
        out['JSD'][ca_feat] = jensenshannon(ref_p, traj_p)

    ### JENSEN SHANNON DISTANCE ON CA BONDS
    for i, bond_feat in enumerate(ca_bond_feats):
        # Use appropriate range for bond lengths (typically 0.35-0.4 nm for CA-CA)
        ref_p = np.histogram(ca_bond_ref[:,i], range=(0.3, 0.5), bins=100)[0]
        traj_p = np.histogram(ca_bond_traj[:,i], range=(0.3, 0.5), bins=100)[0]
        out['JSD'][bond_feat] = jensenshannon(ref_p, traj_p)

    ### JENSEN SHANNON DISTANCE ON CA ANGLES
    for i, angle_feat in enumerate(ca_angle_feats):
        ref_p = np.histogram(ca_angle_ref[:,i], range=(0, np.pi), bins=100)[0]
        traj_p = np.histogram(ca_angle_traj[:,i], range=(0, np.pi), bins=100)[0]
        out['JSD'][angle_feat] = jensenshannon(ref_p, traj_p)

    ############ Torsion decorrelations
    out['md_decorrelation'] = {}
    out['our_decorrelation'] = {}
    if args.no_decorr or args.c_alpha_only:
        pass
    else:
        #### Reference backbone decorrelations
        for i, feat in enumerate(feats):
            nlag = 100000 if not args.truncate_ref else args.truncate_ref / args.stride_ref - 2
            autocorr = acovf(np.sin(ref[:,i]), demean=False, adjusted=True, nlag=nlag) + acovf(np.cos(ref[:,i]), demean=False, adjusted=True, nlag=nlag)
            baseline = np.sin(ref[:,i]).mean()**2 + np.cos(ref[:,i]).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if 'PHI' in feat or 'PSI' in feat:
                axs[0,1].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
            else:
                axs[0,2].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
    
            out['md_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
           
        axs[0,1].set_title('Backbone decorrelation')
        axs[0,2].set_title('Sidechain decorrelation')
        axs[0,1].set_xscale('log')
        axs[0,2].set_xscale('log')
    
        #### Generated backbone decorrelations
        for i, feat in enumerate(feats):
            
            autocorr = acovf(np.sin(traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000) + acovf(np.cos(traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000)
            baseline = np.sin(traj[:,i]).mean()**2 + np.cos(traj[:,i]).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if 'PHI' in feat or 'PSI' in feat:
                axs[1,1].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
            else:
                axs[1,2].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
    
            out['our_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
    
        axs[1,1].set_title('Backbone decorrelation')
        axs[1,2].set_title('Sidechain decorrelation')
        axs[1,1].set_xscale('log')
        axs[1,2].set_xscale('log')

    if args.no_decorr:
        pass
    else:
        for i, feat in enumerate(ca_torsion_feats):
            nlag = 100000 if not args.truncate_ref else args.truncate_ref / args.stride_ref - 2
            autocorr = acovf(np.sin(ca_torsion_ref[:,i]), demean=False, adjusted=True, nlag=nlag) + acovf(np.cos(ca_torsion_ref[:,i]), demean=False, adjusted=True, nlag=nlag)
            baseline = np.sin(ca_torsion_ref[:,i]).mean()**2 + np.cos(ca_torsion_ref[:,i]).mean()**2
            lags = 1 + np.arange(len(autocorr))
            axs[4,0].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
            out['md_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
           
        axs[4,0].set_title('Ref CA torsion decorrelation')
        axs[4,0].set_xscale('log')

        for i, feat in enumerate(ca_torsion_feats):
            autocorr = acovf(np.sin(ca_torsion_traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000) + acovf(np.cos(ca_torsion_traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000)
            baseline = np.sin(ca_torsion_traj[:,i]).mean()**2 + np.cos(ca_torsion_traj[:,i]).mean()**2
            lags = 1 + np.arange(len(autocorr))
            axs[5,0].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
    
            out['our_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
    
        axs[5,0].set_title('Gen CA torsion decorrelation')
        axs[5,0].set_xscale('log')

        # Bond length decorrelations (MD reference)
        for i, bond_feat in enumerate(ca_bond_feats):
            nlag = 100000 if not args.truncate_ref else args.truncate_ref // args.stride_ref - 2
            # For bond lengths, we use the values directly (no sin/cos transformation)
            autocorr = acovf(ca_bond_ref[:,i], demean=False, adjusted=True, nlag=nlag)
            baseline = ca_bond_ref[:,i].mean()**2
            lags = 1 + np.arange(len(autocorr))
            axs[4,1].plot(lags, (autocorr - baseline) / (ca_bond_ref[:,i].var()), color=colors[i%len(colors)])
            out['md_decorrelation'][bond_feat] = (autocorr.astype(np.float16) - baseline) / (ca_bond_ref[:,i].var())
        
        axs[4,1].set_title('CA Bond Length Decorrelation (MD)')
        axs[4,1].set_xscale('log')
        
        # Bond length decorrelations (generated trajectories)
        for i, bond_feat in enumerate(ca_bond_feats):
            autocorr = acovf(ca_bond_traj[:,i], demean=False, adjusted=True, nlag=1 if args.ito else 1000)
            baseline = ca_bond_traj[:,i].mean()**2
            lags = 1 + np.arange(len(autocorr))
            axs[5,1].plot(lags, (autocorr - baseline) / (ca_bond_traj[:,i].var()), color=colors[i%len(colors)])
            out['our_decorrelation'][bond_feat] = (autocorr.astype(np.float16) - baseline) / (ca_bond_traj[:,i].var())
        
        axs[5,1].set_title('CA Bond Length Decorrelation (Generated)')
        axs[5,1].set_xscale('log')
        
        # Bond angle decorrelations (MD reference)
        for i, angle_feat in enumerate(ca_angle_feats):
            nlag = 100000 if not args.truncate_ref else args.truncate_ref // args.stride_ref - 2
            # For angles, use sin/cos transformation like other angular features
            autocorr = acovf(np.sin(ca_angle_ref[:,i]), demean=False, adjusted=True, nlag=nlag) + acovf(np.cos(ca_angle_ref[:,i]), demean=False, adjusted=True, nlag=nlag)
            baseline = np.sin(ca_angle_ref[:,i]).mean()**2 + np.cos(ca_angle_ref[:,i]).mean()**2
            lags = 1 + np.arange(len(autocorr))
            axs[4,2].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
            out['md_decorrelation'][angle_feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
        
        axs[4,2].set_title('CA Bond Angle Decorrelation (MD)')
        axs[4,2].set_xscale('log')
        
        # Bond angle decorrelations (generated trajectories)
        for i, angle_feat in enumerate(ca_angle_feats):
            autocorr = acovf(np.sin(ca_angle_traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000) + acovf(np.cos(ca_angle_traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000)
            baseline = np.sin(ca_angle_traj[:,i]).mean()**2 + np.cos(ca_angle_traj[:,i]).mean()**2
            lags = 1 + np.arange(len(autocorr))
            axs[5,2].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
            out['our_decorrelation'][angle_feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
        
        axs[5,2].set_title('CA Bond Angle Decorrelation (Generated)')
        axs[5,2].set_xscale('log')

    if not args.c_alpha_only:
        ####### TICA #############
        feats, traj = mdgen.analysis_deeptime.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=True, cossin=True)
        if args.truncate: traj = traj[:args.truncate]
        feats, ref = mdgen.analysis_deeptime.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True, cossin=True)

        tica, _ = mdgen.analysis_deeptime.get_tica(ref)
        ref_tica = tica.transform(ref)
        traj_tica = tica.transform(traj)
        
        tica_0_min = min(ref_tica[:,0].min(), traj_tica[:,0].min())
        tica_0_max = max(ref_tica[:,0].max(), traj_tica[:,0].max())

        tica_1_min = min(ref_tica[:,1].min(), traj_tica[:,1].min())
        tica_1_max = max(ref_tica[:,1].max(), traj_tica[:,1].max())
        
        ref_p = np.histogram(ref_tica[:,0], range=(tica_0_min, tica_0_max), bins=100)[0]
        traj_p = np.histogram(traj_tica[:,0], range=(tica_0_min, tica_0_max), bins=100)[0]
        out['JSD']['TICA-0'] = jensenshannon(ref_p, traj_p)

        ref_p = np.histogram2d(*ref_tica[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]
        traj_p = np.histogram2d(*traj_tica[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]
        out['JSD']['TICA-0,1'] = jensenshannon(ref_p.flatten(), traj_p.flatten())
        
        #### 1,0, 1,1 TICA FES
        if args.plot:
            fes_ref = deeptime.util.energy2d(ref_tica[::100, 0], ref_tica[::100, 1])
            fes_traj = deeptime.util.energy2d(traj_tica[:, 0], traj_tica[:, 1])
            deeptime.plots.plot_energy2d(fes_ref, ax=axs[2,0], cbar=False, contourf_kws={'cmap': 'nipy_spectral'})
            deeptime.plots.plot_energy2d(fes_traj, ax=axs[2,1], cbar=False, contourf_kws={'cmap': 'nipy_spectral'})
            axs[2,0].set_title('TICA FES (MD)')
            axs[2,1].set_title('TICA FES (ours)')


        ####### TICA decorrelation ########
        if args.no_decorr:
            pass
        else:
            # x, adjusted=False, demean=True, fft=True, missing='none', nlag=None
            autocorr = acovf(ref_tica[:,0], nlag=100000, adjusted=True, demean=False)
            out['md_decorrelation']['tica'] = autocorr.astype(np.float16)
            if args.plot:
                axs[0,3].plot(autocorr)
                axs[0,3].set_title('MD TICA')
            
        
            autocorr = acovf(traj_tica[:,0], nlag=1 if args.ito else 1000, adjusted=True, demean=False)
            out['our_decorrelation']['tica'] = autocorr.astype(np.float16)
            if args.plot:
                axs[1,3].plot(autocorr)
                axs[1,3].set_title('Traj TICA')

        ###### Markov state model stuff #################
        if not args.no_msm:
            kmeans, ref_kmeans = mdgen.analysis_deeptime.get_kmeans(tica.transform(ref))
            try:
                msm, pcca, cmsm = mdgen.analysis_deeptime.get_msm(ref_kmeans, nstates=10)
        
                out['kmeans'] = kmeans
                out['msm'] = msm
                out['pcca'] = pcca
                out['cmsm'] = cmsm
            
                traj_discrete = mdgen.analysis_deeptime.discretize(tica.transform(traj), kmeans, pcca)
                ref_discrete = mdgen.analysis_deeptime.discretize(tica.transform(ref), kmeans, pcca)
                out['traj_metastable_probs'] = (traj_discrete == np.arange(10)[:,None]).mean(1)
                out['ref_metastable_probs'] = (ref_discrete == np.arange(10)[:,None]).mean(1)
                ######### 
            
                msm_transition_matrix = mdgen.analysis_deeptime.get_full_transition_matrix(cmsm)

                out['msm_transition_matrix'] = msm_transition_matrix
                out['pcca_pi'] = pcca.coarse_grained_stationary_probability
            
                msm_pi = np.zeros(10)
                msm_pi[cmsm.state_symbols()] = cmsm.stationary_distribution

                out['msm_pi'] = msm_pi
                
                if args.no_traj_msm:
                    pass
                else:
                    traj_msm = deeptime.markov.msm.MaximumLikelihoodMSM(lagtime=args.msm_lag).fit(traj_discrete).fetch_model()
                    out['traj_msm'] = traj_msm
            
                    traj_transition_matrix = mdgen.analysis_deeptime.get_full_transition_matrix(traj_msm)
                    out['traj_transition_matrix'] = traj_transition_matrix
                
                    traj_pi = np.zeros(10)
                    traj_pi[traj_msm.state_symbols()] = traj_msm.stationary_distribution
                    out['traj_pi'] = traj_pi
                    
            except Exception as e:
                print('ERROR', e, name, flush=True)
    
    # Calculate backbone JSD as single performance metric
    if args.c_alpha_only:
        # For CA-only: use CA dihedral angles as primary metric
        backbone_jsd = np.mean([v for k, v in out['JSD'].items() 
                               if 'CA_DIHEDRAL' in k])
        metric_name = "CA Dihedral JSD"
    else:
        # For all-atom: use PHI/PSI angles
        backbone_jsd = np.mean([v for k, v in out['JSD'].items() 
                               if any(x in k for x in ['PHI', 'PSI']) and '|' not in k])
        metric_name = "Backbone JSD"
    
    out['backbone_jsd'] = backbone_jsd
    print(f"{metric_name} for {name}: {backbone_jsd:.4f}")
    
    if args.plot:
        # Add JSD summary plot
        plot_jsd_by_feature_type(out['JSD'], axs[2,3])
        
        # Add backbone JSD as text annotation
        fig.suptitle(f'{name} - {metric_name}: {backbone_jsd:.4f}', fontsize=16, y=0.98)
        
        fig.savefig(f'{args.pdbdir}/{name}.pdf')
    
    return name, out

if args.pdb_id:
    pdb_id = args.pdb_id
else:
    pdb_id = [nam.split('.')[0] for nam in os.listdir(args.pdbdir) if '.pdb' in nam and not '_traj' in nam]
pdb_id = [nam for nam in pdb_id if os.path.exists(f'{args.pdbdir}/{nam}.xtc')]
print('number of trajectories', len(pdb_id))

    
if args.num_workers > 1:
    p = Pool(args.num_workers)
    p.__enter__()
    __map__ = p.imap
else:
    __map__ = map
out = dict(tqdm.tqdm(__map__(main, pdb_id), total=len(pdb_id)))
if args.num_workers > 1:
    p.__exit__(None, None, None)

if args.save:
    with open(f"{args.pdbdir}/{args.save_name}", 'wb') as f:
        f.write(pickle.dumps(out))


