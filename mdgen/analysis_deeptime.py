import json
import os

import mdtraj
import numpy as np
import deeptime
from tqdm import tqdm

def get_backbone_angle_identifier(atom_indices, traj):
    """
    Generate an identifier for the angle based on the bond vector in it's rotation axis. 
    """
    atom2 = traj.topology.atom(atom_indices[1])
    atom2_to_angle_name = {
        'N': 'PHI',
        'CA': 'PSI',
        'C': 'OMEGA',
    }
    angle_name = atom2_to_angle_name[atom2.name]
    residue_name = atom2.residue.name
    residue_index = atom2.residue.index
    chainID = atom2.residue.chain.index

    return f'{angle_name} {chainID} {residue_name} {residue_index}'

def get_sidechain_angle_identifier(atom_indices, traj):
    """
    Generate an identifier for the sidechain angle based on the bond vector in it's rotation axis.
    """
    atom2 = traj.topology.atom(atom_indices[1])
    atom2_to_chi_index = {
        'CA': 1,
        'CB': 2,
        'CG': 3,
        'CD': 4,
    }
    chi_index = atom2_to_chi_index[atom2.name]
    residue_name = atom2.residue.name
    residue_index = atom2.residue.index
    chainID = atom2.residue.chain.index

    return f'CHI{chi_index} {chainID} {residue_name} {residue_index}'

def get_featurized_traj(name, sidechains=False, cossin=True):
    traj = mdtraj.load(name + '.xtc', top=name + '.pdb')
    
    # Compute backbone torsions
    phi_indices, phi_angles = mdtraj.compute_phi(traj)
    psi_indices, psi_angles = mdtraj.compute_psi(traj)

    # Create identifiers for the angles
    phi_identifiers = [get_backbone_angle_identifier(indices, traj) for indices in phi_indices]
    psi_identifiers = [get_backbone_angle_identifier(indices, traj) for indices in psi_indices]

    features = [phi_angles, psi_angles]
    feature_labels = [phi_identifiers, psi_identifiers]

    # Compute sidechain torsions if requested
    if sidechains:
        chi1_indices, chi1_angles = mdtraj.compute_chi1(traj)
        chi2_indices, chi2_angles = mdtraj.compute_chi2(traj)
        chi3_indices, chi3_angles = mdtraj.compute_chi3(traj)
        chi4_indices, chi4_angles = mdtraj.compute_chi4(traj)

        # Create identifiers for the sidechain angles
        chi1_identifiers = [get_sidechain_angle_identifier(indices, traj) for indices in chi1_indices]
        chi2_identifiers = [get_sidechain_angle_identifier(indices, traj) for indices in chi2_indices]
        chi3_identifiers = [get_sidechain_angle_identifier(indices, traj) for indices in chi3_indices]
        chi4_identifiers = [get_sidechain_angle_identifier(indices, traj) for indices in chi4_indices]
        
        sidechain_torsions = [chi1_angles, chi2_angles, chi3_angles, chi4_angles]
        sidechain_labels = [chi1_identifiers, chi2_identifiers, chi3_identifiers, chi4_identifiers]

        for angles, labels in zip(sidechain_torsions, sidechain_labels):
            if angles.shape[1] > 0:
                features.append(angles)
                feature_labels.append(labels)

    # Combine features
    traj_features = np.concatenate(features, axis=1)
    feature_labels = [label for sublist in feature_labels for label in sublist]  # Flatten the list of labels

    # If cossin is requested, apply cosine and sine transformations
    if cossin:
        cos_features = np.cos(traj_features)
        sin_features = np.sin(traj_features)
        stacked_cossin = np.stack([cos_features, sin_features], axis=2)
        traj_features = stacked_cossin.reshape(traj_features.shape[0], -1)
        label_tuples = [(f'COS({label})', f'SIN({label})') for label in feature_labels]
        feature_labels = [item for sublist in label_tuples for item in sublist]  # Flatten the list of tuples

    return feature_labels, traj_features

def get_featurized_omega_traj(name, cossin=True):
    """
    Load a trajectory and compute the omega angles, returning the features and their labels.
    """
    traj = mdtraj.load(name + '.xtc', top=name + '.pdb')

    # Compute omega angles
    omega_indices, omega_angles = mdtraj.compute_omega(traj)

    # Create identifiers for the omega angles
    omega_identifiers = [get_backbone_angle_identifier(indices, traj) for indices in omega_indices]

    # Combine features
    traj_features = omega_angles
    feature_labels = omega_identifiers

    # If cossin is requested, apply cosine and sine transformations
    if cossin:
        cos_features = np.cos(traj_features)
        sin_features = np.sin(traj_features)
        stacked_cossin = np.stack([cos_features, sin_features], axis=2)
        traj_features = stacked_cossin.reshape(traj_features.shape[0], -1)
        label_tuples = [(f'COS({label})', f'SIN({label})') for label in feature_labels]
        feature_labels = [item for sublist in label_tuples for item in sublist]  # Flatten the list of tuples

    return feature_labels, traj_features

def get_tica(traj, lag=1000):
    tica = deeptime.decomposition.TICA(lagtime=lag, scaling='kinetic_map', var_cutoff=0.95)
    tica.fit(traj)
    return tica, tica.transform(traj)

def normalize_sequence(sequence):
    """
    Converts a sequence into its canonical representation.
    E.g., [11, 11, 88, 20] -> [0, 0, 1, 2]
    """
    mapping = {}
    next_canonical_id = 0
    normalized = []
    
    for item in sequence:
        if item not in mapping:
            mapping[item] = next_canonical_id
            next_canonical_id += 1
        normalized.append(mapping[item])
        
    return normalized

def get_kmeans(traj):
    kmeans = deeptime.clustering.KMeans(n_clusters=100, max_iter=100, fixed_seed=137)
    kmeans.fit(traj)
    return kmeans, kmeans.transform(traj)

def get_msm(traj, lag=1000, nstates=10):
    msm = deeptime.markov.msm.MaximumLikelihoodMSM(lagtime=lag).fit(traj).fetch_model()
    pcca = msm.pcca(n_metastable_sets=nstates)
    assert len(pcca.assignments) == 100
    cmsm = deeptime.markov.msm.MaximumLikelihoodMSM(lagtime=lag).fit(pcca.assignments[traj]).fetch_model()
    return msm, pcca, cmsm

def discretize(traj, kmeans, pcca):
    discrete_traj = pcca.assignments[kmeans.transform(traj)]
    return discrete_traj

def get_full_transition_matrix(msm):
    """
    Converts the MSM transition matrix of active states into a full transition matrix including inactive states. 
    """
    n_states_full = msm.count_model.n_states_full
    msm_transition_matrix = np.eye(n_states_full)
    for a, i in enumerate(msm.state_symbols()):
        for b, j in enumerate(msm.state_symbols()):
            msm_transition_matrix[i,j] = msm.transition_matrix[a,b]
    return msm_transition_matrix


# The rest of the functions (load_tps_ensemble, sample_tp, etc.) do not directly use pyemma and can remain as they are.
# However, load_tps_ensemble needs to be redefined here to use the new get_featurized_traj
def load_tps_ensemble(name, directory):
    metadata = json.load(open(os.path.join(directory, f'{name}_metadata.json'),'rb'))
    all_feats = []
    all_traj = []
    for i, meta_dict in tqdm(enumerate(metadata)):
        feats, traj = get_featurized_traj(f'{directory}/{name}_{i}', sidechains=True)
        all_feats.append(feats)
        all_traj.append(traj)
    return all_feats, all_traj