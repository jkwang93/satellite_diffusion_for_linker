import re

import matplotlib.pyplot
from rdkit import Chem
import os
import numpy as np
import torch
from rdkit.Chem import AllChem, rdMolTransforms, Draw
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse
import collections
import pickle
import os
import json
from tqdm import tqdm
# from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
from zinc.analyze import check_stability
from zinc.rdkit_functions import BasicMolecularMetrics
import configs.datasets_config

atomic_number_list = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]
inverse = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 13: 6, 14: 7, 15: 8, 16: 9, 17: 10, 33: 11, 35: 12, 53: 13,
           80: 14, 83: 15}
# atom_name = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']
atom_name= ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'N', 'N', 'O', 'O', 'S', 'S', 'S']

n_atom_types = len(atomic_number_list)
n_bond_types = 4


# 芳香键的数量比例
def Calc_ARR(mh):
    m = Chem.RemoveHs(mh)
    num_bonds = m.GetNumBonds()
    num_aromatic_bonds = 0
    for bond in m.GetBonds():
        if bond.GetIsAromatic():
            num_aromatic_bonds += 1
    ARR = num_aromatic_bonds / num_bonds
    return ARR


def Calc_AROM(mh):
    m = Chem.RemoveHs(mh)
    ring_info = m.GetRingInfo()
    atoms_in_rings = ring_info.AtomRings()
    num_aromatic_ring = 0
    for ring in atoms_in_rings:
        aromatic_atom_in_ring = 0
        for atom_id in ring:
            atom = m.GetAtomWithIdx(atom_id)
            if atom.GetIsAromatic():
                aromatic_atom_in_ring += 1
        if aromatic_atom_in_ring == len(ring):
            num_aromatic_ring += 1
    return num_aromatic_ring


def AROM_num():


    path = './data/molecules_zinc_valid_final.json'  # train/valid/test文件格式一致
    with open(path, 'r') as f:
        train_data = json.load(f)
    AROM_num = []
    for idx, line in enumerate(train_data):
        try:
            print(idx)
            smiles_in = line['smiles_in']

            smiles = line['smiles_out']
            mol_out = Chem.MolFromSmiles(smiles)

            smiles_in_list = smiles_in.split('.')
            for smiles_in in smiles_in_list:
                # remove [*]
                smiles_in = re.sub('\[\*.\d\]', '', smiles_in)
                smiles_in = smiles_in.replace('()', '')
                # smiles_frag = Chem.GetMolFrags(smiles_in,asMols=True)
                mol_in = Chem.MolFromSmarts(smiles_in)
                mol_out = Chem.DeleteSubstructs(mol_out, mol_in)

                # Draw rdkit
                # matches = mol_out.GetSubstructMatches(mol_in)
                # from rdkit.Chem import AllChem, Draw
                # draw = Draw.MolToImage(mol_out, (600, 600), highlightAtoms=matches[0],dpi=600)
                # draw.save('./test_fig/'+smiles_in[:5]+'.jpg')

            linker_smiles = Chem.MolToSmiles(mol_out)
            AROM_num.append(Calc_AROM(mol_out))

        except Exception as e:
            print(e)

    np.save('data/geom/AROM_num.npy', AROM_num)


def cal_bond_dist(args):
    '''计算同一个原子内，相同bond的标准差，均值。以及linker里面的bond与fragment之间的关系（todo）'''
    # bond_dict = collections.defaultdict(list)
    # bond_length_dict_rdkit = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}
    bond_dict = {}
    all_bond_dict = {}
    have_saved_smiles=[]
    repeat_smiles = 0

    path = './data/molecules_zinc_train_final.json'  # train/valid/test文件格式一致
    with open(path, 'r') as f:
        train_data = json.load(f)
    for idx, line in enumerate(train_data):
        try:
            print(idx)
            smiles = line['smiles_out']
            if smiles in have_saved_smiles:
                repeat_smiles+=1
                continue
            else:
                have_saved_smiles.append(smiles)

            mol = Chem.MolFromSmiles(smiles)
            '''rdkit MMFFOptimizeMolecule'''
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)

            # GetDistanceMatrix
            conf = mol.GetConformer()
            mol = AllChem.RemoveAllHs(mol)

            bonds_rdkit = [bond for bond in mol.GetBonds()]
            atoms_rdkit = mol.GetAtoms()
            atom_nums_rdkit = []
            for atom in atoms_rdkit:
                atom_nums_rdkit.append(atom.GetSymbol())

            # for i in range(len(bonds_rdkit)):
            #     bond = bonds_rdkit[i]
            #     atom1, atom2 = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            #
            #     bond_length = rdMolTransforms.GetBondLength(conf, atom1, atom2)
            #     mol.GetBondWithIdx(i).SetProp("bondNote", str(int(bond_length*100)))
            # draw = Draw.MolToImage(mol, (1200, 2400),dpi=600)
            # draw.save('./test_fig/'+'test'+'.jpg')

            # ssr = Chem.GetSymmSSSR(mol)
            for bond in bonds_rdkit:
                atom1, atom2 = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                at1_type = atom_nums_rdkit[atom1]
                at2_type = atom_nums_rdkit[atom2]
                bond_length = rdMolTransforms.GetBondLength(conf, atom1, atom2)
                bond_length = int(bond_length * 100)

                if at1_type > at2_type:  # Sort the pairs to avoid redundancy
                    temp = at2_type
                    at2_type = at1_type
                    at1_type = temp

                bond_type = bond.GetBondType().name.lower()

                bond_key = bond_type + '_' + at1_type + '_' + at2_type
                if bond_key in bond_dict.keys():
                    bond_dict[bond_key].append(bond_length)
                else:
                    bond_dict[bond_type + '_' + at1_type + '_' + at2_type] = [bond_length]
            for key in bond_dict.keys():
                bond_dict[key] = [np.mean(bond_dict[key]),np.std(bond_dict[key],dtype=np.float16),np.max(bond_dict[key]),np.min(bond_dict[key])]
                if key in all_bond_dict.keys():
                    all_bond_dict[key].append(bond_dict[key])
                else:
                    all_bond_dict[key]=[bond_dict[key]]
            bond_dict.clear()
        except Exception as e:
            print(e)
    print('repeat_smiles:',repeat_smiles)
    print('have_saved_smiles:',len(have_saved_smiles))
    with open('./data/geom/bond_dist_each_molecule', 'wb') as bond_dictionary_file:
        pickle.dump(all_bond_dict, bond_dictionary_file)


def cal_dist_data(args):
    Counter = collections.Counter
    bond_length_dict = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}

    path = './data/molecules_zinc_valid_final.json'  # train/valid/test文件格式一致
    with open(path, 'r') as f:
        train_data = json.load(f)
    for idx, line in enumerate(train_data):

        print(idx)
        coords = line['positions_out']
        bonds = line['graph_out']
        atom_nums = line['node_features_out']

        for bond in bonds:
            atom1, atom2 = bond[0], bond[2]
            type = bond[1]
            # bond length
            c1 = coords[atom1]
            c2 = coords[atom2]
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            # Bin the distance
            dist = int(dist * 100)

            # atom types
            at1_type = atom_nums[atom1]
            at2_type = atom_nums[atom2]
            at1_type = atom_name[np.argmax(at1_type)]
            at2_type = atom_name[np.argmax(at2_type)]

            bond_length_dict[type][(at1_type, at2_type, dist)] += 1
        print()


def extract_conformers(args):
    Counter = collections.Counter
    bond_length_dict = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}
    bond_length_dict_rdkit = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}

    path = './data/molecules_zinc_train_final.json'  # train/valid/test文件格式一致
    with open(path, 'r') as f:
        train_data = json.load(f)
    for idx, line in enumerate(train_data):
        try:
            print(idx)
            coords = line['positions_out']
            bonds = line['graph_out']
            atom_nums = line['node_features_out']
            smiles = line['smiles_out']
            mol = Chem.MolFromSmiles(smiles)
            '''rdkit MMFFOptimizeMolecule'''
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)

            # GetDistanceMatrix
            conf = mol.GetConformer()
            mol = AllChem.RemoveAllHs(mol)

            bonds_rdkit = [bond for bond in mol.GetBonds()]
            atoms_rdkit = mol.GetAtoms()
            atom_nums_rdkit = []
            for atom in atoms_rdkit:
                atom_nums_rdkit.append(atom.GetAtomicNum())

            # ssr = Chem.GetSymmSSSR(mol)
            for bond in bonds_rdkit:
                atom1, atom2 = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                at1_type = atom_nums_rdkit[atom1]
                at2_type = atom_nums_rdkit[atom2]
                bond_length = rdMolTransforms.GetBondLength(conf, atom1, atom2)
                bond_length = int(bond_length * 100)

                if at1_type > at2_type:  # Sort the pairs to avoid redundancy
                    temp = at2_type
                    at2_type = at1_type
                    at1_type = temp

                bond_type = bond.GetBondType().name.lower()
                if bond_type == 'single':
                    type = 0
                elif bond_type == 'double':
                    type = 1
                elif bond_type == 'triple':
                    type = 2
                elif bond_type == 'aromatic':
                    type = 3
                else:
                    raise ValueError("Unknown bond type", bond_type)

                bond_length_dict_rdkit[type][(at1_type, at2_type, bond_length)] += 1
        except Exception as e:
            print(e)

        # for bond in bonds:
        #     atom1, atom2 = bond[0], bond[2]
        #     type = bond[1]
        #     # bond length
        #     c1 = coords[atom1]
        #     c2 = coords[atom2]
        #     dist = np.linalg.norm(np.array(c1) - np.array(c2))
        #     # Bin the distance
        #     dist = int(dist * 100)
        #
        #     # atom types
        #     at1_type = atom_nums[atom1]
        #     at2_type = atom_nums[atom2]
        #     at1_type = atom_name[np.argmax(at1_type)]
        #     at2_type = atom_name[np.argmax(at2_type)]
        #
        #
        #     # if at1_type > at2_type:  # Sort the pairs to avoid redundancy
        #     #     temp = at2_type
        #     #     at2_type = at1_type
        #     #     at1_type = temp
        #
        #     # bond_type = bond.GetBondType().name.lower()
        #     # if bond_type == 'single':
        #     #     type = 0
        #     # elif bond_type == 'double':
        #     #     type = 1
        #     # elif bond_type == 'triple':
        #     #     type = 2
        #     # elif bond_type == 'aromatic':
        #     #     type = 3
        #     # else:
        #     #     raise ValueError("Unknown bond type", bond_type)
        #
        #     bond_length_dict[type][(at1_type, at2_type, dist)] += 1
        #
        # print()

    # print("Current state of the bond length dictionary", bond_length_dict_rdkit)
    # if os.path.exists('bond_length_dict.pkl'):
    #     os.remove('bond_length_dict.pkl')
    with open('data/geom/bond_length_dict', 'wb') as bond_dictionary_file:
        pickle.dump(bond_length_dict_rdkit, bond_dictionary_file)


def create_matrix(args):
    with open('data/geom/bond_length_dict', 'rb') as bond_dictionary_file:
        all_bond_types = pickle.load(bond_dictionary_file)
    x = np.zeros((n_atom_types, n_atom_types, n_bond_types, 350))
    for bond_type, d in all_bond_types.items():
        for key, count in d.items():
            at1, at2, bond_len = key
            x[inverse[at1], inverse[at2], bond_type, bond_len - 50] = count

    np.save('bond_length_matrix', x)


def create_histograms(args):
    x = np.load('./data/geom/bond_length_matrix.npy')
    x = x[:, :, :, :307]
    label_list = ['single', 'double', 'triple', 'aromatic']
    for j in range(n_atom_types):
        for i in range(j + 1):
            if np.sum(x[i, j]) == 0:  # Bond does not exist
                continue

            # Remove outliers
            y = x[i, j]
            y[y < 0.02 * np.sum(y, axis=0)] = 0

            plt.figure()
            existing_bond_lengths = np.array(np.nonzero(y))[1]
            mini, maxi = existing_bond_lengths.min(), existing_bond_lengths.max()
            y = y[:, mini: maxi + 1]
            x_range = np.arange(mini, maxi + 1)
            for k in range(n_bond_types):
                if np.sum(y[k]) > 0:
                    plt.plot(x_range, y[k], label=label_list[k])
            plt.xlabel("Bond length")
            plt.ylabel("Count")
            plt.title(f'{atom_name[i]} - {atom_name[j]} bonds')
            plt.legend()
            plt.savefig(f'./figures/{atom_name[i]}-{atom_name[j]}-hist.png')
            plt.close()


def analyse_geom_stability():
    data_file = './data/geom/geom_drugs_30.npy'
    dataset_info = configs.datasets_config.get_dataset_info('geom', remove_h=False)
    atom_dict = dataset_info['atom_decoder']
    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]

    x = np.load(data_file)
    mol_id = x[:, 0].astype(int)
    all_atom_types = x[:, 1].astype(int)
    all_positions = x[:, 2:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    all_atom_types_split = np.split(all_atom_types, split_indices)
    all_positions_split = np.split(all_positions, split_indices)

    atomic_nb_list = torch.Tensor(dataset_info['atomic_nb'])[None, :].long()
    num_stable_mols = 0
    num_mols = 0
    num_stable_atoms_total = 0
    num_atoms_total = 0
    formatted_data = []
    for i, (p, at_types) in tqdm(enumerate(zip(all_positions_split, all_atom_types_split))):
        p = torch.from_numpy(p)
        at_types = torch.from_numpy(at_types)[:, None]
        one_hot = torch.eq(at_types, atomic_nb_list).int()
        at_types = torch.argmax(one_hot, dim=1)  # Between 0 and 15
        formatted_data.append([p, at_types])

        mol_is_stable, num_stable_atoms, num_atoms = check_stability(p, at_types, dataset_info)
        num_mols += 1
        num_stable_mols += mol_is_stable
        num_stable_atoms_total += num_stable_atoms
        num_atoms_total += num_atoms
        if i % 5000 == 0:
            print(f"IN PROGRESS -- Stable molecules: {num_stable_mols} / {num_mols}"
                  f" = {num_stable_mols / num_mols * 100} %")
            print(
                f"IN PROGRESS -- Stable atoms: {num_stable_atoms_total} / {num_atoms_total}"
                f" = {num_stable_atoms_total / num_atoms_total * 100} %")

    print(f"Stable molecules: {num_stable_mols} / {num_mols} = {num_stable_mols / num_mols * 100} %")
    print(
        f"Stable atoms: {num_stable_atoms_total} / {num_atoms_total} = {num_stable_atoms_total / num_atoms_total * 100} %")

    metrics = BasicMolecularMetrics(dataset_info)
    metrics.evaluate(formatted_data)


def debug_geom_stability(num_atoms=100000):
    data_file = './data/geom/geom_drugs_30.npy'
    dataset_info = configs.datasets_config.get_dataset_info('geom', remove_h=False)
    atom_dict = dataset_info['atom_decoder']
    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]

    x = np.load(data_file)
    x = x[:num_atoms]

    # Print non hydrogen atoms
    x = x[x[:, 1] != 1.0, :]

    mol_id = x[:, 0].astype(int)
    max_mol_id = mol_id.max()
    may_be_incomplete = mol_id == max_mol_id
    x = x[~may_be_incomplete]
    mol_id = mol_id[~may_be_incomplete]
    all_atom_types = x[:, 1].astype(int)
    all_positions = x[:, 2:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    all_atom_types_split = np.split(all_atom_types, split_indices)
    all_positions_split = np.split(all_positions, split_indices)

    atomic_nb_list = torch.Tensor(dataset_info['atomic_nb'])[None, :].long()

    formatted_data = []
    for p, at_types in zip(all_positions_split, all_atom_types_split):
        p = torch.from_numpy(p)
        at_types = torch.from_numpy(at_types)[:, None]
        one_hot = torch.eq(at_types, atomic_nb_list).int()
        at_types = torch.argmax(one_hot, dim=1)  # Between 0 and 15
        formatted_data.append([p, at_types])

    metrics = BasicMolecularMetrics(atom_dict, bond_dict, dataset_info)
    m, smiles_list = metrics.evaluate(formatted_data)
    print(smiles_list)


def compute_n_nodes_dict(file='./data/geom/geom_drugs_30.npy', remove_hydrogens=True):
    all_data = np.load(file)
    atom_types = all_data[:, 1]
    if remove_hydrogens:
        hydrogens = np.equal(atom_types, 1.0)
        all_data = all_data[~hydrogens]

    # Get the size of each molecule
    mol_id = all_data[:, 0].astype(int)
    max_id = mol_id.max()
    mol_id_counter = np.zeros(max_id + 1, dtype=int)
    for id in mol_id:
        mol_id_counter[id] += 1

    unique_sizes, size_count = np.unique(mol_id_counter, return_counts=True)

    sizes_dict = {}
    for size, count in zip(unique_sizes, size_count):
        sizes_dict[size] = count

    print(sizes_dict)
    return sizes_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformations", type=int, default=30,
                        help="Max number of conformations kept for each molecule.")
    parser.add_argument("--data_dir", type=str, default='/home/vignac/diffusion/data/geom/')
    parser.add_argument("--data_file", type=str, default="rdkit_folder/summary_drugs.json")
    args = parser.parse_args()
    AROM_num()
    # cal_dist_data(args)
    # cal_bond_dist(args)
    # extract_conformers(args)
    # create_matrix(args)
    # create_histograms(args)
    # #
    # analyse_geom_stability()
    # compute_n_nodes_dict()
