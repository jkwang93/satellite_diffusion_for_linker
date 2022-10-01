from rdkit import Chem
import numpy as np
from zinc.bond_analyze import get_bond_order, geom_predictor, bonds1, margin1, bonds2, margin2, bonds3, margin3
from . import dataset_qm9 as dataset
# from . import dataset

import torch
from configs.datasets_config import get_dataset_info
import pickle
import os


def compute_qm9_smiles(dataset_name, remove_h):
    '''

    :param dataset_name: zinc or qm9_second_half
    :return:
    '''
    print("\tConverting QM9 dataset to SMILES ...")

    class StaticArgs:
        def __init__(self, dataset, remove_h):
            self.dataset = dataset
            self.batch_size = 1
            self.num_workers = 1
            self.filter_n_atoms = None
            self.datadir = 'zinc/temp'
            self.remove_h = remove_h
            self.include_charges = True

    args_dataset = StaticArgs(dataset_name, remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_dataset)
    dataset_info = get_dataset_info(args_dataset.dataset, args_dataset.remove_h)
    n_types = 4 if remove_h else 5
    mols_smiles = []
    for i, data in enumerate(dataloaders['train']):

        positions = data['positions'][0].view(-1, 3).numpy()
        one_hot = data['one_hot'][0].view(-1, n_types).type(torch.float32)
        atom_type = torch.argmax(one_hot, dim=1).numpy()

        mol = build_molecule(torch.tensor(positions), torch.tensor(atom_type), dataset_info)
        mol = mol2smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i) / len(dataloaders['train'])))
    return mols_smiles


def retrieve_qm9_smiles(dataset_info):
    dataset_name = dataset_info['name']
    if dataset_info['with_h']:
        pickle_name = dataset_name
    else:
        pickle_name = dataset_name + '_noH'

    file_name = 'zinc/temp/%s_smiles.pickle' % pickle_name
    try:
        with open(file_name, 'rb') as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        try:
            os.makedirs('zinc/temp')
        except:
            pass
        qm9_smiles = compute_qm9_smiles(dataset_name, remove_h=not dataset_info['with_h'])
        with open(file_name, 'wb') as f:
            pickle.dump(qm9_smiles, f)
        return qm9_smiles


#### New implementation ####

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for zinc currently.
        if dataset_smiles_list is None and 'zinc' in dataset_info['name']:
            self.dataset_smiles_list = retrieve_qm9_smiles(
                self.dataset_info)

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        wrong_count_ = 0
        for graph in generated:
            mol,wrong_count = build_molecule(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            wrong_count_+=wrong_count
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)
        print(wrong_count_)
        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        return [validity, uniqueness, novelty], unique


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(positions, atom_types, graph_out, smiles_out, dataset_info):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E,wrong_count = build_xae_molecule(positions, atom_types, graph_out, smiles_out, dataset_info)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    return mol,wrong_count


def build_xae_molecule(positions, atom_types, graph_out, smiles_out, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    smiles = 'C[C@H](NC(=O)NCCc1ccccn1)c1ccc(F)cc1Cl'

    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)

    edge_dict = []
    for i in graph_out:
        start = i[0]
        end = i[2]
        edge = i[1]
        if start==end==0:
            continue
        edge_dict.append(sorted([start.detach().cpu().numpy().tolist(), end.detach().cpu().numpy().tolist()]))
        # distance_ = torch.dist(positions[start], positions[end], p=2)
        #
        # atom1 = atom_decoder[atom_types[start]]
        # atom2 = atom_decoder[atom_types[end]]
        #
        # print(atom1, atom2, edge, distance_)
        # distance_ *= 100
        # if distance_ < bonds1[atom1][atom2] + margin1:
        #     # Check if atoms in bonds2 dictionary.
        #     if atom1 in bonds2 and atom2 in bonds2[atom1]:
        #         thr_bond2 = bonds2[atom1][atom2] + margin2
        #         if distance_ < thr_bond2:
        #             if atom1 in bonds3 and atom2 in bonds3[atom1]:
        #                 thr_bond3 = bonds3[atom1][atom2] + margin3
        #                 if distance_ < thr_bond3:
        #                     print(3 - 1)  # Triple
        #                     continue
        #             print(2 - 1)  # Double
        #             continue
        #     print(1 - 1)  # Single
        #     continue
        # print(0 - 1)
    edge_dict.sort()
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    predict_edge = []
    edge_count = 0
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'zinc' or dataset_info['name'] == 'qm9_second_half' or dataset_info[
                'name'] == 'qm9_first_half':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'zinc':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j],
                                       limit_bonds_to_one=False)
            elif dataset_info['name'] == 'geom':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j],
                                       limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                # for k in edge_dict:
                #     if sorted([i, j]) == k:
                #         edge_count+=1
                predict_edge.append(sorted([i, j]))
                A[i, j] = 1
                E[i, j] = order
            predict_edge.sort()
    if predict_edge != edge_dict:
        wrong_count = 1
    else:
        wrong_count=0
    return X, A, E, wrong_count


if __name__ == '__main__':
    smiles_mol = 'C1CCC1'
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)
