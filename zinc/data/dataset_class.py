import torch
from torch.utils.data import Dataset


class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """

    def __init__(self, data, shuffle=True):
        self.data = data
        self.max_graph_len = 38

        if shuffle:
            self.perm = torch.randperm(len(data))
        else:
            self.perm = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]

        train_dict = {}
        mol_dict = self.data[idx]

        # todo remove data which linker > 10
        in_len = len(mol_dict['positions_in'])
        out_len = len(mol_dict['positions_out'])
        linker_len = out_len - in_len

        positions_out = torch.tensor(mol_dict['positions_out'])
        # center molecules position
        mean = torch.sum(positions_out, dim=0, keepdim=True) / out_len
        positions_out = positions_out - mean

        node_features_out = torch.tensor(mol_dict['node_features_out'])

        # padding to max_graph_len=38
        # ZeroPad = torch.nn.ZeroPad2d(padding=(0, 0, 0, self.max_graph_len - out_len))
        # positions_out = ZeroPad(positions_out)
        # node_features_out = ZeroPad(node_features_out)

        linker_idx = torch.arange(in_len, out_len)

        y_atom_type = node_features_out[-linker_len:, :]
        # y_charge = charges[:, :, -linker_len:]
        y = positions_out[-linker_len:, :]

        train_dict['positions'] = positions_out
        train_dict['one_hot'] = node_features_out
        train_dict['linker_idx'] = linker_idx
        train_dict['linker_len'] = linker_len

        train_dict['y'] = y
        train_dict['y_atom_type'] = y_atom_type
        train_dict['out_len'] = out_len
        # print(mol_dict['smiles_out'])
        # print(mol_dict['smiles_in'])
        # print('\n')


        # train_dict['smiles_out'] = mol_dict['smiles_out']
        train_dict['graph_out'] = torch.tensor(mol_dict['graph_out'])
        train_dict['smiles_out'] = train_dict['graph_out']




        return train_dict
