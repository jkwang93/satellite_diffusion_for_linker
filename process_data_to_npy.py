import networkx as nx
from networkx.algorithms import tree

# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from zinc import dataset_generate_npy

import torch
import pickle

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_concat')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='zinc',
                    help='zinc | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='zinc/temp',
                    help='zinc directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_false', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
# parser.add_argument("--conditioning", nargs='+', default=['alpha'],
#                     help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument("--conditioning", nargs='+', default=['alpha'],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
# parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
#                     help='normalize factors for [x, categorical, integer]')
parser.add_argument('--normalize_factors', type=eval, default=[1, 5, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
# e3_diffusion
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset_generate_npy.retrieve_dataloaders(args)
import numpy as np


def remove_mean_with_mask(x, node_mask, atom_num):
    masked_max_abs_value = np.sum(np.abs(x * (1 - node_mask)))
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    # N = node_mask.sum(1, keepdims=True)
    mean = np.sum(x, axis=0,keepdims=True) / atom_num
    x = x - mean * node_mask

    return x


def main():
    # load_npy = np.load("./processed_data/qm9_test.npy",allow_pickle=True)

    data_all = []
    # total_num = 0

    for i, data in enumerate(dataloaders['train']):
        # print(i)
        # transform to trajectory, there need a end token add to one_hot
        for mol_dict in data:

            atom_mask = mol_dict['charges'] > 0
            atom_mask = atom_mask.reshape(-1,1)

            num_atom = mol_dict['num_atoms']

            # print(num_atom+1)
            # total_num = total_num+num_atom+1


            mol_dict['positions'] = remove_mean_with_mask(mol_dict['positions'], atom_mask, num_atom)
            position = mol_dict['positions']

            # exclude padding atom
            position_exclude_padding = position[:num_atom]
            mol_dict['one_hot'] = np.concatenate(
                (mol_dict['one_hot'], np.zeros((len(mol_dict['one_hot']), 1), dtype=bool)), axis=1)

            # build tree based on 3d distances
            squared_dist = np.sum(
                np.square(position_exclude_padding[:, None, :] - position_exclude_padding[None, :, :]), axis=-1)
            # if self.perm == 'minimum_spanning_tree':
            nx_graph = nx.from_numpy_matrix(squared_dist)
            edges = list(tree.minimum_spanning_edges(nx_graph, algorithm='prim', data=False))
            focus_node_id, target_node_id = zip(*edges)
            node_perm = np.concatenate(([0], target_node_id))

            "add start graph, this graph include a H atom with (0,0,0,)"
            i_mol_dict = copy.deepcopy(mol_dict)
            i_mol_dict['positions'] = np.zeros((i_mol_dict['positions'].shape), dtype=np.float32)
            i_mol_dict['num_atoms'] = 0
            i_mol_dict['one_hot'] = np.zeros((i_mol_dict['one_hot'].shape), dtype=bool)
            i_mol_dict['charges'] = np.zeros((i_mol_dict['charges'].shape), dtype=np.float32)
            i_mol_dict['insert_idx'] = 0

            for i, node_idx in enumerate(node_perm):
                # input i_mol_dict to zero mol_dict
                i_mol_dict['num_atoms'] = i
                i_mol_dict['insert_idx'] = node_idx

                i_mol_dict['add_atom_type'] = mol_dict['one_hot'][node_idx]
                i_mol_dict['add_atom_position'] = mol_dict['positions'][node_idx]
                i_mol_dict['add_atom_charges'] = mol_dict['charges'][node_idx]

                i_mol_dict['positions'][node_idx] = mol_dict['positions'][node_idx]
                i_mol_dict['one_hot'][node_idx] = mol_dict['one_hot'][node_idx]
                i_mol_dict['charges'][node_idx] = mol_dict['charges'][node_idx]
                data_all.append(copy.deepcopy(i_mol_dict))

            # add end token step, set end to 0
            i_mol_dict['num_atoms'] = len(node_perm)
            i_mol_dict['insert_idx'] = len(node_perm)

            i_mol_dict['add_atom_type'] = np.zeros((len(mol_dict['one_hot'][0])), dtype=bool)
            i_mol_dict['add_atom_type'][-1] = 1

            i_mol_dict['add_atom_position'] = np.zeros((i_mol_dict['positions'][0].shape),
                                                       dtype=i_mol_dict['positions'].dtype)

            i_mol_dict['add_atom_charges'] = 11

            i_mol_dict['charges'][len(node_perm)] = i_mol_dict['add_atom_charges']
            i_mol_dict['positions'][len(node_perm)] = i_mol_dict['add_atom_position']
            i_mol_dict['one_hot'][len(node_perm)] = i_mol_dict['add_atom_type']

            data_all.append(copy.deepcopy(i_mol_dict))

            # data_all.extend(dict_list)
    # print(total_num)

    print(len(data_all))
    np.save("processed_data/" + "qm9_train.npy", data_all)


if __name__ == "__main__":
    main()
