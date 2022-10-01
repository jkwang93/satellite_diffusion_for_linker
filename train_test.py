import copy
import pickle

import wandb

from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask_y, \
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask, assert_mean_zero_with_mask_y
# from data.bond_adding import BondAdder

import numpy as np
import zinc.visualizer as vis
from zinc.analyze import analyze_stability_for_molecules
from zinc.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import zinc.utils as qm9utils
from zinc import losses
import time
import torch

atom_decoder = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'N', 'N', 'O', 'O', 'S', 'S', 'S']

atom_id = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}


def convert_to_mols(one_hot, position, node_mask):
    mol_list = {'symbol':[],'conf':[]}
    atom_num_list = np.sum(one_hot, axis=(2,1)).flatten()
    for mol_idx, atom_num in enumerate(atom_num_list):
        atom_list, position_list = [], []
        for idx in range(int(atom_num)):
            # atom = atom_id[atom_decoder[np.argmax(one_hot[mol_idx][idx])]]
            atom = atom_decoder[np.argmax(one_hot[mol_idx][idx])]
            atom_list.append(atom)
            position_list.append(position[mol_idx][idx].tolist())
        mol_list['symbol'].append(atom_list)
        mol_list['conf'].append(position_list)

        # bond_adder = BondAdder()
        # try:
        #     rd_mol, ob_mol = bond_adder.make_mol(np.array(atom_list), np.array(position_list))
        #     mol_list.append(rd_mol)
        # except Exception as e:
        #     print(e)
        with open('./data/generated_linker_molecules', 'wb') as f:
            pickle.dump(mol_list, f)
    return mol_list

def convert_to_mol(one_hot, position):
    atom_num = len(one_hot)
    atom_list = []
    position_list = []
    for idx in range(atom_num):
        atom = atom_decoder[np.argmax(one_hot[idx])]
        atom_list.append(atom)
        position_list.append(position[idx].tolist())

    return atom_list,position_list

def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        # if i > 5:
        #     break

        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        insert_idx = data['linker_idx'].to(device, dtype)
        sub_graph_mask = data['sub_graph_mask'].to(device, dtype).unsqueeze(2)
        linker_len = data['linker_len'].to(device, dtype)

        y_atom_type = data['y_atom_type'].to(device, dtype)
        y = data['y'].to(device, dtype)

        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        # x, y = remove_mean_with_mask_y(x, condition_node_mask, y)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        # 因为加入了y_label，所以mean并不为0
        # assert_mean_zero_with_mask_y(x, condition_node_mask, y)
        # y_atom_type = one_hot

        h = {'categorical': one_hot, 'integer': charges}
        y_h = {'categorical': y_atom_type, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, y, y_h, node_mask, edge_mask, context, insert_idx,
                                                                sub_graph_mask, linker_len
                                                                )
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
        nll_epoch.append(nll.item())
        # if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
        #     start = time.time()
        #     if len(args.conditioning) > 0:
        #         save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
        #     save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
        #                           batch_id=str(i))
        #     sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
        #                                     prop_dist, epoch=epoch)
        #     print(f'Sampling took {time.time() - start:.2f} seconds')
        #
        #     vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
        #     vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
        #     if len(args.conditioning) > 0:
        #         vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
        #                             wandb=wandb, mode='conditional')
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


@torch.no_grad()
def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    # eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)

            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            insert_idx = data['linker_idx'].to(device, dtype)
            sub_graph_mask = data['sub_graph_mask'].to(device, dtype).unsqueeze(2)
            linker_len = data['linker_len'].to(device, dtype)

            y_atom_type = data['y_atom_type'].to(device, dtype)
            y = data['y'].to(device, dtype)

            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            check_mask_correct([x, one_hot, charges], node_mask)

            h = {'categorical': one_hot, 'integer': charges}
            y_h = {'categorical': y_atom_type, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist,
                                                    x, h, y, y_h, node_mask, edge_mask, context, insert_idx,
                                                    sub_graph_mask, linker_len
                                                    )
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch / n_samples:.2f}")

    return nll_epoch / n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(loader, partition, epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=1000):
    print(f'Analyzing molecule stability at epoch {epoch}...')

    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}

    for i in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, x, node_mask = sample(loader, partition, args, device, model_sample, dataset_info, prop_dist,
                                       nodesxsample=nodesxsample)

        # one_hot = torch.stack(one_hot)
        # x = torch.stack(x)
        mol_list = {'symbol':[],'conf':[]}

        for idxx in range(len(one_hot)):
            one_hot_idxx = one_hot[idxx].detach().cpu().numpy()
            x_idxx = x[idxx].detach().cpu().numpy()
            atom_list,position_list = convert_to_mol(one_hot_idxx, x_idxx)
            mol_list['symbol'].append(atom_list)
            mol_list['conf'].append(position_list)
        with open('./data/generated_linker_molecules', 'wb') as f:
            pickle.dump(mol_list, f)
        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
