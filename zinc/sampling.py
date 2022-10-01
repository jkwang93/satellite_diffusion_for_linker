import copy

import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, \
    assert_correctly_masked
from zinc.analyze import check_stability
import zinc.utils as qm9utils


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample(loader, partition, args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False):
    dtype = torch.float32
    each_batch = 1
    each_mol_linker = 1
    # all_mol_linker_list_h = []
    # all_mol_linker_list_xyz = []
    each_mol_linker_list_h = []
    each_mol_linker_list_xyz = []

    for i, data in enumerate(loader):
        # each_mol_linker_list_h = []
        # each_mol_linker_list_xyz = []
        if i >5:
            break
        for j in range(int(each_mol_linker / each_batch)):
            print(i)
            x = data['positions'].to(device, dtype)

            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)

            insert_idx = data['linker_idx'].to(device, dtype)
            sub_graph_mask = data['sub_graph_mask'].to(device, dtype).unsqueeze(2)
            linker_len = data['linker_len'].to(device, dtype)

            y_atom_type = data['y_atom_type'].to(device, dtype)
            y = data['y'].to(device, dtype)

            # graph_total_x = copy.deepcopy(x)
            # graph_total_h = copy.deepcopy(one_hot)

            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            x = x.repeat(each_batch, 1, 1)

            batch_size = x.size(0)

            node_mask = node_mask.repeat(each_batch, 1, 1)
            edge_mask = edge_mask.repeat(each_batch, 1, 1)

            one_hot = one_hot.repeat(each_batch, 1, 1)

            insert_idx = insert_idx.repeat(each_batch, 1)
            sub_graph_mask = sub_graph_mask.repeat(each_batch, 1, 1)
            linker_len = linker_len.repeat(each_batch)

            y = y.repeat(each_batch, 1, 1)
            y_atom_type = y_atom_type.repeat(each_batch, 1, 1)

            graph_total_x = copy.deepcopy(x)
            graph_total_h = copy.deepcopy(one_hot)

            x_h = {'categorical': one_hot, 'integer': charges}
            y_h = {'categorical': y_atom_type, 'integer': charges}

            # if len(args.conditioning) > 0:
            #     context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            #     assert_correctly_masked(context, node_mask)
            # else:
            #     context = None
            context = None
            max_n_nodes = linker_len.max().long().detach().cpu().numpy().tolist()

            # atom_type = copy.deepcopy(y_atom_type)

            condition_graph_x, condition_graph_h, delta_log_px, y, y_h, delta_log_py = generative_model.normalize(x,
                                                                                                                  x_h,
                                                                                                                  y,
                                                                                                                  y_h,
                                                                                                                  node_mask)

            condition_xh = torch.cat(
                [condition_graph_x, condition_graph_h['categorical'], condition_graph_h['integer']], dim=2)

            # xh = torch.cat([y, y_h['categorical'], y_h['integer']], dim=2)

            if args.probabilistic_model == 'diffusion':
                x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, condition_xh,
                                               sub_graph_mask,
                                               insert_idx, linker_len, fix_noise=fix_noise)

                for index, item in enumerate(insert_idx):
                    insert_idx_without_zero = item[0:linker_len[index].long()].long().detach().cpu().numpy()
                    graph_total_x[index, insert_idx_without_zero, :] = x[index, :linker_len[index].long(), :]
                    graph_total_h[index, insert_idx_without_zero, :] = h['categorical'][index,
                                                                       :linker_len[index].long(), :]
                # h = h['categorical']
            each_mol_linker_list_h.extend(graph_total_h)
            # each_mol_linker_list_h = torch.stack(each_mol_linker_list_h)
            # each_mol_linker_list_h = each_mol_linker_list_h.detach().cpu().numpy()

            each_mol_linker_list_xyz.extend(graph_total_x)


        # all_mol_linker_list_h.append(each_mol_linker_list_h)
        # all_mol_linker_list_xyz.append(each_mol_linker_list_xyz)

    return each_mol_linker_list_h,each_mol_linker_list_xyz, node_mask


def sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None):
    n_samples = 1
    if args.dataset == 'zinc' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom':
        n_nodes = 44
    else:
        raise ValueError()

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
        context = context.repeat(1, n_nodes, 1).to(device)
        # context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError

    return one_hot, charges, x


def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist,
                                            nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask
