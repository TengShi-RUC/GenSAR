import argparse
import collections
import json
import os
from typing import Dict

import torch
from models.rqvae import RQVAE
from modules.dataset import EmbCollator, EmbDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def get_indices(data, data_loader, prefix, semantic=False):

    if semantic:
        print("==========Get Semantic Index==========")
    else:
        print("==========Get Collaborate Index==========")

    all_indices = []
    all_indices_str = []
    for d in tqdm(data_loader):
        # d = d.to(device)
        if isinstance(d, Dict):
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    d[k] = v.to(device)
        else:
            d = d.to(device)
        indices = model.get_indices(d, use_sk=False)

        if semantic:
            indices = indices[0]
        else:
            indices = indices[1]

        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))

    print(all_indices_str[:10])
    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ",
          max(get_indices_count(all_indices_str).values()))

    tt = 0
    #There are often duplicate items in the dataset, and we no longer differentiate them
    while True:
        if tt >= 20 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print('collision_item_groups: ', len(collision_item_groups))
        for collision_items in collision_item_groups:
            d_collision = EmbCollator()([data[x] for x in collision_items])

            if isinstance(d_collision, Dict):
                for k, v in d_collision.items():
                    if isinstance(v, torch.Tensor):
                        d_collision[k] = v.to(device)
            else:
                d_collision = d_collision.to(device)

            indices = model.get_indices(d_collision, use_sk=True)

            if semantic:
                indices = indices[0]
            else:
                indices = indices[1]

            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))

                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ",
          max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices):
        all_indices_dict[item + 1] = list(indices)

    return all_indices_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Index")

    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")

    parser.add_argument("--ckpt_type",
                        type=str,
                        default="best_collision_model",
                        choices=['best_collision_model', 'best_loss_model'])

    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


if __name__ == '__main__':
    generate_args = parse_args()
    for flag, value in generate_args.__dict__.items():
        print('{}: {}'.format(flag, value))

    dataset = generate_args.dataset
    ckpt_name = os.path.basename(generate_args.ckpt_path)
    ckpt_path = os.path.join(generate_args.ckpt_path,
                             f'{generate_args.ckpt_type}.pth')
    device = generate_args.device

    output_dir = f"../data/{dataset}/emb/"

    if generate_args.ckpt_type == 'best_collision_model':
        output_file = f"{ckpt_name}_collision"
    elif generate_args.ckpt_type == 'best_loss_model':
        output_file = f"{ckpt_name}_loss"
    else:
        raise ValueError(
            "ckpt_type must be best_collision_model or best_loss_model")

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(args)

    model = RQVAE(
        semantic_dim=data.semantic_dim,
        collaborate_dim=data.collaborate_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_iters=args.sk_iters,
        shared_n_e_list=args.shared_num_emb_list,
        semantic_n_e_list=args.semantic_num_emb_list,
        collaborate_n_e_list=args.collaborate_num_emb_list,
        shared_e_dim=args.shared_e_dim,
        specific_e_dim=args.specific_e_dim,
        shared_sk_epsilons=args.shared_sk_epsilons,
        specific_sk_epsilons=args.specific_sk_epsilons,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    data_loader = DataLoader(data,
                             num_workers=args.num_workers,
                             batch_size=64,
                             shuffle=False,
                             pin_memory=True,
                             collate_fn=EmbCollator())

    semantic_prefix = [
        f"<shared_{chr(ord('a')+i)}" + "_{}>"
        for i in range(len(args.shared_num_emb_list))
    ] + [
        f"<semantic_{chr(ord('a')+i)}" + "_{}>"
        for i in range(len(args.semantic_num_emb_list))
    ]

    collaborate_prefix = [
        f"<shared_{chr(ord('a')+i)}" + "_{}>"
        for i in range(len(args.shared_num_emb_list))
    ] + [
        f"<collaborate_{chr(ord('a')+i)}" + "_{}>"
        for i in range(len(args.collaborate_num_emb_list))
    ]

    for vq in model.rq.vq_layers_shared:
        vq.sk_epsilon = 0.0

    for vq in model.rq.vq_layers_semantic[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers_semantic[-1].sk_epsilon == 0.0:
        model.rq.vq_layers_semantic[-1].sk_epsilon = 0.003

    for vq in model.rq.vq_layers_collaborate[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers_collaborate[-1].sk_epsilon == 0.0:
        model.rq.vq_layers_collaborate[-1].sk_epsilon = 0.003

    semantic_indices_dict = get_indices(data,
                                        data_loader,
                                        semantic_prefix,
                                        semantic=True)
    semantic_output_file = os.path.join(output_dir,
                                        f"{output_file}_semantic.json")
    with open(semantic_output_file, 'w') as fp:
        json.dump(semantic_indices_dict, fp)

    collaborate_indices_dict = get_indices(data,
                                           data_loader,
                                           collaborate_prefix,
                                           semantic=False)
    collaborate_output_file = os.path.join(output_dir,
                                           f"{output_file}_collaborate.json")
    with open(collaborate_output_file, 'w') as fp:
        json.dump(collaborate_indices_dict, fp)
