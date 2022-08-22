# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:41
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : utils.py
# @Software: PyCharm
# @Note    :
import json
import os
import shutil
import re
import numpy as np
from functools import partial
import dgl
import torch
import torch.nn as nn


def write_json(dict, path):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(dict, file_obj, indent=4, ensure_ascii=False)


def write_post(post_list, path):
    for post in post_list:
        write_json(post[1], os.path.join(path, f'{post[0]}.json'))


def dataset_makedirs(dataset_path):
    train_path = os.path.join(dataset_path, 'train', 'raw')
    val_path = os.path.join(dataset_path, 'val', 'raw')
    test_path = os.path.join(dataset_path, 'test', 'raw')

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(test_path)
    os.makedirs(os.path.join(dataset_path, 'train', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'val', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'test', 'processed'))

    return train_path, val_path, test_path


def clean_comment(comment_text):
    match_res = re.match('回复@.*?:', comment_text)
    if match_res:
        return comment_text[len(match_res.group()):]
    else:
        return comment_text


def create_log_dict_pretrain(args):
    log_dict = {}
    log_dict['dataset'] = args.dataset
    log_dict['unsup train size'] = args.unsup_train_size
    log_dict['runs'] = args.runs

    log_dict['batch size'] = args.batch_size
    log_dict['num heads'] = args.num_heads
    log_dict['num out heads'] = args.num_out_heads
    log_dict['num layers'] = args.num_layers
    log_dict['num hidden'] = args.num_hidden
    log_dict['residual'] = args.residual
    log_dict['in drop'] = args.in_drop
    log_dict['attn drop'] = args.attn_drop
    log_dict['norm'] = args.norm
    log_dict['negative slope'] = args.negative_slope
    log_dict['activation'] = args.activation
    log_dict['mask rate'] = args.mask_rate
    log_dict['drop edge rate'] = args.drop_edge_rate
    log_dict['replace rate'] = args.replace_rate
    log_dict['concat hidden'] = args.concat_hidden
    log_dict['pooling'] = args.pooling

    log_dict['encoder'] = args.encoder
    log_dict['decoder'] = args.decoder
    log_dict['loss fn'] = args.loss_fn
    log_dict['alpha l'] = args.alpha_l

    log_dict['lr'] = args.lr
    log_dict['epochs'] = args.epochs
    log_dict['weight decay'] = args.weight_decay

    log_dict['record'] = []
    return log_dict


def create_log_dict_semisup(args):
    log_dict = {}
    log_dict['dataset'] = args.dataset
    log_dict['unsup train size'] = args.unsup_train_size
    log_dict['runs'] = args.runs

    log_dict['batch size'] = args.batch_size
    log_dict['num heads'] = args.num_heads
    log_dict['num out heads'] = args.num_out_heads
    log_dict['num layers'] = args.num_layers
    log_dict['num hidden'] = args.num_hidden
    log_dict['residual'] = args.residual
    log_dict['in drop'] = args.in_drop
    log_dict['attn drop'] = args.attn_drop
    log_dict['norm'] = args.norm
    log_dict['negative slope'] = args.negative_slope
    log_dict['activation'] = args.activation
    log_dict['mask rate'] = args.mask_rate
    log_dict['drop edge rate'] = args.drop_edge_rate
    log_dict['replace rate'] = args.replace_rate
    log_dict['concat hidden'] = args.concat_hidden
    log_dict['pooling'] = args.pooling

    log_dict['encoder'] = args.encoder
    log_dict['decoder'] = args.decoder
    log_dict['loss fn'] = args.loss_fn
    log_dict['alpha l'] = args.alpha_l

    log_dict['lr'] = args.lr
    log_dict['ft lr'] = args.ft_lr
    log_dict['epochs'] = args.epochs
    log_dict['ft epochs'] = args.ft_epochs
    log_dict['weight decay'] = args.weight_decay

    log_dict['record'] = []
    return log_dict


def write_log(log, str):
    log.write(f'{str}\n')
    log.flush()


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="graphnorm")
    else:
        return None


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
