# -*- coding: utf-8 -*-
# @Time    : 2022/8/22 11:08
# @Author  :
# @Email   :
# @File    : main(semisup).py
# @Software: PyCharm
# @Note    :
import sys
import os.path as osp

dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl
from dgl.dataloading import GraphDataLoader
from Main.dataset import WeiboDataset
from Main.pargs import pargs
from Main.utils import create_log_dict_semisup, write_log, write_json
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.models.edcoder import PreModel
from Main.sort import sort_weibo_dataset, sort_weibo_self_dataset, sort_weibo_2class_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def semisup_train(unsup_train_loader, train_loader, model, optimizer, device, lamda):
    model.train()
    loss_list = []

    for batch_sup, batch_unsup in zip(train_loader, unsup_train_loader):
        optimizer.zero_grad()

        batch_sup_g, y = batch_sup
        batch_sup_g = batch_sup_g.to(device)
        y = y.to(device)
        sup_x = batch_sup_g.ndata["x"]

        batch_unsup_g, _ = batch_unsup
        batch_unsup_g = batch_unsup_g.to(device)
        unsup_x = batch_unsup_g.ndata["x"]

        loss = F.binary_cross_entropy(model.predict(batch_sup_g, sup_x), y.to(torch.float32)) + \
               model(batch_sup_g, sup_x) * lamda + \
               model(batch_unsup_g, unsup_x) * lamda

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    return np.mean(loss_list)


def test(model, dataloader, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []
    for batch in dataloader:
        batch_g, y = batch
        batch_g = batch_g.to(device)
        y = y.to(device)

        x = batch_g.ndata["x"]
        pred = model.predict(batch_g, x)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        error += F.binary_cross_entropy(pred, y.to(torch.float32)).item() * batch_g.batch_size
        y_true += y.tolist()
        y_pred += pred.tolist()
    acc = accuracy_score(y_true, y_pred)
    prec = [precision_score(y_true, y_pred, pos_label=1, average='binary'),
            precision_score(y_true, y_pred, pos_label=0, average='binary')]
    rec = [recall_score(y_true, y_pred, pos_label=1, average='binary'),
           recall_score(y_true, y_pred, pos_label=0, average='binary')]
    f1 = [f1_score(y_true, y_pred, pos_label=1, average='binary'),
          f1_score(y_true, y_pred, pos_label=0, average='binary')]
    return error / len(dataloader.dataset), acc, prec, rec, f1


def test_and_log(model, val_loader, test_loader, device, epoch, lr, loss, train_acc, log_record):
    val_error, val_acc, val_prec, val_rec, val_f1 = test(model, val_loader, device)
    test_error, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Train ACC: {:.3f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
        .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc, test_prec[0], test_prec[1],
                test_rec[0],
                test_rec[1], test_f1[0], test_f1[1])

    log_record['val accs'].append(round(val_acc, 3))
    log_record['test accs'].append(round(test_acc, 3))
    log_record['test prec T'].append(round(test_prec[0], 3))
    log_record['test prec F'].append(round(test_prec[1], 3))
    log_record['test rec T'].append(round(test_rec[0], 3))
    log_record['test rec F'].append(round(test_rec[1], 3))
    log_record['test f1 T'].append(round(test_f1[0], 3))
    log_record['test f1 F'].append(round(test_f1[1], 3))
    return val_error, log_info, log_record


def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1].view(-1) for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


if __name__ == '__main__':
    args = pargs()

    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    k = args.k

    batch_size = args.batch_size
    unsup_bs_ratio = args.unsup_bs_ratio
    weight_decay = args.weight_decay
    lamda = args.lamda
    epochs = args.epochs

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', unsup_dataset, 'dataset')
    model_path = osp.join(dirname, '..', 'Model', f'w2v_{dataset}_{unsup_train_size}_{vector_size}.model')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')

    log = open(log_path, 'w')
    log_dict = create_log_dict_semisup(args)

    if not osp.exists(model_path):
        if dataset == 'Weibo':
            sort_weibo_dataset(label_source_path, label_dataset_path)
        elif 'DRWeibo' in dataset:
            sort_weibo_2class_dataset(label_source_path, label_dataset_path)

        sentences = collect_sentences(label_dataset_path, unlabel_dataset_path, unsup_train_size)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)

    for run in range(runs):
        write_log(log, f'run:{run}')
        log_record = {'run': run, 'val accs': [], 'test accs': [], 'test prec T': [], 'test prec F': [],
                      'test rec T': [], 'test rec F': [], 'test f1 T': [], 'test f1 F': []}

        word2vec = Embedding(model_path)
        unlabel_dataset = WeiboDataset(dataset, unlabel_dataset_path, word2vec, clean=False)
        unsup_train_loader = GraphDataLoader(unlabel_dataset, batch_size=batch_size,
                                             collate_fn=collate_fn, shuffle=True)

        if dataset == 'Weibo':
            sort_weibo_dataset(label_source_path, label_dataset_path, k_shot=k)
        elif 'DRWeibo' in dataset:
            sort_weibo_2class_dataset(label_source_path, label_dataset_path, k_shot=k)

        train_dataset = WeiboDataset(dataset, train_path, word2vec)
        val_dataset = WeiboDataset(dataset, val_path, word2vec)
        test_dataset = WeiboDataset(dataset, test_path, word2vec)

        train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                       shuffle=True)
        test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
        val_loader = GraphDataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

        model = PreModel(
            pooling=args.pooling,
            in_dim=vector_size,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            nhead_out=args.num_out_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            encoder_type=args.encoder,
            decoder_type=args.decoder,
            mask_rate=args.mask_rate,
            norm=args.norm,
            loss_fn=args.loss_fn,
            drop_edge_rate=args.drop_edge_rate,
            replace_rate=args.replace_rate,
            alpha_l=args.alpha_l,
            concat_hidden=args.concat_hidden
        ).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

        val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
                                                       device, 0, args.lr, 0, 0, log_record)
        write_log(log, log_info)

        for epoch in range(1, epochs + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']

            _ = semisup_train(unsup_train_loader, train_loader, model, optimizer, device, lamda)

            train_error, train_acc, _, _, _ = test(model, train_loader, device)
            val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
                                                           device, epoch, lr, train_error, train_acc,
                                                           log_record)
            write_log(log, log_info)

            scheduler.step(val_error)

        log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 3)
        write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
