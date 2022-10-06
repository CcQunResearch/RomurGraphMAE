# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 18:59
# @Author  :
# @Email   :
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import os.path as osp
import json
import torch
from dgl.data import DGLDataset
from dgl import DGLGraph, save_graphs, load_graphs
from Main.utils import clean_comment


class WeiboDataset(DGLDataset):
    def __init__(self, name, root, word2vec, clean=True):
        raw_dir = osp.join(root, 'raw')
        save_dir = osp.join(root, 'processed')
        self.word2vec = word2vec
        self.clean = clean
        super().__init__(name=name, raw_dir=raw_dir, save_dir=save_dir)

    def download(self):
        pass

    def process(self):
        graphs = []
        labels = []
        raw_file_names = os.listdir(self.raw_dir)

        if self.clean:
            limit_num = 600
            pass_comment = ['', '转发微博', '转发微博。', '轉發微博', '轉發微博。']
            for filename in raw_file_names:
                y = 0
                row = []
                col = []
                num_node = 1
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                if 'label' in post['source'].keys():
                    y = post['source']['label']
                pass_num = 0
                id_to_index = {}
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    id_to_index[comment['comment id']] = i
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        pass_num += 1
                        continue
                    post['comment'][i]['comment id'] -= pass_num
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        continue
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    if comment['parent'] == -1:
                        row.append(0)
                    else:
                        row.append(post['comment'][id_to_index[comment['parent']]]['comment id'] + 1)
                    col.append(comment['comment id'] + 1)
                    num_node += 1

                # tdrow = row
                # tdcol = col
                # burow = col
                # bucol = row
                # row = tdrow + burow
                # col = tdcol + bucol

                graph = DGLGraph()
                graph.add_nodes(num_node)
                graph.add_edges(row, col)
                graph.ndata['x'] = x
                graphs.append(graph)
                labels.append(y)
        else:
            for filename in raw_file_names:
                y = 0
                row = []
                col = []
                num_node = 1
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                if 'label' in post['source'].keys():
                    y = post['source']['label']
                for i, comment in enumerate(post['comment']):
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    row.append(comment['parent'] + 1)
                    col.append(comment['comment id'] + 1)
                    num_node += 1

                # tdrow = row
                # tdcol = col
                # burow = col
                # bucol = row
                # row = tdrow + burow
                # col = tdcol + bucol

                graph = DGLGraph()
                graph.add_nodes(num_node)
                graph.add_edges(row, col)
                graph.ndata['x'] = x
                graphs.append(graph)
                labels.append(y)

        self.graphs = graphs
        self.labels = torch.tensor(labels)

    def save(self):
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})

    def load(self):
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        return os.path.exists(graph_path)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)
