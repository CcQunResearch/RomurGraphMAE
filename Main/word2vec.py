# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 17:47
# @Author  :
# @Email   :
# @File    : word2vec.py
# @Software: PyCharm
# @Note    :
from Main.pargs import pargs
import os
import os.path as osp
import json
import random
import torch
from gensim.models import Word2Vec
from Main.utils import clean_comment
from Main.sort import sort_weibo_dataset, sort_weibo_self_dataset


class Embedding():
    def __init__(self, w2v_path):
        self.w2v_path = w2v_path
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = self.make_embedding()

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self):
        self.embedding_matrix = []
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
        for i, word in enumerate(self.embedding.wv.key_to_index):
            # e.g. self.word2index['魯'] = 1
            # e.g. self.index2word[1] = '魯'
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv.get_vector(word, norm=True))
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def sentence_word2idx(self, sen):
        sentence_idx = []
        for word in sen:
            if (word in self.word2idx.keys()):
                sentence_idx.append(self.word2idx[word])
            else:
                sentence_idx.append(self.word2idx["<UNK>"])
        return sentence_idx

    def get_word_embedding(self, sen):
        sentence_idx = self.sentence_word2idx(sen)
        word_embedding = self.embedding_matrix[sentence_idx]
        return word_embedding

    def get_sentence_embedding(self, sen):
        word_embedding = self.get_word_embedding(sen)
        sen_embedding = torch.sum(word_embedding, dim=0)
        return sen_embedding

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)


def collect_sentences(label_dataset_path, unlabel_dataset_path, unsup_train_size):
    train_path = osp.join(label_dataset_path, 'train', 'raw')
    val_path = osp.join(label_dataset_path, 'val', 'raw')
    test_path = osp.join(label_dataset_path, 'test', 'raw')
    unlabel_path = osp.join(unlabel_dataset_path, 'raw')

    sentences = collect_label_sentences(train_path) + collect_label_sentences(val_path) \
                + collect_label_sentences(test_path) + collect_unlabel_sentences(unlabel_path, unsup_train_size)
    return sentences


def collect_label_sentences(path):
    sentences = []
    for filename in os.listdir(path):
        filepath = osp.join(path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        sentences.append(post['source']['content'])
        for commnet in post['comment']:
            sentences.append(clean_comment(commnet['content']))
    return sentences


def collect_unlabel_sentences(path, unsup_train_size):
    sentences = []
    filenames = os.listdir(path)
    random.shuffle(filenames)
    for i, filename in enumerate(filenames):
        if i == unsup_train_size:
            break
        filepath = osp.join(path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        sentences.append(post['source']['content'])
        for commnet in post['comment']:
            sentences.append(clean_comment(commnet['content']))
    return sentences


def train_word2vec(sentences, vector_size):
    model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=5, workers=12, epochs=10, sg=1)
    return model


if __name__ == '__main__':
    args = pargs()
    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    vector_size = args.vector_size

    dirname = osp.dirname(osp.abspath(__file__))
    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', 'Weibo-unsup', 'dataset')
    model_path = osp.join(dirname, '..', 'Log', f'w2v_{dataset}_{unsup_train_size}.model')

    if dataset == 'Weibo':
        sort_weibo_dataset(label_source_path, label_dataset_path)
    elif dataset == 'Weibo-self':
        sort_weibo_self_dataset(label_source_path, label_dataset_path, unlabel_dataset_path)

    sentences = collect_sentences(label_dataset_path, unlabel_dataset_path, unsup_train_size)
    model = train_word2vec(sentences, vector_size)
    model.save(model_path)
