# -*- coding: utf-8 -*
'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-12 16:40:00
@LastEditTime: 2019-08-12 16:40:00
@LastEditors: your name
'''
# -*- coding: utf-8 -*-
# @Time    : 2019/8/12 16:10
# @Author  : 陈旭
# @Software: PyCharm
# @File    : data_loader.py
import json
import os
import numpy as np
import re
from gensim.models import Word2Vec
import numpy as np
import jieba
from random import choice
import codecs
import pyhanlp


class Config:
    def __init__(self):
        self.mode = 0
        self.char_size = 128
        self.maxlen = 512


class Utils(Config):
    def __init__(self):
        super().__init__()
        self.w2v_path = '../wiki/wiki.zh.text.model'
        self.get_word2vec()  # load word2vec

    def get_word2vec(self):
        word2vec = Word2Vec.load(self.w2v_path)
        self.id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
        self.word2id = {j: i for i, j in self.id2word.items()}
        word2vec = word2vec.wv.syn0
        self.word_size = word2vec.shape[1]
        self.word2vec = np.concatenate(
            [np.zeros((1, self.word_size)), word2vec])

    def tokenize(self, s):
        """
        这个就是一个分词，这里改为jiagu，jieba，foolnltk都可以
        :param s:
        :return:
        """
        # return list(jieba.cut(s))
        return [i.word for i in pyhanlp.HanLP.segment(s)]

    def sent2vec(self, S):
        """
        S格式：[[w1, w2]]
        按一个词中字的个数，将这个词向量追加对应多次
        长度统一为序列最长长度

        """
        V = []
        for s in S:
            V.append([])
            for w in s:
                for _ in w:
                    V[-1].append(self.word2id.get(w, 0))
        V = self.seq_padding(V)
        V = self.word2vec[V]
        return V

    def seq_padding(self, X, padding=0):
        """
        将每个序列按最长长度对齐
        :param X:
        :param padding:
        :return:
        """
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])

    def repair(self, d):
        d['text'] = d['text'].lower()
        something = re.findall(u'《([^《》]*?)》', d['text'])
        something = [s.strip() for s in something]
        zhuanji = []
        gequ = []
        for sp in d['spo_list']:
            sp[0] = sp[0].strip(u'《》').strip().lower()
            sp[2] = sp[2].strip(u'《》').strip().lower()
            for some in something:
                if sp[0] in some and d['text'].count(sp[0]) == 1:
                    sp[0] = some
            if sp[1] == u'所属专辑':
                zhuanji.append(sp[2])
                gequ.append(sp[0])
        spo_list = []
        for sp in d['spo_list']:
            if sp[1] in [u'歌手', u'作词', u'作曲']:  # 过滤头实体为专辑的这三类关系三元组 xzk
                if sp[0] in zhuanji and sp[0] not in gequ:
                    continue
            spo_list.append(tuple(sp))
        d['spo_list'] = spo_list


class DataLoader(Utils):

    def __init__(self):
        super().__init__()

        self.total_path = '../data/train_data_me.json'
        self.all_50_schemes_path = '../data/all_50_schemas_me.json'
        self.all_chars_path = '../data/all_chars_me.json'
        self.random_order_vote_path = '../data/random_order_vote.json'
        # self.data_loader()

    def data_loader(self):

        self.total_data = json.load(codecs.open(self.total_path, "r", "utf-8"))
        self.id2predicate, self.predicate2id = json.load(
            codecs.open(self.all_50_schemes_path, "r", "utf-8"))
        self.id2predicate = {int(i): j for i, j in self.id2predicate.items()}
        self.id2char, self.char2id = json.load(
            codecs.open(self.all_chars_path, "r", "utf-8"))
        self.num_classes = len(self.id2predicate)

        if not os.path.exists(self.random_order_vote_path):
            random_order = list(range(len(self.total_data)))
            np.random.shuffle(random_order)
            json.dump(
                random_order,
                codecs.open(self.random_order_vote_path, 'w', "utf-8"),
                indent=4
            )
        else:
            random_order = json.load(codecs.open(
                self.random_order_vote_path, "r", "utf-8"))

        self.train_data = [self.total_data[j]
                           for i, j in enumerate(random_order) if i % 8 != self.mode]
        self.dev_data = [self.total_data[j]
                         for i, j in enumerate(random_order) if i % 8 == self.mode]

        self.predicates = {}  # 格式：{predicate: [(subject, predicate, object)]}

        for d in self.train_data:
            self.repair(d)
            for sp in d['spo_list']:
                if sp[1] not in self.predicates:
                    self.predicates[sp[1]] = []
                self.predicates[sp[1]].append(sp)

        for d in self.dev_data:
            self.repair(d)

    def random_generate(self, d, spo_list_key):
        """
        :param d: 输入数据项
        :param spo_list_key: "spo_list"
        :return: 50%的可能返回原数据，50%的可能将其中一条关系对应的实体进行替换，生成新的数据
        例如原句：'《原因》是由易家扬作词，冯翰铭作曲，alex@the invisible men编曲，余文乐的一首歌曲，收录于专辑《whether or not》中，发行于2005年10月21日'
        替换为：'《眼红红》是由林夕作词，冯翰铭作曲，alex@the invisible men编曲，余文乐的一首歌曲，收录于专辑《whether or not》中，发行于2005年10月21日'
        对应的三元组也进行替换
        """
        r = np.random.random()
        if r > 0.5:
            return d
        else:
            k = np.random.randint(len(d[spo_list_key]))
            spi = d[spo_list_key][k]
            k = np.random.randint(len(self.predicates[spi[1]]))
            spo = self.predicates[spi[1]][k]
            def F(s): return s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
            text = F(d['text'])
            spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]
            return {'text': text, spo_list_key: spo_list}


class DataGenerator():
    def __init__(self, train=True, batch_size=64):

        self.data_loader = DataLoader()
        self.data_loader.data_loader()
        if train:
            self.data = self.data_loader.train_data
        else:
            self.data = self.data_loader.dev_data

        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T1, T2, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], [], []
            for i in idxs:
                spo_list_key = 'spo_list'  # if np.random.random() > 0.5 else 'spo_list_with_pred'
                d = self.data_loader.random_generate(
                    self.data[i], spo_list_key)
                text = d['text'][:self.data_loader.maxlen]
                text_words = self.data_loader.tokenize(text)
                text = ''.join(text_words)
                items = {}
                for sp in d[spo_list_key]:
                    subjectid = text.find(sp[0])
                    objectid = text.find(sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid + len(sp[0]))
                        if key not in items:
                            items[key] = []
                        items[key].append((objectid,
                                           objectid + len(sp[2]),
                                           self.data_loader.predicate2id[sp[1]]))
                if items:
                    T1.append([self.data_loader.char2id.get(c, 1)
                               for c in text])  # 1是unk，0是padding
                    T2.append(text_words)
                    s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1] - 1] = 1
                    k1, k2 = np.array(list(items.keys())).T
                    k1 = choice(k1)
                    k2 = choice(k2[k2 >= k1])
                    o1, o2 = np.zeros((len(text), self.data_loader.num_classes)), np.zeros(
                        (len(text), self.data_loader.num_classes))
                    for j in items.get((k1, k2), []):
                        o1[j[0]][j[2]] = 1
                        o2[j[1] - 1][j[2]] = 1
                    S1.append(s1)
                    S2.append(s2)
                    K1.append([k1])
                    K2.append([k2 - 1])
                    O1.append(o1)
                    O2.append(o2)
                    # t1为字编码，长度为各个句子长度，seq_padding之后长度统一为最长句子的长度
                    # t2为词，长度为各个句子中词的个数，sent2vec之后为 最长句子长度 * 词向量维度
                    # s1 s2为句子中的所有subject的头尾指针向量，one-hot，长度为句子长度, seq_padding之后长度统一为最长句子的长度
                    # k1 k2为随机选中的句子中的某个subject的开头结尾，长度为1，数值为头尾在句子中的位置
                    # o1 o2为矩阵，sent_len * num_classes，o[i][j]为1表示句子中位置i是在关系j下的object,seq_padding之后长度统一为最长句子的长度
                    # 数据到达此处时为list类型，经过下面统一的处理，为ndarry
                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = self.data_loader.seq_padding(T1)
                        T2 = self.data_loader.sent2vec(T2)
                        S1 = self.data_loader.seq_padding(S1)
                        S2 = self.data_loader.seq_padding(S2)
                        O1 = self.data_loader.seq_padding(
                            O1, np.zeros(self.data_loader.num_classes))
                        O2 = self.data_loader.seq_padding(
                            O2, np.zeros(self.data_loader.num_classes))
                        K1, K2 = np.array(K1), np.array(K2)
                        yield [T1, T2, S1, S2, K1, K2, O1, O2], None
                        T1, T2, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], [], []
