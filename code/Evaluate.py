# -*- coding: utf-8 -*
'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-15 13:29:20
@LastEditTime: 2019-08-16 10:33:09
@LastEditors: Please set LastEditors
'''
# -*- coding: utf-8 -*-
# @Time    : 2019/8/12 16:14
# @Author  : 陈旭
# @Software: PyCharm
# @File    : Evaluate.py


from keras.callbacks import Callback
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import json

from tqdm import tqdm

from data_loader import DataLoader


class Evaluate(Callback):

    def __init__(self, train_model, subject_model, object_model, EMAer=None):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0
        self.data_loader = DataLoader()
        self.train_model = train_model
        self.subject_model = subject_model
        self.object_model = object_model
        self.EMAer = EMAer

    def extract_items(self, text_in):
        text_words = self.data_loader.tokenize(text_in.lower())
        text_in = ''.join(text_words)
        R = []
        _t1 = [self.data_loader.char2id.get(c, 1) for c in text_in]
        _t1 = np.array([_t1])
        _t2 = self.data_loader.sent2vec([text_words])
        _k1, _k2 = self.subject_model.predict([_t1, _t2])
        _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
        _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.4)[0]
        _subjects = []
        for i in _k1:
            j = _k2[_k2 >= i]
            if len(j) > 0:
                j = j[0]
                _subject = text_in[i: j + 1]
                _subjects.append((_subject, i, j))
        if _subjects:
            _t1 = np.repeat(_t1, len(_subjects), 0)
            _t2 = np.repeat(_t2, len(_subjects), 0)
            _k1, _k2 = np.array([_s[1:]
                                 for _s in _subjects]).T.reshape((2, -1, 1))
            _o1, _o2 = self.object_model.predict([_t1, _t2, _k1, _k2])
            for i, _subject in enumerate(_subjects):
                _oo1, _oo2 = np.where(_o1[i] > 0.5), np.where(_o2[i] > 0.4)
                for _ooo1, _c1 in zip(*_oo1):
                    for _ooo2, _c2 in zip(*_oo2):
                        if _ooo1 <= _ooo2 and _c1 == _c2:
                            _object = text_in[_ooo1: _ooo2 + 1]
                            _predicate = self.data_loader.id2predicate[_c1]
                            R.append((_subject[0], _predicate, _object))
                            break
            zhuanji, gequ = [], []
            for s, p, o in R[:]:
                if p == u'妻子':
                    R.append((o, u'丈夫', s))
                elif p == u'丈夫':
                    R.append((o, u'妻子', s))
                if p == u'所属专辑':
                    zhuanji.append(o)
                    gequ.append(s)
            spo_list = set()
            for s, p, o in R:
                if p in [u'歌手', u'作词', u'作曲']:
                    if s in zhuanji and s not in gequ:
                        continue
                spo_list.add((s, p, o))
            return list(spo_list)
        else:
            return []

    # def on_batch_begin(self, batch, logs=None):
    #     """第一个epoch用来warmup，不warmup有不收敛的可能。
    #     """
    #     if self.passed < self.params['steps']:
    #         lr = (self.passed + 1.) / self.params['steps'] * 1e-3
    #         K.set_value(self.model.optimizer.lr, lr)
    #         self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        if self.EMAer != None:
            self.EMAer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            self.train_model.save_weights('best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' %
              (f1, precision, recall, self.best))
        if self.EMAer != None:
            self.EMAer.reset_old_weights()
        # if epoch == 2:
            # self.train_model.load_weights('best_model_2.weights')
        # if epoch + 1 == 5 or (
        #         self.stage == 0 and epoch > 10 and
        #         (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        # ):
        #     self.stage = 1
        #     self.train_model.load_weights('best_model.weights')
        #     if self.EMAer != None:
        #         self.EMAer.initialize()
        #     K.set_value(self.model.optimizer.lr, 1e-4)
        #     K.set_value(self.model.optimizer.iterations, 0)
        #     opt_weights = K.batch_get_value(self.model.optimizer.weights)
        #     opt_weights = [w * 0. for w in opt_weights]
        #     K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))

    def evaluate(self):
        orders = ['subject', 'predicate', 'object']
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(self.data_loader.dev_data)):
            R = set(self.extract_items(d['text']))
            T = set(d['spo_list'])
            A += len(R & T)
            B += len(R)
            C += len(T)
            s = json.dumps({
                'text': d['text'],
                'spo_list': [
                    dict(zip(orders, spo)) for spo in T
                ],
                'spo_list_pred': [
                    dict(zip(orders, spo)) for spo in R
                ],
                'new': [
                    dict(zip(orders, spo)) for spo in R - T
                ],
                'lack': [
                    dict(zip(orders, spo)) for spo in T - R
                ]
            }, ensure_ascii=False, indent=4)
            F.write(s + '\n')
        F.close()
        return 2 * A / (B + C), A / B, A / C

    def test(self, test_data):
        """输出测试结果
        """
        orders = ['subject', 'predicate', 'object',
                  'object_type', 'subject_type']
        F = open('test_pred.json', 'w')
        for d in tqdm(iter(test_data)):
            R = set(self.extract_items(d['text']))
            s = json.dumps({
                'text': d['text'],
                'spo_list': [
                    dict(zip(orders, spo + ('', ''))) for spo in R
                ]
            }, ensure_ascii=False)
            F.write(s.encode('utf-8') + '\n')
        F.close()
