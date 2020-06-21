#! -*- coding:utf-8 -*-


from __future__ import print_function
import json
import numpy as np
from random import choice
from tqdm import tqdm
# import pyhanlp
from gensim.models import Word2Vec
import re, os
import jieba

mode = 0
char_size = 128
maxlen = 512

word2vec = Word2Vec.load('../word2vec_baike/word2vec_baike')

id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
word2id = {j: i for i, j in id2word.items()}
word2vec = word2vec.wv.syn0
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])


def tokenize(s):
    return list(jieba.cut(s))
    # return [i.word for i in pyhanlp.HanLP.segment(s)]


def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


total_data = json.load(open('../datasets/train_data_vote_me.json'))
id2predicate, predicate2id = json.load(open('../datasets/all_50_schemas_me.json'))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open('../datasets/all_chars_me.json'))
num_classes = len(id2predicate)

if not os.path.exists('../random_order_vote.json'):
    random_order = range(len(total_data))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('../random_order_vote.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('../random_order_vote.json'))

train_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 != mode]
dev_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 == mode]

predicates = {}  # 格式：{predicate: [(subject, predicate, object)]}


def repair(d):
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
        if sp[1] in [u'歌手', u'作词', u'作曲']:
            if sp[0] in zhuanji and sp[0] not in gequ:
                continue
        spo_list.append(tuple(sp))
    d['spo_list'] = spo_list


for d in train_data:
    repair(d)
    for sp in d['spo_list']:
        if sp[1] not in predicates:
            predicates[sp[1]] = []
        predicates[sp[1]].append(sp)

for d in dev_data:
    repair(d)


def random_generate(d, spo_list_key):
    r = np.random.random()
    if r > 0.5:
        return d
    else:
        k = np.random.randint(len(d[spo_list_key]))
        spi = d[spo_list_key][k]
        k = np.random.randint(len(predicates[spi[1]]))
        spo = predicates[spi[1]][k]
        F = lambda s: s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]
        return {'text': text, spo_list_key: spo_list}


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(idxs)
            T1, T2, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], [], []
            for i in idxs:
                spo_list_key = 'spo_list'  # if np.random.random() > 0.5 else 'spo_list_with_pred'
                d = random_generate(self.data[i], spo_list_key)
                text = d['text'][:maxlen]
                text_words = tokenize(text)
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
                                           predicate2id[sp[1]]))
                if items:
                    T1.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                    T2.append(text_words)
                    s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1] - 1] = 1
                    k1, k2 = np.array(items.keys()).T
                    k1 = choice(k1)
                    k2 = choice(k2[k2 >= k1])
                    o1, o2 = np.zeros((len(text), num_classes)), np.zeros((len(text), num_classes))
                    for j in items.get((k1, k2), []):
                        o1[j[0]][j[2]] = 1
                        o2[j[1] - 1][j[2]] = 1
                    S1.append(s1)
                    S2.append(s2)
                    K1.append([k1])
                    K2.append([k2 - 1])
                    O1.append(o1)
                    O2.append(o2)
                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)
                        T2 = sent2vec(T2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        O1 = seq_padding(O1, np.zeros(num_classes))
                        O2 = seq_padding(O2, np.zeros(num_classes))
                        K1, K2 = np.array(K1), np.array(K2)
                        yield [T1, T2, S1, S2, K1, K2, O1, O2], None
                        T1, T2, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], [], []


from paddle import fluid


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = fluid.layers.cast(idxs, 'int32')
    batch_idxs = fluid.layers.arange(0, fluid.layers.shape(seq)[0])
    batch_idxs = fluid.layers.unsqueeze(batch_idxs, [1])
    idxs = fluid.layers.concat([batch_idxs, idxs], 1)
    return fluid.layers.gather_nd(seq, idxs)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return fluid.layers.reduce_max(seq, 1, keep_dim=True)


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = fluid.layers.shape(seq)[-1]
    h = fluid.layers.sequence_conv(seq, dim * 2, 3)  # 缺少 dilation_rate
    g, h = h[:, :, :dim], h[:, :, dim:]
    g = fluid.layers.dropout(g, dropout_prob=0.1,
                             dropout_implementation="upscale_in_train")
    g = fluid.layers.sigmoid(g)
    seq = g * seq + (1 - g) * h
    seq = fluid.layers.elementwise_mul(seq, mask)
    return seq


t1_in = fluid.layers.data(name='t1_in', shape=[None], dtype='float32')
t2_in = fluid.layers.data(name='t2_in', shape=[None, word_size], dtype='float32')
s1_in = fluid.layers.data(name='s1_in', shape=[None], dtype='float32')
s2_in = fluid.layers.data(name='s2_in', shape=[None], dtype='float32')
k1_in = fluid.layers.data(name='s1_in', shape=[1], dtype='float32')
k2_in = fluid.layers.data(name='s2_in', shape=[2], dtype='float32')

o1_in = fluid.layers.data(name='s1_in', shape=[None, num_classes], dtype='float32')
o2_in = fluid.layers.data(name='s2_in', shape=[None, num_classes], dtype='float32')

t1, t2, s1, s2, k1, k2, o1, o2 \
    = t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in

# 计算mask
mask_h = fluid.layers.unsqueeze(t1, [2])
mask_zero = fluid.layers.zeros(shape=mask_h.shape, dtype="float32")
mask = fluid.layers.cast(fluid.layers.greater_than(mask_h, mask_zero), 'float32')


def position_id(x):
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = fluid.layers.arange(0, fluid.layers.shape(x)[1])
    pid = fluid.layers.unsqueeze(pid, [0])
    pid = fluid.layers.expand(pid, [fluid.layers.shape(x)[0], 1])
    return fluid.layers.abs(pid - fluid.layers.cast(r, 'int32'))


pid = position_id(t1)

pv = fluid.layers.embedding(pid, [maxlen, char_size])

t1 = fluid.layers.embedding(t1, [len(char2id) + 2, char_size])  # 0: padding, 1: unk
t2 = fluid.layers.fc(t2, char_size)  # 词向量也转为同样维度
t = fluid.layers.elementwise_add(t1, t2)  # 字向量、词向量、位置向量相加
t = fluid.layers.elementwise_add(t, pv)
t = fluid.layers.dropout(t, dropout_prob=0.25, dropout_implementation="upscale_in_train")
t = fluid.layers.elementwise_mul(t, mask)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 1)

t_dim = fluid.layers.shape(t)[-1]

pn1 = fluid.layers.fc(t, char_size, act="relu")
pn1 = fluid.layers.fc(pn1, 1, act='sigmoid')

pn2 = fluid.layers.fc(t, char_size, act='relu')
pn2 = fluid.layers.fc(pn2, 1, act='sigmoid')

h = fluid.nets.scaled_dot_product_attention(t, t, t, num_heads=8)
h = fluid.layers.concat(t, h)
h = fluid.nets.sequence_conv_pool(h, char_size, 3, act="relu")

ps1 = fluid.layers.fc(h, 1, act='sigmoid')
ps2 = fluid.layers.fc(h, 1, act='sigmoid')
ps1 = fluid.layers.elementwise_mul(ps1, pn1)
ps2 = fluid.layers.elementwise_mul(ps2, pn2)

subject_model = Model([t1_in, t2_in], [ps1, ps2])  # 预测subject的模型

t_max = seq_maxpool([t, mask])
pc = Dense(char_size, activation='relu')(t_max)
pc = Dense(num_classes, activation='sigmoid')(pc)


def get_k_inter(x, n=6):
    seq, k1, k2 = x
    k_inter = [K.round(k1 * a + k2 * (1 - a)) for a in np.arange(n) / (n - 1.)]
    k_inter = [seq_gather([seq, k]) for k in k_inter]
    k_inter = [K.expand_dims(k, 1) for k in k_inter]
    k_inter = K.concatenate(k_inter, 1)
    return k_inter


k = Lambda(get_k_inter, output_shape=(6, t_dim))([t, k1, k2])
k = Bidirectional(CuDNNGRU(t_dim))(k)
k1v = position_embedding(Lambda(position_id)([t, k1]))
k2v = position_embedding(Lambda(position_id)([t, k2]))
kv = Concatenate()([k1v, k2v])
k = Lambda(lambda x: K.expand_dims(x[0], 1) + x[1])([k, kv])

h = Attention(8, 16)([t, t, t, mask])
h = Concatenate()([t, h, k])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
po = Dense(1, activation='sigmoid')(h)
po1 = Dense(num_classes, activation='sigmoid')(h)
po2 = Dense(num_classes, activation='sigmoid')(h)
po1 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po1, pc, pn1])
po2 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po2, pc, pn2])

object_model = Model([t1_in, t2_in, k1_in, k2_in], [po1, po2])  # 输入text和subject，预测object及其关系

train_model = Model([t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in],
                    [ps1, ps2, po1, po2])

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * mask) / K.sum(mask)

o1_loss = K.sum(K.binary_crossentropy(o1, po1), 2, keepdims=True)
o1_loss = K.sum(o1_loss * mask) / K.sum(mask)
o2_loss = K.sum(K.binary_crossentropy(o2, po2), 2, keepdims=True)
o2_loss = K.sum(o2_loss * mask) / K.sum(mask)

loss = (s1_loss + s2_loss) + (o1_loss + o2_loss)

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """

    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]

    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)

    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))

    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))

    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


EMAer = ExponentialMovingAverage(train_model)
EMAer.inject()


def extract_items(text_in):
    text_words = tokenize(text_in.lower())
    text_in = ''.join(text_words)
    R = []
    _t1 = [char2id.get(c, 1) for c in text_in]
    _t1 = np.array([_t1])
    _t2 = sent2vec([text_words])
    _k1, _k2 = subject_model.predict([_t1, _t2])
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
        _k1, _k2 = np.array([_s[1:] for _s in _subjects]).T.reshape((2, -1, 1))
        _o1, _o2 = object_model.predict([_t1, _t2, _k1, _k2])
        for i, _subject in enumerate(_subjects):
            _oo1, _oo2 = np.where(_o1[i] > 0.5), np.where(_o2[i] > 0.4)
            for _ooo1, _c1 in zip(*_oo1):
                for _ooo2, _c2 in zip(*_oo2):
                    if _ooo1 <= _ooo2 and _c1 == _c2:
                        _object = text_in[_ooo1: _ooo2 + 1]
                        _predicate = id2predicate[_c1]
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


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，不warmup有不收敛的可能。
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * 1e-3
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
        EMAer.reset_old_weights()
        if epoch + 1 == 50 or (
                self.stage == 0 and epoch > 10 and
                (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        ):
            self.stage = 1
            train_model.load_weights('best_model.weights')
            EMAer.initialize()
            K.set_value(self.model.optimizer.lr, 1e-4)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))

    def evaluate(self):
        orders = ['subject', 'predicate', 'object']
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(dev_data)):
            R = set(extract_items(d['text']))
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
            F.write(s.encode('utf-8') + '\n')
        F.close()
        return 2 * A / (B + C), A / B, A / C


def test(test_data):
    """输出测试结果
    """
    orders = ['subject', 'predicate', 'object', 'object_type', 'subject_type']
    F = open('test_pred.json', 'w')
    for d in tqdm(iter(test_data)):
        R = set(extract_items(d['text']))
        s = json.dumps({
            'text': d['text'],
            'spo_list': [
                dict(zip(orders, spo + ('', ''))) for spo in R
            ]
        }, ensure_ascii=False)
        F.write(s.encode('utf-8') + '\n')
    F.close()


train_D = data_generator(train_data)
evaluator = Evaluate()

if __name__ == '__main__':
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=120,
                              callbacks=[evaluator]
                              )
else:
    train_model.load_weights('best_model.weights')
