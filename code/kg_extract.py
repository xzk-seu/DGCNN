'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-15 10:33:08
@LastEditTime: 2019-08-16 17:25:34
@LastEditors: Please set LastEditors
'''
#! -*- coding:utf-8 -*-


from __future__ import print_function
from Evaluate import Evaluate

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from data_loader import DataGenerator
from keras.callbacks import ReduceLROnPlateau


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。

    取出seq矩阵中，seq[batch_idxs][idxs]对应的向量
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1, keepdims=True)


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = Conv1D(dim * 2, 3, padding='same', dilation_rate=dilation_rate)(seq)

    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        # 把隐层输出分为前一半g和后一半h
        g, h = h[:, :, :dim], h[:, :, dim:]
        # 训练阶段对g进行drop_out
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h

    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq


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


class Attention(Layer):
    """多头注意力机制
    """

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(
            qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(
            kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(
            vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class PCNN():

    def __init__(self, data_generate):
        self.data_generate = data_generate

    def build_model(self):
        t1_in = Input(shape=(None,))
        t2_in = Input(shape=(None, self.data_generate.data_loader.word_size))
        s1_in = Input(shape=(None,))
        s2_in = Input(shape=(None,))
        k1_in = Input(shape=(1,))
        k2_in = Input(shape=(1,))
        o1_in = Input(shape=(None, self.data_generate.data_loader.num_classes))
        o2_in = Input(shape=(None, self.data_generate.data_loader.num_classes))

        t1, t2, s1, s2, k1, k2, o1, o2 = t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in

        # 因为输入字编码被统一为最长句子长度，不足此长度的通过padding补0达到次长度，musk为句子中不为0的不分，即可获取句子的真实长度
        mask = Lambda(lambda x: K.cast(
            K.greater(K.expand_dims(x, 2), 0), 'float32'))(t1)

        pid = Lambda(self.position_id)(t1)
        position_embedding = Embedding(self.data_generate.data_loader.maxlen, self.data_generate.data_loader.char_size,
                                       embeddings_initializer='zeros')
        pv = position_embedding(pid)

        t1 = Embedding(len(self.data_generate.data_loader.char2id) + 2, self.data_generate.data_loader.char_size)(
            t1)  # 0: padding, 1: unk
        t2 = Dense(self.data_generate.data_loader.char_size,
                   use_bias=False)(t2)  # 词向量也转为同样维度
        # 自向量和位置向量用Embedding转为了128维（char_size），256维的词向量用Dense转为了128维，在下面进行相加

        t = Add()([t1, t2, pv])  # 字向量、词向量、位置向量相加
        t = Dropout(0.25)(t)
        t = Lambda(lambda x: x[0] * x[1])([t, mask])
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
        t_dim = K.int_shape(t)[-1]

        pn1 = Dense(self.data_generate.data_loader.char_size,
                    activation='relu')(t)
        pn1 = Dense(1, activation='sigmoid')(pn1)
        pn2 = Dense(self.data_generate.data_loader.char_size,
                    activation='relu')(t)
        pn2 = Dense(1, activation='sigmoid')(pn2)

        h = Attention(8, 16)([t, t, t, mask])
        h = Concatenate()([t, h])
        h = Conv1D(self.data_generate.data_loader.char_size,
                   3, activation='relu', padding='same')(h)
        ps1 = Dense(1, activation='sigmoid')(h)
        ps2 = Dense(1, activation='sigmoid')(h)
        ps1 = Lambda(lambda x: x[0] * x[1])([ps1, pn1])
        ps2 = Lambda(lambda x: x[0] * x[1])([ps2, pn2])
        # pn1 pn2为实体指针
        # ps1 ps2为subject指针
        # 一个字要满足pn和ps同时为1才课看作实体头或尾

        subject_model = Model([t1_in, t2_in], [ps1, ps2])  # 预测subject的模型

        # t为共享编码层输出，即12层DGCNN的输出 128维

        t_max = Lambda(seq_maxpool)([t, mask])

        # pc为句子的分类输出
        pc = Dense(self.data_generate.data_loader.char_size,
                   activation='relu')(t_max)
        pc = Dense(self.data_generate.data_loader.num_classes,
                   activation='sigmoid')(pc)

        # 取到k1到k2之间长度为6的编码层向量序列，作为GRU的输入
        # GRU输入为（batch, 6, 128）输出为（batch, 256）
        k = Lambda(self.get_k_inter, output_shape=(6, t_dim))([t, k1, k2])
        # k = Bidirectional(CuDNNGRU(t_dim))(k)
        k = Bidirectional(GRU(t_dim))(k)
        k1v = position_embedding(Lambda(self.position_id)([t, k1]))  # (batch, len, 128)
        k2v = position_embedding(Lambda(self.position_id)([t, k2]))  # (batch, len, 128)
        kv = Concatenate()([k1v, k2v])  # 默认按最后一维连接 (batch, len, 256)  相对位置嵌入
        k = Lambda(lambda x: K.expand_dims(x[0], 1) + x[1])([k, kv])

        h = Attention(8, 16)([t, t, t, mask])
        h = Concatenate()([t, h, k])  # (batch, len, 128+128+256) = (batch, len, 512)
        h = Conv1D(self.data_generate.data_loader.char_size,
                   3, activation='relu', padding='same')(h)  # (batch, len, 128)
        po = Dense(1, activation='sigmoid')(h)  # 预测句子中该位置是否有obj
        po1 = Dense(self.data_generate.data_loader.num_classes,
                    activation='sigmoid')(h)
        po2 = Dense(self.data_generate.data_loader.num_classes,
                    activation='sigmoid')(h)

        po1 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po1, pc, pn1])
        po2 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po2, pc, pn2])

        object_model = Model([t1_in, t2_in, k1_in, k2_in], [
                             po1, po2])  # 输入text和subject，预测object及其关系

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

        # train_model.compile(optimizer=Adam(lr=1e-3))
        train_model.compile(optimizer=Adam(lr=1e-5))
        train_model.summary()
        self.train_model = train_model
        self.subject_model = subject_model
        self.object_model = object_model

    def position_id(self, x):
        """
        r 为偏移量，模型中会将subject头尾指针k1 k2作为偏移量，得到每个字符相对k1 k2的位置，从而进行位置嵌入
        首先获取一个位置列表 [0, 1, 2, 3.....] shape=(sent_len,)
        然后增加维度 [[0, 1, 2, 3.....]]  shape=(1,sent_len)
        接着第一个维度平铺batch倍，第二维度不变 [[0, 1, 2, 3.....], [0, 1, 2, 3.....], [0, 1, 2, 3.....],....]
        最后减去偏移量r

        该函数只用到了输入矩阵的维度，没有用到输入矩阵的内容
        :param x:
        :return:
        """
        if isinstance(x, list) and len(x) == 2:
            x, r = x
        else:
            r = 0
        pid = K.arange(K.shape(x)[1])
        pid = K.expand_dims(pid, 0)
        pid = K.tile(pid, [K.shape(x)[0], 1])
        return K.abs(pid - K.cast(r, 'int32'))

    def get_k_inter(self, x, n=6):
        seq, k1, k2 = x

        # [k2, 0.8*k2+0.2*k1, 0.6*k2+0.4*k1, 0.4*k2+0.6*k1, 0.2*k2+0.8*k1, k1]
        k_inter = [K.round(k1 * a + k2 * (1 - a))
                   for a in np.arange(n) / (n - 1.)]
        k_inter = [seq_gather([seq, k]) for k in k_inter]
        k_inter = [K.expand_dims(k, 1) for k in k_inter]
        k_inter = K.concatenate(k_inter, 1)
        return k_inter

    def train(self, data=None):
        EMAer = ExponentialMovingAverage(self.train_model)
        EMAer.inject()
        if data == None:
            data = self.data_generate
        # self.train_model.load_weights('best_train_model.weights')
        # self.subject_model.load_weights("best_subject_model.weights")
        # self.object_model.load_weights("best_object_model.weights")
        evaluator = Evaluate(
            self.train_model, self.subject_model, self.object_model, EMAer)
        # reduce_lr = ReduceLROnPlateau(monitor='loss', patience=1, mode='auto')
        self.train_model.fit_generator(data.__iter__(),
                                       steps_per_epoch=len(data),
                                       epochs=60,
                                       callbacks=[evaluator]
                                       )


if __name__ == '__main__':
    data_generate = DataGenerator(train=True, batch_size=64)
    pcnn_model = PCNN(data_generate)
    pcnn_model.build_model()
    pcnn_model.train()
