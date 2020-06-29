import configparser
import os

import numpy as np
import paddle.fluid as fluid
from tqdm import tqdm

import attention_model as att
from BiGRU import BiGRU
from conv1d import Conv1d, DilatedGatedConv1d
from data_loader import DataGeneratorPaddle

cf = configparser.ConfigParser()
conf_path = os.path.join(os.getcwd(), os.path.pardir, "config", "dgcnn_paddle.conf")
cf.read(conf_path)
EPOCH = cf.getint("model", "EPOCH")
BATCH_SIZE = cf.getint("model", "BATCH_SIZE")
USE_GPU = cf.getboolean("model", "USE_GPU")
PRINT_PER_BATCH = cf.getint("model", "PRINT_PER_BATCH")
LEARNING_RATE = cf.getfloat("model", "LEARNING_RATE")

"""
预测部分还没写
"""


class MyModel(fluid.dygraph.Layer):
    def __init__(self, max_len, char_size, char_id_len, class_num):
        super(MyModel, self).__init__(None)
        self.max_len = max_len  # 256
        self.char_size = char_size  # 128
        self.char_id_len = char_id_len  # 7028
        self.wv_len = 256
        self.h_dim = 128
        self.class_num = class_num

        t1_emb_size = [self.char_id_len + 2, self.char_size]
        self.t1_embedding = fluid.dygraph.Embedding(size=t1_emb_size)

        self.pos_emb_size = [self.max_len, self.char_size]
        self.position_embedding = fluid.dygraph.Embedding(size=self.pos_emb_size)

        self.t2_fc = fluid.dygraph.Linear(self.wv_len, char_size)

        self.h_conv1d = Conv1d(input_dim=256, output_dim=self.char_size, kernel_size=3, activation="relu")
        self.h_conv1d_2 = Conv1d(input_dim=512, output_dim=self.char_size, kernel_size=3, activation="relu")

        self.self_att_1 = att.MultiHead_Attention(8, 16)
        self.self_att_2 = att.MultiHead_Attention(8, 16)

        # add
        self.drop_out1 = fluid.dygraph.Dropout(p=0.5)

        self.dgcnn1 = DilatedGatedConv1d(char_size, 1)
        self.dgcnn2 = DilatedGatedConv1d(char_size, 2)
        self.dgcnn3 = DilatedGatedConv1d(char_size, 5)
        self.dgcnn4 = DilatedGatedConv1d(char_size, 1)
        self.dgcnn5 = DilatedGatedConv1d(char_size, 2)
        self.dgcnn6 = DilatedGatedConv1d(char_size, 5)
        self.dgcnn7 = DilatedGatedConv1d(char_size, 1)
        self.dgcnn8 = DilatedGatedConv1d(char_size, 2)
        self.dgcnn9 = DilatedGatedConv1d(char_size, 5)
        self.dgcnn10 = DilatedGatedConv1d(char_size, 1)
        self.dgcnn11 = DilatedGatedConv1d(char_size, 1)
        self.dgcnn12 = DilatedGatedConv1d(char_size, 1)

        self.fc_pn1 = fluid.dygraph.Linear(char_size, char_size, act="relu")
        self.fc_pn2 = fluid.dygraph.Linear(char_size, 1, act="sigmoid")
        self.fc_pn3 = fluid.dygraph.Linear(char_size, char_size, act="relu")
        self.fc_pn4 = fluid.dygraph.Linear(char_size, 1, act="sigmoid")

        self.fc_pc1 = fluid.dygraph.Linear(char_size, char_size, act="relu")
        self.fc_pc2 = fluid.dygraph.Linear(char_size, self.class_num, act="sigmoid")

        self.fc_ps1 = fluid.dygraph.Linear(char_size, 1, act="sigmoid")
        self.fc_ps2 = fluid.dygraph.Linear(char_size, 1, act="sigmoid")

        self.bigru = BiGRU(self.h_dim)
        self.k1v_emb = fluid.dygraph.Embedding(size=self.pos_emb_size)
        self.k2v_emb = fluid.dygraph.Embedding(size=self.pos_emb_size)

        self.fc_po = fluid.dygraph.Linear(char_size, 1, act="sigmoid")
        self.fc_po1 = fluid.dygraph.Linear(char_size, self.class_num, act="sigmoid")
        self.fc_po2 = fluid.dygraph.Linear(char_size, self.class_num, act="sigmoid")

    def forward(self, inputs: list):
        t1, t2, k1, k2, mask = inputs

        pid = position_id(t1)

        pv = self.position_embedding(pid)
        t1 = self.t1_embedding(t1)

        t2 = t2.astype("float32")
        t2 = self.t2_fc(t2)

        t = fluid.layers.elementwise_add(t1, t2)
        t = fluid.layers.elementwise_add(t, pv)
        t = self.drop_out1(t)
        t = t * mask

        # dilation_rates = [1, 2, 5, 1, 2, 5, 1, 2, 5, 1, 1, 1]
        # dgcnns = [DilatedGatedConv1d(dim=t_dim, dilation_rate=x) for x in dilation_rates]
        # for dgcnn in dgcnns:
        #     t = dgcnn(t, mask)

        t = self.dgcnn1(t, mask)
        t = self.dgcnn2(t, mask)
        t = self.dgcnn3(t, mask)
        t = self.dgcnn4(t, mask)
        t = self.dgcnn5(t, mask)
        t = self.dgcnn6(t, mask)
        t = self.dgcnn7(t, mask)
        t = self.dgcnn8(t, mask)
        t = self.dgcnn9(t, mask)
        t = self.dgcnn10(t, mask)
        t = self.dgcnn11(t, mask)
        t = self.dgcnn12(t, mask)

        pn1 = self.fc_pn1(t)
        pn1 = self.fc_pn2(pn1)
        pn2 = self.fc_pn3(t)
        pn2 = self.fc_pn4(pn2)

        h = self.self_att_1([t, t, t, mask])
        h = fluid.layers.concat([t, h], axis=-1)
        h = self.h_conv1d(h)

        ps1 = self.fc_ps1(h)
        ps2 = self.fc_ps2(h)

        ps1 = ps1 * pn1
        ps2 = ps2 * pn2
        # subject end

        t_max = seq_maxpool([t, mask])
        pc = self.fc_pc1(t_max)
        pc = self.fc_pc2(pc)

        # k = Lambda(self.get_k_inter, output_shape=(6, t_dim))([t, k1, k2])
        k = get_k_inter([t, k1, k2], 6)
        k = self.bigru(k)

        # k1v = position_embedding(Lambda(self.position_id)([t, k1]))  # (batch, len, 128)
        # k2v = position_embedding(Lambda(self.position_id)([t, k2]))  # (batch, len, 128)
        # kv = Concatenate()([k1v, k2v])  # 默认按最后一维连接 (batch, len, 256)  相对位置嵌入
        # k = Lambda(lambda x: K.expand_dims(x[0], 1) + x[1])([k, kv])
        k1v = position_id([t, k1])
        k1v = self.k1v_emb(k1v)
        k2v = position_id([t, k2])
        k2v = self.k2v_emb(k2v)
        kv = fluid.layers.concat([k1v, k2v], -1)
        k = fluid.layers.unsqueeze(k, 1) + kv

        # h = Attention(8, 16)([t, t, t, mask])
        # h = Concatenate()([t, h, k])  # (batch, len, 128+128+256) = (batch, len, 512)
        # h = Conv1D(self.data_generate.data_loader.char_size,
        #            3, activation='relu', padding='same')(h)  # (batch, len, 128)
        h = self.self_att_1([t, t, t, mask])
        h = fluid.layers.concat([t, h, k], -1)
        h = self.h_conv1d_2(h)

        # po = Dense(1, activation='sigmoid')(h)  # 预测句子中该位置是否有obj
        # po1 = Dense(self.data_generate.data_loader.num_classes,
        #             activation='sigmoid')(h)
        # po2 = Dense(self.data_generate.data_loader.num_classes,
        #             activation='sigmoid')(h)
        #
        # po1 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po1, pc, pn1])
        # po2 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po2, pc, pn2])
        po = self.fc_po(h)
        po1 = self.fc_po1(h)
        po2 = self.fc_po2(h)
        po1 = po * po1 * pc * pn1
        po2 = po * po2 * pc * pn2

        return ps1, ps2, po1, po2


def get_k_inter(x, n=6):
    seq, k1, k2 = x
    k1 = k1.astype("float32")
    k2 = k2.astype("float32")

    # [k2, 0.8*k2+0.2*k1, 0.6*k2+0.4*k1, 0.4*k2+0.6*k1, 0.2*k2+0.8*k1, k1]
    # k_inter = [K.round(k1 * a + k2 * (1 - a))
    #            for a in np.arange(n) / (n - 1.)]
    k_inter = [fluid.layers.round(k1 * a + k2 * (1 - a)) for a in np.arange(n) / (n - 1.)]
    k_inter = [seq_gather([seq, k]) for k in k_inter]
    # k_inter = [K.expand_dims(k, 1) for k in k_inter]
    k_inter = [fluid.layers.unsqueeze(k, 1) for k in k_inter]
    # k_inter = K.concatenate(k_inter, 1)
    k_inter = fluid.layers.concat(k_inter, 1)
    return k_inter


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。

    取出seq矩阵中，seq[batch_idxs][idxs]对应的向量
    """
    seq, idxs = x
    # idxs = K.cast(idxs, 'int32')
    idxs = idxs.astype("int32")
    # batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = fluid.layers.arange(0, seq.shape[0])
    # batch_idxs = K.expand_dims(batch_idxs, 1)
    batch_idxs = fluid.layers.unsqueeze(batch_idxs, 1).astype("int32")
    # idxs = K.concatenate([batch_idxs, idxs], 1)
    idxs = fluid.layers.concat([batch_idxs, idxs], 1)
    r = fluid.layers.gather_nd(seq, idxs)
    return r


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    r = fluid.layers.reduce_max(seq, dim=1, keep_dim=True)
    return r


def position_id(x):
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
        r = fluid.dygraph.to_variable(np.array([r], dtype="int64"))

    # batch_size, sent_len = fluid.layers.cast(fluid.layers.shape(x), dtype="int64")
    batch_size = x.shape[0]
    sent_len = x.shape[1]
    pid = fluid.layers.arange(0, sent_len, dtype="int64")
    pid = fluid.layers.unsqueeze(pid, [0])
    pid = fluid.layers.expand(pid, [int(batch_size), 1])
    pid = fluid.layers.abs(pid - r)

    return pid


def train(data_generate):
    place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        model = MyModel(max_len=data_generate.data_loader.maxlen,
                        char_size=data_generate.data_loader.char_size,
                        char_id_len=len(data_generate.data_loader.char2id),
                        class_num=data_generate.data_loader.num_classes)

        # optimizer = fluid.optimizer.SGDOptimizer(learning_rate=LEARNING_RATE,
        # parameter_list=model.parameters())
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=LEARNING_RATE,
                                                  parameter_list=model.parameters())

        for epoch in range(EPOCH):
            data_loader = fluid.io.DataLoader.from_generator(capacity=64)
            data_loader.set_batch_generator(data_generate.batch_generator_creator(), places=place)
            for bt_id, data in tqdm(enumerate(data_loader())):
                # for data in data_loader():
                t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in = data
                t1 = fluid.dygraph.to_variable(t1_in)
                t2 = fluid.dygraph.to_variable(t2_in)
                s1 = fluid.dygraph.to_variable(s1_in)
                s2 = fluid.dygraph.to_variable(s2_in)
                k1 = fluid.dygraph.to_variable(k1_in)
                k2 = fluid.dygraph.to_variable(k2_in)
                o1 = fluid.dygraph.to_variable(o1_in)
                o2 = fluid.dygraph.to_variable(o2_in)

                mask = fluid.layers.unsqueeze(t1, [2])
                mask = fluid.layers.cast((mask > 0), "float32")

                ps1, ps2, po1, po2 = model([t1, t2, k1, k2, mask])

                # s1 = fluid.layers.cast(s1, "float32")
                # s2 = fluid.layers.cast(s2, "float32")
                s1 = s1.astype("float32")
                s2 = s2.astype("float32")
                s1 = fluid.layers.unsqueeze(s1, [2])
                s2 = fluid.layers.unsqueeze(s2, [2])

                # s1_loss = fluid.layers.cross_entropy(ps1, s1)
                # s2_loss = fluid.layers.cross_entropy(ps2, s2)
                s1_loss = fluid.layers.cross_entropy(ps1, s1, soft_label=True)
                s2_loss = fluid.layers.cross_entropy(ps2, s2, soft_label=True)
                s1_loss = fluid.layers.reduce_sum(s1_loss * mask) / fluid.layers.reduce_sum(mask)
                s2_loss = fluid.layers.reduce_sum(s2_loss * mask) / fluid.layers.reduce_sum(mask)

                # o1_loss = K.sum(K.binary_crossentropy(o1, po1), 2, keepdims=True)
                # o1_loss = K.sum(o1_loss * mask) / K.sum(mask)
                # o2_loss = K.sum(K.binary_crossentropy(o2, po2), 2, keepdims=True)
                # o2_loss = K.sum(o2_loss * mask) / K.sum(mask)
                o1 = o1.astype("float32")
                o2 = o2.astype("float32")
                o1_loss = fluid.layers.cross_entropy(po1, o1, soft_label=True)
                o1_loss = fluid.layers.reduce_sum(o1_loss * mask) / fluid.layers.reduce_sum(mask)
                o2_loss = fluid.layers.cross_entropy(po2, o2, soft_label=True)
                o2_loss = fluid.layers.reduce_sum(o2_loss * mask) / fluid.layers.reduce_sum(mask)

                # ave_loss = fluid.layers.mean(s1_loss+s2_loss)
                ave_loss = s1_loss + s2_loss + o1_loss + o2_loss
                # losss1 = s1_loss.numpy()
                # losss2 = s2_loss.numpy()
                # losso1 = o1_loss.numpy()
                # losso2 = o2_loss.numpy()
                # losssum = ave_loss.numpy()

                if bt_id % PRINT_PER_BATCH == 0:
                    print('epoch:{}\tbatch:{}\tloss:{}'.format(epoch, bt_id, ave_loss.numpy()))

                ave_loss.backward()
                optimizer.minimize(ave_loss)
                model.clear_gradients()

            save_path = os.path.join(os.getcwd(), os.pardir, "dygraph_model")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = "%d_sub_model_%.7f" % (epoch, float(ave_loss.numpy()))
            save_path = os.path.join(save_path, file_name)
            fluid.save_dygraph(model.state_dict(), save_path)


def my_test():
    generator = generate_img()

    with fluid.dygraph.guard():
        model = Mnist_dense()
        model_dict, _ = fluid.load_dygraph('mnist_test.model')
        model.load_dict(model_dict)

        model.eval()

        for i in range(10):
            # image, label = generate_img()
            image, label = next(generator)
            # print (
            #     'bug', image
            # )

            # x = input()

            # show_img(image[0])

            x_input = fluid.dygraph.to_variable(image)

            predict = model(x_input)

            print(np.argmax(predict.numpy(), axis=1))
            print(label.astype('int64'))


def main():
    data_generate = DataGeneratorPaddle(train=True, batch_size=64)
    train(data_generate)


if __name__ == '__main__':
    main()
