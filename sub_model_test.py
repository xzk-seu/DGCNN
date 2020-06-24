import numpy as np
import paddle.fluid as fluid
from data_loader import DataGenerator
import paddle

EPOCH = 1
BATCH_SIZE = 64
USE_GPU = True


class Conv1d(fluid.dygraph.Layer):
    def __init__(self, input_dim, output_dim, kernel_size, activation=None, dilation_rate=1):
        super(Conv1d, self).__init__(None)

        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.rec_field = self.kernel_size + (self.kernel_size - 1) * (self.dilation_rate - 1)  # 感受野 等效核
        self.pad = self.rec_field // 2

        self.con1d = fluid.dygraph.Conv2D(num_channels=1,
                                          num_filters=output_dim,
                                          filter_size=(3, input_dim),
                                          padding=(self.pad, 0),
                                          dilation=(dilation_rate, 1),
                                          act=activation)

    def forward(self, seq):
        h = fluid.layers.unsqueeze(seq, axes=[1])
        h = self.con1d(h)
        h = fluid.layers.squeeze(h, axes=[3])
        h = fluid.layers.transpose(h, perm=[0, 2, 1])
        return h


class DilatedGatedConv1d(fluid.dygraph.Layer):
    def __init__(self, dim, dilation_rate):
        super(DilatedGatedConv1d, self).__init__(None)
        self.dim = int(dim)
        self.con1d = Conv1d(input_dim=self.dim, output_dim=2 * self.dim, kernel_size=3, dilation_rate=dilation_rate)

    def forward(self, seq, mask):
        """
        膨胀门卷积（残差式）
        """
        h = self.con1d(seq)

        def _gate(x):
            dropout_rate = 0.1
            s, h = x
            g, h = h[:, :, :self.dim], h[:, :, self.dim:]
            drop_out = fluid.dygraph.Dropout(p=dropout_rate)
            g = drop_out(g)

            g = fluid.layers.sigmoid(g)
            return g * s + (1 - g) * h
        seq = _gate([seq, h])
        seq = seq * mask
        return seq


class SubjectModel(fluid.dygraph.Layer):
    def __init__(self, max_len, char_size, char_id_len):
        super(SubjectModel, self).__init__(None)
        self.max_len = max_len  # 256
        self.char_size = char_size  # 128
        self.char_id_len = char_id_len  # 7028
        self.wv_len = 256
        self.h_dim = 128

        t1_emb_size = [self.char_id_len + 2, self.char_size]
        self.t1_embedding = fluid.dygraph.Embedding(size=t1_emb_size)

        pos_emb_size = [self.max_len, self.char_size]
        self.position_embedding = fluid.dygraph.Embedding(size=pos_emb_size)

        self.t2_fc = fluid.dygraph.Linear(self.wv_len, char_size)

        self.h_conv1d = Conv1d(input_dim=256, output_dim=self.char_size, kernel_size=3, activation="relu")

    def forward(self, inputs: list):
        t1, t2 = inputs
        mask = fluid.layers.unsqueeze(t1, [2])
        mask = fluid.layers.cast((mask > 0), "float32")

        pid = position_id(t1)

        pv = self.position_embedding(pid)
        t1 = self.t1_embedding(t1)

        t2 = fluid.layers.cast(t2, "float32")
        t2 = self.t2_fc(t2)

        t = fluid.layers.elementwise_add(t1, t2)
        t = fluid.layers.elementwise_add(t, pv)
        t = fluid.layers.dropout(t, 0.5)
        t = t * mask
        t_dim = fluid.layers.shape(t)[-1]

        dilation_rates = [1, 2, 5, 1, 2, 5, 1, 2, 5, 1, 1, 1]
        dgcnns = [DilatedGatedConv1d(dim=t_dim, dilation_rate=x) for x in dilation_rates]
        for dgcnn in dgcnns:
            t = dgcnn(t, mask)

        pn1 = fluid.layers.fc(t, self.char_size, num_flatten_dims=2, act="relu")
        pn1 = fluid.layers.fc(pn1, 1, num_flatten_dims=2, act="sigmoid")
        pn2 = fluid.layers.fc(t, self.char_size, num_flatten_dims=2, act="relu")
        pn2 = fluid.layers.fc(pn2, 1, num_flatten_dims=2, act="sigmoid")

        # h = Attention(8, 16)([t, t, t, mask])
        # h = Concatenate()([t, h])
        # h = Conv1D(self.data_generate.data_loader.char_size,
        #            3, activation='relu', padding='same')(h)
        # ps1 = Dense(1, activation='sigmoid')(h)
        # ps2 = Dense(1, activation='sigmoid')(h)
        # ps1 = Lambda(lambda x: x[0] * x[1])([ps1, pn1])
        # ps2 = Lambda(lambda x: x[0] * x[1])([ps2, pn2])
        h = fluid.nets.scaled_dot_product_attention(t, t, t, num_heads=8)
        h = fluid.layers.concat([t, h], axis=-1)
        h = self.h_conv1d(h)
        ps1 = fluid.layers.fc(h, 2, num_flatten_dims=2, act="sigmoid")
        ps2 = fluid.layers.fc(h, 2, num_flatten_dims=2, act="sigmoid")
        ps1 = ps1 * pn1
        ps2 = ps2 * pn2

        return ps1, ps2


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
    batch_size, sent_len = fluid.layers.cast(fluid.layers.shape(x), dtype="int64")
    pid = fluid.layers.arange(0, sent_len, dtype="int64")
    pid = fluid.layers.unsqueeze(pid, [0])
    pid = fluid.layers.expand(pid, [int(batch_size), 1])
    pid = fluid.layers.abs(pid - r)

    return pid


class MyModel(object):
    def __init__(self, data: DataGenerator):
        self.data_generate = data

    def train(self):
        place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()

        with fluid.dygraph.guard(place):
            sub_model = SubjectModel(max_len=self.data_generate.data_loader.maxlen,
                                     char_size=self.data_generate.data_loader.char_size,
                                     char_id_len=len(self.data_generate.data_loader.char2id))
            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=sub_model.parameters())

            for epoch in range(EPOCH):
                for bt_id, data in enumerate(self.data_generate.__iter__()):
                    data, _ = data
                    t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in = data
                    t1 = fluid.dygraph.to_variable(t1_in)
                    t2 = fluid.dygraph.to_variable(t2_in)
                    s1 = fluid.dygraph.to_variable(s1_in)
                    s2 = fluid.dygraph.to_variable(s2_in)
                    ps1, ps2 = sub_model([t1, t2])

                    s1 = fluid.layers.cast(s1, "int64")
                    s2 = fluid.layers.cast(s2, "int64")
                    s1_loss = fluid.layers.cross_entropy(ps1, s1)
                    s2_loss = fluid.layers.cross_entropy(ps2, s2)

                    ave_loss = fluid.layers.mean(s1_loss+s2_loss)

                    if bt_id % 10 == 0:
                        print('epoch:{}\tbatch:{}\tloss:{}'.format(epoch, bt_id, ave_loss.numpy()))

                    ave_loss.backward()
                    optimizer.minimize(ave_loss)
                    sub_model.clear_gradients()

            fluid.save_dygraph(sub_model.state_dict(), 'sub.model')


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
    data_generate = DataGenerator(train=True, batch_size=64)
    my_model = MyModel(data_generate)
    my_model.train()


if __name__ == '__main__':
    main()
