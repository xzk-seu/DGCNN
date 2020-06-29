import numpy as np
import paddle.fluid as fluid


class MultiHead_Attention(fluid.dygraph.Layer):
    """
    多头注意力机制
    """

    def __init__(self, nb_head, size_per_head, **kwargs):

        super(MultiHead_Attention, self).__init__()

        self.n_heads = nb_head
        self.hid_dim = nb_head * size_per_head

        # d_model // h 仍然是要能整除  - 换个名字仍然意义不变

        self.w_q = fluid.dygraph.Linear(self.hid_dim, self.hid_dim)
        self.w_k = fluid.dygraph.Linear(self.hid_dim, self.hid_dim)
        self.w_v = fluid.dygraph.Linear(self.hid_dim, self.hid_dim)

        self.fc = fluid.dygraph.Linear(self.hid_dim, self.hid_dim)

        # self.dorp_out = fluid.dygraph.Dropout(dropout_rate)

        self.scale = fluid.layers.sqrt(fluid.dygraph.to_variable(
            np.array([self.hid_dim // self.n_heads], "float32")))

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(len(x.shape) - len(mask.shape)):
                mask = fluid.layers.unsqueeze(mask, len(mask.shape))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def forward(self, inputs):

        # Q,K,V 计算与变形：
        bsz = inputs[0].shape[0]
        Q, K, V = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]

        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)

        Q = fluid.layers.reshape(Q, shape=[bsz, -1, self.n_heads, self.hid_dim //
                                           self.n_heads])
        Q = fluid.layers.transpose(Q, perm=[0, 2, 1, 3])

        K = fluid.layers.reshape(K, shape=[bsz, -1, self.n_heads, self.hid_dim //
                                           self.n_heads])
        K = fluid.layers.transpose(K, perm=[0, 2, 1, 3])

        V = fluid.layers.reshape(Q, shape=[bsz, -1, self.n_heads, self.hid_dim //
                                           self.n_heads])
        V = fluid.layers.transpose(V, perm=[0, 2, 1, 3])

        # Q, K相乘除以scale
        energy = fluid.layers.matmul(Q, fluid.layers.transpose(K, perm=[0, 1, 3, 2])) / (self.scale)

        # 第一个 : V mask
        energy = fluid.layers.transpose(energy, perm=[0, 3, 2, 1])
        energy = self.mask(energy, v_mask, 'add')
        energy = fluid.layers.transpose(energy, perm=[0, 3, 2, 1])

        # 然后对Q,K相乘的结果计算softmax ( -- 一般这里最外面会加上dropout的 然鹅 苏神代码目测没有加)
        attention = fluid.layers.softmax(energy, axis=-1)

        # attention结果与V相乘
        x = fluid.layers.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了

        x = fluid.layers.transpose(x, perm=[0, 2, 1, 3])
        x = fluid.layers.reshape(
            x, shape=[bsz, -1, self.n_heads * (self.hid_dim // self.n_heads)])
        # 第二个Q mask
        x = self.mask(x, q_mask, 'mul')

        # x = self.fc(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)
