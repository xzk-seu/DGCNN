import paddle.fluid as fluid


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
