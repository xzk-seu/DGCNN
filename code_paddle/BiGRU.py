import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers


class BiGRU(fluid.dygraph.Layer):
    def __init__(self, hidden_dim):
        super(BiGRU, self).__init__(None)
        self.step = 6

        self.fwd_grus = [fluid.dygraph.GRUUnit(size=hidden_dim * 3) for _ in range(self.step)]
        self.bwd_grus = [fluid.dygraph.GRUUnit(size=hidden_dim * 3) for _ in range(self.step)]

    def forward(self, input_x):
        input_x_split = layers.split(input_x, num_or_sections=self.step, dim=1)
        input_x_split = [layers.squeeze(x, axes=[1]) for x in input_x_split]
        hidden_input_fwd = np.random.normal(size=input_x_split[0].shape).astype('float32')
        hidden_input_bwd = np.random.normal(size=input_x_split[0].shape).astype('float32')
        hidden_input_fwd = fluid.dygraph.to_variable(hidden_input_fwd)
        hidden_input_bwd = fluid.dygraph.to_variable(hidden_input_bwd)
        input_x_split = [layers.expand(x, expand_times=[1, 3]) for x in input_x_split]

        hidden = hidden_input_fwd
        for i in range(self.step):
            hidden, reset_hidden_pre, gate = self.fwd_grus[i](input_x_split[i], hidden)
        fwd_output = hidden

        hidden = hidden_input_bwd
        for i in reversed(range(self.step)):
            hidden, reset_hidden_pre, gate = self.bwd_grus[i](input_x_split[i], hidden)
        bwd_output = hidden

        output = layers.concat([fwd_output, bwd_output], axis=-1)
        return output
