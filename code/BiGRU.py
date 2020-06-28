import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy


class BiGRU(fluid.dygraph.Layer):
    def __init__(self, hidden_dim):
        super(BiGRU, self).__init__(None)
        self.encoder_fwd_cell = fluid.dygraph.GRUUnit(size=hidden_dim)
        self.encoder_bwd_cell = fluid.dygraph.GRUUnit(size=hidden_dim)

    def forward(self, src_embedding, src_sequence_length):
        # encoder_fwd_output, fwd_state = layers.rnn(cell=self.encoder_fwd_cell,
        #                                            inputs=src_embedding,
        #                                            sequence_length=src_sequence_length,
        #                                            time_major=False,
        #                                            is_reverse=False)
        # encoder_bwd_output, bwd_state = layers.rnn(cell=self.encoder_bwd_cell,
        #                                            inputs=src_embedding,
        #                                            sequence_length=src_sequence_length,
        #                                            time_major=False,
        #                                            is_reverse=True)

        hidden_input = numpy.random.randn(src_embedding.shape.numpy()).astype('float32')
        encoder_fwd_output = self.encoder_fwd_cell(src_embedding, hidden_input)
        # encoder_bwd_output =
        encoder_output = layers.concat(input=[encoder_fwd_output, encoder_bwd_output], axis=2)
        encoder_state = layers.concat(input=[fwd_state, bwd_state], axis=1)
        return encoder_output, encoder_state


# def encoder(src_embedding, src_sequence_length):
#     # 使用GRUCell构建前向RNN
#     encoder_fwd_cell = layers.GRUCell(hidden_size=hidden_dim)
#     encoder_fwd_output, fwd_state = layers.rnn(
#         cell=encoder_fwd_cell,
#         inputs=src_embedding,
#         sequence_length=src_sequence_length,
#         time_major=False,
#         is_reverse=False)
#     # 使用GRUCell构建反向RNN
#     encoder_bwd_cell = layers.GRUCell(hidden_size=hidden_dim)
#     encoder_bwd_output, bwd_state = layers.rnn(
#         cell=encoder_bwd_cell,
#         inputs=src_embedding,
#         sequence_length=src_sequence_length,
#         time_major=False,
#         is_reverse=True)
#     # 拼接前向与反向GRU的编码结果得到h
#     encoder_output = layers.concat(
#         input=[encoder_fwd_output, encoder_bwd_output], axis=2)
#     encoder_state = layers.concat(input=[fwd_state, bwd_state], axis=1)
#     return encoder_output, encoder_state


