

def scaled_dot_product_attention(queries,
                                 keys,
                                 values,
                                 num_heads=1,
                                 dropout_rate=0.):
    check_variable_and_dtype(queries, 'queries', ['float32', 'float64'],
                             "scaled_dot_product_attention")
    check_variable_and_dtype(keys, 'keys', ['float32', 'float64'],
                             "scaled_dot_product_attention")
    check_variable_and_dtype(values, 'values', ['float32', 'float64'],
                             "scaled_dot_product_attention")

    if not (queries.dtype == keys.dtype == values.dtype):
        raise TypeError(
            "The dtype of keys, values and queries should be the same."
            "But received queries.dtype = %s, "
            " keys.dtype = %s, values.dtype) = %s." %
            (convert_dtype(queries.dtype), convert_dtype(keys.dtype),
             convert_dtype(values.dtype)))

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs queries, keys and values should all be 3-D tensors."
            "But received len(queries.shape) = %d, "
            "len(keys.shape) = %d, len(values.shape) = %d." %
            (len(queries.shape), len(keys.shape), len(values.shape)))

    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(
            "The hidden size of queries and keys should be the same."
            "But received queries' hidden size = %d and keys' hidden size = %d."
            % (queries.shape[-1], keys.shape[-1]))
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError(
            "The max sequence length in value batch and in key batch "
            "should be the same. But received max sequence length in value batch "
            "= %d, in key batch = %d." % (values.shape[-2], keys.shape[-2]))
    if keys.shape[-1] % num_heads != 0:
        raise ValueError("The hidden size of keys (%d) must be divisible "
                         "by the number of attention heads (%d)." %
                         (keys.shape[-1], num_heads))
    if values.shape[-1] % num_heads != 0:
        raise ValueError("The hidden size of values (%d) must be divisible "
                         "by the number of attention heads (%d)." %
                         (values.shape[-1], num_heads))

    def __compute_qkv(queries, keys, values, num_heads):
        """
        Add linear projection to queries, keys, and values.
        Args:
            queries(Tensor): a 3-D input Tensor.
            keys(Tensor): a 3-D input Tensor.
            values(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads. Linearly project the inputs
                            ONLY when num_heads > 1.
        Returns:
            Tensor: linearly projected output Tensors: queries', keys' and
                    values'. They have the same shapes with queries, keys and
                    values.
        """

        if num_heads == 1:
            return queries, keys, values

        q = layers.fc(input=queries, size=queries.shape[-1], num_flatten_dims=2)
        k = layers.fc(input=keys, size=keys.shape[-1], num_flatten_dims=2)
        v = layers.fc(input=values, size=values.shape[-1], num_flatten_dims=2)
        return q, k, v

    def __split_heads(x, num_heads):
        """
        Reshape the last dimension of input tensor x so that it becomes two
        dimensions.
        Args:
            x(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads.
        Returns:
            Tensor: a Tensor with shape [..., n, m/num_heads], where m is size
                    of the last dimension of x.
        """
        if num_heads == 1:
            return x

        hidden_size = x.shape[-1]
        # reshape the 3-D input: [batch_size, max_sequence_length, hidden_dim]
        # into a 4-D output:
        # [batch_size, max_sequence_length, num_heads, hidden_size_per_head].
        reshaped = layers.reshape(
            x=x,
            shape=list(x.shape[:-1]) + [num_heads, hidden_size // num_heads])

        # permute the dimensions into:
        # [batch_size, num_heads, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Reshape the last two dimensions of input tensor x so that it becomes
        one dimension.
        Args:
            x(Tensor): a 4-D input Tensor with shape
                       [bs, num_heads, max_sequence_length, hidden_dim].
        Returns:
            Tensor: a Tensor with shape
                    [bs, max_sequence_length, num_heads * hidden_dim].
        """

        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        return layers.reshape(
            x=trans_x,
            shape=list(
                map(int, [
                    trans_x.shape[0], trans_x.shape[1], trans_x.shape[2] *
                    trans_x.shape[3]
                ])))

    q, k, v = __compute_qkv(queries, keys, values, num_heads)

    q = __split_heads(q, num_heads)
    k = __split_heads(k, num_heads)
    v = __split_heads(v, num_heads)

    key_dim_per_head = keys.shape[-1] // num_heads
    scaled_q = layers.scale(x=q, scale=key_dim_per_head**-0.5)
    product = layers.matmul(x=scaled_q, y=k, transpose_y=True)

    weights = layers.reshape(
        x=layers.reshape(
            x=product, shape=[-1, product.shape[-1]], act="softmax"),
        shape=product.shape)
    if dropout_rate:
        weights = layers.dropout(
            weights, dropout_prob=dropout_rate, is_test=False)
    ctx_multiheads = layers.matmul(weights, v)
    return __combine_heads(ctx_multiheads)