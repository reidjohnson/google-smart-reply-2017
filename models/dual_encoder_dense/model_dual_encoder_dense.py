import tensorflow as tf


def dot_product_scoring(inputs_x, inputs_y, is_training):
    """Creates a two-channel network that encodes inputs x and y into a hidden
    representation and calculates a dot product to determine their scoring.

    :param inputs_x: Summed word embeddings for the context (1 x emb_dim).
    :param inputs_y: Summed word embeddings for the utterance (1 x emb_dim).
    :param is_training: Training phase.
    :return:
    """

    # Encode x as a (1 x embedding_size) vector.
    x = network_channel(inputs=inputs_x, is_training=is_training, name='x_channel')

    # Encode y as a (1 x embedding_size) vector.
    y = network_channel(inputs=inputs_y, is_training=is_training, name='y_channel')

    # Compute scores for all pairs of input texts and labels.
    S = tf.matmul(x, y, transpose_b=True)

    return S


def network_channel(inputs, is_training, name):
    """Creates an n-layer feedforward network that encodes the inputs into a
    k-dimensional representational space.

    :param inputs: Inputs.
    :param is_training: Training phase.
    :param name: Scope name.
    :return:
    """
    with tf.variable_scope(name) as scope:
        net = tf.layers.dense(inputs, units=300, activation=tf.nn.tanh)
        net = tf.layers.batch_normalization(net, training=is_training)

        net = tf.layers.dense(net, units=300, activation=tf.nn.tanh)
        net = tf.layers.batch_normalization(net, training=is_training)

        net = tf.layers.dense(net, units=500, activation=tf.nn.tanh, name='encoding')

        return net
