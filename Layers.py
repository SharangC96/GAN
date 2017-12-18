import tensorflow as tf

def dense_layer(x, units, activation=tf.nn.relu, name='dense'):
    prev_units = int(x.shape[1])

    init = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope(name_or_scope=name):
        b = tf.get_variable('bias', shape=[units, 1], initializer=init, dtype=tf.float32)
        w = tf.get_variable('weight', shape=[prev_units, units], initializer=init, dtype=tf.float32)

    return activation(tf.add(tf.matmul(x, w), b))


def conv_layer(x, kernels, kernel_size=3, strides=1, padding='VALID', name='ConvLayer', activation=tf.nn.relu,
               use_bias=True):

    init = tf.contrib.layers.xavier_initializer()

    batch_size, dim1, dim2, in_channels = x.shape

    with tf.variable_scope(name_or_scope=name):
        b = tf.constant(0)
        if use_bias:
            b = tf.get_variable('bias', shape=[batch_size, 1, 1, kernels], initializer=init, dtype=tf.float32)
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, batch_size, kernels])

    return activation(tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, name='convolution') + b)


def deconv_layer(x, kernels, kernel_size=3, strides=1, padding='VALID', name='DeconvLayer', activation=tf.nn.relu):
    init = tf.contrib.layers.xavier_initializer()

    return tf.layers.conv2d_transpose(x, kernels, kernel_size, (strides, strides), padding, activation=activation,
                                      kernel_initializer=init, name=name)


def batch_norm(x,name):
    pass
