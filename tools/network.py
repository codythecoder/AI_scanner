import tensorflow as tf
from setup import *
from numpy import product

__all__ = ['y_conv', 'train_step', 'accuracy',]

# create the weight and bias variables for FCLs
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.3)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# create the weight and bias for the convolutional network
#   important that we create it like this so that we can use the
#   same CNN to get the same features from both images.
def conv_weight_variable(name, shape):
    initialiser = tf.random_normal_initializer(stddev=0.3)
    return tf.get_variable(name, shape, initializer=initialiser)

def conv_bias_variable(name, shape):
    initialiser = tf.constant_initializer(0.1)
    return tf.get_variable(name, shape, initializer=initialiser)

def conv2d(x, W, stride=1):
    """Create a convolutional neural network layer.
    x is the input and W is the initialised weights."""
    # W = filter_height, filter_width, in_channels, out_channels
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def max_pool_2x2(x, padding="VALID"):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding=padding)

def tensor_size(t):
    """Get the flattened size of a tensor."""
    return product([int(a) for a in t.shape[1:]])

def conv_chain(x, x_features, layer_sizes, layer_features, layer_stride=None):
    """Create a chain of CNN layers.

    params:
    x              - Input tensor
    x_features     - Number of features in x (probably number of color channels)
    layer_sizes    - the filter size for each CNN layer (e.g. (5, 3, 3))
    layer_features - The number of features to calculate for each layer (e.g. (8, 16, 32))

    output:
    final layer of the CNN"""
    layer_sizes = (None,) + tuple(layer_sizes)
    if layer_stride is None:
        layer_stride = (1,)*len(layer_stride)
    else:
        layer_stride = (None,) + tuple(layer_stride)
    layer_features = (x_features,) + tuple(layer_features)
    last_layer = x
    for i in range(1, len(layer_sizes)):
        W_conv = conv_weight_variable('weight{}'.format(i), [layer_sizes[i], layer_sizes[i], layer_features[i-1], layer_features[i]])
        b_conv = conv_bias_variable('bias{}'.format(i), [layer_features[i]])

        h_conv = tf.nn.relu(conv2d(last_layer, W_conv, layer_stride[i]) + b_conv)
        # h_conv = tf.nn.tanh(conv2d(last_layer, W_conv) + b_conv)
        # last_layer = max_pool_2x2(h_conv)
        last_layer = h_conv
    return last_layer

def fcl_chain(x, layer_sizes, dropout_keep_prob):
    last_layer = x
    layer_sizes = (tensor_size(x),) + tuple(layer_sizes)

    for i in range(1, len(layer_sizes)):
        W_fc1 = weight_variable([layer_sizes[i-1], layer_sizes[i]])
        b_fc1 = bias_variable([layer_sizes[i]])

        h_fc1 = tf.nn.relu(tf.matmul(last_layer, W_fc1) + b_fc1)

        last_layer = tf.nn.dropout(h_fc1, dropout_keep_prob)

    return last_layer


# image1, image2, output
x1 = tf.placeholder(tf.float32, shape=[None, input_size])
x2 = tf.placeholder(tf.float32, shape=[None, input_size])
y_ = tf.placeholder(tf.float32, shape=[None, nn_outputs])

# training droput
keep_prob = tf.placeholder(tf.float32)

# x1 and x2 take a flattened image, but CNNs take an nd array
# so we're going to have to reshape things a lot
x1_image = tf.reshape(x1, input_shape)
x2_image = tf.reshape(x2, input_shape)
with tf.variable_scope("image_filters") as scope:
    # create a filter that operates on BOTH x1 and x2
    #   while this technically shouldn't be nescecarry
    #   (as image order shouldn't affect anything), it simplifies the problem
    #   and theoretically makes life easier
    x1_conv_chain = conv_chain(x1_image, color_channels, CNN_filter_sizes, CNN_features, CNN_strides)
    x1_size = tensor_size(x1_conv_chain)
    scope.reuse_variables()
    x2_conv_chain = conv_chain(x2_image, color_channels, CNN_filter_sizes, CNN_features, CNN_strides)
    x2_size = tensor_size(x2_conv_chain)

# after extracting features from x1 and x2, we should concatenate the outputs
#   for use in the FCLs
concat = tf.concat([x1_conv_chain, x2_conv_chain], 3)
concat_size = x1_size + x2_size

concat = tf.reshape(concat, [-1, concat_size])

y_conv = fcl_chain(concat, FCL_sizes, keep_prob)

# training
loss = tf.reduce_mean(tf.square(y_ - y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
accuracy = tf.reduce_mean(tf.abs(y_ - y_conv))
