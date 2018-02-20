import tensorflow as tf
from data import Training
import argparse
import os
from setup import *
from numpy import product

parser = argparse.ArgumentParser()
parser.add_argument('--training_folder', default='../training_data', help='The training data folder')

args = parser.parse_args()

# how many epochs until we print the accuracy, how many till we save a checkpoint
print_time = 100
save_time = 2000


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

def conv2d(x, W):
    """Create a convolutional neural network layer.
    x is the input and W is the initialised weights."""
    # W = filter_height, filter_width, in_channels, out_channels
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x, padding="VALID"):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding=padding)

def tensor_size(t):
    """Get the flattened size of a tensor."""
    return product([int(a) for a in t.shape[1:]])

def conv_chain(x, x_features, layer_sizes, layer_features):
    """Create a chain of CNN layers.

    params:
    x              - Input tensor
    x_features     - Number of features in x (probably number of color channels)
    layer_sizes    - the catchment area for each CNN layer (e.g. (5, 3, 3))
    layer_features - The number of features to calculate for each layer (e.g. (8, 16, 32))

    output:
    final layer of the CNN"""
    layer_sizes = (None,) + tuple(layer_sizes)
    layer_features = (x_features,) + tuple(layer_features)
    last_layer = x
    for i in range(1, len(layer_sizes)):
        W_conv = conv_weight_variable(f'weight{i}', [layer_sizes[i], layer_sizes[i], layer_features[i-1], layer_features[i]])
        b_conv = conv_bias_variable(f'bias{i}', [layer_features[i]])

        h_conv = tf.nn.relu(conv2d(last_layer, W_conv) + b_conv)
        # h_conv = tf.nn.tanh(conv2d(last_layer, W_conv) + b_conv)
        last_layer = max_pool_2x2(h_conv)
    return last_layer

def rearrange_batch(batch):
    """Rearrange the batch from [(image1, image2, output), ...] to
    [(image1, ...), (image2, ...), (output, ...)]
    """
    return list(zip(*batch))


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
    x1_conv_chain = conv_chain(x1_image, color_channels, (5, 3, 3), (9, 32, 64))
    x1_size = tensor_size(x1_conv_chain)
    scope.reuse_variables()
    x2_conv_chain = conv_chain(x2_image, color_channels, (5, 3, 3), (9, 32, 64))
    x2_size = tensor_size(x2_conv_chain)

# after extracting features from x1 and x2, we should concatenate the outputs
#   for use in the FCLs
concat = tf.concat([x1_conv_chain, x2_conv_chain], 0)
concat_size = x1_size + x2_size

#fully connected layer 1
W_fc1 = weight_variable([concat_size, 1024])
b_fc1 = bias_variable([1024])

concat_flat = tf.reshape(concat, [-1, concat_size])
h_fc1 = tf.nn.relu(tf.matmul(concat_flat, W_fc1) + b_fc1)

#dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#fully connected layer 2
W_fc2 = weight_variable([1024, 128])
b_fc2 = bias_variable([128])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#dropout
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#fully connected layer 3
W_fc3 = weight_variable([128, nn_outputs])
b_fc3 = bias_variable([nn_outputs])

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3


# training
loss = tf.reduce_mean(tf.square(y_ - y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
accuracy = tf.reduce_mean(tf.abs(y_ - y_conv))

# training data loader
training_data = Training(args.training_folder)

saver = tf.train.Saver()
model_path = os.path.join(args.training_folder, 'model.ckpt')

# debugging to keep track of accuracy even through checkpoints
graph = []

with tf.Session() as sess:
    # if a chekcpoint exists, use that
    restore = False
    if os.path.isfile(model_path + '.index'):
        if True or input('model already exists, continue? (y/n) ') == 'y':
            restore = True
            saver.restore(sess, model_path)
    if not restore:
        sess.run(tf.global_variables_initializer())
    i = 0
    while True:
        print (f'step {i:,}', end='\r')
        # load up to 10 data points with random distribution of
        #   true/false data in random proportions (averaging at 50/50 True/False)
        batch = rearrange_batch(training_data.load(10))

        # dubugging accuracy printout/
        if i % print_time == 0:
            batch = rearrange_batch(training_data.load(100))
            train_accuracy = accuracy.eval(feed_dict={
                    x1: batch[0], x2: batch[1], y_: batch[2], keep_prob: 1.0})
            print(f'step {i:,}, training accuracy {train_accuracy:,.6f}')
            graph.append(train_accuracy)
        # save graph
        if i and i % save_time == 0:
            print ('saving')
            saver.save(sess, model_path)
            with open(model_path + '.csv', 'a') as f:
                # Save the accuracies so that we can check them later on
                f.write('\n'.join(str(s) for s in graph) + '\n')
                graph = []

        # training
        train_step.run(feed_dict={x1: batch[0], x2: batch[1], y_: batch[2], keep_prob: 0.4})
        i += 1
