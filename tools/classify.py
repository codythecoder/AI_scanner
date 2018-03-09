import tensorflow as tf
from data import Training
import argparse
import os
from setup import *
from numpy import product
import matplotlib.pyplot as plt
import math
import network as net

parser = argparse.ArgumentParser()
parser.add_argument('--training_folder', default='../training_data', help='The training data folder')

args = parser.parse_args()

def rearrange_batch(batch):
    """Rearrange the batch from [(image1, image2, output), ...] to
    [(image1, ...), (image2, ...), (output, ...)]
    """
    return list(zip(*batch))

def getActivations(layer, feed_dict):
    units = sess.run(layer,feed_dict=feed_dict)
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        # plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.show()



training_data = Training(args.training_folder)

saver = tf.train.Saver()
model_path = os.path.join(args.training_folder, 'model.ckpt')


best_accuracy = float('inf')

with tf.Session() as sess:
    saver.restore(sess, model_path)
    data = rearrange_batch(training_data.load(1000000))
    feed_dict = {net.x1: data[0], net.x2:data[1], net.keep_prob:1}
    feed_dict_1 = {net.x1: [data[0][0]], net.x2:[data[1][0]], net.keep_prob:1}
    # for i in range(len(temp1)):
    #     getActivations(temp1[i],feed_dict_1)
    #     getActivations(temp2[i],feed_dict_1)
    output = net.y_conv.eval(feed_dict)
    output = [x[0] for x in output]
    check = [x[0] for x in data[2]]
    print (output)
    print ([int(round(x)) for x in output])
    print (check)
    print ([int((0 if x < 0.5 else 1) == y) for x, y in zip(output, check)])
    print (sum(int((0 if x < 0.5 else 1) == y) for x, y in zip(output, check))/len(output))
    # print ([(0 if x < avg else 1) == check for x, y in zip(output, check)])
