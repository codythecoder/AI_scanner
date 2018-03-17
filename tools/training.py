import tensorflow as tf
from data import Training
import argparse
import os
from setup import *
from numpy import product
import network as net

parser = argparse.ArgumentParser()
parser.add_argument('--training_folder', default='../training_data', help='The training data folder')

args = parser.parse_args()

# how many epochs until we print the accuracy, how many till we save a checkpoint
print_time = 100
save_time = 2000


def rearrange_batch(batch):
    """Rearrange the batch from [(image1, image2, output), ...] to
    [(image1, ...), (image2, ...), (output, ...)]
    """
    return list(zip(*batch))


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
        print ('step {:,}'.format(i), end='\r')
        # load up to 10 data points with random distribution of
        #   true/false data in random proportions (averaging at 50/50 True/False)
        batch = rearrange_batch(training_data.load(10))

        # debugging accuracy printout
        if i % print_time == 0:
            batch = rearrange_batch(training_data.load(100))
            train_accuracy = net.accuracy.eval(feed_dict={
                    net.x1: batch[0], net.x2: batch[1], net.y_: batch[2], net.keep_prob: 1.0})
            print('step {:,}, training accuracy {:,.6f}'.format(i, train_accuracy))
            graph.append(train_accuracy)
        # save graph
        if i and i % save_time == 0:
            print ('step {:,}        saving (average = {:,.5f})'.format(i, sum(graph)/len(graph)))
            saver.save(sess, model_path)
            with open(model_path + '.csv', 'a') as f:
                # Save the accuracies so that we can check them later on
                f.write('\n'.join(str(s) for s in graph) + '\n')
                graph = []

        # training
        net.train_step.run(feed_dict={net.x1: batch[0], net.x2: batch[1], net.y_: batch[2], net.keep_prob: 0.5})
        i += 1
