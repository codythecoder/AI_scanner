import tensorflow as tf
from data import Training
import argparse
import os
from setup import *
from network import Network
from misc import rearrange_batch

parser = argparse.ArgumentParser()
parser.add_argument('--training_folder', default='../training_data', help='The training data folder')

args = parser.parse_args()

# how many epochs until we print the accuracy, how many till we save a checkpoint
print_time = 2000
save_time = 2000


# training data loader
training_data = Training(args.training_folder)

model_path = os.path.join(args.training_folder, 'model.ckpt')
net = Network(model_path, training_dropout=0.5)

# debugging to keep track of accuracy even through checkpoints
graph = []

i = 0
while True:
    print ('step {:,}'.format(i), end='\r')
    # load up to 10 data points with random distribution of
    #   true/false data in random proportions (averaging at 50/50 True/False)
    batch = rearrange_batch(training_data.load(10))

    # debugging accuracy printout
    if i % print_time == 0:
        batch = rearrange_batch(training_data.load(1000000))
        train_accuracy = net.accuracy(batch[0], batch[1], batch[2])
        print('step {:,}, training accuracy {:,.6f}'.format(i, train_accuracy))
        graph.append(train_accuracy)
    # save graph
    if i and i % save_time == 0:
        print ('step {:,}        saving (average = {:,.5f})'.format(i, sum(graph)/len(graph)))
        net.save()
        with open(model_path + '.csv', 'a') as f:
            # Save the accuracies so that we can check them later on
            f.write('\n'.join(str(s) for s in graph) + '\n')
            graph = []

    # training
    net.train(batch[0], batch[1], batch[2])
    i += 1
