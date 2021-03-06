"""Print out the predictions and expectations of the training data"""

from data import Training
import argparse
import os
from setup import *
import matplotlib.pyplot as plt
import math
from network import Network
from misc import rearrange_batch


parser = argparse.ArgumentParser()
parser.add_argument('--training_folder', default='../training_data', help='The training data folder')

args = parser.parse_args()
model_path = os.path.join(args.training_folder, 'model.ckpt')
net = Network(model_path)


training_data = Training(args.training_folder)

data = rearrange_batch(training_data.load(1000000))

output = net.classify(data[0], data[1])
print (output)
check = [x[0] for x in data[2]]
print (output)
print ([int(round(x)) for x in output])
print (check)
print ([int((0 if x < 0.5 else 1) == y) for x, y in zip(output, check)])
print (sum(int((0 if x < 0.5 else 1) == y) for x, y in zip(output, check))/len(output))
# print ([(0 if x < avg else 1) == check for x, y in zip(output, check)])
