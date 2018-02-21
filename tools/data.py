import os
import random
import numpy as np
from PIL import Image
from misc import get_all_files_walk
from setup import *

class Training:
    def __init__(self, output, cache_limit=50000):
        self.output = output
        files = os.path.join(output, 'sample_true')
        files0 = os.path.join(output, 'sample_false')
        # get all the true, and all the false data points
        # don't cache them until we request them though
        self.files = get_all_files_walk(files)
        self.files0 = get_all_files_walk(files0)

        self.images = {}
        self.cache_limit = cache_limit

    def load(self, n=100):
        """Get a random combination of true and false data totalling n data points"""
        n = min(n, len(self.files) + len(self.files0))
        n_data = random.randint(0, n)
        n0 = n - n_data
        if n >= len(self.files) + len(self.files0):
            #TODO get an algorithm that works when len(files) <= n <= len(self.files) + len(self.files0)
            n_data = len(self.files)
            n0 = len(self.files0)

        load_images = random.sample(self.files, n_data)
        parts = []
        for filename in load_images:
            part = self._load_images(filename)
            if part is not None:
                parts.append(part)

        load_images0 = random.sample(self.files0, n0)
        parts0 = []
        for filename in load_images0:
            part = self._load_images(filename)
            if part is not None:
                parts0.append(part)

        # output the images and shuffle the order
        output = [(p[0], p[1], [1]) for p in parts]
        output += [(p[0], p[1], [0]) for p in parts0]
        random.shuffle(output)
        return output

    def _load_images(self, filename):
        """load the requested file as its two separate images"""
        # cache the image
        if filename not in self.images:
            im = Image.open(filename)
            arr = np.array(im)
            a, b = np.split(arr, 2, 0)
            a = np.reshape(a, [input_size])
            b = np.reshape(b, [input_size])
            self.images[filename] = a, b

        # randomly show images in reverse order
        if random.getrandbits(1):
            return self.images[filename]
        else:
            return self.images[filename][1], self.images[filename][0]



if __name__ == '__main__':
    # test that loading all the images works
    a = Training('../training_data')
    a.load(1000000)
