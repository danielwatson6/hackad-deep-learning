import os
import struct
from array import array

import numpy as np


class MNIST(object):
  """MNIST data parser. Borrowed from GitHub repository sorki/python-mnist:
     https://github.com/sorki/python-mnist/blob/master/mnist/loader.py"""

  def __init__(self):
    self.load_train()
    self.load_valid()
  
  def load_train(self):
    """Load the training dataset."""
    images, labels = self.load(os.path.join('mnist', 'train', 'images'),
                               os.path.join('mnist', 'train', 'labels'))
    self.train_data = list(zip(images, labels))
  
  def load_valid(self):
    """Load the validation dataset."""
    images, labels = self.load(os.path.join('mnist', 'valid', 'images'),
                               os.path.join('mnist', 'valid', 'labels'))
    self.valid_data = list(zip(images, labels))
  
  @classmethod
  def load(cls, path_img, path_lbl):
    with open(path_lbl, 'rb') as file:
      magic, size = struct.unpack(">II", file.read(8))
      if magic != 2049:
        raise ValueError('Magic number mismatch, expected 2049,'
                         'got {}'.format(magic))
      labels = array("B", file.read())
    
    with open(path_img, 'rb') as file:
      magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
      if magic != 2051:
        raise ValueError('Magic number mismatch, expected 2051,'
                         'got {}'.format(magic))
      image_data = array("B", file.read())

    images = []
    for i in range(size):
      images.append([0] * rows * cols)
    
    for i in range(size):
      # Normalize data to have mean 0 and variance 1
      images[i][:] = np.array(
        image_data[i * rows * cols : (i + 1) * rows * cols]) / 256 - .5

    return images, labels

  @classmethod
  def display(cls, img, threshold=200/256 - .5):
    """Render one of the images in the dataset."""
    render = ''
    for i in range(len(img)):
      if i % 28 == 0:
        render += '\n'
      if img[i] > threshold:
        render += '#'
      else:
        render += '.'
    print(render)
