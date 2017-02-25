from __future__ import division
from __future__ import print_function

import os
import struct
from array import array

import numpy as np


def sigmoid(z):
  """Sigmoid curve. Accepts both scalar and `np.array` inputs, where the
     sigmoid function is applied component-wise."""
  return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z) * (1 - sigmoid(z))


class MNIST(object):
  """MNIST data parser. Borrowed from GitHub repository sorki/python-mnist:
     https://github.com/sorki/python-mnist/blob/master/mnist/loader.py"""

  def __init__(self):
    self.train_data = []
    self.valid_data = []
  
  def load_train(self):
    """Load the training dataset."""
    images, labels = self.load(os.path.join('mnist', 'train', 'images'),
                               os.path.join('mnist', 'train', 'labels'))
    self.train_data = zip(images, labels)
  
  def load_valid(self):
    """Load the validation dataset."""
    images, labels = self.load(os.path.join('mnist', 'valid', 'images'),
                               os.path.join('mnist', 'valid', 'labels'))
    self.valid_data = zip(images, labels)
  
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


### Task 1:
### To get familiarized with the data, you must at least understand how the
### MNIST loader above works. Try displaying some images in the terminal.
### Hint: load the validation dataset (the train dataset is big and takes a
### while to load), and use the `display` method. Try playing with the
### `threshold` keyword argument and see its effect on the output.

mnist = MNIST()
mnist.load_valid()
for img, label in mnist.valid_data[:10]:
  print(label)
  mnist.display(img)


### Task 2:
### 1. Implement the `feedforward` method in the class below. Your code should
###    take as input a list of images (`np.array`s) and for each image output
###    a `np.array` of 10 numbers in the range (0, 1).
### 2. Implement the `accuracy` method in the class below. Your code should use
###    the `feedforward` method, and then output the percentage of correctly
###    guessed images. A guess is the index of the maximum component of the
###    output vector, and it is correct if it is equal to that image's label.

class NeuralNet(object):
  """2-layer neural network."""
  
  def __init__(self, learning_rate=1., num_hidden=100):
    self.num_hidden = num_hidden
    self.W1 = np.random.randn(num_hidden, 28*28)
    self.b1 = np.random.randn(num_hidden)
    self.W2 = np.random.randn(10, num_hidden)
    self.b2 = np.random.randn(10)
  
  def feedforward(self, images):
    ### Your code here.
    pass
  
  def accuracy(self, data):
    images, labels = zip(*data)
    ### Your code here.
    ### Hint: use `np.argmax`
  
  def train(self, data):
    ### Your code here.
    pass
  
  def backprop(self, image, label):
    """WEEK 2."""
    pass


### Task 3:
### Test your implementation of the `accuracy` method a few times and show that
### in expectation the neural net will guess correctly ~10% of the images.

### Your code here.


### Task 4:
### 1. Implement the `train` method in the `NeuralNet` class. This should use
###    the already provided `backprop` method to get the gradients and use
###    them to update W1, W2, b1 and b2 in a way that will improve accuracy.
###
###    Extra challenge: instead of looping, try using matrix multiplications.
###    This will be much faster since numpy computes the results in C++.
###
### 2. Test the results of the `accuracy` method after iterating over train().
###    Print out the results after, say, every 100 iterations and show how it
###    improves until it reaches a threshold. How good does the model get?

### Your code here.
