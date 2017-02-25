from __future__ import division
from __future__ import print_function

import os
import random
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
    
    images, labels = self.load('mnist/train/images', 'mnist/train/labels')
    ### WINDOWS USERS: replace the line above with the one below:
    # images, labels = self.load('mnist\train\images', 'mnist\train\labels')
    self.train_data = zip(images, labels)
  
  def load_valid(self):
    """Load the validation dataset."""
    images, labels = self.load('mnist/valid/images', 'mnist/valid/labels')
    ### WINDOWS USERS: replace the line above with the one below:
    # images, labels = self.load('mnist\valid\images', 'mnist\valid\labels')
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
    for _ in xrange(size):
      images.append([0] * rows * cols)
    
    for i in xrange(size):
      # Normalize data to have mean 0 and variance 1
      images[i][:] = np.array(
        image_data[i * rows * cols : (i + 1) * rows * cols]) / 256 - .5

    return images, labels

  @classmethod
  def display(cls, img, threshold=200/256 - .5):
    """Render one of the images in the dataset."""
    render = ''
    for i in xrange(len(img)):
      if i % 28 == 0:
        render += '\n'
      if img[i] > threshold:
        render += '#'
      else:
        render += '.'
    print(render)


### Task 1: (week 1)
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
###    a `np.array` of 10 numbers in the range (0, 1). Note that the `sigmoid`
###    method has already been provided.
### 2. Implement the `recall` method in the class below. Your code should
###    output the percentage of correctly guessed images. A guess is the index
###    of the maximum component of the output vector, and it is correct if it
###    is equal to that image's label.

class NeuralNet(object):
  """L-layer neural network."""
  
  def __init__(self, layers, learning_rate=1.):
    """The `layers` argument should be an array containing the output sizes.
       E.g. 1-layer nn would have layers=[10], 2-layer nn [?, 10], etc."""
    self.layers = layers
    self.weight_matrices = []
    self.bias_vectors = []
    input_sizes = [28*28] + layers[:-1]
    for l in xrange(layers):
      self.weight_matrices.append(np.random.randn(layers[l], input_sizes[l]))
      self.bias_vectors.append(np.random.randn(layers[l]))
    # Just for better performance
    self.weight_matrices = np.array(self.weight_matrices)
    self.bias_vectors = np.array(self.bias_vectors)
  
  def feedforward(self, image):
    ### Your code here.
    pass
  
  def recall(self, outputs, labels):
    ### Your code here.
    ### Hint: use `np.argmax`
    pass
  
  def loss_prime(self, output, label):
    ### Your code here.
    pass
  
  def backprop(self, feedforward_output, label):
    ### Your code here.
    pass
  
  def train(self, image_label_pairs, recall=100):
    """Recall keyword argument: every how many steps to compute `recall`."""
    step = 0
    for img, label in image_label_pairs:
      ### Call `feedforward`. Your code here.
      if step % recall == 0:
        ### Print the results from `recall`. Your code here.
        pass
      ### Call `backprop`. Your code here.
      step += 1
    ### Use gradient descent to update all the weights and biases, averaging
    ### the gradients of all the training instances. Your code here.


### Task 3:
### Test your implementation of the `feedforward` and  `recall` methods a few
### times and show that in expectation the recall is ~10%.

### Your code here.


### Task 4:
### Implement the `backprop` method in the `NeuralNet` class above. Modify
### the `feedforward` and `recall` methods so that `feedforward` returns all
### the logits and activations. Fill in the `loss_prime` method for equation 1.
### All the activations will be sigmoids, and both the `sigmoid` and
### `sigmoid_prime` methods are provided (they work with vectors as well).


### Task 5:
### Implement the `train` method. Put all your work together and use gradient
### descent to update the neural net. Run your implementation over all the
### training data and see how the recall improves. Keep increasing the number
### of times you call `train` until the recall gets capped. Don't forget to
### load the training data as well!

### Your code here.


### Task 6:
### Implement stochastic gradient descent. Put the `train` method inside a loop
### and before preparing the mini-batches, use `random.shuffle` on the dataset.
### Experiment with the batch size-- this is a hyperparameter to the model.

batch_size = 10
### Your code here.
