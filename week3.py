from __future__ import division
from __future__ import print_function
from six.moves import xrange

import os
import random
import struct
from array import array

import numpy as np
import tensorflow as tf


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
    self.train_data = list(zip(images, labels))
  
  def load_valid(self):
    """Load the validation dataset."""
    images, labels = self.load('mnist/valid/images', 'mnist/valid/labels')
    ### WINDOWS USERS: replace the line above with the one below:
    # images, labels = self.load('mnist\valid\images', 'mnist\valid\labels')
    self.valid_data = list(zip(images, labels))
  
  @classmethod
  def load(cls, path_img, path_lbl):
    # Read labels
    with open(path_lbl, 'rb') as file:
      magic, size = struct.unpack(">II", file.read(8))
      if magic != 2049:
        raise ValueError('Magic number mismatch, expected 2049,'
                         'got {}'.format(magic))
      labels = array("B", file.read())
    # Read images
    with open(path_img, 'rb') as file:
      magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
      if magic != 2051:
        raise ValueError('Magic number mismatch, expected 2051,'
                         'got {}'.format(magic))
      image_data = array("B", file.read())
    # Create arrays for the images
    images = []
    for _ in xrange(size):
      images.append([0] * rows * cols)
    # Normalize data to have mean 0 and variance 1
    for i in xrange(size):
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


### YOUR CODE HERE

mnist = MNIST()
mnist.load_train()
mnist.load_valid()
# You may disable this, but play with it to ensure that the data is being
# loaded correctly and to understand the `mnist` variable.
for img, label in mnist.valid_data[:10]:
  print(label)
  mnist.display(img)


LEARNING_RATE = 1.

graph = tf.Graph()
with graph.as_default():
  
  ### TASK 1: make this a 2-layer deep neural network
  
  inputs = tf.placeholder(tf.float32, [None, 28*28])
  labels = tf.placeholder(tf.int64, [None])
  
  W1 = tf.Variable(tf.zeros([28*28, 10]))
  b1 = tf.Variable(tf.zeros([10]))
  
  z1 = tf.matmul(inputs, W1) + b1
  y = tf.nn.softmax(z1)
  
  one_hot_labels = tf.one_hot(labels, 10)
  loss = tf.losses.mean_squared_error(one_hot_labels, y)
  
  train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  bool_predictions = tf.equal(tf.argmax(y, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(bool_predictions, tf.float32)) * 100


### TASK 2: change this to use stochastic gradient descent

with tf.Session(graph=graph) as sess:
  step = 0
  num_steps_per_eval = 10
  valid_i, valid_l = list(zip(*mnist.valid_data))  # this is not shuffled
  
  tf.global_variables_initializer().run()
  
  # Training loop (press ctrl+c in terminal to stop)
  while True:
    # Shuffle the (input, label) pairs to avoid memorizing the data
    random.shuffle(mnist.train_data)
    # Convert pairs into individual arrays
    train_i, train_l = list(zip(*mnist.train_data))
    sess.run(train_op, feed_dict={inputs: train_i, labels: train_l})
    
    if step % num_steps_per_eval == 0:
      print("Training set accuracy (step={}): {}%".format(step,
        sess.run(accuracy, feed_dict={inputs: train_i, labels: train_l})))
      print("Validation set accuracy (step={}): {}%".format(step,
        sess.run(accuracy, feed_dict={inputs: valid_i, labels: valid_l})))
    
    step += 1
