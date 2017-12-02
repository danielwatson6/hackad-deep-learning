import random

import tensorflow as tf

from mnist import MNIST


mnist = MNIST()


LEARNING_RATE = 1e-3
BATCH_SIZE = 20
HIDDEN_SIZE = 512


graph = tf.Graph()
with graph.as_default():

  inputs_placeholder = tf.placeholder(tf.float32, shape=[None, 784])
  labels_placeholder = tf.placeholder(tf.int64, shape=[None])
  
  W1 = tf.get_variable('W1', shape=[784, HIDDEN_SIZE])
  b1 = tf.get_variable('b1', shape=[HIDDEN_SIZE])
  
  W2 = tf.get_variable('W2', shape=[HIDDEN_SIZE, 10])
  b2 = tf.get_variable('b2', shape=[10])
  
  hidden_layer = tf.nn.relu(tf.matmul(inputs_placeholder, W1) + b1)
  logits = tf.matmul(hidden_layer, W2) + b2
  
  
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels_placeholder, logits=logits)
  
  backprop_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
  
  accuracy = tf.reduce_mean(
    tf.to_float( tf.equal(tf.argmax(logits, axis=1), labels_placeholder) ))
  init_op = tf.global_variables_initializer()


with tf.Session(graph=graph) as sess:
  sess.run(init_op)
  
  valid_inputs, valid_labels = zip(*mnist.valid_data)
  valid_fd = {
    inputs_placeholder: valid_inputs,
    labels_placeholder: valid_labels,
  }
  
  for epoch in range(200):
    
    for i in range(0, len(mnist.train_data) // BATCH_SIZE, BATCH_SIZE):
      
      train_inputs, train_labels = zip(*mnist.train_data[i:i+BATCH_SIZE])
      train_fd = {
        inputs_placeholder: train_inputs,
        labels_placeholder: train_labels,
      }
      sess.run(backprop_op, feed_dict=train_fd)
    
    random.shuffle(mnist.train_data)
    print("Epoch", epoch, "complete")
    print("Test set accuracy:", sess.run(accuracy, feed_dict=valid_fd))
  
