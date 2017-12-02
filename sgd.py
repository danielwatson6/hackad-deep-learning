import random

import tensorflow as tf

from mnist import MNIST


mnist = MNIST()


LEARNING_RATE = 1.
BATCH_SIZE = 20
HIDDEN_SIZE = 256


graph = tf.Graph()
with graph.as_default():

  inputs_placeholder = tf.placeholder(tf.float32, shape=[None, 784])
  labels_placeholder = tf.placeholder(tf.int64, shape=[None])
  
  one_hot_labels = tf.one_hot(labels_placeholder, 10)
  
  
  # By default, TensorFlow uses a glorot uniform initializer [-limit, limit]
  # where limit=sqrt(6 / (num_inputs + num_outputs))
  
  # Trick: do xW instead of Wx to preserve batch size
  
  W1 = tf.get_variable('W1', shape=[784, HIDDEN_SIZE])
  b1 = tf.get_variable('b1', shape=[HIDDEN_SIZE])
  
  W2 = tf.get_variable('W2', shape=[HIDDEN_SIZE, 10])
  b2 = tf.get_variable('b2', shape=[10])
  
  
  hidden_layer = tf.sigmoid(tf.matmul(inputs_placeholder, W1) + b1)
  predictions = tf.sigmoid(tf.matmul(hidden_layer, W2) + b2)
  
  
  loss = tf.losses.mean_squared_error(one_hot_labels, predictions)
  backprop_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  
  accuracy = tf.reduce_mean(
    tf.to_float( tf.equal(tf.argmax(predictions, axis=1), labels_placeholder) )
  )
  
  init_op = tf.global_variables_initializer()





with tf.Session(graph=graph) as sess:
  sess.run(init_op)
  
  valid_inputs, valid_labels = zip(*mnist.valid_data)
  valid_fd = {
    inputs_placeholder: valid_inputs,
    labels_placeholder: valid_labels,
  }
  
  for epoch in range(20):
    
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
  
