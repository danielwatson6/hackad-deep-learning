import tensorflow as tf

from mnist import MNIST



mnist = MNIST()
for img, label in mnist.valid_data[:10]:
  print(label)
  mnist.display(img)


