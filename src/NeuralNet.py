'''
Author: Ryan Schachte
Sources: Morvan
Description:
The purpose of this neural network is to model the relationship of pixels in digits
to properly classify the handwritten digits using neural networks
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
-Basic Architecture for the Network: (Feed Forward Neural Net)
    input data > weight > hidden layer 1(activation function) >
    hidden layer 2(activation function) > weights > outputs layer

- Compare the output to the intended output using cost/loss function

- Utilize an optimization function to minimize the cost

********************************
Feed Forward + Backprop = epoch
********************************
'''

'''
10 classes 0-9 for the digits

The 1 hot classification means something like this
For 0 : [1, 0 0 0 0 0 0 0 0 0]
For 1 : [0, 1 0 0 0 0 0 0 0 0]
'''

# Load the handwritten digit data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define the number of nodes in the hidden layers/ 500 nodes in each
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# Digits 0-9
n_classes = 10

# Represents batches of 100's of features to manipulate features
batch_size = 100

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


def neural_network_model(data):

    '''
    The function modeled by (input_data * weights) + biases
    Bias: Some neurons still fire if all neurons on 0
    '''

    # Creates a tensor/array using a bunch of random numbers
    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # This is modeling Wx + Biases
    # We are adding the multiplication of the input data vector with the weights, then adding the biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])

    # Apply the activation function to see which neurons fire
    l1 = tf.nn.relu(l1)

    '''Layer 2'''
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    '''Layer 3'''
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    '''Output Layer'''
    output = tf.add(tf.matmul(l3, output_layer['weights']) , output_layer['biases'])

    return output


def train_neural_network(x):
    '''Specify how you want to run data through the computation graph'''

    # Outputs a 1-hot tensor to predict the digit
    prediction = neural_network_model(x)

    # Using cross-entropy with logits as the loss function
    # Similar to mean squared error...
    # Looks at the prediction one-hot with the true value
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # Now we want to minimize the cost
    # Default learning rate is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # No. of times you go forward and backward to reweight neuron weights
    # Cycle count
    epoch_count = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for counter in range(epoch_count):
            epoch_loss = 0

            # How many we process based on input size
            for _ in range(int(mnist.train.num_examples/batch_size)):

                # Gets batch size for x and y
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # Get the cost
                # How does this work?
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', counter, 'completed out of', epoch_count, 'loss: ', epoch_loss)

        # Return indexes of both arrays (both should be the same if we are right and good)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)
