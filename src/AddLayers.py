'''
Author: Ryan Schachte
Sources: Morvan
Description: Simple neural network that will learn the weights and bypass of a two dimensional function
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Inputs represents the last layer (information processed by the previous layer
# In-Size represents the size for the input information (how many hidden units in last layer , hidden neurons)
# Out-Size represents the size of the output information (number of neurons in the current layer)
# Activation function represents the.... activation function
def add_layer(inputs, in_size, out_size, activation_function=None):

    # Randomized weights can improve the network weights better
    weights = tf.Variable(tf.random_normal([in_size, out_size]))

    # 0 + 0.1 for all variable biases
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    # Perform the multiplcation (matrix)
    Wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


'''Make up some real data'''
# 300 samples of values -1, 1
x_data = np.linspace(-1, 1, 500)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# plt.scatter(x_data, y_data)
# plt.show()

'''Define placeholders for inputs to the network'''

# None represents accepting all possible number of samples
# The 1 represents the number of features
xs = tf.placeholder(tf.float32, [None ,1])
ys = tf.placeholder(tf.float32, [None ,1])

'''Add a hidden layer'''

# We call the add layer for our x data with 1 feature
# We will have output size of about 10 neurons (random)
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

'''Add the output layer'''

# Output layer is a regression problem, so we don't need an activator since it's linear function
prediction = add_layer(l1, 10, 1, activation_function=None)

'''Compute the error between the prediction and the real data'''

# Compute the square error of the prediction and real-data, then sum all the samples together
# Loss of all samples and all features
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

# Use grad. desc optimizer with .1 learning rate to minimize the loss function
train_step = tf.train.GradientDescentOptimizer(.1).minimize(loss)

'''Initialize all the variables'''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

'''Begin visualization process'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)

#unblocks the plotting process
plt.ion()
plt.show()

for _ in range(5000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    '''Visualize the data'''
    if _%20 == 0:

        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_values = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_values, 'r', lw=5)
        plt.pause(0.1)

