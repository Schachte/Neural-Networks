'''
Author: Ryan Schachte
Sources: Morvan
Description: Simple neural network that will learn the weights and bypass of a two dimensional function
'''

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
    '''Our goal is to predict the y data using tensorflow structure'''

    #Create the data
    x_dat = np.random.rand(100).astype(np.float32)
    y_dat = x_dat * .1 + .3 #The .1 is the weight and the .3 is the bias in the function


    '''Define the tensorflow structure'''
    # This builds a number from a random uniform distribution
    # Generates 1 value between the range of -1 and +1
    weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

    # We just want to initialize the bias to zero for just one value
    biases = tf.Variable(tf.zeros([1]))

    # This represents the predicted y-value
    # Clearly, right now, with a random weight (which is random and wrong) and a bias of 0 (which is wrong)
    # we will try and improve these values using our training techniques in the neural network
    y = weights * x_dat + biases

    # In order to improve these values we need to compare the predicted value with the actual value
    # The actual value can be generated from running the function above, the predicted value is generated from our
    # prediction model

    # The difference between the prediction and the actual is the loss function (error)

    '''Calculate the square error between the predicted value and the actual value then get the mean of all samples'''
    # Keep in mind, we are passing in a vector of 100 samples, so we take the mean of the squared error of all of them
    loss = tf.reduce_mean(tf.square(y-y_dat))

    # We want to optimize/minimize the loss (Gradient descent)
    optimizer = tf.train.GradientDescentOptimizer(.5) # The 0.5 is a value (between 0 and 1) that is the learning rate

    # Use the optimizer to minimize the loss
    train = optimizer.minimize(loss)

    '''Very important step'''
    '''Once you define any tf variables, you must initialize all of them by calling this function'''
    init = tf.global_variables_initializer()

    # Basically a point to point to the structures you have already built up in the graph structure above
    sess = tf.Session()

    # Execute a run on all the above structures
    sess.run(init)

    # We will minimize the loss 201 times
    for step in range(201):
        sess.run(train)
        # for every 20 steps
        if step % 20 == 0:
            #Get the step count, access the weights object in the sessions and get the biases
            print(step, sess.run(weights), sess.run(biases))


if __name__ == "__main__":
    main()

