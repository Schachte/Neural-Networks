'''
Author: Ryan Schachte
Sources: Morvan
Description: How to build tensorflow variables for neural networks
'''

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Create a variable named counter with an initial value of 0
state = tf.Variable(0, name="counter")

# This will get you the name of the variable (attribute)
print(state.name)

# Define a constant called one with a value of one
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Alternatively
# update = tf.assign(state, new_value)

# Once you define any variable, you have to initialize it
init = tf.global_variables_initializer() # Must be done or nothing will work

with tf.Session() as sess:

    # We want to run the variable initialize function and update operation
    sess.run(init) # Allows us to initialize the variables
    sess.run(update) # Allows us to run the operation
    print("counter:"),
    print(sess.run(state))
