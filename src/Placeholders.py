'''
Author: Ryan Schachte
Sources: Morvan
Description: Using placeholders in tensorflow
'''

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output_string = tf.placeholder(tf.string)

# Simple math function that will multiply whatever we feed in to the placeholders
output = tf.multiply(input1, input2)
string_print = tf.Print(None, None, output_string)

with tf.Session() as sess:
    # We will run the output operation

    # The feed_dict will feed in the values into the input variables
    # and subsequently run the output operation on them
    print(sess.run(output, feed_dict={input1: [55], input2: [10]}))
    print(sess.run(string_print, feed_dict={output_string}))
