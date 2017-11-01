'''
Author: Ryan Schachte
Sources: Morvan
Description: Understanding matrix multiplication using sessions in tensorflow
'''

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Define the dimensions of two matricies
'''
Good source for understanding matrix structure
https://hackernoon.com/machine-learning-with-tensorflow-8873fdee2b68
'''

# This represents one row containing two columns 3,3 [[3, 3]]
matrix1 = tf.constant([[3,3]])

# This represents two rows with one column containing two columns [[2]
#                                                                  [2]]
matrix2 = tf.constant([[2],[2]])

# The product for the matrix multiplication should be 12
product = tf.matmul(matrix1, matrix2)


'''
@METHOD 1 of Running Session Function
'''
# Define the session object
sess = tf.Session()

# We don't need to run tf.initilizevariables because we haven't explicitly defined any Variables
result = sess.run(product)
print("Result1: "),
print(result)

# We can actually print the structure of the matrix to better understand what it looks like
print(sess.run(matrix1))
print(sess.run(matrix2))

# Usually ok to ignore this, but more formal
sess.close()

'''
@METHOD 2 of Running Session Function
'''

# Basically just open a session
with tf.Session() as sess:
    result2 = sess.run(product)
    print("Result2: "),
    print(result2)
    # Auto closes sessions because of with call
