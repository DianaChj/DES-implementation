# Author:   Diana Chajkovska
# Date:     20th May, 2019

import math

import tensorflow as tf
import numpy as np

# Restor defaul graph
tf.reset_default_graph()
tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()

output_data = np.array([[0.], [1.], [1.], [0.]], dtype='uint8')
# Create some variables.
n_input = tf.placeholder(tf.float32, shape=[None, 2], name="n_input")
n_output = tf.placeholder(tf.float32, shape=[None, 1], name="n_output")

hidden_nodes = 2

b_hidden = tf.Variable(tf.zeros([hidden_nodes]), name="hidden_bias")
b2 = tf.Variable(tf.zeros([1]), name="Biases2")
W_hidden = tf.Variable(tf.random_normal([2, hidden_nodes]), name="hidden_weights")
hidden = tf.sigmoid(tf.matmul(n_input, W_hidden) + b_hidden)

W_output = tf.Variable(tf.random_normal([hidden_nodes, 1]), name="output_weights")
output = tf.sigmoid(tf.matmul(hidden, W_output) + b2)

# Add ops to restore all the variables.
saver = tf.train.Saver()
sess1 = tf.Session()
saver.restore(sess1, "/tmp/model.ckpt")

# Bitwise XOR function
def XOR_func(x):
    with sess1.as_default() as sess:
        res = []
        for i in range(len(x)):
            res.append(int(np.round(sess.run(output, feed_dict={n_input: np.array([x[i]])}))[0][0]))
        return res


# Function to close current session
def ses_close():
    sess1.close()


# Unite bits in pair for XOR
def bin_pair(x, y):
    return list(zip(x, y))


# Function to get XOR value from input
def getval(x, y):
    bin_pair_arr = bin_pair(x, y)
    ascii_arr = ''
    for i in XOR_func(bin_pair_arr):
        ascii_arr += str(i)
    return ascii_arr
