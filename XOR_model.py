# Author:   Diana Chajkovska
# Date:     20th May, 2019

import tensorflow as tf
import numpy as np

input_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype='uint8')
output_data = np.array([[0.], [1.], [1.], [0.]], dtype='uint8')

# Number of input|output neurons
n_input = tf.placeholder(tf.float32, shape=[None, 2], name="n_input")
n_output = tf.placeholder(tf.float32, shape=[None, 1], name="n_output")

hidden_nodes = 3

b_hidden = tf.Variable(tf.random_normal([hidden_nodes]), name="hidden_bias")
W_hidden = tf.Variable(tf.random_normal([2, hidden_nodes]), name="hidden_weights")
hidden = tf.sigmoid(tf.matmul(n_input, W_hidden) + b_hidden)

# Output layer`s weight matrix
W_output = tf.Variable(tf.random_normal([hidden_nodes, 1]), name="output_weights")
# Calculate output layer`s activation
output = tf.sigmoid(tf.matmul(hidden, W_output))

cross_entropy = tf.square(n_output - output)

# Calculate the mean of cross_entropy
loss = tf.reduce_mean(cross_entropy)
# Optimize loss function
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Create session
sess = tf.Session()
# Initialize all variables
sess.run(init)

print("Training the model: ")
for epoch in range(0, 20001):
    # run the training operation
    cvalues = sess.run([train, loss, W_hidden, b_hidden, W_output],
                       feed_dict={n_input: input_data, n_output: output_data})
    if epoch % 5000 == 0:
        print("")
        print("Step#{:>3}".format(epoch))
        print("Loss function: {}".format(cvalues[1]))
        print('Weights: ', *W_output.eval(session=sess))
        print('Bias: ', *b_hidden.eval(session=sess))

print("\nCheck the model:")
for i in range(len(input_data)):
    print("input: {} | output: {}".format(input_data[i], np.ndarray.round(sess.run(output, feed_dict={n_input: [input_data[i]]}))))

# Save the variables to disk.
save_path = saver.save(sess, "/tmp/model.ckpt")
print("\nModel saved in path: %s" % save_path)
