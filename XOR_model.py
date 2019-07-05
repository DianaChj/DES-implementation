# Author:   Diana Chajkovska
# Date:     20th May, 2019

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def analyze_classifier(sess, i, w1, b1, w2, XOR_X, XOR_T):
    """Visualize the classification."""
    print('\nEpoch %i' % i)
    print('Hypothesis %s' % sess.run(output,
                                     feed_dict={n_input: XOR_X,
                                                n_output: XOR_T}))
    print('w1=%s' % sess.run(w1))
    print('b1=%s' % sess.run(b1))
    print('w2=%s' % sess.run(w2))
    print('cost (ce)=%s' % sess.run(cross_entropy,
                                    feed_dict={n_input: XOR_X,
                                               n_output: XOR_T}))
    # Visualize classification boundary
    xs = np.linspace(-1, 2)
    ys = np.linspace(-1, 2)
    pred_classes = []
    for x in xs:
        for y in ys:
            pred_class = sess.run(output,
                                  feed_dict={n_input: [[x, y]]})
           # print('x - ', x, 'y - ', y, 'c - ', pred_class)
            pred_classes.append((x, y, pred_class.round()))
    xs_p, ys_p = [], []
    xs_n, ys_n = [], []
    for x, y, c in pred_classes:
        if c == 0:
            xs_n.append(x)
            ys_n.append(y)
        else:
            xs_p.append(x)
            ys_p.append(y)
    plt.plot(xs_p, ys_p, 'ro', xs_n, ys_n, 'bo')
    plt.show()

input_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype='uint8')
output_data = np.array([[0.], [1.], [1.], [0.]], dtype='uint8')

# Number of input|output neurons
n_input = tf.placeholder(tf.float32, shape=[None, 2], name="n_input")
n_output = tf.placeholder(tf.float32, shape=[None, 1], name="n_output")

hidden_nodes = 2

b_hidden = tf.Variable(tf.zeros([hidden_nodes]), name="hidden_bias")
b2 = tf.Variable(tf.zeros([1]), name="Biases2")
W_hidden = tf.Variable(tf.random_normal([2, hidden_nodes]), name="hidden_weights")
hidden = tf.sigmoid(tf.matmul(n_input, W_hidden) + b_hidden)

# Output layer`s weight matrix
W_output = tf.Variable(tf.random_normal([hidden_nodes, 1]), name="output_weights")
# Calculate output layer`s activation
output = tf.sigmoid(tf.matmul(hidden, W_output) + b2)

# Calculate the mean of cross_entropy
cross_entropy = tf.square(n_output - output)
# Calculate the mean of cross_entropy
loss = tf.reduce_mean(cross_entropy)
# Optimize loss function
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Create session
sess = tf.Session()
# Initialize all variables
sess.run(init)

print("Training the model: ")
for epoch in range(0, 1001):
    # run the training operation
    sess.run(optimizer, feed_dict={n_input: input_data, n_output: output_data})
    if epoch % 200 == 0:
        analyze_classifier(sess, epoch, W_hidden, b_hidden, W_output, input_data, output_data)

print("\nCheck the model:")
for i in range(len(input_data)):
    print("input: {} | output: {}".format(input_data[i], np.ndarray.round(sess.run(output, feed_dict={n_input: [input_data[i]]}))))

# Save the variables to disk.
save_path = saver.save(sess, "/tmp/model.ckpt")
print("\nModel saved in path: %s" % save_path)
