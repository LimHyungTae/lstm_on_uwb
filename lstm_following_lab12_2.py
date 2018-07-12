# Lab 12 Character Sequence RNN
import tensorflow as tf
import numpy as np
from dataset_paser import Data
tf.set_random_seed(777)  # reproducibility

# hyper parameters
number_of_uwb = 4
data_dim = number_of_uwb  # RNN input size (one hot size)
hidden_size = 2 # RNN output size
output_dim = 2  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = 10 # number of lstm rollings (unit #)
learning_rate = 0.001

# sample_idx = [char2idx[c] for c in sample]  # char to index
# x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
# y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello
x = []
y = []
x_data= []
y_data= []

file_name = 'data_sample.txt'
total_length = 0
with open(file_name) as f:
    for i, l in enumerate(f):  # For large data, enumerate should be used!
        pass
    total_length = i
file = open(file_name , 'r')
for i in range(total_length):
    line = file.readline()[:-1].split(' ')
    line_x=[]
    line_y=[]
    print (line)
    for j in range(4):
        line_x.append(float(line[j]))
    for k in range(4,6):
        line_y.append(float(line[k]))
    x.append(line_x)
    y.append(line_y)
print(x)
print(y)
x_data.append(x)
y_data.append(y)
file.close()
x_data = np.array(x_data)#, dtype = tf.float32)
y_data = np.array(y_data)#, dtype = tf.float32)
print(x_data.shape)
print(y_data.shape)
# data_parser =Data('./data.txt',1)

X = tf.placeholder(tf.float32, [None, sequence_length, data_dim])  # X data
Y = tf.placeholder(tf.float32, [None, sequence_length, hidden_size])  # Y label
#
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_size, state_is_tuple=True, activation = None)
# initial_state = cell.zero_state(batch_size, tf.float32)
# initial_state = np.array([[10.0,0.0],[20.0,0.0],[80.0,0.0],[10.0,0.0],[10.0,0.0],
#                           [70.0,0.0],[40.0,0.0],[60.0,0.0],[0.0,0.0],[100.0,0.0]])

# cell = tf.nn.rnn_cell.MultiRNNCell([cell]*2)

# outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X,  dtype=tf.float32)
## FC layer

X_for_fc = tf.reshape(outputs, [-1, hidden_size]) #-1 for flatten
# outputs = tf.contrib.layers.fully_connected(X_for_fc, 100, activation_fn=None)

Y_pred = tf.contrib.layers.fully_connected(X_for_fc, num_outputs= 10)
Y_pred = tf.contrib.layers.fully_connected(Y_pred, output_dim, activation_fn=None)
Y_pred = tf.reshape(Y_pred, [batch_size, sequence_length, output_dim])
print( Y.shape)


# weights = tf.ones([batch_size, sequence_length])
# sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)

loss = tf.losses.mean_squared_error(Y, Y_pred, weights = 1, reduction=tf.losses.Reduction.MEAN)
# loss = tf.reduce_mean(tf.square(Y-Y_pred))
# loss = (tf.square(Y-Y_pred))
# loss = tf.reduce_sum(tf.square(Y-Y_pred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

#prediction = tf.argmax(outputs, axis=2)
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iter =5000
    for i in range(iter):
        l, _, y = sess.run([loss, train,Y_pred], feed_dict={X: x_data, Y: y_data})
        print ( l ,y )
        #result = sess.run(prediction, feed_dict={X: x_data})
        # if (i>iter-2):
        print(l)

    test_predict = sess.run(Y_pred, feed_dict = {X: x_data, Y: y_data})
    print(test_predict)


