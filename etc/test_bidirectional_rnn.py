import tensorflow as tf
import pprint
import numpy as np
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

hidden_size = 2

cell1 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
cell2 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)

x_data = np.array([[[1.0,0.0]]],dtype = np.float32)

outputs, _state = tf.nn.bidirectional_dynamic_rnn(cell1, cell2, x_data, dtype = tf.float32)
# outputs, _state = tf.nn.dynamic_rnn(cell1, x_data, dtype = tf.float32)
concat_output = tf.concat([outputs[0], outputs[1]], axis = 1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output1, output2 = sess.run([outputs, concat_output])

    print (output1)
    print (output2)


# print((outputs))
# pp.pprint(outputs.eval())
# print(outputs)
#