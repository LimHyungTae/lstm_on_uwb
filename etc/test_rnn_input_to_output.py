import tensorflow as tf
import pprint
import numpy as np
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

hidden_size = 2

cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
list =[[[4,4,4]]]
list = np.array(list)
print(list.shape)
x_data = np.array([[[718.8802343511221, 5516.393965257707, 7772.465594137369, 5524.400034867455],
[724.5600703104905, 5505.717965987669, 7768.0007166714695, 5528.63995436672],
[731.9048703037521, 5497.330074172925, 7759.800210408214, 5529.000159792103  ],
[736.2090515491228, 5485.448867852925, 7753.160929655976, 5527.74840892488  ],
[743.6404575648415, 5474.361941484512, 7744.716569165959, 5530.232101099491  ],
[753.9795172728893, 5467.638708799926, 7736.275792213155, 5530.826157136589]]],dtype = np.float32)
print(x_data.shape)

outputs, _state = tf.nn.dynamic_rnn(cell, x_data, dtype = tf.float32)

sess.run(tf.global_variables_initializer())
print(outputs.shape)
x_for_softmax = tf.reshape(outputs, [-1,hidden_size]) # -1 for flatten

print(x_for_softmax.shape)

# print((outputs))
pp.pprint(outputs.eval())
# print(outputs)