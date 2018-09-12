import numpy as np
import tensorflow as tf
b= []
sess = tf.InteractiveSession()

a = np.array([[2,2],[1,3]])
c = np.array([[1,3],[4,3]])
d = tf.concat([a,c], axis = 1 )
d = tf.add(a,c)
print(d.eval())
e = tf.Print(d, [d], message = "This is a :")
print (d)

