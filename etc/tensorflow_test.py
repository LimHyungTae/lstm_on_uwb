import tensorflow as tf

# x = tf.constant([35, 40, 30], name = 'x' )
# y = tf.Variable(x+1, name = 'y')

W =tf.Variable([.3],tf.float32)
b =tf.Variable([.3],tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

predicted_y = W*x+b
loss = tf.reduce_sum(tf.square(y - predicted_y))
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

x_data = [2, 4, 6, 9]
y_data = [0, 1, 2, 3]

init = tf.global_variables_initializer()
sess =tf.Session()
sess.run(init)
for i in range(10):
    _ ,cur_w, cur_b , cur_loss=  sess.run([train,W,b,loss], feed_dict={x:x_data, y : y_data})
    print( cur_w, cur_b, cur_loss)




