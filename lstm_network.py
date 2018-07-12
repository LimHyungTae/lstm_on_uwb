import tensorflow as tf

class BidirectionalLSTM:
    def __init__(self, args): # batch_size, input_size,sequence_length, hidden_size, output_size):
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.sequence_length = args.sequence_length

        self.X_data = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.sequence_length, self.input_size],
                                           name='input_placeholder')
        self.Y_data = tf.placeholder(dtype=tf.float32,
                                        shape=[None, self.sequence_length,self.hidden_size],
                                        name='output_placeholder')

        self.build_model()
    def setUnidirectionalLSTM(self):
        with tf.variable_scope("unidirectional_lstm"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_size)

            self.isbidirectional = 0
            # outputs : tuple
            a = tf.contrib.seq2seq.python.ops.decoder()
            return tf.nn.dynamic_rnn(cell, self.X_data, dtype=tf.float32)


    def setBidirectionalLSTM(self):
        with tf.variable_scope("bidirectional_lstm"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_size)
            # cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 1.0)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_size)
            # cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 1.0)

            self.isbidirectional = 1
            # outputs : tuple
            return tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.X_data, dtype=tf.float32)

    def build_model(self):
            outputs, _states = self.setBidirectionalLSTM()
            if (self.isbidirectional):
                print ("It's bidirectional")
                # outputs = tf.reduce_sum(outputs, axis=0)
                outputs = tf.concat([outputs[0], outputs[1]], axis = 1)
            ## FC layer
            X_for_fc = tf.reshape(outputs, [-1, self.hidden_size*2])  # -1 for flatten
            # outputs = tf.contrib.layers.fully_connected(X_for_fc, 100, activation_fn=None)
            Y_pred = tf.contrib.layers.fully_connected(X_for_fc, 100)
            Y_pred = tf.contrib.layers.fully_connected(Y_pred, self.output_size, activation_fn=None)

            # Y_pred = tf.reshape(Y_pred, [batch_size, sequence_length, num_classes])
            self.Y_pred = tf.reshape(Y_pred, [-1, self.sequence_length, self.output_size])


    def build_loss(self, lr, lr_decay_rate, lr_decay_step):
        self.init_lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        batch_size = self.batch_size

        with tf.variable_scope('lstm_loss'):
            # loss = tf.losses.mean_squared_error(Y, outputs, weights=weights)#reduction=tf.losses.Reduction.MEAN)
            # loss = tf.reduce_mean(tf.square(Y-outputs))
            self.loss = tf.reduce_sum(tf.square(self.Y_data - self.Y_pred))
            tf.summary.scalar('lstm_loss', self.loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.cur_lr = tf.train.exponential_decay(self.init_lr,
                                                     global_step=self.global_step,
                                                     decay_rate=self.lr_decay_rate,
                                                     decay_steps=self.lr_decay_step)

            tf.summary.scalar('global learning rate', self.cur_lr)

            self.train = tf.train.AdamOptimizer(learning_rate= self.cur_lr).minimize(self.loss)
