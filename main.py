# Lab 12 Character Sequence RNN
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from lstm_network import LSTM
import numpy as np
import DataPreprocessing
import trilateration
from tqdm import tqdm, trange
import os
import argparse
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.set_random_seed(777)  # reproducibility
# hyper parameters
p =argparse.ArgumentParser()
p.add_argument('--train_data', type=str, default="inputs/train_data_2D_zigzag_error_10.csv")
p.add_argument('--save_dir', type=str, default="model/RiTA/bidirectional_LSTM_model/")
p.add_argument('--load_dir', type=str, default="model/bidirectional_LSTM_model/")
p.add_argument('--lr', type=float, default = 0.008)
p.add_argument('--decay_rate', type=float, default = 0.85)
p.add_argument('--epoches', type=int, default = 10000)
p.add_argument('--batch_size', type=int, default = 149998)
p.add_argument('--hidden_size', type=int, default = 2) # RNN output size
p.add_argument('--input_size', type=int, default = 4) #RNN input size : number of uwb
p.add_argument('--sequence_length', type=int, default = 5) # # of lstm rolling
p.add_argument('--output_size', type=int, default = 2) #final output size (RNN or softmax, etc)
p.add_argument('--mode', type=str, default = "train") #train or test
args = p.parse_args()




data_parser = DataPreprocessing.DataPreprocessing(args.train_data, args.sequence_length)
data_parser.fit_data()

X_data,Y_data =data_parser.set_data()
# data : size of data - sequence length + 1
LSTM = LSTM(args) #batch_size, dic_size, sequence_length, hidden_size, num_classes)
print(X_data.shape) #Data size / sequence length / uwb num

#terms for learning rate decay
global_step = tf.Variable(0, trainable=False)
iter = int(len(X_data)/args.batch_size)
num_total_steps = args.epoches*iter
# boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps), np.int32((9/10) * num_total_steps)]
# values = [learning_rate, learning_rate / 2, learning_rate / 4, learning_rate/8]
# learning_rate_decay = tf.train.piecewise_constant(global_step, boundaries, values)
LSTM.build_loss(args.lr, args.decay_rate, num_total_steps/5)
saver = tf.train.Saver(max_to_keep = 5)

# Use simple momentum for the optimization.
###########for using tensorboard########
merged = tf.summary.merge_all()
########################################
with tf.Session() as sess:
    if (args.mode=='train'):

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('./board/lstm_RiTA_train_bidirectional',sess.graph)
        step = 0
        min_loss = 2
        tqdm_range = trange(args.epoches, desc = 'Loss', leave = True)
        for ii in tqdm_range:
            loss_of_epoch = 0
            for i in range(iter): #iter = int(len(X_data)/batch_size)
                step = step + 1
                idx = i* args.batch_size
                l, _,gt, prediction, summary = sess.run([LSTM.loss, LSTM.train, LSTM.Y_data, LSTM.Y_pred, merged ],
                                                        feed_dict={LSTM.X_data: X_data[idx : idx + args.batch_size], LSTM.Y_data: Y_data[idx : idx + args.batch_size]})
                writer.add_summary(summary, step)
                loss_of_epoch += l/args.batch_size
            loss_of_epoch /=iter
            if (loss_of_epoch < min_loss):
                min_loss = loss_of_epoch
                saver.save(sess, args.save_dir + 'model_'+'{0:.5f}'.format(loss_of_epoch).replace('.','_'), global_step=step)
            tqdm_range.set_description('Loss ' +'{0:.7f}'.format(loss_of_epoch)+'  ')
            tqdm_range.refresh()

    elif (args.mode =='test'):
   #For save diagonal data
        saver.restore(sess, args.load_dir + 'model_0_001-10000')
   # tf.train.latest_checkpoint(

        diagonal_data = 'inputs/data_diagonal_w_big_error.csv'
        data_parser.dir = diagonal_data
        X_test, Y_test = data_parser.set_test_data()
        prediction = sess.run([LSTM.Y_pred], feed_dict={LSTM.X_data: X_test}) #prediction : type: list, [ [[[hidden_size]*sequence_length] ... ] ]

        data_parser.write_file_data('results/result_diagonal.csv', prediction)
    #trilateration

        trilateration = trilateration.Trilateration(diagonal_data, 'results/result_diagonal_w_trilateration.csv')
        trilateration.write_file_data2D()

        #For save one round path data
        round_data =  'inputs/data_round1_w_big_error.csv'
        data_parser.dir = round_data
        X_test, Y_test = data_parser.set_test_data()
        prediction1 = sess.run([LSTM.Y_pred], feed_dict={LSTM.X_data: X_test})
        data_parser.write_file_data('results/result_round1.csv', prediction1)

        #trilateration
        # trilateration = trilateration.Trilateration(diagonal_data, 'results/result_round1_w_trilateration.csv')
        trilateration.dir =round_data
        trilateration.output_dir = 'results/result_round1_w_trilateration.csv'
        trilateration.write_file_data2D()

