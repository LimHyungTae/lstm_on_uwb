import cv2
import numpy as np
import os
import random

class Data:
    def __init__(self, dir, batch_size ):
        self.dir = dir
        self.batch_size = batch_size

        self.start_point = 0
        self.remain_task = False
        self.batch_input = []
        self.batch_gt = []

        self.debug_i=[]
        with open(self.dir) as f:
            for i,l in enumerate(f): #For large data, enumerate should be used!
                pass
            self.file_length = i+1

    def get_data(self):
        del self.batch_input[:]
        del self.batch_gt[:]
        del self.debug_i[:]
        end_point = self.start_point + self.batch_size
        # print(self.start_point)
        # print (end_point)
        with open (self.dir) as data:
            for i, line in enumerate(data):
                if (i >= self.start_point and i< end_point):
                    # print ('hi ' + str(i))
                    edited_line = line[:-1]
                    edited_line = edited_line.split(' ')
                    batch_input = []
                    batch_gt = []
                    for j in edited_line[:4]:
                        batch_input.append(float(j))
                    for j in edited_line[4:6]:
                        batch_gt.append(float(j))

                    #print (i, batch_input, batch_gt)
                    self.batch_input.append(batch_input)
                    self.batch_gt.append(batch_gt)
                    self.debug_i.append(i)

                    if ((i+1) == self.file_length): # i arrives to file length
                        remain_length = end_point - self.file_length
                        self.remain_task =True
                        break
            # print(self.batch_input)
            # print(self.batch_input[0])
            #
            # print(self.batch_gt)
        if (self.remain_task):
            data =open(self.dir , 'r')
            for i in range(end_point - self.file_length):
                line = data.readline()
                edited_line = line[:-1]
                edited_line = edited_line.split(' ')
                batch_input = edited_line[:4]
                batch_gt = edited_line[4:]
                self.batch_input.append(batch_input)
                self.batch_gt.append(batch_gt)
                self.debug_i.append(i)

            data.close()
            self.start_point = end_point - self.file_length
            self.remain_task = False
        else:
            self.start_point = end_point
        assert len(self.batch_input) == len(self.batch_gt)
        # print (len(self.batch_input), len(self.batch_input[0]), len(self.batch_gt[0]))
        # #print(self.batch_input)
        # print(self.debug_i[0], self.batch_input[0])
        # print(self.debug_i[-1], self.batch_input[-1])
        # #print(self.batch_gt)
        # print('-------------------\n')
        # #print(self.batch_input)
        # batchintput = np.asarray(self.batch_input, dtype=np.float32)
        # #batchintput = batchintput/255.0 - 0.5
        #
        # batchgt = np.asarray(self.batch_gt, dtype=np.float32)
        # #batchgt = batchgt / 255.0 - 0.5
        #
        return self.batch_input, self.batch_gt


