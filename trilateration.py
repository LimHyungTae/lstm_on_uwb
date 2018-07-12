import csv
import sys
import numpy as np
from DataPreprocessing import DataPreprocessing


class Trilateration:
    def __init__(self, dir, output_dir):

        self.dir = dir
        self.output_dir = output_dir
        self.loadData()
    def trilaterate2D(self):
        self.position =  np.array([[-500.0,-500.0],
                                   [5500.0 ,-500.0],
                                   [5500.0 , 5500.0],
                                   [-500.0, 5500.0]])
        A = []
        B = []
        ##trilateration using SVD
        for idx in range(4):
            if idx == 0: #i:1 j:4
                x_coefficient = self.position[3][0] - self.position[idx][0] #x1-xidx
                y_coefficient = self.position[3][1] - self.position[idx][1] #y1-yidx
                b = 1/2*(self.distances[idx]**2 - self.distances[3]**2 -
                         ((self.position[idx][0]-self.position[3][0])**2 + (self.position[idx][1]-self.position[3][1])**2))\
                        +x_coefficient*self.position[3][0] + y_coefficient*self.position[3][1]
                A.append([x_coefficient, y_coefficient])
                B.append([b])
            else:
                x_coefficient = self.position[0][0] - self.position[idx][0] #x1-xidx
                y_coefficient = self.position[0][1] - self.position[idx][1] #y1-yidx
                b = 1/2*(self.distances[idx]**2 - self.distances[0]**2 -
                         ((self.position[idx][0]-self.position[0][0])**2 + (self.position[idx][1]-self.position[0][1])**2))\
                        +x_coefficient*self.position[0][0] + y_coefficient*self.position[0][1]
                A.append([x_coefficient, y_coefficient])
                B.append([b])
        B = np.array(B)
        A_pseudo = np.linalg.pinv(A)
        position = np.dot(A_pseudo, B)
        # return x, y position
        return position
        # print (position, position[0],position[1])
    def loadData(self):
        xy = np.loadtxt(self.dir, delimiter=',')
        self.distances_data = xy[:,:4] # [[l1,l2,l3,l4], ... ]

    def getDistances(self,distances):
        self.distances = distances

    def write_file_data2D(self):
        result_file = open(self.output_dir, 'w', encoding='utf-8', newline='')
        wr = csv.writer(result_file)
        for distances in self.distances_data: #refer to loadData fn!
            self.getDistances(distances)
            position = self.trilaterate2D()
            wr.writerow([position[0][0],position[1][0]])

        result_file.close()
####################
# target = 'diagonal'
# trilateration = Trilateration('./data_'+ target +'.csv')
# trilateration.loadData()
# trilateration.write_file_data('results/result_'+ target +'_w_trilateration11111.csv')

