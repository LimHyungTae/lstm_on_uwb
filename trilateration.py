import csv
import sys
import numpy as np
from DataPreprocessing import DataPreprocessing


class Trilateration:

    def trilaterate2D(self):
        self.position =  np.array([[-0.5,-0.5],
                                   [5.5 ,-0.5],
                                   [5.5 , 5.5],
                                   [-0.5, 5.5]])
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
    def setInput(self,dir):
        xy = np.loadtxt(dir, delimiter=',')
        self.distances_data = xy[:,:4] # [[l1,l2,l3,l4], ... ]

    def setOutputDir(self,out_dir):
        self.output_dir =out_dir

    def getDistances(self,distances):
        self.distances = distances

    def write_file_data2D(self):
        result_file = open(self.output_dir, 'w', encoding='utf-8', newline='')
        wr = csv.writer(result_file)
        for distances in self.distances_data: #refer to setInput fn!
            self.getDistances(distances)
            position = self.trilaterate2D()
            wr.writerow([position[0][0],position[1][0]])

        result_file.close()
####################
if __name__ == '__main__':
    input = 'inputs/test_data_arbitrary_square_uwb_2D_e10.csv'
    output_dir = 'results/RiTA/trilateration.csv'

    trilateration = Trilateration()
    trilateration.setInput(input)
    trilateration.setOutputDir(output_dir)
    trilateration.write_file_data2D()

