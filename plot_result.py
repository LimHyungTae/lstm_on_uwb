import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np
import os
p =argparse.ArgumentParser()
p.add_argument('--save_dir', type=str, default="results/RiTA/graphs/")
p.add_argument('--gt_dir', type=str, default="inputs/test_data_arbitrary_square_uwb_2D_e10.csv")
p.add_argument('--bidirectional_LSTM_csv', type=str, default="results/RiTA/bidirectional_wo_fcn.csv")
p.add_argument('--unidirectional_LSTM_csv', type=str, default= "results/RiTA/unidirectional_wo_fcn.csv")
p.add_argument('--trilateration_csv', type=str, default="model/RiTA/bidirectional_LSTM_model/")
p.add_argument('--save_MSE_name', type=str, default="Distance_error_result.png")
p.add_argument('--save_trajectory_name', type=str, default="Trajectory_result.png")

args = p.parse_args()

class Visualization:
    def __init__(self, args):
        self.args = args
        self.folder_name = "results/"
        if not os.path.isdir(self.folder_name):
            os.mkdir(self.folder_name)
        self.setGT()

    def setGT(self):
        gt_xy = np.loadtxt(args.gt_dir, delimiter=',')
        #x_array: gt_xy[:,0]
        #y_array: gt_xy[:,1]
        self.gt_xy = gt_xy[4:,4:]

    def _calDistanceError(self, predicted_result_dir):
        predicted_xy = np.loadtxt(predicted_result_dir, delimiter=',')
        # gt_xy = np.random.randint(3,size = (4,2))
        # predicted_xy = np.random.randint(3, size = (4,2))
        dx_dy_array = self.gt_xy - predicted_xy

        distance_square = np.square(dx_dy_array[:,0]) + np.square(dx_dy_array[:,1])
        MSE = np.sum(distance_square)/distance_square.shape
        print (MSE)

        return np.sqrt(distance_square)

    def plotDistanceError(self, *target_files_csv):
        saved_file_name = self.args.save_MSE_name
        plot_title = "Distance Error"
        plt.title(plot_title)

        for i, csv in enumerate(target_files_csv):
            distance_error = self._calDistanceError(csv)
            #marker o x + * s:square d:diamond p:five_pointed star

            if i == 0:
                color = 'g'
                marker = 'x'
                linestyle = '-'
                label = 'Prediciton'
            elif i == 1:
                color = 'b'
                marker = 's'
                linestyle = '--'
                label = 'Prediciton'
            elif i == 2:
                color = 'r'
                marker = '*'
                linestyle = '--'
                label = 'GTn'
            plt.plot(range(distance_error.shape[0]), distance_error, color= color, marker = marker,
                            linestyle = linestyle,label = label)

        plt.legend()
        # plt.xlim(-0.5,1.5)
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.ylim(-0.5,1.5)
        plt.xlabel("Time Step t")
        plt.ylabel("Distance Error (cm)")
        fig = plt.gcf()
        plt.show()
        fig.savefig(saved_file_name)
        print ("Done")

    def plot2DTrajectory(self, *target_files_csv):
        saved_file_name = self.args.save_trajectory_name
        plot_title = "Trajectory"
        # plt.title(plot_title)
        gt_x = self.gt_xy[:,0]
        gt_y = self.gt_xy[:,1]

        plt.plot(gt_x, gt_y,'r',linestyle='--' , label = 'GT')

        for i, csv in enumerate(target_files_csv):
            predicted_xy = np.loadtxt(csv, delimiter = ',')
            predicted_x = predicted_xy[:,0]
            predicted_y = predicted_xy[:,1]
            #marker o x + * s:square d:diamond p:five_pointed star
            if i == 0:
                color ='g'
                marker ='x'
                linestyle = '-'
                label = 'Bidirectional LSTM'
            elif i == 1:
                color ='b'
                marker ='s'
                linestyle = '--'
                label = 'Unidirectional LSTM'
            elif i == 2:
                color ='r'
                marker ='*'
                linestyle = '--'
                label = 'Trilateration'
            plt.plot(predicted_x, predicted_y, color = color, #marker= marker,
                            linestyle = linestyle,label = label)

        plt.legend()
        # plt.xlim(-0.5,1.5)
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.ylim(-0.5,1.5)
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        fig = plt.gcf()
        plt.show()
        fig.savefig(saved_file_name)
        print ("Done")

    def drawResult3D(self, X_list, Y_list, Z_list):

        self.fig = plt.figure()
        # plt.subplot(221)
        # self.ax1 = self.fig.gca(projection = '3d') #add_subplot(111, projection='3d')
        # self.ax1.scatter(X_list, Y_list, Z_list)
        plt.subplot(222)
        plt.scatter(X_list, Y_list)
        plt.xlabel("X_axis")
        plt.ylabel("Y_axis")
        plt.subplot(223)
        plt.scatter(Y_list, Z_list)
        plt.xlabel("Y_axis")
        plt.ylabel("Z_axis")
        plt.subplot(224)
        plt.scatter(X_list, Z_list)
        plt.xlabel("X_axis")
        plt.ylabel("Z_axis")
        self.fig = plt.gcf()
        self.fig.savefig(self.folder_name +"/Results.png")

viz = Visualization(args)
# viz.plotDistanceError(args.bidirectional_LSTM_csv)
viz.plot2DTrajectory(args.bidirectional_LSTM_csv, args.unidirectional_LSTM_csv)
