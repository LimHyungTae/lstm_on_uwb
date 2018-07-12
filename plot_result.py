import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np
parser = argparse.ArgumentParser()


# gt_file = open('results/test_diagonal_gt.csv','r',encoding = 'utf-8')
# prediction_file = open('results/result_diagonal.csv','r',encoding = 'utf-8')
def extract_x_y(dir):
    file = csv.reader(open(dir,'r'),delimiter=',')
    position_list = [x for x in file]
    position_x =[]
    position_y =[]
    step =0
    for position in position_list:
       step +=1
       if (step% 25 == 0):
           position_x.append(round(float(position[0]),3))
           position_y.append(round(float(position[1]),3))
    return position_x, position_y

basic_dir='results/'
##############################
index = 1
dic_target = {1: 'round1', 2: 'diagonal'}
dic_plot_title = {1: 'One round', 2: 'Diagonal'}
target = dic_target[index]
w_trilateration = '_w_trilateration'
##############################
target_file = basic_dir + 'result_'+target+'.csv'
trilateration_file = basic_dir +'result_' +target + w_trilateration+'.csv'
gt_file = basic_dir+ 'test_' + target + '_gt.csv'

plot_title = dic_plot_title[index]+" path"
saved_file_name = basic_dir+ target+"_path_result_more_iter"+w_trilateration

test_diagonal_x, test_diagonal_y= extract_x_y(target_file)
test_tri_x, test_tri_y= extract_x_y(trilateration_file)

gt_diagonal_x, gt_diagonal_y= extract_x_y(gt_file)
# print (max(test_diagonal_x))
# print (max(test_diagonal_y))
# print (max(gt_diagonal_x))
# print (max(gt_diagonal_y))
plt.title(plot_title)
plt.plot(test_diagonal_x, test_diagonal_y,'gx',linestyle ='-',label = 'Prediction')
#test trilateration
plt.plot(test_tri_x, test_tri_y,'bs',linestyle ='--',label = 'Trilateration')

plt.plot(gt_diagonal_x, gt_diagonal_y,'r*',linestyle='--' , label = 'GT')
# marker o x + * s:square d:diamond p:five_pointed star
plt.legend()
# plt.xlim(-0.5,1.5)
# plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
# plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
# plt.ylim(-0.5,1.5)
plt.xlabel("X_axis")
plt.ylabel("Y_axis")
fig = plt.gcf()
plt.show()
fig.savefig(saved_file_name)
print ("Done")


