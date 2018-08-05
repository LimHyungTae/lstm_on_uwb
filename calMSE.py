import numpy as np

gt_xy = np.loadtxt('test_data_arbitrary_path2D.csv', delimiter=',')
gt_xy = gt_xy[4:,4:]
predicted_xy = np.loadtxt('results/RiTA/result_bidirectional.csv', delimiter=',')

# gt_xy = np.random.randint(3,size = (4,2))
# predicted_xy = np.random.randint(3, size = (4,2))
dx_dy_array = gt_xy - predicted_xy

distance_square = np.square(dx_dy_array[:,0]) + np.square(dx_dy_array[:,1])
MSE = np.sum(distance_square)/distance_square.shape
print (MSE)
distance = np.sqrt(distance_square)

def argtest(*args):
    for i in range(len(args)):
        print (i)
        print (args)
lsit = [4,6,1]
argtest(1,2,3, None)


argtest(lsit)
