import numpy as np
import matplotlib as plt

import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import spline
import numpy as np
# x = np.array([6,7,8,6,10])
# y = np.array([10,20,6,30,15])
#
# # xnew = np.linspace(x.min(),x.max(),100) #300 represents number of points to make between T.min and T.max
# # print (xnew)
# # ynew = spline(x,y,xnew)
# tck, u = interpolate.splprep([x, y], s=0)
# unew = np.arange(0, 1.01, 0.01)
# out = interpolate.splev(unew, tck)
# plt.plot(out[0], out[1])
# # plt.plot(xnew,ynew)
# plt.xlabel("X Axis")
# plt.ylabel("Y Axis")
#
# fig = plt.gcf()
#
# plt.show()
#
# fig.savefig("just_test.png")
a= [1 for _ in range(3)]
print (a)
#
# unew = np.arange(0, 1.01, 0.01)
# out = interpolate.splev(unew, tck)
# plt.figure()
# plt.plot(x, y, 'x', out[0], out[1], np.sin(2*np.pi*unew), np.cos(2*np.pi*unew), x, y, 'b')
# plt.legend(['Linear', 'Cubic Spline', 'True'])
# plt.axis([-1.05, 1.05, -1.05, 1.05])
# plt.title('Spline of parametrically-defined curve')
# plt.show()
a = [1,2,3,4]
b = ['a', 'b', 'c', 'd']
c = np.array(['a', 'b', 'c', 'd'])
s = np.arange(len(a))
np.random.shuffle(s)
print (s)

c = c[s]
print (c)
