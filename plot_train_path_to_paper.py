import numpy as np

import matplotlib.pyplot as plt

plt.xlim(-4.0, 4.0)
plt.ylim(-4.0, 4.0)
# option = 'horizontal zigzag'
option = 'vertical zigzag'

# if option == 'horizontal zigzag':
#     ax = plt.axes()
#     ax.arrow(0, 0, 4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax2 = plt.axes()
#     ax2.arrow(5, 0, 0, 1.05, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax3 = plt.axes()
#     ax3.arrow(5, 1.25, -4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax4 = plt.axes()
#     ax4.arrow(0, 1.25, 0, 1.05, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax5 = plt.axes()
#     ax5.arrow(0, 2.5, 4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax6 = plt.axes()
#     ax6.arrow(5, 2.5, 0, 1.05, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax7= plt.axes()
#     ax7.arrow(5, 3.75, -4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax8 = plt.axes()
#     ax8.arrow(0, 3.75, 0, 1.05, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax9 = plt.axes()
#     ax9.arrow(0, 5, 4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
# else:
#     ax = plt.axes()
#     ax.arrow(5, 5, 0, -4.80, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax2 = plt.axes()
#     ax2.arrow(5, 0, -1.05, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax3 = plt.axes()
#     ax3.arrow(3.75, 0, 0, 4.8, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax4 = plt.axes()
#     ax4.arrow(3.75, 5, -1.05, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax5 = plt.axes()
#     ax5.arrow(2.5, 5, 0, -4.80, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax6 = plt.axes()
#     ax6.arrow(2.5, 0, -1.05, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax7 = plt.axes()
#     ax7.arrow(1.25, 0, 0, 4.8, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax8 = plt.axes()
#     ax8.arrow(1.25, 5, -1.05, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')
#
#     ax9 = plt.axes()
#     ax9.arrow(0, 5, 0, -4.8, head_width=0.15, head_length=0.2, fc='k', ec='k')

def return_actual_position(x, y, direction):
    real_x = x*0.45
    real_y = y*0.45
    if direction == 'u':
        real_y += 0.058
    elif direction == 'd':
        real_y -= 0.058
    elif direction == 'l':
        real_x -= 0.058
    elif direction == 'r':
        real_x += 0.058

    return real_x, real_y


# plt.xlim(-4.0, 4.0)
# plt.ylim(-4.0, 4.0)


plt.xlim(-3.5, 3.5)
plt.ylim(-3.0, 3.5)

offset_x = - 0.8
offset_y = 0.2
data_list = [(-5, 5 ,'r'), (-3, 0, 'u'), (-6, -3, 'r'), (0, 2, 'd'), (1, -5, 'l'), (3, -2, 'u'), (6,6, 'd'), (6, -4, 'l')]
for data in data_list:
    real_x, real_y = return_actual_position(data[0], data[1], data[2])
    plt.scatter(real_x, real_y, c='r', marker='^', s=100)
    axis_string = '(' + str(round(real_x, 2)) +', ' + str(round(real_y,2)) + ')'
    plt.text(real_x + offset_x, real_y + offset_y, axis_string, fontsize=15)
plt.grid()
plt.xlabel("X Axis [m]", fontsize=15)
plt.ylabel("Y Axis [m]", fontsize=15)
fig =plt.gcf()
plt.show()
fig.savefig(option + ".png")