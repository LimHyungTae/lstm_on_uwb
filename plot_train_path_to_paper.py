import numpy as np

import matplotlib.pyplot as plt

plt.xlim(-1.0, 6.0)
plt.ylim(-1.0, 6.0)
# option = 'horizontal zigzag'
option = 'vertical zigzag'

if option == 'horizontal zigzag':
    ax = plt.axes()
    ax.arrow(0, 0, 4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax2 = plt.axes()
    ax2.arrow(5, 0, 0, 1.05, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax3 = plt.axes()
    ax3.arrow(5, 1.25, -4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax4 = plt.axes()
    ax4.arrow(0, 1.25, 0, 1.05, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax5 = plt.axes()
    ax5.arrow(0, 2.5, 4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax6 = plt.axes()
    ax6.arrow(5, 2.5, 0, 1.05, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax7= plt.axes()
    ax7.arrow(5, 3.75, -4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax8 = plt.axes()
    ax8.arrow(0, 3.75, 0, 1.05, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax9 = plt.axes()
    ax9.arrow(0, 5, 4.8, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')

else:
    ax = plt.axes()
    ax.arrow(5, 5, 0, -4.80, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax2 = plt.axes()
    ax2.arrow(5, 0, -1.05, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax3 = plt.axes()
    ax3.arrow(3.75, 0, 0, 4.8, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax4 = plt.axes()
    ax4.arrow(3.75, 5, -1.05, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax5 = plt.axes()
    ax5.arrow(2.5, 5, 0, -4.80, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax6 = plt.axes()
    ax6.arrow(2.5, 0, -1.05, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax7 = plt.axes()
    ax7.arrow(1.25, 0, 0, 4.8, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax8 = plt.axes()
    ax8.arrow(1.25, 5, -1.05, 0, head_width=0.15, head_length=0.2, fc='k', ec='k')

    ax9 = plt.axes()
    ax9.arrow(0, 5, 0, -4.8, head_width=0.15, head_length=0.2, fc='k', ec='k')



plt.scatter(-0.5, -0.5, c='r',marker ='^', s=100)
plt.scatter(-0.5, 5.5, c='r',marker ='^', s=100)
plt.scatter(5.5, 5.5, c='r',marker ='^', s=100)
plt.scatter(5.5, -0.5, c='r',marker ='^', s=100)
plt.xlabel("X Axis [m]")
plt.ylabel("Y Axis [m]")
fig =plt.gcf()
plt.show()
fig.savefig(option + ".png")