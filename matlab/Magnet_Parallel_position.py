# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:00:51 2018

@author: brucelau
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *


A = np.array([1000,1000,0,0,500,500,-500,-500,200,1000,1000,-100,-200,-300,-1000,-1000,-100,100])+100
B = np.array([8,7,7,6,6,5,5,4,4,4,3,3,3,3,3,2,2,2])



fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111)
# fc: filling color
# ec: edge color

plt.scatter([100, 600, 300, 0, -100, -200, 0,  200], 
            [7,6,4,3,3,3,2,2], 
            marker = 'x', color = 'r',edgecolors='b',  s = 300) 

for i in np.array([2,4,6,8,9,11,12,13,14,16,17]).astype(int)-1:
    ax.arrow(A[i], B[i], A[i+1]-A[i], B[i+1]-B[i],
    length_includes_head=True,# 增加的长度包含箭头部分
    head_width=0.25, head_length=20, fc='b', ec='b',linewidth=3)
    
ax.arrow(100, 8, 1000, 0,
    length_includes_head=True,# 增加的长度包含箭头部分
    head_width=0.25, head_length=20, fc='b', ec='b',linestyle='-.',linewidth=3)

plt.plot([1100,1100],[8,7],'b',
         [100,100],[7,6],'b',
         [600,600],[6,5],'b',
         [-400,-400],[5,4],'b',
         [1100,1100],[4,3],'b',
         [-900,-900],[3,2],'b',
         linestyle='--',linewidth=3)

x_ticks = np.array([-1000, -500,-300,-200,-100,0,100,200,500,1000])+100
plt.xticks(x_ticks,fontsize=15)
y_ticks = np.arange(1, 9 )
plt.yticks(y_ticks, color='w')

xlabel(r'$B_{\parallel}(Gs)$',fontsize=20,labelpad=15)

ax.grid()
ax = gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none')
# Example:
#    ax = fig.add_subplot(122)
#    ax.annotate("", xy=(B[0], B[1]), xytext=(A[0], A[1]),arrowprops=dict(arrowstyle="->"))
#    ax.set_xlim(0,5)
#    ax.set_ylim(0,5)
#    ax.grid()
#    ax.set_aspect('equal') #x轴y轴等比例
#    plt.show()
#    plt.tight_layout()
#保存图片，通过pad_inches控制多余边缘空白
plt.savefig('Magnet_Parallel_position.jpg', transparent = True, bbox_inches = 'tight', pad_inches = 0.25) 
