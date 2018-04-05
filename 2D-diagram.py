# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *

#定义从.txt文档中读取实验数据的函数
def ReadData(name,path):
    
    file = open(path)
    All=file.readlines()
    for i in All:
        if i[16:20] == "Time":
            position=All.index(i)+1
            print('\n Your data begins from Row %d, The meaning of each column is: \n %s' %(position,i,))
    data=np.loadtxt(name,skiprows=position)
    print('There are %d rows and %d columns.' %(data.shape[0],data.shape[1]))
    file.close()
    return data

#定义将微分电阻与偏流的关系转化为微分电导与偏压的关系的函数,该函数认为偏置电流的数列最中间点是电流零点
def dr2dc(I_bias,dr):

    #
    if dr.size!=I_bias.size:
        print('Error! The array of differential groups and bias currents must have the same dimensions!')
        return None
    else:
        size=dr.size

    I0=int(size/2)#寻找偏置电流零点(实际位置为I0+1)
    
    #积分得到电压：
    V=np.zeros(size)#定义电压数组
    V[0]=0
    for i in np.linspace(1,size-1,size-1):
        i=int(i)
        V[i]=V[i-1]+dr[i-1]*(I_bias[i]-I_bias[i-1])
    V=V-V[I0]

    #求微分电导
    dc=np.divide(1,dr)
    # dc=6.62607004e-34*dc/(2* 1.6021766**2 * 1e-38) #转化单位从\Omega^{-1}为2e^2/h

    #插值求并使数据点均匀
    f=interpolate.interp1d(V,dc,kind='linear')
    V_new=np.linspace(V.min(),V.max(),size)
    dc_new=f(V_new)

    ##绘图
    # plt.figure(999,figsize=(10,6))
    # plt.plot(V_new,dc_new)
    # xlabel(r'$V(\mu V)$',fontsize=12)
    # ylabel(r'$dI/dV (2e^2/h)$',fontsize=12,labelpad=12)
    # title('Differential Conduction-Bias Voltage Relationship',fontsize=16)

    return V_new,dc_new
    
#二维图：
data2=ReadData('180323.004.txt',r'C:\Users\孙晓培\AnacondaProjects\DataProcessing\180323.004.txt')

#判断扫场的方式以及寻找断点
for i in np.linspace(0,data2.shape[0]-1,data2.shape[0]):
    if data2[int(i)][-1]==data2[0][-1]:
        continue
    else:
        y_points=int(i)
        x_points=int(data2.shape[0]/i)
        print('There are %d points in \'I_bias\' and %d points in Magnetic Filed \'B\'.' %(y_points,x_points))
        break

for i in np.linspace(0,data2.shape[0]-1,data2.shape[0]):
    if data2[int(i)][4]==0:
        print('The point %d means zero magnet' %int(i))
        break

fig3=plt.figure(3,figsize=(10,8))

# x=np.linspace(data2[0][-1],data2[-1][-1],x_points)
# y=np.linspace(data2[0][3],data2[-1][3],y_points)
# R=data2[:,1].reshape(x_points,y_points)
# R=R.T
# X,Y=np.meshgrid(x,y)

x=np.linspace(data2[0][-1],data2[-1][-1],x_points)    #提取平行的磁场数据(unit:T)
Ib=np.linspace(data2[0][2],data2[-1][2],y_points)/1e5 #提取锁相测量的数据并将其转化为偏置电流值(unit: A)
R=data2[:,1].reshape(x_points,y_points)
R=R.T/1e-6

G=np.zeros((y_points,x_points))
for i in np.linspace(0,x_points-1,x_points):
    i=int(i)
    y,G[:,i]=dr2dc(Ib,R[:,i])#将微分电阻与偏流换位微分电导与偏压(unit: \Omega^{-1} & V)
y=y*1e6#将偏置电压单位转化为 \mu V
X,Y=np.meshgrid(x,y)

# cmaps = ['viridis', 'plasma', 'inferno', 'magma',
#             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
#             'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
#             'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
#             'hot', 'afmhot', 'gist_heat', 'copper',
#             'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
#             'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
#             'Pastel1', 'Pastel2', 'Paired', 'Accent',
#             'Dark2', 'Set1', 'Set2', 'Set3',
#             'tab10', 'tab20', 'tab20b', 'tab20c',
#             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
#             'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
# for i in np.linspace(0,78,79):
#     i=int(i)
#     plt.subplot(8,10,i+1)
#     pcolor(X,Y,R,cmap=cmaps[i])#颜色类型详见：https://matplotlib.org/examples/color/colormaps_reference.html

pcolor(X,Y,G,cmap='jet')#颜色类型详见：https://matplotlib.org/examples/color/colormaps_reference.html
colorbar()

xlabel(r'$B(T)$',fontsize=12)
ylabel(r'$V_{Bias}(\mu V)$',fontsize=12,labelpad=12)
# text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(a)')
fig3.savefig('2-D.jpg')

plt.show()