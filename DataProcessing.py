# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
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

#定义将微分电阻转化为微分电导的函数,注意偏置电流的数列最中间点必须是零点,偏置电流值需要为uA
def dr2dc(I_bias,dr):

    if dr.size!=I_bias.size:
        print('Error! The array of differential groups and bias currents must have the same dimensions!')
        return None
    else:
        size=dr.size

    I0=int(size/2)#寻找偏置电流零点(实际位置为I0+1)
    
    V=np.zeros(size)#定义电压数组
    V[0]=0
    for i in np.linspace(1,size-1,size-1):
        i=int(i)
        V[i]=V[i-1]+dr[i-1]*(I_bias[i]-I_bias[i-1])
    V=V-V[I0]

    plt.figure(3,figsize=(10,6))
    plt.plot(I_bias,V)
    xlabel(r'$I_{Bias}(\mu A)$',fontsize=12)
    ylabel(r'$V(\mu V)$',fontsize=12,labelpad=12)
    title('V-I Relationship',fontsize=16)
    return None

# #随磁场变化四个器件微分电阻值的对比：
data1=ReadData('180322.001.txt',r'C:\Users\孙晓培\AnacondaProjects\DataProcessing\180322.001.txt')

#判断扫场的方式以及寻找断点
for i in np.linspace(0,data1.shape[0]-1,data1.shape[0]):
    if data1[int(i)][4]==data1[0][4]:
        continue
    else:
        y_points=int(i)
        x_points=int(data1.shape[0]/i)
        print('There are %d points in \'I_bias\' and %d points in Magnetic Filed \'B\'.' %(y_points,x_points))
        break

for i in np.linspace(0,data1.shape[0]-1,data1.shape[0]):
    if data1[int(i)][4]==0:
        zero_mgnet=int(i)
        print('The point %d means zero magnet' %int(i))
        break

#各项数据随时间的变化，用于初步判断：
# for i in np.linspace(2,data1.shape[1],data1.shape[1]-1):
#     plt.subplot(data1.shape[1]-1,1,i-1)
#     x=data1[:,0]
#     y=data1[:,int(i)-1]
#     plt.plot(x,y)
#     #plt.title('Column %d' %i)
x=data1[zero_mgnet:zero_mgnet+y_points,3]/1e-6
y=data1[zero_mgnet:zero_mgnet+y_points,1]/1e-7

dr2dc(x,y)

fig1=plt.figure(1,figsize=(10,6), dpi=600)#绘制四个器件微分电阻随磁场的变化
labelsize=12

plt.subplot(4,1,1)
x=data1[0:-1:501,4]
y=data1[251:-1:501,1]/1e-7
plt.plot(x,y,'b')
xlabel(r'$B(T)$',fontsize=labelsize)
ylabel(r'$dV/dI(\Omega)$',fontsize=labelsize,labelpad=12)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(a)')
title('Differential Resistance with Parallel Magnetic Field',fontsize=16)

plt.subplot(4,1,2)
x=data1[0:-1:501,4]
y=data1[251:-1:501,2]/1e-7
plt.plot(x,y,'r')
xlabel(r'$B(T)$',fontsize=labelsize)
ylabel(r'$dV/dI(\Omega)$',fontsize=labelsize,labelpad=20)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(b)')

plt.subplot(4,1,3)
x=data1[0:-1:501,4]
y=data1[251:-1:501,5]/1e-9
plt.plot(x,y,'g')
xlabel(r'$B(T)$',fontsize=labelsize)
ylabel(r'$dV/dI(\Omega)$',fontsize=labelsize,labelpad=5)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(c)')

plt.subplot(4,1,4)
x=data1[0:-1:501,4]
y=data1[251:-1:501,6]/1e-7
plt.plot(x,y,'c')
xlabel(r'$B(T)$',fontsize=labelsize)
ylabel(r'$dV/dI(\Omega)$',fontsize=labelsize,labelpad=16)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(d)')

plt.subplots_adjust(wspace =0, hspace =0.3)#调整子图间距

fig1.savefig('Resistance-Magnet.jpg')

fig2=plt.figure(2,figsize=(10,6), dpi=600)#绘制四个器件微分电阻随偏置电流的变化

plt.subplot(4,1,1)
x=data1[zero_mgnet:zero_mgnet+y_points,3]
y=data1[zero_mgnet:zero_mgnet+y_points,1]/1e-7
plt.plot(x,y,'b')
xlabel(r'$I_{Bias}(\mu A)$',fontsize=labelsize)
ylabel(r'$dV/dI(\Omega)$',fontsize=labelsize,labelpad=12)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(a)')
title('Differential Resistance with Bias Current',fontsize=12)

plt.subplot(4,1,2)
x=data1[zero_mgnet:zero_mgnet+y_points,3]
y=data1[zero_mgnet:zero_mgnet+y_points,2]/1e-7
plt.plot(x,y,'r')
xlabel(r'$I_{Bias}(\mu A)$',fontsize=labelsize)
ylabel(r'$dV/dI(\Omega)$',fontsize=labelsize,labelpad=18)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(b)')

plt.subplot(4,1,3)
x=data1[zero_mgnet:zero_mgnet+y_points,3]
y=data1[zero_mgnet:zero_mgnet+y_points,5]/1e-9
plt.plot(x,y,'g')
xlabel(r'$I_{Bias}(\mu A)$',fontsize=labelsize)
ylabel(r'$dV/dI(\Omega)$',fontsize=labelsize,labelpad=5)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(c)')

plt.subplot(4,1,4)
#！！！第四个锁相一开始测量微分电阻值随偏置电流变化时发现数据并不是很好，
# 变化的很缓慢（比锁相一的数据还要缓慢），所以就没有测，直接拔掉了，所以这里显示的其实是是噪音!!!
x=data1[zero_mgnet:zero_mgnet+y_points,3]
y=data1[zero_mgnet:zero_mgnet+y_points,6]/1e-7
plt.plot(x,y,'c')
xlabel(r'$I_{Bias}(\mu A)$',fontsize=labelsize)
ylabel(r'$dV/dI(\Omega)$',fontsize=labelsize,labelpad=7)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(d)')

plt.subplots_adjust(wspace =0, hspace =0.3)#调整子图间距

fig2.savefig('Resistance-BiasCurrent.jpg')

#二维图：
# data2=ReadData('180322.001.txt',r'C:\Users\孙晓培\AnacondaProjects\DataProcessing\180322.001.txt')

#判断扫场的方式以及寻找断点
# for i in np.linspace(0,data2.shape[0]-1,data2.shape[0]):
#     if data2[int(i)][-1]==data2[0][-1]:
#         continue
#     else:
#         y_points=int(i)
#         x_points=int(data2.shape[0]/i)
#         print('There are %d points in \'I_bias\' and %d points in Magnetic Filed \'B\'.' %(y_points,x_points))
#         break

# for i in np.linspace(0,data2.shape[0]-1,data2.shape[0]):
#     if data2[int(i)][4]==0:
#         print('The point %d means zero magnet' %int(i))
#         break

# plt.figure()

# x=np.linspace(data2[0][-1],data2[-1][-1],x_points)
# y=np.linspace(data2[0][3],data2[-1][3],y_points)
# X,Y=np.meshgrid(x,y)
# R=data2[:,1].reshape(x_points,y_points)

# pcolor(X,Y,R.T)
# colorbar()

plt.show()