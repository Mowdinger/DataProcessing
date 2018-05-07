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



#一维图：
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

#绘制四个器件微分电阻随磁场的变化：
fig1=plt.figure(1,figsize=(10,6), dpi=600)
labelsize=12

plt.subplot(4,1,1)
x=data1[0:-1:501,4]#提取磁场数据 （unit: T）
y=data1[251:-1:501,1]/1e-7#提取锁相放大器测量的数据并将其转化为微分电阻值（unit: \Omega）
y=np.divide(1,y)#将微分电阻换位微分电导(unit: \Omega^{-1})
# y=6.62607004e-34*y/(2* 1.6021766**2 * 1e-38)#转化单位为量子电导
plt.plot(x,y,'b')
xlabel(r'$B(T)$',fontsize=labelsize)
ylabel(r'$dI/dV(S)$',fontsize=labelsize,labelpad=22)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(a)')
# title('Differential Conductance with Parallel Magnetic Field',fontsize=16)

plt.subplot(4,1,2)
x=data1[0:-1:501,4]
y=data1[251:-1:501,2]/1e-7
y=np.divide(1,y)#将微分电阻换位微分电导
# y=6.62607004e-34*y/(2* 1.6021766**2 * 1e-38)#转化单位为量子电导
plt.plot(x,y,'r')
xlabel(r'$B(T)$',fontsize=labelsize)
ylabel(r'$dI/dV(S)$',fontsize=labelsize,labelpad=22)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(b)')

plt.subplot(4,1,3)
x=data1[0:-1:501,4]
y=data1[251:-1:501,5]/1e-9
y=np.divide(1,y)#将微分电阻换位微分电导
# y=6.62607004e-34*y/(2* 1.6021766**2 * 1e-38)#转化单位为量子电导
plt.plot(x,y,'g')
xlabel(r'$B(T)$',fontsize=labelsize)
ylabel(r'$dI/dV(S)$',fontsize=labelsize,labelpad=2)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(c)')

plt.subplot(4,1,4)
x=data1[0:-1:501,4]
y=data1[251:-1:501,6]/1e-7
y=np.divide(1,y)#将微分电阻换位微分电导
# y=6.62607004e-34*y/(2* 1.6021766**2 * 1e-38)#转化单位为量子电导
plt.plot(x,y,'c')
xlabel(r'$B(T)$',fontsize=labelsize)
ylabel(r'$dI/dV(S)$',fontsize=labelsize,labelpad=20)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(d)')

plt.subplots_adjust(wspace =0, hspace =0.3)#调整子图间距

fig1.savefig('Differential Conductance-Parallel Magnetic Field.jpg')

#绘制四个器件微分电阻随偏置电流的变化
fig2=plt.figure(2,figsize=(10,6), dpi=600)

plt.subplot(4,1,1)
x=data1[zero_mgnet:zero_mgnet+y_points,3]/1e6 #提取直流电压数据并将其转化为偏置电流值(unit: A)
y=data1[zero_mgnet:zero_mgnet+y_points,1]/1e-7 #提取锁相放大器测量的数据并将其转化为微分电阻值（unit: \Omega）
x,y=dr2dc(x,y)#将微分电阻与偏流换位微分电导与偏压(unit: \Omega^{-1} & V)
x=x*1e6#将偏置电压单位转化为 \mu V
plt.plot(x,y,'b')
xlabel(r'$V_{Bias}(\mu V)$',fontsize=labelsize)
ylabel(r'$dI/dV(S)$',fontsize=labelsize,labelpad=14)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(a)')
# title('Differential Conductance with Bias Voltage',fontsize=12)

plt.subplot(4,1,2)
x=data1[zero_mgnet:zero_mgnet+y_points,3]/1e6
y=data1[zero_mgnet:zero_mgnet+y_points,2]/1e-7
x,y=dr2dc(x,y)
x=x*1e6
plt.plot(x,y,'r')
xlabel(r'$V_{Bias}(\mu V)$',fontsize=labelsize)
ylabel(r'$dI/dV(S)$',fontsize=labelsize,labelpad=14)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(b)')

plt.subplot(4,1,3)
x=data1[zero_mgnet:zero_mgnet+y_points,3]/1e6
y=data1[zero_mgnet:zero_mgnet+y_points,5]/1e-9
x,y=dr2dc(x,y)
x=x*1e6
plt.plot(x,y,'g')
xlabel(r'$V_{Bias}(\mu V)$',fontsize=labelsize)
ylabel(r'$dI/dV(S)$',fontsize=labelsize,labelpad=2)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(c)')

plt.subplot(4,1,4)
#！！！第四个锁相一开始测量微分电阻值随偏置电流变化时发现数据并不是很好，
# 变化的很缓慢（比锁相一的数据还要缓慢），所以就没有测，直接拔掉了，所以这里显示的其实是是噪音!!!
x=data1[zero_mgnet:zero_mgnet+y_points,3]/1e6
y=data1[zero_mgnet:zero_mgnet+y_points,6]/1e-7
x,y=dr2dc(x,y)
x=x*1e6
plt.plot(x,y,'c')
xlabel(r'$V_{Bias}(\mu V)$',fontsize=labelsize)
ylabel(r'$dI/dV(S)$',fontsize=labelsize,labelpad=8)
text(x.min()*1.08,y.max()-(y.max()-y.min())*0.15,'(d)')

plt.subplots_adjust(wspace =0, hspace =0.3)#调整子图间距

fig2.savefig('Differential Conductance-Bias Voltage.jpg')

# plt.show()