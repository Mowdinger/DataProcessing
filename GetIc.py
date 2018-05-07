# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *

data = sio.loadmat('sxp_Ibias_Imag_45.mat')  # 读取mat文件

Z0=data['dr13'] #unit: \Omega

x=data['value_imag'].T*0.8 #unit: Gs
y=data['value_ibais']/1e5*1e6  #偏置电流 unit: uA
y=np.linspace(y.min(),y.max(),Z0.shape[0])

Z0=data['dr13'] #unit: \Omega
Z1=Z0[0:int(y.size/2),:]##numpy数组切片太蛋疼了！！！最后一位竟然要加个1才行
Z2=Z0[int(y.size/2):y.size+1,:]

#X,Y=meshgrid(x,y)
#pcolor(X,Y,Z0,cmap='jet')
#colorbar
plt.figure(1,figsize=(10,6))
X,Y=meshgrid(x[0:-1:2],y[0:-1:20])
pcolor(X,Y,Z0[0:-1:20,0:-1:2],cmap='jet')
colorbar

dy0=diff(Z0.T).T
dy0=np.r_[dy0,np.zeros((1,x.size))]

dy1=dy0[0:int(y.size/2),:]
dy2=dy0[int(y.size/2):y.size+1,:]

IcP1=np.zeros(x.size)
IcP2=np.zeros(x.size)

for i in np.linspace(0,x.size-1,x.size):
    i=int(i)
    j1,k = max( enumerate(dy1[:,i]), key=(lambda x: x[1]) )
    dy1[int(j1)][i]=0
    j2,k = min( enumerate(dy1[:,i]), key=(lambda x: x[1]) )
    #print(j,k)
    IcP1[i]=j2

Ic1=np.zeros(x.size)
for i in np.linspace(0,x.size-1,x.size):
    i=int(i)
    Ic1[i]=y.T[ int(IcP1[i]) ]

plt.plot(x,Ic1,'w',linewidth=1.5)
#    
for i in np.linspace(0,x.size-1,x.size):
    i=int(i)
    j1,k = min( enumerate(dy2[:,i]), key=(lambda x: x[1]) )
    dy2[int(j1)][i]=0
    j2,k = max( enumerate(dy2[:,i]), key=(lambda x: x[1]) )
    #print(j,k)
    IcP2[i]=j2
    
Ic2=np.zeros(x.size)
for i in np.linspace(0,x.size-1,x.size):
    i=int(i)
    Ic2[i]=y.T[ int(IcP2[i]) + int(y.size/2) ]
plt.plot(x,Ic2,'w',linewidth=1.5)


#分析某一列的数据及其导数
def analysis(x_p):#输入该列的横坐标
    c=int((x_p-x.min())*x.size/(x.max()-x.min()))
    plt.plot(dy0[:,c]*100+c*2*0.8-120,y.T,'r',linewidth=1.5)#导数放大100倍画出来
    plt.plot(np.zeros(y.size)+c*2*0.8-120,y.T,'r--',linewidth=1.5)#参考线
    plt.plot(Z0[:,c]*10+c*2*0.8-120,y.T,'c',linewidth=2.5)#数据放大十倍画出来
    return None

#Ic2[51:90]=Ic1[51:90]
#plt.plot(x,Ic2,'w',linewidth=3)


#扩充数据（存在问题）
#for i in np.linspace(0,x.size-1,x.size):
#    i=int(i)
#    for j in np.linspace(0,int(StopPoint1[i]),int(StopPoint1[i])+1):
#        j=int(j)
#        Z1[j,i]=(Z1[int(StopPoint1[i])+1][i]+Z1[int(StopPoint1[i])+1][i]*(random.random()-0.5)/3)
#    
#for i in np.linspace(0,x.size-1,x.size):
#    i=int(i)
#    for j in np.linspace(int(StopPoint2[i])+1,int(y.size/2),int(y.size/2)-int(StopPoint2[i])):
#        j=int(j)
#        Z2[int(StopPoint2[i])+1:int(y.size/2)+1,i]=(Z2[int(StopPoint2[i])+1][i]+Z2[int(StopPoint2[i])+1][i]*(random.random()-0.5)/3)

#for i in np.linspace(0,x.size-1,x.size):
#    i=int(i)
#    for j in np.linspace(0,int(StopPoint2[i])+1,int(StopPoint2[i])+2):
#        j=int(j)
#        Z2[0:int(StopPoint2[i])+2,i]=(Z2[int(StopPoint2[i])+1][i]+Z2[int(StopPoint2[i])+1][i]*(random.random()-0.5)/3)

    
    
    
    
#for i in np.linspace(0,x.size-1,x.size):
#    i=int(i)
#    f=interpolate.UnivariateSpline(np.linspace(int(StopPoint1[i])+1,int(y.size/2),
#        int(y.size/2)-int(StopPoint1[i])),Z1[int(StopPoint1[i]):int(y.size/2),i])
#    Z1[0:int(StopPoint1[i])+1,i]=f(np.linspace(0,int(StopPoint1[i]),int(StopPoint1[i])+1))
#    
#for i in np.linspace(0,x.size-1,x.size):
#    i=int(i)
#    f=interpolate.UnivariateSpline(np.linspace(0,int(StopPoint2[i]),int(StopPoint2[i])+1),
#                                   Z2[0:int(StopPoint2[i])+1,i])
#    Z2[int(StopPoint2[i]):int(y.size/2)+1,i]=f(np.linspace(int(StopPoint2[i]),int(y.size/2),
#      int(y.size/2)-int(StopPoint2[i])+1))
    
#Z=np.r_[Z1,Z2]
#X,Y=meshgrid(x,y)
#pcolor(X,Y,Z,cmap='jet')
#colorbar

#sio.savemat('sxp_Ibias_Imag_21.mat',{'dr13':Z})