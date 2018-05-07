# -*- coding: utf-8 -*-
#This script was created to fill the missing data in 'sxp_Ibias_Imag_45.mat' 
#with 'sxp_Ibias_Imag_46.mat'. 

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *

data = sio.loadmat('sxp_Ibias_Imag_45.mat')  # 读取mat文件
# print(data.keys()) 
#['__header__', '__version__', '__globals__', 'value_imag', 'Iac13', 
#'H_mag', 'value_ibais', 'vol13', 'vvalue_ibias', 'vvol13', 
#'plovalue_ibiass']

Z0=np.nan_to_num(data['vvol13'])/data['Iac13']

x=data['value_imag'].T*0.8 #unit: Gs
y=data['value_ibais']/1e5*1e6 #偏置电流 unit:uA
y=np.linspace(y.min(),y.max(),Z0.shape[0])

#plt.figure(1,figsize=(10,6))
#X,Y=meshgrid(x[0:-1:2],y[0:-1:20])
#pcolor(X,Y,Z0[0:-1:20,0:-1:2],cmap='jet')
#colorbar

data1 = sio.loadmat('sxp_Ibias_Imag_46.mat')
Z1=np.nan_to_num(data1['vol13'])/data['Iac13']

x1=data1['value_imag'].T*0.8 #unit: Gs
y1=data1['value_ibais']/1e5*1e6 #偏置电流 unit:uA
x2=x1
y2=np.linspace(y1.min(),y1.max(),Z0.shape[0])
Z2=np.zeros((y2.size,x2.size))
y1=y1.flatten()
for i in np.linspace(0,Z1.shape[1]-1,Z1.shape[1]):
    i=int(i)
    f=interpolate.interp1d(y1,Z1[:,i].T,kind='linear')
    Z2[:,i]=f(y2)

plt.figure(2,figsize=(2,6))
X,Y=meshgrid(x2[0:-1:2],y2[0:-1:20])
pcolor(X,Y,Z2[0:-1:20,0:-1:2],cmap='jet')
colorbar

x11=-17
x12=-15
x21=-6.5
x22=-3.5
y11=-20
y22=-28

c11_0=int((x11-x.min())*x.size/(x.max()-x.min()))
c12_0=int((x12-x.min())*x.size/(x.max()-x.min()))
c21_0=int((x21-x.min())*x.size/(x.max()-x.min()))
c22_0=int((x22-x.min())*x.size/(x.max()-x.min()))

c11_1=int(((x11+1)-x1.min())*x1.size/(x1.max()-x1.min()))
c12_1=int(((x12+1)-x1.min())*x1.size/(x1.max()-x1.min()))
c21_1=int(((x21-1.5)-x1.min())*x1.size/(x1.max()-x1.min()))
c22_1=int(((x22-1.5)-x1.min())*x1.size/(x1.max()-x1.min()))

r1_0=int((y11-y.min())*y.size/(y.max()-y.min()))
r2_0=int((y22-y.min())*y.size/(y.max()-y.min()))

r1_1=int((y11-y2.min())*y2.size/(y2.max()-y2.min()))
r2_1=int((y22-y2.min())*y2.size/(y2.max()-y2.min()))

middle=int(y.size/2)

Z0[0:middle,c11_0-1:c12_0]=Z2[0:middle,c11_1:c12_1:2]
Z0[0:middle,c21_0-1:c22_0]=Z2[0:middle,c21_1:c22_1:2]

#plt.figure(3,figsize=(10,6))
#X,Y=meshgrid(x[0:-1:2],y[0:-1:20])
#pcolor(X,Y,Z0[0:-1:20,0:-1:2],cmap='jet')
#colorbar
#plt.figure(3,figsize=(10,6))
#X,Y=meshgrid(x,y)
#pcolor(X,Y,Z0,cmap='jet')
#colorbar
data['dr13']=Z0
#sio.savemat('sxp_Ibias_Imag_45.mat',data)