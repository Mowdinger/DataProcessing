# -*- coding: utf-8 -*-

import numpy as np

import scipy.io as sio
import matplotlib.pyplot as plt
from pylab import *
from scipy.fftpack import fft,ifft
from scipy.optimize import curve_fit


data = sio.loadmat('sxp_Ibias_Imag_45.mat')  # 读取mat文件
y=data['Ic'].flatten()
x=data['value_imag'].T.flatten()*0.8 #unit: Gs

#定义一个函数将fft后的函数重新拆分合并
#def fftarrange(a0):
#    a1=a0[0:int(a0.size/2)+1]
#    a2=a0[int(a0.size/2)+1:a0.size]
#    a=np.append(a2,a1)
#    return a
def func(B,a,b):
    n=B.size #采样点
    jx=np.ones((n,n))*a
    Ic=np.zeros(n)
    k0=np.linspace(0,2,n)
    k=(np.divide( 1,np.exp( 3000*(k0-1.13) )+1 )*1.4+0.3)*a#给定一个超流的分布变化
    
    x=np.linspace(-1.5,1.5,n)
    fai=(B+9.92)/5
    g=(np.divide( 1,np.exp( 1000*(k0-0.8) )+1 )*3+3)#给定一个磁畴的分布变化
    #g=6
    for i in np.arange(0,n-1).astype(int):
        jx[i,0:int(n/g[i])]=np.ones(int(n/g[i]))*k[i]   
        jx[i,int(n-n/g[i])+1:n]=np.ones(n-int(n-n/g[i])-1)*k[i]
        Ic[i]=abs( jx[i,:]*mat( cos( fai[i]*x ) ).T  )    
 
        
    return Ic

def func2(B,a,b):
    n=B.size #采样点
    jx=np.ones((n,n))*a
    Ic=np.zeros(n)
    k0=np.linspace(0,2,n)
    k=(np.divide( 1,np.exp( 3000*(k0-1.13) )+1 )*1.4+0.3)*a#给定一个超流的分布变化
    
    x=np.linspace(-1.5,1.5,n)
    fai=(B+9.92)/5
    g=(np.divide( 1,np.exp( 1000*(k0-0.8) )+1 )*3+3)#给定一个磁畴的分布变化
    #g=6
    for i in np.arange(0,n-1).astype(int):
        jx[i,0:int(n/g[i])]=np.ones(int(n/g[i]))*k[i]   
        jx[i,int(n-n/g[i])+1:n]=np.ones(n-int(n-n/g[i])-1)*k[i]
        Ic[i]=abs( jx[i,:]*mat( cos( fai[i]*x ) ).T  )    
 
        
    return Ic,jx[0,:],jx[n-2,:],x

popt, pcov = curve_fit(func,x,y)
a=popt[0]#popt里面是拟合系数
b=popt[1]


Ic,jx1,jx2,x_spatial=func2(x,a,b)


fig1=plt.figure(1,dpi=300)
plt.plot( x, Ic, 'b',linewidth=1.5,linestyle='--',label='Fitting Curve')
plt.plot( x, y ,'r',linewidth=1.5,label='Experimental Data')

labelsize=15

x_ticks = np.linspace(x.min(),x.max(),5)
plt.xticks(x_ticks,fontsize=13)
y_ticks = np.linspace(0,50,6)
plt.yticks(y_ticks,fontsize=13)

xlabel(r'$B_{\perp}(Gs)$',fontsize=labelsize)
ylabel(r'$I_{Critical}(\mu A)$',fontsize=labelsize,labelpad=12)

plt.legend(loc='upper right')

fig1.savefig('Fitting.png', transparent = True, bbox_inches = 'tight', pad_inches = 0.25) 



fig2=plt.figure(2,dpi=600)
plt.plot(x_spatial,jx1*1000,'k',linewidth=1.5)

labelsize=25

x_ticks = np.linspace(x_spatial.min(),x_spatial.max(),3)
plt.xticks(x_ticks,fontsize=20)
y_ticks = np.linspace(0,0.15,3)*1000
plt.yticks(y_ticks,fontsize=20)

xlabel(r'$x(\mu m)$',fontsize=labelsize)
ylabel(r'$j_{x}(mA/m)$',fontsize=labelsize)

fig2.savefig('jx1.png', transparent = True, bbox_inches = 'tight', pad_inches = 0.25) 


fig3=plt.figure(3,dpi=600)
plt.plot(x_spatial,jx2*1000,'k',linewidth=1.5)

labelsize=25

x_ticks = np.linspace(x_spatial.min(),x_spatial.max(),3)
plt.xticks(x_ticks,fontsize=20)
y_ticks = np.linspace(0,0.15,3)*1000
plt.yticks(y_ticks,fontsize=20)

xlabel(r'$x(\mu m)$',fontsize=labelsize)
ylabel(r'$j_{x}(mA/m)$',fontsize=labelsize)

fig3.savefig('jx2.png', transparent = True, bbox_inches = 'tight', pad_inches = 0.25) 

