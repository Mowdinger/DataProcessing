# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.fftpack import fft,ifft

#定义一个函数将fft后的函数重新拆分合并
#def fftarrange(a0):
#    a1=a0[0:int(a0.size/2)+1]
#    a2=a0[int(a0.size/2)+1:a0.size]
#    a=np.append(a2,a1)
#    return a

n=501 #采样点

jx=np.ones((n,n))*1
Ic=np.zeros(n)
k0=np.linspace(0,2,n)
k=np.divide( 1,exp( 3000*(k0-1.21) )+1 )*1.4+0.3
#k[0:int(k.size/2)+50]=15
#k[int(k.size/2)+50:k.size]=5

x=np.linspace(-1.5,1.5,n)
fai=np.linspace(-18,18,n)

g=3

for i in np.linspace(0,n-1,n).astype(int):
#    jx[i,0:int(2*n/5)]=np.ones(int(2*n/5))*k[i]   
#    jx[i,int(3*n/5):jx.size]=np.ones(int(2*n/5))*k[i]
#    jx[i,0:int(n/3)]=np.ones(int(n/3))*k[i]   
#    jx[i,int(2*n/3)+1:jx.size]=np.ones(int(n/3))*k[i]
    jx[i,0:int(n/g)]=np.ones(int(n/g))*k[i]   
    jx[i,int(n-n/g)+1:n]=np.ones(n-int(n-n/g)-1)*k[i]
    Ic[i]=abs( jx[i,:]*mat( cos( fai[i]*x ) ).T  )
#这个傅里叶变换写成自己写的积分，因为快速傅里叶变换会把原函数先周期化，
#但是实际上我们就应该在这一个周期内积分
#周期化后会导致电流常数的傅里叶变换是一个\delta函数，而不周期化的话，
#这个电流常数就相当于是一个方波,方波傅里叶变换是夫琅禾费包络
#所以用fft的函数得到的波峰值很高，不符合实际情况，应该改用自己写积分
#自己写积分的话可以利用矩阵的内积，这样可能会方便一些。
    

plt.figure(1)
plot(np.arange(Ic.size),Ic)

#plt.figure(2)
#X,Y=meshgrid(k,k)
#pcolor(X,Y,jx)
#colorbar()
