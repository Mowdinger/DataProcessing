# -*- coding: utf-8 -*-

#绘制搓衣板势能的变化
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import scipy.io as sio

#画加速度与斜率的关系
I=np.linspace(-100,100,10000)
a=abs(I/sqrt(I**2+1))
fig3=plt.figure(3,dpi=600)

labelsize=12

plt.plot(I,a,linewidth=1.5)

y_ticks=[0]
plt.yticks(y_ticks,fontsize=labelsize)
    
x_ticks=[0]
plt.xticks(x_ticks,fontsize=labelsize)

xlabel(r'$I$',fontsize=labelsize)
ylabel(r'$dV/dI \propto d^2\varphi/dt^2$',fontsize=labelsize)

fig3.savefig('Washboard a_I Relation.jpg') 

##画实验数据
#data = sio.loadmat('sxp_Ibias_Imag_16.mat')
#
#y=np.nan_to_num(data['vol13'])/data['Iac13']#unit: \Omega
#
#x=data['value_ibais']/1e5*1e6#偏置电流 unit: uA
#y=y[:,int(y.shape[1]/2)+1]
#
#y=y[0:y.size-30]
#
#fig2=plt.figure(2,dpi=600)
#
#plt.plot(x.T,y,'k',linewidth=1.5)
#
#labelsize=12
#
#y_ticks=np.arange(0,y.max()+y.max()/3,round(y.max()/3,2))
#
#plt.yticks(y_ticks,fontsize=labelsize)
#    
#xx1=np.arange(0,x.max()+1,int(x.max()/2))
#xx2=-xx1
#x_ticks=np.r_[xx1,xx2]
#plt.xticks(x_ticks,fontsize=labelsize)
#
#xlabel(r'$I_{Bias}(\mu A)$',fontsize=labelsize)
#ylabel(r'$dV/dI(\Omega)$',fontsize=labelsize)
#
#fig2.savefig('Washboard Data.jpg') 
##画搓衣板
#x=np.linspace(-10,10,100)
#I=[-0.9,0.9,-0.4,0.4]
#fig1=plt.figure(1,figsize=(10,8),dpi=600)
#labelsize=20
#label=['(a)','(b)','(c)','(d)']
#y0=-cos(x)
#for i in np.arange(4):
#    plt.subplot(2,2,i+1)
#    y=-I[i]*x-cos(x)
#    plt.plot(x,y,'b',linewidth=1.5,label=r'$I=%sI_{0}$'%I[i])
#    plt.plot(x,y0,'r',linewidth=1.5,linestyle='--',label=r'$I=0$')
#    
#    yy1=np.arange(0,y.max(),int(y.max()/2))
#    yy2=-yy1
#    y_ticks=np.r_[yy1,yy2]
#    plt.yticks(y_ticks,fontsize=labelsize)
#    
#    xx1=np.arange(0,x.max()+1,int(x.max()/2))
#    xx2=-xx1
#    x_ticks=np.r_[xx1,xx2]
#    plt.xticks(x_ticks,fontsize=labelsize)
#    
#    
#    xlabel(r'$\varphi$',fontsize=labelsize)
#    ylabel(r'$U(I_{0})$',fontsize=labelsize)
#    
#    text(x.min()*1.5,y.max(),label[i],fontsize=labelsize)
#    plt.legend(loc='upper center')
#
#plt.subplots_adjust(wspace =0.5, hspace =0.3)#调整子图间距
#
#fig1.savefig('WashBoard Potential.jpg') 
