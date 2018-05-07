# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *

#将图像中的坏点通过插值修正
def complete(x,Ic,s1,e1):#输入图像的横坐标，数据，坏点起点与终点的横坐标
    s1=int((s1-x.min())*x.size/(x.max()-x.min()))
    e1=int((e1-x.min())*x.size/(x.max()-x.min()))
    Ic_new=np.delete(Ic,np.linspace(s1,e1,e1-s1+1).astype(int) )
    x_new=np.delete(x,np.linspace(s1,e1,e1-s1+1).astype(int) )
    f=interpolate.interp1d(x_new, Ic_new,kind='cubic')
    Ic[s1:e1]=f(x[s1:e1])
    return Ic

#分析某一列的数据及其导数
def analysis(x_p):#输入该列的横坐标
    c=int((x_p-x.min())*x.size/(x.max()-x.min()))
    plt.plot(dy0[:,c]*100+x_p,y.T,'r',linewidth=1.5)#导数放大100倍画出来
    plt.plot(np.zeros(y.size)+x_p,y.T,'r--',linewidth=1.5)#参考线
    plt.plot(Z0[:,c]*10+x_p,y.T,'c',linewidth=2.5)#数据放大十倍画出来
    return None


data1 = sio.loadmat('sxp_Ibias_Imag_52.mat')  # 读取mat文件
data2 = sio.loadmat('sxp_Ibias_Imag_53.mat')  # 读取mat文件
# print(data.keys()) #读取.mat文件中的所有变量
#['__header__', '__version__', '__globals__', 'value_imag', 'Iac13', 
#'H_mag', 'value_ibais', 'vol13', 'vvalue_ibias', 'vvol13', 
#'plovalue_ibiass']

##读取数据及横纵坐标（适用于初版改良程序）

Z1=np.nan_to_num(data1['vvol13'])/data1['Iac13']#unit: \Omega
Z2=np.nan_to_num(data2['vvol13'])/data2['Iac13']#unit: \Omega
Z2=Z2[:,10:Z2.shape[1]]

y1=data1['value_ibais']/1e5*1e6#偏置电流 unit: uA
y2=data1['value_ibais']/1e5*1e6#偏置电流 unit: uA
y1=np.linspace(y1.min(),y1.max(),Z1.shape[0])
y2=np.linspace(y2.min(),y2.max(),Z2.shape[0])

x1=data1['value_imag'].T*0.8 #unit: Gs
x2=data2['value_imag'].T*0.8 #unit: Gs
x2=x2[10:x2.size,:]


Z2_new=np.zeros((Z1.shape[0],Z2.shape[1]))
for i in np.linspace(0,Z2.shape[1]-1,Z2.shape[1]).astype(int):
    f=interpolate.interp1d(y2, Z2[:,i],kind='linear')
    Z2_new[:,i]=f(y1)

Z0=np.c_[Z1,Z2_new]
Z0=Z0[int(Z0.shape[0]/3):int(2*Z0.shape[0]/3),:]

x=np.linspace(x1.min(),x2.max(),Z0.shape[1])
y=y1[int(y1.size/3):int(2*y1.size/3)]

##读取数据及横纵坐标（适用于最终改良版程序）
#Z0=np.nan_to_num(data['vvol13'])/data['Iac13']#unit: \Omega
#
#x=data['value_imag'].T*0.8 #unit: Gs
#y=data['value_ibais']/1e5*1e6  #偏置电流 unit: uA
#y=np.linspace(y.min(),y.max(),Z0.shape[0])

#分别对上下两个矩阵求临界超流值，并给出平均后的结果
Z1=Z0[0:int(y.size/2),:]##numpy数组切片太蛋疼了！！！最后一位竟然要加个1才行
Z2=Z0[int(y.size/2):y.size+1,:]

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
    if j2==int(y.size/2)-1:
        dy1[int(j2)][i]=0
        j2,k = min( enumerate(dy1[:,i]), key=(lambda x: x[1]) )
    IcP1[i]=j2

Ic1=np.zeros(x.size)
for i in np.linspace(0,x.size-1,x.size):
    i=int(i)
    Ic1[i]=y.T[ int(IcP1[i]) ]

#    
for i in np.linspace(0,x.size-1,x.size):
    i=int(i)
    j1,k = min( enumerate(dy2[:,i]), key=(lambda x: x[1]) )
    dy2[int(j1)][i]=0
    j2,k = max( enumerate(dy2[:,i]), key=(lambda x: x[1]) )
    IcP2[i]=j2
    
Ic2=np.zeros(x.size)
for i in np.linspace(0,x.size-1,x.size):
    i=int(i)
    Ic2[i]=y.T[ int(IcP2[i]) + int(y.size/2) ]

##对坏点进行修正
Ic1=complete(x,Ic1,100.288,101.647)
Ic2=complete(x,Ic2,100.288,101.647)
#Ic2=complete(x,Ic2,-10.23,-3.08)
#Ic2=complete(x,Ic2,9.904,11.57)
#Ic2=complete(x,Ic2,28.0,29.65)

##拟合超流数据
#z1 = np.polyfit(x.flatten(), Ic1, 100)#用6次多项式拟合
#p1 = np.poly1d(z1)
#Ic1=p1(x).T
#
#z2 = np.polyfit(x.flatten(), Ic2, 100)#用6次多项式拟合
#p2 = np.poly1d(z2)
#Ic2=p2(x).T
    
#绘制临界超流曲线
Ic1=mat(Ic1).T
plt.plot(x,Ic1,'m',linestyle='--',linewidth=1.5)
#plt.plot(x,Ic1,'m',linewidth=0.5)

Ic2=mat(Ic2).T
plt.plot(x,Ic2,'m',linestyle='--',linewidth=1.5)
#plt.plot(x,Ic2,'m',linewidth=0.5)

Ic=(Ic2-Ic1)/2
plt.plot(x,Ic,'w',linewidth=1.5)



#绘制全部数据点，数据较多时会很慢
fig1=plt.figure(1)
#Z0=np.r_[ np.zeros((int(y.size/4),x.size)),Z0,np.zeros((int(y.size/4),x.size)) ]#由于拟合后画出的临界电流可能超出了原图的范围，所以需要给原图补一块儿
#y=np.linspace(y.min()+y.min()/2,y.max()+y.max()/2,Z0.shape[0])
X,Y=meshgrid(x,y)
diagram_2D=pcolor(X,Y,Z0,cmap='jet')

##快速简易绘图，用于分析数据
#fig1=plt.figure(1,figsize=(10,8))
##Z0=np.r_[ Z0,np.zeros((int(y.size/14),x.size)) ]
##y=np.linspace(y.min(),y.max()+y.max()/7,Z0.shape[0])
#X,Y=meshgrid(x[0:-1:2],y[0:-1:20])
#diagram_2D=pcolor(X,Y,Z0[0:-1:20,0:-1:2],cmap='jet')

##修饰图像：

#标注平行场大小
plt.text(x.max()-(x.max()-x.min())/5, y.max()*7/8, '$B_{\parallel}=200 Gs$',color='w',family='Times New Roman')

#设置横纵轴
labelsize=12
xlabel(r'$B_{\perp}(Gs)$',fontsize=labelsize)
ylabel(r'$I_{Bias}(\mu A)$',fontsize=labelsize,labelpad=12)

#设置colorbar位置
position=fig1.add_axes([0.3, 0.9, 0.6, 0.03])#位置[左,下,宽，高]
cb=plt.colorbar(diagram_2D,orientation='horizontal',cax=position)
#设置colorbar的ticklabels
min=round(  Z0.min()*2/3+Z0.max()/3 ,2)
max=round(Z0.min()/3+Z0.max()*2/3,2)
mid=round((min+max)/2,2)
cb.set_ticks([min,mid,max])
cb.set_ticklabels([min,mid,max])
cb.ax.xaxis.set_ticks_position('top')
#设置colorbar的标注
cb.ax.text(-0.1,0.5,'$dV/dI(\Omega)$',fontsize=labelsize,horizontalalignment='center',verticalalignment='center')


#保存图片
fig1.savefig('100gauss.jpg',dpi=600)

#将临界超流值保存到数据文件中
data1['Ic1']=Ic1
data1['Ic2']=Ic2
data1['Ic']=Ic
data1['dr']=Z0
data1['Bparallel']=x
data1['Ibias']=y

sio.savemat('sxp_Ibias_Imag_52.mat',data1)


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