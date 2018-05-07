import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from scipy.fftpack import fft,ifft

#sio.savemat('sxp_Ibias_Imag_21.mat',{'dr13':Z})

data = sio.loadmat('sxp_Ibias_Imag_21.mat')  # 读取mat文件
# print(data.keys())   
# ['__header__', '__version__', '__globals__', 'Iac10', 'Iac11', 'Iac13', 'value_imag', 'value_ibais', 'vol10', 'vol11', 'vol13']
xx=data['value_imag'].T*0.8 #unit: Gs
y=data['value_ibais']#更改单位？？？
yy=y[:,0:500]
X,Y = np.meshgrid(xx,y)

Z0=np.nan_to_num(data['vol13'])/data['Iac13'] #unit: \Omega
Z=np.zeros(Z0.shape)
#for i in Z0[]:
    
    
Z=Z0[0:int(1001/2),:]

dy=diff(Z0.T).T
dy=np.r_[dy,np.zeros((1,151))]

#plt.figure(2)
#pcolor(X,Y,dy,cmap='jet')
#colorbar()

#plt.figure(3,dpi=1200)
#for i in np.linspace(0,29,30):
#    i=int(i)
#    plt.subplot(6,5,i+1)
#    plt.plot(y.T,dy[:,i*5])
#    plt.savefig('dy.jpg')
##以数组中的最大值为临界电流值
##Ic=np.zeros(xx.size)
##for i in np.linspace(0,xx.size-1,xx.size):
##    i=int(i)
##    j,k = max( enumerate(Z[:,i]), key=(lambda x: x[1]) )
##    #print(j,k)
##    Ic[i]=yy[0,j]
#
plt.figure(4)
pcolor(X,Y,Z0,cmap='jet')
colorbar()
plt.plot(dy[:,31]*100+31*2*0.8-120,y.T,'w',linewidth=1.5)
plt.plot(Z0[:,31]*100+31*2*0.8-120,y.T,'r',linewidth=1.5)
#plot(xx,Ic,'w',linewidth=2.5)
#plot(xx,Z0[int(y.size/2)+1,:],'w',linewidth=2.5)
#
##jx0=fft(Ic)
##jx=jx0
##def fftarrange(a0):
##    a1=a0[0:int(a0.size/2+0.5)]
##    a2=a0[int(a0.size/2+0.5):a0.size-1]
##    a=np.append(a2,a1)
##    return a
##jx1=fftarrange(jx)
##fig5=plt.figure(5)
##plot(np.linspace(1,jx.size,jx.size),abs(jx),'b')
##plot(np.linspace(1,jx1.size,jx1.size),abs(jx1),'r')
##
#plt.figure(6)
#Icc=abs(ifft(jx1))
#plot(np.linspace(1,Icc.size,Icc.size),Icc,'r')
#plot(np.linspace(1,jx.size,jx.size),abs(ifft(jx)),'b' )


# plt.figure(7)
# jxjxjx=np.append(jx,jx)
# plot(np.linspace(1,jxjxjx.size,jxjxjx.size),abs(jxjxjx),'b')

# plt.figure(8)
# plot(np.linspace(1,jxjxjx.size,jxjxjx.size),abs(ifft(jxjxjx)),'b' )

#fig5.savefig('new.jpg')

plt.show()

