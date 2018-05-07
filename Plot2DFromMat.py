import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from scipy.fftpack import fft,ifft

data = sio.loadmat('sxp_Ibias_Imag_16.mat')  # 读取mat文件
# print(data.keys())   
# ['__header__', '__version__', '__globals__', 'Iac10', 'Iac11', 'Iac13', 'value_imag', 'value_ibais', 'vol10', 'vol11', 'vol13']
xx=data['value_imag']
y=data['value_ibais']
yy=y[:,0:500]
X,Y = np.meshgrid(xx,yy)

Z=np.nan_to_num(data['vol13'])
Z=Z[0:int(1001/2),:]

Ic=np.zeros(xx.size)
for i in np.linspace(0,xx.size-1,xx.size):
    i=int(i)
    j,k = max( enumerate(Z[:,i]), key=(lambda x: x[1]) )
    #print(j,k)
    Ic[i]=yy[0,j]

plt.figure(4)
pcolor(X,Y,Z,cmap='jet')
colorbar()
plot(xx.T,Ic,'w')

jx0=fft(Ic)
jx=jx0
def fftarrange(a0):
    a1=a0[0:int(a0.size/2+0.5)]
    a2=a0[int(a0.size/2+0.5):a0.size]
    a=np.append(a2,a1)
    return a
jx1=fftarrange(jx)
fig5=plt.figure(5,dpi=600)
plot(np.linspace(1,jx.size,jx.size),abs(jx),'b')
plot(np.linspace(1,jx1.size,jx1.size),abs(jx1),'r')

plt.figure(6)
Icc=abs(ifft(jx1))
plot(np.linspace(1,Icc.size,Icc.size),Icc,'r')
plot(np.linspace(1,jx.size,jx.size),abs(ifft(jx)),'b' )

fig5.savefig('old.jpg')

# plt.figure(7)
# jxjxjx=np.append(jx,jx)
# plot(np.linspace(1,jxjxjx.size,jxjxjx.size),abs(jxjxjx),'b')

# plt.figure(8)
# plot(np.linspace(1,jxjxjx.size,jxjxjx.size),abs(ifft(jxjxjx)),'b' )

plt.show()

