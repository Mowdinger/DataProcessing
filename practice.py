import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from scipy.fftpack import fft,ifft

def fftarrange(a0):
    a1=a0[0:int(a0.size/2+0.5)]
    a2=a0[int(a0.size/2+0.5):a0.size-1]
    a=np.append(a2,a1)
    return a

plt.figure(6)
xxx=np.linspace(-16*np.pi,16*np.pi,400)
yyy0=np.divide(np.sin(10*xxx),xxx)
yyy=fft(yyy0)
# plot(xxx,yyy0)
plot(xxx,abs(yyy),'r')
plt.figure(7)
xyz=fftarrange(abs(fft(abs(yyy))))
plot(np.linspace(0,xyz.size-1,xyz.size),xyz,'b')

# plt.figure(7)
# size=10000
# x1 = np.arange(2, 4, 2.0/size)
# y1 = np.where(x1<3, 1, 0)

# x2 = np.arange(0, 2, 2.0/size)
# y2 = np.where(x2>1, 1, 0)

# x=np.append(x2,x1)
# y=np.append(y2,y1)
# yy=ifft(y)
# plot(x,y,'b')
# plot(x,abs(yy),'r')

plt.show()