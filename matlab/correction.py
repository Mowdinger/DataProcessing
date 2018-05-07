# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.optimize import curve_fit


data = sio.loadmat('sxp_Ibias_Imag_45.mat')  # 读取mat文件
Ic=data['Ic'].flatten()
x=data['value_imag'].T.flatten()*0.8 #unit: Gs

x1=-16.08
x2=-2.83871

c1=int((x1-x.min())*x.size/(x.max()-x.min()))
c2=int((x2-x.min())*x.size/(x.max()-x.min()))

xx=x[c1:c2+1]
yy=Ic[c1:c2+1]

def func(x,a,b,c):
    return a*np.sin(b*(x+c))/(b*(x+c))

popt, pcov = curve_fit(func,xx,yy)

a=popt[0]#popt里面是拟合系数
b=popt[1]
c=popt[2]

Icnew=abs(func(x,a,b,c))
#correction=Icnew-Ic
correction=np.divide(Icnew,Ic)
#不管是相减还是相除都很不靠谱

data['correction']=correction

figure(1)
plt.plot(x,Ic,'b')
plt.plot(x,Icnew,'r')
plt.plot(x,correction,'k')

sio.savemat('sxp_Ibias_Imag_45.mat',data)
