# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *

data0 = sio.loadmat('sxp_Ibias_Imag_45.mat')
correction=data0['correction'].T
x0=data0['value_imag'].T*0.8 

data=sio.loadmat('sxp_Ibias_Imag_48.mat')
Ic0=data['Ic']
x=data['value_imag'].T*0.8 

min=int((x.min()-x0.min())*x0.size/(x0.max()-x0.min()))
max=int((x.max()-x0.min())*x0.size/(x0.max()-x0.min()))
correction=correction[  min:max+1 ,:]

Ic=np.multiply(Ic0,correction)
plot(x,Ic)
plot(x,Ic0)

#画出来的是什么鬼，简直太不靠谱了