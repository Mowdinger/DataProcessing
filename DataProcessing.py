# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import pylab


def ReadData(name,path):
    
    file = open(path)
    All=file.readlines()
    for i in All:
        if i[16:20] == "Time":
            position=All.index(i)+1
            print('\n Your data begins from Row %d, The meaning of each column is: \n %s' %(position,i,))
    data=np.loadtxt(name,skiprows=position)
    print('There are %d rows and %d columns.' %(data.shape[0],data.shape[1]))
    file.close()
    return data

data=ReadData('180322.003.txt',r'C:\Users\孙晓培\AnacondaProjects\DataProcessing\180322.003.txt')
#print(data[:,2])
plt.figure(1)
for i in np.linspace(1,data.shape[1],data.shape[1]):
    plt.subplot(data.shape[1],1,i)
    x=data[:,3]
    y=data[:,int(i)-1]
    plt.plot(x,y)
    #plt.title('Column %d' %i)
plt.show()
