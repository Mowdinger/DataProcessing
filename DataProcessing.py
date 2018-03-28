# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import pylab as plb


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

# data1=ReadData('180322.001.txt',r'C:\Users\孙晓培\AnacondaProjects\DataProcessing\180322.001.txt')
# #print(data[:,2])
# plt.figure(1)
# for i in np.linspace(2,data1.shape[1],data1.shape[1]-1):
#     plt.subplot(data1.shape[1],1,i-1)
#     x=data1[:,0]
#     y=data1[:,int(i)-1]
#     plt.plot(x,y)
#     #plt.title('Column %d' %i)
# plt.show()

data2=ReadData('180323.004.txt',r'C:\Users\孙晓培\AnacondaProjects\DataProcessing\180323.004.txt')

for i in np.linspace(0,data2.shape[0]-1,data2.shape[0]):
    if data2[int(i)][-1]==data2[0][-1]:
        continue
    else:
        y_points=int(i)
        x_points=int(data2.shape[0]/i)
        print('There are %d points in Magnetic Filed \'B\' and %d points in \'I_bias\'.' %(y_points,x_points))
        break

plt.figure(2)

x=np.linspace(data2[0][-1],data2[-1][-1],x_points)
y=np.linspace(data2[0][3],data2[-1][3],y_points)
X,Y=np.meshgrid(x,y)
R=data2[:,1].reshape(x_points,y_points)

plb.pcolor(X,Y,R.T)
plb.colorbar()
plt.show()