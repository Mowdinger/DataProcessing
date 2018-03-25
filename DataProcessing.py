# -*- coding: utf-8 -*-
import os
import numpy
import matplotlib
import pylab


def ReadData(name,path):
    
    file = open(path)
    All=file.readlines()
    for i in All:
        if i[16:20] == "Time":
            position=All.index(i)+1
            print('Your data begins from Row %d, The meanings of each column are: \n %s' %(position,i))
    data=numpy.loadtxt(name,skiprows=position)
    #print(data[0][0])
    file.close()
    return data

data=ReadData('180322.002.txt',r'C:\Users\孙晓培\AnacondaProjects\DataProcessing\180322.002.txt')
print(data[0][0])
