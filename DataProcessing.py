# -*- coding: utf-8 -*-
import os
import numpy
import matplotlib
import pylab
print('Hello world!')

def readFile():
    readData = []
    f = open(r"C:\Users\孙晓培\AnacondaProjects\DataProcessing\180322.002.txt");
    for i in f.readlines():
        if i[:5] == "Added":
            readData.append(i)
    f.close()
    return readData
 
def writeFile():
    data = readFile()
    f = open(r"new file path","w")
    f.writelines(data)
    f.close()
 
if __name__ == '__main__':
    writeFile()