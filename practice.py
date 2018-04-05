from pylab import *
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

f = np.arange(0,101)                 # frequency 
t = np.arange(11,245)                # time
z = 20*np.sin(f**0.56)+22            # function
z = np.reshape(z,(1,max(f.shape)))   # reshape the function
Z = z*np.ones((max(t.shape),1))      # make the single vector to a mxn matrix
T, F = meshgrid(f,t)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.pcolor(F,T,Z, norm=LogNorm(vmin=z.min(),vmax=z.max()))
plt.xlim((t.min(),t.max()))
mn=int(np.floor(Z.min()))        # colorbar min value
mx=int(np.ceil(Z.max()))         # colorbar max value
md=(mx-mn)/2                     # colorbar midpoint value
cbar=plt.colorbar()              # the mystery step ???????????
cbar.set_ticks([10.1,30,50])
cbar.set_ticklabels([10.1,30,50]) # add the labels
plt.show()