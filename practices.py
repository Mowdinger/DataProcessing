import numpy as np  
import matplotlib.pyplot as plt   
data=np.random.rand(10,10)  
fig, ax=plt.subplots()  
data[data==-1]=np.nan#去掉缺省值-1  
im =ax.imshow(data,interpolation='none',cmap='Reds_r',vmin=0.6,vmax=.9)#不插值  
#去掉边框  
# ax.spines['top'].set_visible(False)  
# ax.spines['right'].set_visible(False)  
# ax.spines['bottom'].set_visible(False)  
# ax.spines['left'].set_visible(False)  
########################################################################  
position=fig.add_axes([0.15, 0.05, 0.7, 0.03])  
cb=plt.colorbar(im,cax=position,orientation='horizontal')#方向
cb.
plt.show()