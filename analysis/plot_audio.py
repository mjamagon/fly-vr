import numpy as np
from matplotlib import pyplot as plt 
import h5py 
import sys 

path = sys.argv[1]
data = h5py.File(path,'r')
x = data['daq']['input']['samples'][:,-1]
fig,ax = plt.subplots()
ax.plot(x)
plt.show()