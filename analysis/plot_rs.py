import numpy as np
from matplotlib import pyplot as plt 
import h5py 
import glob
import os 
import sys 
from scipy.ndimage import label

def normalize(x):
	rng = np.max(x)-np.min(x)
	x = (x-np.min(x))/rng
	return x 

path = sys.argv[1]
baseDir = os.path.dirname(path)

# Get stimulus data 
daqPath = glob.glob(os.path.join(baseDir,'*daq.h5'))[0]
daq = h5py.File(daqPath,'r')
stim = daq['daq']['input']['samples'][:,0]

# Get fictrac data 
data = h5py.File(path,'r')
fictrac = data['fictrac']
rs = fictrac['output'][:,2]

# Timestamps for stim and fictrac 
sync = daq['daq']['input']['synchronization_info']
fictracSync = sync[:,0]
stimSync = sync[:,1]

# Find stim times 
stimBinary = np.where(stim>=0.1,1,0)
labels = label(stimBinary)[0]
events = np.sort(np.unique(labels))[1:]
times = np.arange(len(stimBinary))
stimOnsets = [times[labels==e][0] for e in events]
stimOffsets = [times[labels==e][-1] for e in events]

# Get stim times closest to sync times 
syncOnsets = [] 
syncOffsets = []

for onset,offset in zip(stimOnsets,stimOffsets):
	closestOnsetTime = min(stimSync,key=lambda x:abs(x-onset))
	closestOffsetTime = min(stimSync,key=lambda x:abs(x-offset))
	onsetBin = np.argwhere(stimSync==closestOnsetTime)
	offsetBin = np.argwhere(stimSync==closestOffsetTime)
	syncOnsets.append(onsetBin)
	syncOffsets.append(offsetBin)

syncOnsets = np.ravel(syncOnsets)
syncOffsets = np.ravel(syncOffsets)

# Now use sync stim times to plot stimuli on fictrac data 
onsetFictrac = fictracSync[syncOnsets]
offsetFictrac = fictracSync[syncOffsets]

# Plot data 
fig,ax = plt.subplots()
ax.plot(rs)
for onset,offset in zip(onsetFictrac,offsetFictrac):
	ax.axvspan(onset,offset,color='r',alpha=0.5)
ax.set_xlabel('fictrac frame')
ax.set_ylabel('rotational speed (a.u.)')

plt.savefig('rs.png',dpi=300)



