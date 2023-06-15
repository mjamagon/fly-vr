

import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
import argparse
import pandas as pd 


def scale(x,r1,r2):
    '''Scale input between range.'''
    return (x-min(x))/(max(x)-min(x)) * (r2-r1) + r1

parser = argparse.ArgumentParser(
description = '''Plot tracking index for experiment.'''
)
parser.add_argument('--dir',type=str,required=True,help='directory containing experiments, or directory for single experiment')
parser.add_argument('--window',type=int,default=90,help='correlation window')

args = parser.parse_args()

# Root directory
rootDir = args.dir

# Read data
fictracPath = glob.glob(f'{rootDir}/*[!video_server,!daq].h5')[0]
vidServerPath = glob.glob(f'{rootDir}/*video_server.h5')[0]
daqPath = glob.glob(f'{rootDir}/*daq.h5')[0]
dset = h5py.File(fictracPath,'r') # fictrac data
vid = h5py.File(vidServerPath,'r') # vid data
daq = h5py.File(daqPath,'r') # daq data

# Get rotational speed
rs = -dset['fictrac']['output'][:,2]

# Get stimulus direction direction
stimDirection = -vid['video']['stimulus']['actuator'][:,4] # neg. because right translates to left for male 

# Delta timestamps for fictrac
deltaTimestamps = dset['fictrac']['output'][:,21]
avgDT = np.mean(np.diff(deltaTimestamps))/1000 # average time/frame (s/frame)

# Load video synchronization info
# Index is vid number, element is fictrac frame
vidSync = vid['video']['synchronization_info']

# Resample stimulus position to match fictrac
originalTime = vidSync[:,0] # fictrac frame for each video frame
desiredTime = np.arange(len(rs)) # want one vid frame for each fictrac frame
stimDirection = np.interp(desiredTime,originalTime,stimDirection)
stimDirection = scale(stimDirection,-1,1) # scale between -1 and 1

# get tracking fidelity 
rs = pd.Series(rs)
direction = pd.Series(stimDirection)
trackingCorrelation = rs.rolling(args.window).corr(direction)

# tracking vigor
binaryDirection = np.where(direction>0,1,0) # actuator direction
binarySpeed = np.where(rs>0,1,0) # rotational speed direction
speedMask = np.squeeze(np.argwhere(binaryDirection!=binarySpeed))
maskedSpeed = abs(rs.copy())
maskedSpeed[speedMask] = 0 # zero out speed when directions don't match up
maskedSpeed = pd.Series(maskedSpeed)
vigor = maskedSpeed.rolling(args.window).mean()
vigor/=np.max(vigor) # normalize 

# plot 
# import pdb; pdb.set_trace()
TI = vigor * trackingCorrelation
tiSTD = np.std(TI)
vigorSTD = np.std(vigor)
corrSTD = np.std(trackingCorrelation)
tracking = np.where(TI>2*tiSTD,1,0)
# tracking = np.where(vigor>2*vigorSTD,1,0)
# tracking = np.where(trackingCorrelation>2*corrSTD,1,0)

fig,ax = plt.subplots(1,4)
ax[0].plot(vigor)
ax[1].plot(trackingCorrelation)
ax[2].plot(TI)
ax[3].plot(tracking)
ax[0].set_title('tracking vigor')
ax[1].set_title('tracking correlation')
ax[2].set_title('tracking index')
ax[3].set_title('tracking times ')
plt.tight_layout()
plt.show()