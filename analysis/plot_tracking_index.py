import numpy as np
from matplotlib import pyplot as plt
import h5py
import sys
import glob
import yaml
import os

def getYaml(dir,fileName,key):
    '''Load config yaml file.
    %% Inputs %%
    - dir (str): directory containing yaml
    - filename (str): name of yaml file
    - key (str): field of yaml file to use
    %% Outputs %%
    - params (dict): parameters
    '''
    # Check if a motion correction parameter file exists. If so, use it.
    paramFiles = glob.glob(os.path.join(dir,fileName))
    params = {}

    if len(paramFiles)==1:
        paramFile = paramFiles[0]
        with open(paramFile,'r') as stream:
            try:
                params = yaml.load(stream,Loader=yaml.CLoader)[key]
                print('Parameter file successfully loaded.')
            except:
                print('Something is wrong with the parameter .yaml file.')
                pass
    else:
        print('No parameter .yaml file found.')

    return params

def getMovingAverage(x,win):
    '''Compute moving average'''
    cv = np.convolve(x,np.ones(win),'valid')/win
    return cv

def scale(x,r1,r2):
    return (x-min(x))/(max(x)-min(x)) * (r2-r1) + r1

def computeTrackingIndex(stimPos,speed,scaling,xOffset,rollWin,exclude=-1):
    '''Calculate tracking index: product of fidelity and vigor.
    %% Inputs %%
    - stimPos (list-like): stimulus x position
    - speed (list-like): animal's rotational speed on ball
    - scaling (float): inverse scaling for backnforth angular span
    - rollWin (int): window for computing rolling average
    '''
    vigor = scale(getMovingAverage(speed[:exclude],rollWin),0,1)
    fidelity = getMovingAverage(1 - abs(stimPos[:exclude])*scaling,rollWin)
    trackingIndex = vigor*fidelity
    return trackingIndex

# Read data
rootDir = sys.argv[1]
fictracPath = glob.glob(f'{rootDir}/*[!video_server,!daq].h5')[0]
vidServerPath = glob.glob(f'{rootDir}/*video_server.h5')[0]
daqPath = glob.glob(f'{rootDir}/*daq.h5')[0]
dset = h5py.File(fictracPath,'r') # fictrac data
vid = h5py.File(vidServerPath,'r') # vid data
daq = h5py.File(daqPath,'r') # daq data

# Get tracking fidelity
stimPos = vid['video']['stimulus']['backnforth'][:,3] # x position of stimulus

# Tracking vigor (rotational speed)
rs = abs(dset['fictrac']['output'][:,2])

# Delta timestamps for fictrac
deltaTimestamps = dset['fictrac']['output'][:,21]
avgDT = np.mean(np.diff(deltaTimestamps))/1000 # average time/prame (s/frame)
framesToSmooth = int(3/avgDT) # number of frames for smoothing over 4 second window

# Load video synchronization info
# Index is vid number, element is fictrac frame
vidSync = vid['video']['synchronization_info']

# Resample stimulus position to match fictrac
originalTime = vidSync[:,0] # fictrac frame for each video frame
desiredTime = np.arange(len(rs)) # want one vid frame for each fictrac frame
stimPos = np.interp(desiredTime,originalTime,stimPos)

# Compute tracking index
params = getYaml(rootDir,'config_backnforth.yaml','playlist')['video'][0]['backnforth']
scaling = params['scaling']
xOffset = params['offset'][0]
trackingIndex = computeTrackingIndex(stimPos,rs,scaling,xOffset,framesToSmooth,exclude=-300)

# Plot results
fig,ax = plt.subplots()
ax.plot(trackingIndex)
ax.set_xlabel('fictrac frame')
ax.set_ylabel('tracking index')
plt.savefig(f'{rootDir}/tracking_index.png',dpi=300)
