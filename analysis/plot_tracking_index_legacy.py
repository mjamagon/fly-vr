import numpy as np
from matplotlib import pyplot as plt
import h5py
import sys
import glob
import yaml
import os
import pandas as pd
from tqdm import tqdm
from scipy.stats.stats import pearsonr

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

def computeOLTI(stimPos,speed,rollWin,makePlot=False):
    '''In OL, fidelity is correlation between stimulus position
    and animal rotational speed. See Sten et al. 2021.
    %% Inputs %%
    - stimPos (list-like): stimulus x position
    - speed (list-like): animal's rotational speed on ball
    - rollWin (int): window for computing correlation
    %% Ouputs %%
    - fidelity (array): tracking fidelity
    '''
    # stimPos = pd.Series(scale(stimPos,-1,1))
    # speedUnscaled = rollWin/3 * getMovingAverage(-speed,rollWin//3) # left ball rotation -> rightward turn
    # speed = pd.Series(speedUnscaled)/max(abs(speedUnscaled))
    stimPos = scale(stimPos,-1,1)
    speed = -speed[:len(speed)-len(speed)%rollWin]
    stimPos = stimPos[:len(stimPos)-len(stimPos)%rollWin]
    speed = np.reshape(speed,(-1,rollWin))
    stimPos = np.reshape(stimPos,(-1,rollWin))
    speed = speed/np.max(abs(speed),axis=-1)[:,None]
    fidelity = [pearsonr(s,x)[0] for (s,x) in zip(speed,stimPos)]
    stimSign = np.where(stimPos>0,1,-1)
    speedSign = np.where(speed>0,1,-1)
    vigorMask = [[(s1==s2) for (s1,s2) in zip(stimRow,speedRow)] for (stimRow,speedRow) in zip(stimSign,speedSign)]
    vigor = [sum(abs(speedRow)[maskRow]) for speedRow,maskRow in zip(speed,vigorMask)]
    vigor/=max(vigor)
    TI = fidelity*vigor
    TI[TI<0] = 0

    # # Make plot
    # if makePlot:
    #     # Plot results
    #     fig,ax = plt.subplots(1,2)
    #     ax[0].plot(stimPos[10000:12000],label='stimPos (a.u.)')
    #     ax[0].plot(speed[10000:12000],label='rotational speed (a.u.)')
    #     ax[0].set_ylim(top=1)
    #     ax[0].set_xlabel('fictrac frame')
    #     ax[0].set_ylabel('amplitude (a.u.)')
    #     ax[0].legend()
    #     ax[1].hist(speedUnscaled,bins=100,density=True)
    #     ax[1].set_xlabel('rotational speed (rad/s)')
    #     ax[1].set_ylabel('probability')
    #     plt.savefig(f'{subDir}/speed_data.png',dpi=300)
    #     plt.close()
    return TI

def computeTrackingIndex(stimPos,speed,scaling,xOffset,rollWin,isCL,exclude=-1):
    '''Calculate tracking index: product of fidelity and vigor.
    %% Inputs %%
    - stimPos (list-like): stimulus x position
    - speed (list-like): animal's rotational speed on ball
    - scaling (float): inverse scaling for backnforth angular span
    - rollWin (int): window for computing rolling average
    - isCL (bool): flag for closed loop experiment
    '''
    vigor = scale(getMovingAverage(abs(speed)[:exclude],rollWin),0,1)
    if isCL:
        fidelity = getMovingAverage(1 - abs(stimPos[:exclude])*scaling,rollWin)
        minLen = min(len(vigor),len(fidelity))
        trackingIndex = vigor[:minLen]*fidelity[:minLen]
    else:
        trackingIndex = computeOLTI(stimPos[:exclude],speed[:exclude],rollWin,makePlot=True)

    return trackingIndex

# Root directory
rootDir = sys.argv[1]

# Process data in all subdirectories
subDirs = glob.glob(rootDir + '/[!test]*')

# If only only experiment folder given, only process this folder
if not [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,name))]:
    subDirs = [rootDir]

for subDir in tqdm(subDirs):
    # Read data
    fictracPath = glob.glob(f'{subDir}/*[!video_server,!daq].h5')[0]
    vidServerPath = glob.glob(f'{subDir}/*video_server.h5')[0]
    daqPath = glob.glob(f'{subDir}/*daq.h5')[0]
    dset = h5py.File(fictracPath,'r') # fictrac data
    vid = h5py.File(vidServerPath,'r') # vid data
    daq = h5py.File(daqPath,'r') # daq data

    # Get tracking fidelity
    stimPos = vid['video']['stimulus']['backnforth'][:,3] # x position of stimulus

    # Tracking vigor (rotational speed)
    rs = dset['fictrac']['output'][:,2]

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
    params = getYaml(subDir,'config_backnforth.yaml','playlist')['video'][0]['backnforth']
    scaling = params['scaling']
    xOffset = params['offset'][0]
    isCL = params['CL']
    trackingIndex = computeTrackingIndex(stimPos,rs,scaling,xOffset,framesToSmooth,isCL,exclude=-1)

    # Plot results
    fig,ax = plt.subplots()
    ax.plot(trackingIndex)
    ax.set_ylim(top=1)
    ax.set_xlabel('fictrac frame')
    ax.set_ylabel('tracking index')
    plt.savefig(f'{subDir}/tracking_index.png',dpi=300)
    plt.close()
