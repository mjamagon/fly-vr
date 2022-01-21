import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
import os
import sys
from tqdm import tqdm
from scipy.signal import find_peaks, resample

def getMovingAverage(x,win):
    '''Compute moving average'''
    cv = np.convolve(x,np.ones(win),'valid')/win
    return cv

def scale(x,r1,r2):
    '''Scale input between range.'''
    return (x-min(x))/(max(x)-min(x)) * (r2-r1) + r1

def resampleRows(x,nSamples):
    '''Equalize number of samples in each row of input data.
    %% Inputs %%
    - x (array): input data (each row is a trial)
    - nSamples (int): number of samples in resampled row
    %% Outputs %%
    - resampled (array): resampled array with equal samples per row.
    '''
    resampled = []

    for row in x:
        resampledRow = resample(row,nSamples) # resapmled row
        resampled.append(resampledRow)

    resampled = np.array(resampled)

    return resampled

def chunkData(x,nCycles,dt,peaks=None):
    '''Chunk vectorized data into trials.
    %% Inputs %%
    - x (list-like): vector data
    - nCycles (int): number of stimulus cycles per chunk
    - dT (float): average seconds/frame
    - peaks (list-like): pre-computed peaks used for chunking data
    %% Outputs %%
    - chunkedData (array): chunked data
    - peaks (list): discovered peaks
    '''
    # First, scale data
    x = scale(x,-1,1)

    # Three peaks per cycle: high, low
    nPeaks = 2*nCycles

    # Identify peaks in data - peaks in second derivative (discontinuities)
    if peaks is None:
        peaks = find_peaks(abs(np.diff(np.diff(x))))[0]

    # Chunk data by peaks
    xChunked = [x[peaks[ii*nPeaks]:peaks[(ii+1)*nPeaks]] for ii,_ in enumerate(peaks) if (ii+1)*nPeaks<len(peaks)]

    # Compute average time per trial
    avgTrialTime = np.mean([peaks[(ii+1)*nPeaks] -peaks[ii*nPeaks] for ii,_ in enumerate(peaks) if (ii+1)*nPeaks<len(peaks)])*dt

    # Resample chunked data to equalize samples per row
    xResampled = resampleRows(xChunked,nSamples=1000)

    return xResampled,peaks,avgTrialTime

def getTurningVigor(stimChunks,speedChunks):
    '''Vigor is the sum of rotational speed ipsilateral to stimulus.
    %% Inputs %%
    - stimChunks (array): each row is a stimulus cycle or integer multiple of cycles. zero-centered
    - speedChunks (array): stimulus-aligned speed data (each row is one stimulus period)'''
    stimSign = np.where(stimChunks>0,1,-1)
    speedSign = -np.where(speedChunks>0,1,-1) # negative because clockwise rotation = left turn
    vigorMask = stimSign==speedSign # check sign consistency
    vigor = [sum(abs(speedRow)[maskRow]) for speedRow,maskRow in zip(speedChunks,vigorMask)] # speed ipsilateral to grating rotation
    vigor/=max(vigor)
    vigor=np.array(vigor)

    return vigor

# Root directory
rootDir = sys.argv[1]

# Process data in all subdirectories
subDirs = glob.glob(rootDir + '/[!__pycache__,!test]*')

# If only only experiment folder given, only process this folder
if not [name for name in subDirs if os.path.isdir(name)]:
    subDirs = [rootDir]

for subDir in tqdm(subDirs):
    # Read data
    fictracPath = glob.glob(f'{subDir}/*[!video_server,!daq].h5')[0]
    vidServerPath = glob.glob(f'{subDir}/*video_server.h5')[0]
    daqPath = glob.glob(f'{subDir}/*daq.h5')[0]
    dset = h5py.File(fictracPath,'r') # fictrac data
    vid = h5py.File(vidServerPath,'r') # vid data
    daq = h5py.File(daqPath,'r') # daq data

    # Get grating direction
    gratingDirection = vid['video']['stimulus']['grating'][:,-2]

    # Get rotational speed
    rs = dset['fictrac']['output'][:,2]

    # Delta timestamps for fictrac
    deltaTimestamps = dset['fictrac']['output'][:,21]
    avgDT = np.mean(np.diff(deltaTimestamps))/1000 # average time/prame (s/frame)
    framesToSmooth = int(0.2/avgDT) # number of frames for smoothing over 200ms window

    # Load video synchronization info
    # Index is vid number, element is fictrac frame
    vidSync = vid['video']['synchronization_info']

    # Resample stimulus position to match fictrac
    originalTime = vidSync[:,0] # fictrac frame for each video frame
    desiredTime = np.arange(len(rs)) # want one vid frame for each fictrac frame
    gratingDirection = np.interp(desiredTime,originalTime,gratingDirection)

    # Chunk grating direction
    gratingChunks,peaks,trialTime = chunkData(gratingDirection,nCycles=1,dt=avgDT)

    # Chunk rotational speed
    speedChunks,_,_ = chunkData(rs,nCycles=1,peaks=peaks,dt=avgDT)

    # Compute tracking vigor
    vigor = getTurningVigor(gratingChunks,speedChunks)
    trialIDs = np.arange(len(vigor))
    trialTimestamps = trialTime*trialIDs/60

    # Overlay rotational speed and grating direction
    fig = plt.figure(constrained_layout=True)
    spec = fig.add_gridspec(10, 3) # down, across
    ax0 = fig.add_subplot(spec[:2,:2])
    ax1 = fig.add_subplot(spec[2:,:2])
    ax2 = fig.add_subplot(spec[2:,2])
    ax0.plot(gratingChunks[0],color='k')
    ax0.axis('off')
    ax1.imshow(speedChunks,aspect='auto')
    ax1.set_xticks([0,1000])
    ax1.set_xticklabels([0,f'{trialTime:.0f}'])
    ax1.set_yticks(trialIDs[::2])
    ax1.set_yticklabels(trialTimestamps[::2].astype(int))
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('time (min)',rotation=-90)
    ax2.plot(vigor,trialIDs,color='k')
    ax2.invert_yaxis()
    ax2.axis('off')
    fig.suptitle('Rotational speed')
    plt.show()
