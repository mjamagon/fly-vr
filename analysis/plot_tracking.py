import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
import os
import sys
from tqdm import tqdm
from scipy.signal import find_peaks, resample
from scipy.stats import zscore, pearsonr
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import argparse

parser = argparse.ArgumentParser(
description = '''Plot tracking index for experiment.'''
)
parser.add_argument('--dir',type=str,required=True,help='directory containing experiments, or directory for single experiment')
parser.add_argument('--nCycles',type=int,default=3,help='number of stimulus cycles per trial')
parser.add_argument('--nSamples',type=int,default=1000,help='samples per trial (resampled)')
args = parser.parse_args()

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

def chunkData(x,nCycles,dt,peaks=None,nSamples=1000,isSpeed=False,isBacknforth=False):
    '''Chunk vectorized data into trials.
    %% Inputs %%
    - x (list-like): vector data
    - nCycles (int): number of stimulus cycles per chunk
    - dT (float): average seconds/frame
    - peaks (list-like): pre-computed peaks used for chunking data
    - isSpeed (bool): flag indicating input is speed data
    - isBacknforth (bool): flag indicating if stimulus is backnforth (vs. grating)
    %% Outputs %%
    - chunkedData (array): chunked data
    - peaks (list): discovered peaks
    '''
    # First, scale data (only if not speed)
    if not isSpeed:
        x = scale(x,-1,1)

    # Two peaks per cycle: high, low
    nPeaks = 2*nCycles

    # Identify peaks in data - peaks in second derivative (discontinuities)
    if peaks is None:
        if not isBacknforth:
            d2x = abs(np.diff(np.diff(x)))
            d2x[d2x>0.2] = 1
            peaks = find_peaks(d2x,distance=100)[0][2:]
        # Anything else
        else:
            # Stationary stimulus control: we'll pretend there are 7 peaks every 10 seconds
            if np.isnan(x).all():
                peaks = []
                ii = 1
                shamTime = 0
                while shamTime < len(x):
                    shamTime = int(ii*(1/(0.7*dt)))
                    peaks.append(shamTime)
                    ii+=1

            # If moving stimulus, find the peaks
            else:
                peaks = find_peaks(abs(x),distance=100)[0][2:]
    # Chunk data by peaks
    xChunked = [x[peaks[ii*nPeaks]:peaks[(ii+1)*nPeaks]] for ii,_ in enumerate(peaks) if (ii+1)*nPeaks<len(peaks)]

    # Compute average time per trial
    avgTrialTime = np.mean([peaks[(ii+1)*nPeaks] -peaks[ii*nPeaks] for ii,_ in enumerate(peaks) if (ii+1)*nPeaks<len(peaks)])*dt

    # Resample chunked data to equalize samples per row
    xResampled = resampleRows(xChunked,nSamples=nSamples)
    return xResampled,peaks,avgTrialTime

def getTurningVigor(stimChunks,speedChunks):
    '''Vigor is the sum of rotational speed ipsilateral to stimulus.
    %% Inputs %%
    - stimChunks (array): each row is a stimulus cycle or integer multiple of cycles. zero-centered
    - speedChunks (array): stimulus-aligned speed data (each row is one stimulus period)'''
    stimSign = np.where(stimChunks<0,-1,0) # only pull out stimuli presented to male's left eye
    speedSign = -np.where(speedChunks>0,1,-1) # negative because clockwise rotation = left turn
    vigorMask = stimSign==speedSign # check sign consistency
    vigor = [sum(abs(speedRow)[maskRow]) for speedRow,maskRow in zip(speedChunks.copy(),vigorMask)] # speed ipsilateral to grating rotation
    vigor/=max(vigor)
    vigor=np.array(vigor)

    return vigor

def getFidelity(stimChunks,speedChunks):
    '''Compute tracking fidelity.
    %% Inputs %%
    - stimChunks (array): each row is a stimulus cycle or integer multiple of cycles. zero-centered
    - speedChunks (array): stimulus-aligned speed data (each row is one stimulus period)
    %% Outputs %%
    - fidelity (array): pearson correlation between stimulus position and turning speed
    '''
    fidelityMask = np.where(stimChunks<0,1,0).astype(bool) # only pull out stimuli presented to male's left eye
    fidelity = [pearsonr(-s[mask],x[mask])[0] for (s,x,mask) in zip(stimChunks,speedChunks,fidelityMask)]
    return fidelity

# Root directory
rootDir = args.dir

# Process data in all subdirectories
subDirs = glob.glob(rootDir + '/[!__pycache__,!test]*')

# If only only experiment folder given, only process this folder
if not [name for name in subDirs if os.path.isdir(name)]:
    subDirs = [rootDir]

for subDir in tqdm(subDirs):
    isBacknforth = False
    noStim = False

    # Check stimulus type
    if 'backnforth' in subDir:
        isBacknforth = True
    if 'nostim' in subDir:
        noStim = True

    # Read data
    fictracPath = glob.glob(f'{subDir}/*[!video_server,!daq].h5')[0]
    vidServerPath = glob.glob(f'{subDir}/*video_server.h5')[0]
    daqPath = glob.glob(f'{subDir}/*daq.h5')[0]
    dset = h5py.File(fictracPath,'r') # fictrac data
    vid = h5py.File(vidServerPath,'r') # vid data
    daq = h5py.File(daqPath,'r') # daq data

    # Get rotational speed
    rs = dset['fictrac']['output'][:,2]

    # Get stimulus direction direction
    if isBacknforth and not noStim:
        stimDirection = vid['video']['stimulus']['backnforth'][:,3]
    else:
        stimDirection = vid['video']['stimulus']['grating'][:,-2]

    # Delta timestamps for fictrac
    deltaTimestamps = dset['fictrac']['output'][:,21]
    avgDT = np.mean(np.diff(deltaTimestamps))/1000 # average time/frame (s/frame)

    # Load video synchronization info
    # Index is vid number, element is fictrac frame
    vidSync = vid['video']['synchronization_info']

    # If stimulus is "NoStim", position is all zeros
    if noStim:
        stimDirection = np.zeros(len(vidSync[:,0]))

    # Resample stimulus position to match fictrac
    originalTime = vidSync[:,0] # fictrac frame for each video frame
    desiredTime = np.arange(len(rs)) # want one vid frame for each fictrac frame
    stimDirection = np.interp(desiredTime,originalTime,stimDirection)

    # Chunk grating direction
    nCycles = args.nCycles
    stimChunks,peaks,trialTime = chunkData(stimDirection,nCycles=nCycles,dt=avgDT,nSamples=args.nSamples,isBacknforth=isBacknforth)

    # Chunk rotational speed
    speedChunks,_,_ = chunkData(rs,nCycles=nCycles,peaks=peaks,dt=avgDT,nSamples=args.nSamples,isSpeed=True,isBacknforth=isBacknforth)

    # Compute tracking vigor
    if not noStim:
        vigor = getTurningVigor(stimChunks,speedChunks)
    else:
        vigor = np.zeros(len(speedChunks))


    # Compute tracking fidelity of tracking experiment
    if isBacknforth and not noStim:
        fidelity = getFidelity(stimChunks,speedChunks)
        TI = vigor*fidelity

        # Save tracking index, vigor, and rotational speed
        np.save(os.path.join(subDir,'tracking_index.npy'),TI)
        np.save(os.path.join(subDir,'tracking_vigor.npy'),vigor)
        np.save(os.path.join(subDir,'rotational_speed.npy'),rs)

    else:
        TI = vigor
    # TI = vigor

    trialIDs = np.arange(len(vigor))
    trialTimestamps = (trialTime*trialIDs/60).astype(int)
    uniqueTimestamps = [np.ravel(np.argwhere(trialTimestamps==t)[0]) \
        for t in np.sort(np.unique(trialTimestamps))]
    uniqueTimestamps = np.ravel(uniqueTimestamps)

    # Convert rad/frame to rad/s.
    # speedConversion = -1/avgDT * (2*np.pi*4.5)/(2*np.pi) # will be rad/frame * frame/s * mm/rad -> mm/s
    speedConversion = -1/avgDT # will be rad/frame * frame/s -> rad/s

    # If no stim, just plot speed over time
    if noStim:
        fig,ax1 = plt.subplots()
        im = ax1.imshow(speedConversion*speedChunks,aspect='auto',cmap='bwr',norm=colors.CenteredNorm())
        ax1.set_xticks([0,args.nSamples])
        ax1.set_xticklabels([0,f'{trialTime:.0f}'])
        ax1.set_yticks(trialIDs[uniqueTimestamps])
        ax1.set_yticklabels(trialTimestamps[uniqueTimestamps].astype(int))
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('time (min)')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical',label='rotational speed (rad/s)')
        plt.savefig(os.path.join(subDir,'rotational_speed.png'),dpi=300)
        plt.close()
        continue

    # Overlay rotational speed and grating direction
    fig = plt.figure(constrained_layout=True)
    spec = fig.add_gridspec(10, 3) # down, across
    ax0 = fig.add_subplot(spec[:2,:2])
    ax1 = fig.add_subplot(spec[2:,:2])
    ax2 = fig.add_subplot(spec[2:,2])
    ax0.plot(stimChunks[1],color='k')
    ax0.axis('off')
    im = ax1.imshow(speedConversion*speedChunks,aspect='auto',cmap='bwr',norm=colors.CenteredNorm())
    ax1.set_xticks([0,args.nSamples])
    ax1.set_xticklabels([0,f'{trialTime:.0f}'])
    ax1.set_yticks(trialIDs[uniqueTimestamps])
    ax1.set_yticklabels(trialTimestamps[uniqueTimestamps].astype(int))
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('time (min)')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical',label='rotational speed (rad/s)')
    ax2.plot(TI,trialIDs,color='k')
    sns.despine(ax=ax2)
    ax2.invert_yaxis()
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel('tracking index (a.u.)')
    plt.tight_layout()
    plt.savefig(os.path.join(subDir,'tracking_index.png'),dpi=300)
    plt.close()

    # Plot aligned trials for alignment sanity check
    fig,ax = plt.subplots()
    ax.imshow(stimChunks,aspect='auto')
    ax.axis('off')
    plt.savefig(os.path.join(subDir,'trial_chunks.png'),dpi=300)
    plt.close()
