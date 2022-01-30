import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import sys
from tqdm import tqdm
import seaborn as sns
from scipy.stats import ttest_ind,ttest_1samp
import pandas as pd
import pickle
import argparse

parser = argparse.ArgumentParser(
description = '''Compare tracking experiments with and without female.'''
)
parser.add_argument('--dir',type=str,required=True,help='directory containing experiments')
parser.add_argument('--dataName',type=str,required=True,help='type of experiment (e.g. rotational_speed.npy)')
args = parser.parse_args()

def pToStar(p):
    '''Return asterisks corresponding to p value.'''
    if p<0.001:
        return '***'
    elif p<0.01:
        return '**'
    elif p<0.05:
        return '*'
    else:
        return 'ns'

def plotWithSig(df,hbuff=0.2,vbuff=20,title=''):
    # Make plot
    fig,ax = plt.subplots()
    sns.set(style='ticks', palette='pastel')
    sns.violinplot(x='fly',y='trackingIndex',
                hue='paradigm', palette=['b','r'],
                cut=0,legend=False,data=df,ax=ax)
    sns.despine(offset=10, trim=True)
    plt.show()

    # Make significance bars
    angles = np.unique(df['angle'])
    ticks = ax.get_xticks()

    for tick,angle in zip(ticks,angles):
        data = df[df['angle']==angle]
        dataLeft = data[data['side']=='left']['angular displacement (deg)'].values.tolist()
        dataRight = data[data['side']=='right']['angular displacement (deg)'].values.tolist()

        if len(dataLeft)<2 or len(dataRight)<2:
            continue

        # Perform two sample statistical test
        t,p = ttest_ind(dataLeft,dataRight)
        text = pToStar(p)
        x1,x2 = tick-hbuff,tick+hbuff
        y = max([max(dataLeft),max(dataRight)]) + vbuff
        if text!='ns':
            ax.plot([x1,x1,x2,x2],[y,y+hbuff,y+hbuff,y],lw=1.5,color='k')
            ax.text((x1+x2)*0.5,y,text,ha='center',va='bottom',color='k')

        # Perform one sample statistical test
        _,pL = ttest_1samp(dataLeft,0) # left
        _,pR = ttest_1samp(dataRight,0) # right
        textL = pToStar(pL)
        textR = pToStar(pR)
        yL = max(dataLeft) + vbuff/2
        yR = max(dataRight) + vbuff/2
        if textL!='ns':
            ax.text(x1,yL,textL,ha='center',va='bottom',color='k')
        if textR!='ns':
            ax.text(x2,yR,textR,ha='center',va='bottom',color='k')

    ax.legend(ncol=2,loc='lower center')
    fig.tight_layout()
    fig.savefig(title,dpi=300)
    plt.close(fig)

def loadTrackingData(rootDir,dataName):
    # Get data subdirectories
    subDirs = glob.glob(rootDir + '/[!__pycache__,!test,!female_tracking,!legacy]*')
    trackingData = {'fly':[],'trackingIndex':[],'paradigm':[]}
    nFlies = 0 # number of flies processed so far

    # Get no female and female condition for each experiment
    for subDir in tqdm(subDirs):

        # Find all sub-subdirectories with tracking_index.npy file
        trackingExps = np.array(glob.glob(subDir + f'/**/{dataName}',recursive=True))

        # Get experiment names (e.g. 101_backnforth_female)
        expNames = np.array([os.path.basename(os.path.dirname(exp)) for exp in trackingExps])

        # Get fly associated with each experiment.
        flyIDs = np.array([f'fly_{str(name.split("_")[0])[0]}' for name in expNames if 'pipdist' not in name])

        # Get experiment condition (female or no female)
        expType = np.array(['female' if 'nofemale' not in name else 'nofemale' for name in expNames if 'pipdist' not in name])

        # Place data in trackingData dictionary
        for ii,id in enumerate(np.unique(flyIDs)):
            whichExps = np.ravel(np.argwhere(flyIDs==id)) # experiment indices belonging to fly
            flyExps = trackingExps[whichExps.astype(int)] # experiments belonging to fly
            paradigms = expType[whichExps.astype(int)] # female or no female paradigm for fly's experiments

            # Put data in correct place
            for (exp,paradigm) in zip(flyExps,paradigms):
                flyData = np.load(exp) # load tracking data for this experiment
                trackingData['paradigm'].append(paradigm)
                trackingData['fly'].append(ii+nFlies)
                trackingData['trackingIndex'].append(flyData)

        # Update number of processed flies
        nFlies+=len(np.unique(flyIDs))

    # Save data
    pklPath = os.path.join(rootDir,f'{dataName.split(".")[0]}.pkl')
    with open(pklPath,'wb') as output:
        pickle.dump(trackingData,output)

    return trackingData

# Plot data
dataName = args.dataName
pklPath = os.path.join(args.dir,f'{dataName.split(".")[0]}.pkl')
if os.path.exists(pklPath):
    with open(pklPath,'rb') as input:
        trackingData = pickle.load(input)
else:
    trackingData = loadTrackingData(args.dir,dataName=dataName)

# Make dataframe
df = pd.DataFrame.from_dict(trackingData)
df = df.explode('trackingIndex')

# Make plot
fig,ax = plt.subplots()
sns.boxplot(x='fly',
    y='trackingIndex',
    data=df,
    hue='paradigm',
    showfliers=False,
    ax=ax)
ylabel = dataName.split(".")[0]
ax.set_ylabel(ylabel)
sns.despine(ax=ax)
plt.savefig(args.dir + f'/{ylabel}' + '.png')
plt.close()
