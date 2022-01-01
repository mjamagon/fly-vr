
import time
import random
import pkg_resources
from flyvr.control.experiment import Experiment
import os
import numpy as np
import random

class _MyExperiment(Experiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t = time.time()

        # Define stimuli
        self.optoStims = ['opto_2','opto_2']

        # Load HC stim time data
        dataPath = pkg_resources.resource_filename('flyvr',os.path.join('data','itis_all_data.npy'))
        self.itis = list(np.load(dataPath))
        self.stimTime = None
        self.stimType = None
        self.drawSample()

        # Play pipstim
        self.play_playlist_item(Experiment.BACKEND_VIDEO,'pipstim')

    def drawSample(self,minTime=10,maxTime=60):
        self.stimTime = min(maxTime,random.sample(self.itis,1)[0])
        self.stimTime = max(self.stimTime,minTime)
        self.stimType = np.random.choice(self.optoStims)
        print(self.stimTime)

    def process_state(self, state):
        dt = time.time() - self._t
        if dt > self.stimTime:
            stim = self.stimType
            self.drawSample()
            self._t = time.time()
            self.play_playlist_item(Experiment.BACKEND_DAQ,stim)

experiment = _MyExperiment()
