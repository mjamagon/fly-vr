
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
        self.visualStims = ['nostim','pipstim']
        self.currentStim  = 'nostim'
        self.play_playlist_item(Experiment.BACKEND_VIDEO,self.currentStim)

    def process_state(self, state):
        dt = time.time() - self._t
        if dt > 60:
            self.currentStim = [s for s in self.visualStims if s!=self.currentStim][0]
            self.play_playlist_item(Experiment.BACKEND_VIDEO,self.currentStim)
            self._t = time.time()

experiment = _MyExperiment()
