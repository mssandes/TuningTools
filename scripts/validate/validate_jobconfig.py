#!/usr/bin/python

import sys
import os
import pickle
import numpy as np
from FastNetTool.CrossValid import CrossValid
from FastNetTool.util       import include, normalizeSumRow, reshape, load

DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid.data.ringer.npy'
JobConfigLocationInput            = '/afs/cern.ch/user/j/jodafons/public/valid.jobconfig.ringer.pic'


print 'Opening data and normalize ...'

#Data configuration
objLoad                           = np.load( DatasetLocationInput )
Data                              = normalizeSumRow( reshape( objLoad[0] ) )
Target                            = reshape(objLoad[1])

#job configuration
objLoad                           = pickle.load(open(JobConfigLocationInput,'r'))
Cross                             = objLoad[3]
Inits                             = objLoad[2]
minSort                           = objLoad[1][0]
maxSort                           = objLoad[1][1]
minNeuron                         = objLoad[0]
maxNeuron                         = objLoad[0]
print minNeuron
print maxNeuron
print minSort
print maxSort

OutputName                        = 'valid_train'
DoMultiStop                       = True
ShowEvo                           = 4
Epochs                            = 1000

from FastNetTool.TrainJob import TrainJob
trainjob = TrainJob()

for neuron in range( minNeuron, maxNeuron+1):
  for sort in range( minSort, maxSort+1):
    trainjob( Data, Target, Cross, 
                        neuron=neuron, 
                        sort=sort,
                        inits=Inits, 
                        epochs=Epochs,
                        showEvo=ShowEvo, 
                        output=OutputName,
                        doMultiStop=DoMultiStop,
                        doPerf=False)


