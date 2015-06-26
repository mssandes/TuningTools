#!/usr/bin/env python

import sys
import os
import pickle
import numpy as np
from FastNetTool.CrossValid import CrossValid
from FastNetTool.util       import include, normalizeSumRow, reshape

DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid.data.ringer.npy'
JobConfigLocationInput            = '../../data/job.mc14_13TeV.129160.147406.n0005.i0100.s0000.s0001.pic'


print 'Opening data and normalize ...'

#Data configuration
objLoad                           = np.load( DatasetLocationInput )
Data                              = normalizeSumRow( reshape( objLoad[0] ) )
Target                            = reshape(objLoad[1])
objLoad                           = pickle.load(open(JobConfigLocationInput,'r'))
Cross                             = objLoad[3]
Inits                             = objLoad[2]
minSort                           = objLoad[1][0]
maxSort                           = objLoad[1][1]
neuron                            = objLoad[0]
OutputName                        = 'train.mc14_13TeV.129160.147406'
DoMultiStop                       = True
ShowEvo                           = 4
Epochs                            = 1000

from FastNetTool.TrainJob import TrainJob
trainjob = TrainJob()

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

