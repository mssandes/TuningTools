#!/usr/bin/python

import sys
import os
import pickle
import numpy as np
from FastNetTool.CrossValid import CrossValid
from FastNetTool.util       import include, normalizeSumRow, reshape, load

#DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid.data.ringer.npy'
DatasetLocationInput              ='/tmp/jodafons/mc14_13TeV.147406.129160.sgn.truth.bkg.truth.offline.npy'
print 'openning data and normalize ...'

objDataFromFile                   = np.load( DatasetLocationInput )
#Job option configuration
Data                              = normalizeSumRow( reshape( objDataFromFile[0] ) )
Target                            = reshape(objDataFromFile[1])
Cross                             = CrossValid( Target, nSorts=50, nBoxes=10, nTrain=6, nValid=4)
OutputName                        = 'output'
DoMultiStop                       = True
ShowEvo                           = 4
Epochs                            = 1000

#job configuration
Inits                             = 1
minSort                           = 0
maxSort                           = 0
minNeuron                         = 2
maxNeuron                         = 2

del objDataFromFile

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


