#!/usr/bin/env python

import sys
import os
import pickle
import numpy as np
from FastNetTool.CrossValid import CrossValid
from FastNetTool.util       import include, normalizeSumRow, reshape, load, getModuleLogger

#DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid.data.ringer.npy'
DatasetLocationInput              ='/afs/cern.ch/work/w/wsfreund/public/mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH.npy'

mainLogger = getModuleLogger(__name__)
mainLogger.info('Opening data...')
objDataFromFile                   = np.load( DatasetLocationInput )

#Job option configuration
Data                              = normalizeSumRow( reshape( objDataFromFile[0] ) )
Target                            = reshape(objDataFromFile[1])
Cross                             = CrossValid( nSorts=50, nBoxes=10, nTrain=6, nValid=4)
OutputName                        = 'output'
DoMultiStop                       = True
ShowEvo                           = 5
Epochs                            = 1000
#job configuration
Inits                             = 1
minNeuron                         = 5
maxNeuron                         = 5

del objDataFromFile
from FastNetTool.TrainJob import TrainJob
trainjob = TrainJob()

for neuron in range( minNeuron, maxNeuron+1):
  trainjob( Data, Target, Cross, 
                  neuron=neuron, 
                  sort=0,
                  inits=Inits, 
                  epochs=Epochs,
                  showEvo=ShowEvo, 
                  output=OutputName,
                  doMultiStop=DoMultiStop,
                  doPerf=False)

