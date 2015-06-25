#!/usr/bin/env python

import sys
import os
import pickle
import numpy as np
from FastNetTool.CrossValid import CrossValid
from FastNetTool.util       import include, normalizeSumRow, reshape, load, getModuleLogger


#DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid.data.ringer.npy'
DatasetLocationInput              = sys.argv[1] 
Neuron                            = int(sys.argv[2])
Sort                              = int(sys.argv[3])
Inits                             = int(sys.argv[4])

mainLogger = getModuleLogger(__name__)

mainLogger.info('DatasetLocationInput %s', DatasetLocationInput)
mainLogger.info('Number of neurons %d', Neuron)
mainLogger.info('Sort number %d', Sort)
mainLogger.info('Inits %d', Inits)

mainLogger.info('Opening data...')
objDataFromFile                   = np.load( DatasetLocationInput )

#Job option configuration
Data                              = normalizeSumRow( reshape( objDataFromFile[0] ) )
Target                            = reshape(objDataFromFile[1])
Cross                             = CrossValid( Target, nSorts=50, nBoxes=10, nTrain=6, nValid=4)
OutputName                        = 'RingerOfflineNN_ZeeTruth_JF17Truth'
DoMultiStop                       = True
ShowEvo                           = 20
Epochs                            = 3000


del objDataFromFile

from FastNetTool.TrainJob import TrainJob
trainjob = TrainJob()

trainjob( Data, Target, Cross, 
                neuron=Neuron, 
                sort=Sort,
                inits=Inits, 
                epochs=Epochs,
                showEvo=ShowEvo, 
                output=OutputName,
                doMultiStop=DoMultiStop,
                doPerf=False)


