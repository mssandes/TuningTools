#!/usr/bin/env python

import sys
import os
import pickle
from FastNetTool.Preprocess import Normalize, RingerRp
from FastNetTool.util       import include, normalizeSumRow, reshape, load, getModuleLogger, sourceEnvFile
#sourceEnvFile()
import numpy as np
from FastNetTool.CrossValid import CrossValid

mainLogger = getModuleLogger(__name__)

#DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid.data.ringer.npy'
DatasetLocationInput              = sys.argv[1] 
JobConfiguration                  = load(sys.argv[2])

Neuron                            = JobConfiguration[0]
SortMin                           = JobConfiguration[1][0]
SortMax                           = JobConfiguration[1][1]
InitBounds                        = JobConfiguration[2]
Cross                             = JobConfiguration[3]
output                            = sys.argv[3] 

mainLogger.info('Output %s', output)

mainLogger.info('DatasetLocationInput %s', DatasetLocationInput)
mainLogger.info('Opening data...')
objDataFromFile                   = np.load( DatasetLocationInput )

#Job option configuration
Data                              = reshape(objDataFromFile[0])
Target                            = reshape(objDataFromFile[1])
DoMultiStop                       = True
ShowEvo                           = 1000
Epochs                            = 10000

mainLogger.info('Normalizing...')
prepTool = Normalize(Norm='totalEnergy')
Data     = prepTool(Data)

del objDataFromFile

from FastNetTool.TrainJob import TrainJob
trainjob = TrainJob()

for sort in range(SortMin, SortMax+1):
  mainLogger.info('Number of neurons %d', Neuron)
  mainLogger.info('Sort number %d', sort)
  mainLogger.info('InitBounds %r', InitBounds)
  trainjob( Data, Target, Cross, 
                  neuron=Neuron, 
                  sort=sort,
                  initBounds=InitBounds, 
                  epochs=Epochs,
                  showEvo=ShowEvo, 
                  output=output,
                  doMultiStop=DoMultiStop,
                  prepTools=[],
                  doPerf=False)




