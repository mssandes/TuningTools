#!/usr/bin/env python

import sys
import os
import pickle
import numpy as np
from FastNetTool.CrossValid import CrossValid
from FastNetTool.util       import include, normalizeSumRow, reshape, load, getModuleLogger

DatasetLocationInput              ='/afs/cern.ch/work/w/wsfreund/public/mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH.npy'
JobConfigLocationInput            = '/afs/cern.ch/user/w/wsfreund/public/job.n0019.sl0000.su0001.il0000.iu0001.pic'


mainLogger = getModuleLogger(__name__)
mainLogger.info('Opening data...')
objDataFromFile                   = np.load( DatasetLocationInput )

Data                              = normalizeSumRow( reshape( objDataFromFile[0] ) )
Target                            = reshape(objDataFromFile[1])
JobConfiguration                  = load(JobConfigLocationInput)
Neuron                            = JobConfiguration[0]
SortMin                           = JobConfiguration[1][0]
SortMax                           = JobConfiguration[1][1]
InitBounds                        = JobConfiguration[2]
Cross                             = JobConfiguration[3]
OutputName                        = 'train.mc14_13TeV.129160.147406'
DoMultiStop                       = True
ShowEvo                           = 4
Epochs                            = 1000

from FastNetTool.TrainJob import TrainJob
trainjob = TrainJob()

for sort in range( SortMin, SortMax+1):
  trainjob( Data, Target, Cross, 
                neuron=Neuron, 
                sort=sort,
                initBounds=InitBounds, 
                epochs=Epochs,
                showEvo=ShowEvo, 
                output=OutputName,
                doMultiStop=DoMultiStop,
                doPerf=False)

