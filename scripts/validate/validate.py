#!/usr/bin/python

import sys
import os
import pickle
import numpy as np
from FastNetTool.CrossValid import CrossValid
from FastNetTool.util       import include, normalizeSumRow, reshape, load

#DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid.data.ringer.npy'
#JobConfigLocationInput            = '/afs/cern.ch/user/j/jodafons/public/valid.jobConfig.ringer.pic'
DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid_ringer_sample.pic'

print 'openning data and normalize ...'

objDataFromFile                   = load( DatasetLocationInput )
#objDataFromFile                   = np.load( DatasetLocationInput )
#Job option configuration
Data                              = objDataFromFile[0]
#Data                              = reshape(Data) 
Data                              = normalizeSumRow( Data )
#Target                            = reshape(objDataFromFile[1])
Target                            = objDataFromFile[1]
Cross                             = CrossValid( Target, nSorts=50, nBoxes=10, nTrain=6, nValid=4)
OutputName                        = 'output'
DoMultiStop                       = True
ShowEvo                           = 4
Epochs                            = 1000
Inits                             = 5
nSorts                            = 2
minNeuron                         = 10
maxNeuron                         = 10

#include('FastNetTool/TrainJob.py')
from FastNetTool.TrainJob import TrainJob
trainjob = TrainJob()

for neuron in range( minNeuron, maxNeuron+1):
  for sort in range(nSorts):
    trainjob( Data, Target, cross=Cross, 
                        neuron=neuron, 
                        sort=sort,
                        inits=Inits, 
                        epochs=Epochs,
                        showEvo=ShowEvo, 
                        output=OutputName,
                        doMultiStop=DoMultiStop,
                        doPerf=False)


