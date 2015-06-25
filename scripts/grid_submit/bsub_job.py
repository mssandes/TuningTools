#!/usr/bin/env python

import sys
import os
import pickle
import numpy as np
from FastNetTool.CrossValid import CrossValid
from FastNetTool.util       import include, normalizeSumRow, reshape, load


#DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid.data.ringer.npy'
DatasetLocationInput              = sys.arg[1] 
Neuron                            = int(sys.arg[2])
Sort                              = int(sys.arg[3])

print 'DatasetLocationInput %s' % DatasetLocationInput
print 'Number of neurons %d' % Neuron
print 'Sort number %d' % Sort

print 'Opening data and normalize ...'
objDataFromFile                   = np.load( DatasetLocationInput )

#Job option configuration
Data                              = normalizeSumRow( reshape( objDataFromFile[0] ) )
Target                            = reshape(objDataFromFile[1])
Cross                             = CrossValid( Target, nSorts=50, nBoxes=10, nTrain=6, nValid=4)
OutputName                        = 'RingerOfflineNN_ZeeTruth_JF17Truth'
DoMultiStop                       = True
ShowEvo                           = 20
Epochs                            = 3000


#job configuration
Inits                             = 100

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


