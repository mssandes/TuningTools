#!/usr/bin/python

import sys
import os
import pickle
from FastNetTool.CrossValid import *
from FastNetTool.util       import *
from FastNetTool.FastNet    import Msg

DatasetLocationInput              = '/afs/cern.ch/user/w/wsfreund/public/dataset_ringer_offline_test.pic'

print 'openning data and normalize ...'

objectsFromFile                   = load( DatasetLocationInput )

#Job option configuration
Data                              = normalizeSumRow( objectsFromFile[0] )
Target                            = objectsFromFile[1]
CrossValidObject                  = objectsFromFile[2]
OutputName                        = 'output.save'
MonitoringLevel                   = Msg.INFO
NumberOfInitsPerSort              = 1
NumberOfSortsPerConfigurationMin  = 1
NumberOfSortsPerConfigurationMax  = 1
NumberOfNeuronsInHiddenLayerMin   = 2
NumberOfNeuronsInHiddenLayerMax   = 2
DoMultiStops                      = True
ShowEvolution                     = 4
Epochs                            = 1000

include('FastNetTool/jobs/NeuralTrainingLoop_fastnet_topOption.py')



