#!/usr/bin/python

import sys
import os
import pickle
from FastNetTool.CrossValid import *
from FastNetTool.util       import *
from FastNetTool.defines    import *

DatasetLocationInput              = '/afs/cern.ch/user/j/jodafons/public/valid_ringer_sample.pic'


print 'openning data and normalize ...'

objectsFromFile                   = load( DatasetLocationInput )

#Job option configuration
Data                              = normalizeSumRow( objectsFromFile[0] )
Target                            = objectsFromFile[1]
CrossValidObject                  = objectsFromFile[2]
OutputName                        = 'output.save'
MonitoringLevel                   = INFO
NumberOfInitsPerSort              = 1
NumberOfSortsPerConfigurationMin  = 1
NumberOfSortsPerConfigurationMax  = 1
NumberOfNeuronsInHiddenLayerMin   = 2
NumberOfNeuronsInHiddenLayerMax   = 2
DoMultiStops                      = True
ShowEvolution                     = 4
Epochs                            = 1000

include('../python/jobs/NeuralTrainingLoop_fastnet_topOption.py')



