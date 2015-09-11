#!/usr/bin/env python

import sys
import os
import pickle
import numpy as np
from TuningTools.Preprocess import Normalize, RingerRp
from TuningTools.CrossValid import CrossValid
from RingerCore.util       import include, normalizeSumRow, reshape, load

DatasetLocationInput              ='/afs/cern.ch/work/j/jodafons/public/mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH.npy'
#DatasetLocationInput              ='/afs/cern.ch/work/w/wsfreund/public/mc14_13TeV.147406.129160.sgn.truth.bkg.truth.off.npy'

from RingerCore.Logger import Logger
mainLogger = Logger.getModuleLogger(__name__)

mainLogger.info('Opening data...')
objDataFromFile                   = np.load( DatasetLocationInput )
#Job option configuration
Data                              = reshape( objDataFromFile[0] ) 
Target                            = reshape( objDataFromFile[1] )
Cross                             = CrossValid( nSorts=50, nBoxes=10, nTrain=6, nValid=4)
OutputName                        = 'output'
DoMultiStop                       = True
ShowEvo                           = 1
Epochs                            = 7

#job configuration
Inits                             = 1
minNeuron                         = 5
maxNeuron                         = 5
alpha                             = 1


del objDataFromFile
from TuningTools.TrainJob import TrainJob
trainjob = TrainJob()



for beta in np.arange(10)/10:
  for neuron in range( minNeuron, maxNeuron+1):
    trainjob( Data, Target, Cross, 
                    neuron=neuron, 
                    sort=0,
                    initBounds=Inits, 
                    epochs=Epochs,
                    showEvo=ShowEvo, 
                    output=OutputName,
                    doMultiStop=DoMultiStop,
                    prepTools=[ RingerRp( alpha=alpha, beta=beta) ],
                    doPerf=True)

    word = trainjob.get_output_filename()
    prefix = ('.alpha%04d.beta%4d.pic') % (alpha,beta)
    os.rename(word,word[0:len(word)-4]+prefix)

  
