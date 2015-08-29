#!/usr/bin/env python

from timeit import default_timer as timer

start = timer()

DatasetLocationInput = '/afs/cern.ch/work/j/jodafons/public/mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH.npy'

try:
  from FastNetTool.Logger import Logger, LoggingLevel
  from FastNetTool.TuningJob import TuningJob
  mainLogger = Logger.getModuleLogger(__name__)

  tuningJob = TuningJob()

  mainLogger.info("Entering main job.")

  tuningJob( DatasetLocationInput, 
             neuronBoundsCol = [5, 5], 
             sortBoundsCol = [1, 2],
             initBoundsCol = 500, 
             epochs = 100,
             showEvo = 0, 
             doMultiStop = True,
             doPerf = False,
             seed = 0,
             crossValidSeed = 66,
             level = LoggingLevel.INFO )

  mainLogger.info("Finished.")
except ImportError,e:

  import sys
  import os
  import pickle
  import numpy as np
  from FastNetTool.Preprocess import Normalize, RingerRp
  from FastNetTool.CrossValid import CrossValid
  from FastNetTool.util       import include, normalizeSumRow, reshape, load, getModuleLogger

  mainLogger = getModuleLogger(__name__)
  mainLogger.info('Opening data...')
  objDataFromFile                   = np.load( DatasetLocationInput )
  #Job option configuration
  Data                              = reshape( objDataFromFile[0] ) 
  Target                            = reshape( objDataFromFile[1] )
  preTool                           = Normalize( Norm='totalEnergy' )
  Data                              = preTool( Data )
  Cross                             = CrossValid( nSorts=50, nBoxes=10, nTrain=6, nValid=4, seed = 66)
  OutputName                        = 'output'
  DoMultiStop                       = True
  ShowEvo                           = 99
  Epochs                            = 100

  #job configuration
  Inits                             = 500
  minNeuron                         = 5
  maxNeuron                         = 5

  del objDataFromFile
  from FastNetTool.TrainJob import TrainJob
  trainjob = TrainJob()

  for neuron in range( minNeuron, maxNeuron+1):
    trainjob( Data, Target, Cross, 
                    neuron=neuron, 
                    sort=1,
                    initBounds=Inits, 
                    epochs=Epochs,
                    showEvo=ShowEvo, 
                    doMultiStop=DoMultiStop,
                    prepTools=[],
                    doPerf=False,
                    seed=0)

end = timer()
print(end - start)      
