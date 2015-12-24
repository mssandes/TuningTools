#!/usr/bin/env python

from timeit import default_timer as timer

start = timer()

DatasetLocationInput = '/afs/cern.ch/work/j/jodafons/public/validate_tuningtool/mc14_13TeV.147406.129160.sgn.offLikelihood.bkg.truth.trig.e24_lhmedium_L1EM20VH_etBin_0_etaBin_0.npz'

#try:
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.TuningJob import TuningJob
from TuningTools.PreProc import *
mainLogger = Logger.getModuleLogger(__name__)

tuningJob = TuningJob()

mainLogger.info("Entering main job.")

tuningJob( DatasetLocationInput, 
           neuronBoundsCol = [5, 5], 
           sortBoundsCol = [0, 2],
           initBoundsCol = 2, 
           epochs = 30,
           showEvo = 50, 
           #doMultiStop = True,
           #doPerf = True,
           #seed = 0,
           ppCol = PreProcCollection( PreProcChain( MapStd() ) ),
           crossValidSeed = 66,
           level = LoggingLevel.DEBUG )

mainLogger.info("Finished.")
#except:
  #import sys
  #import os
  #import pickle
  #import numpy as np
  #from TuningTools.Preprocess import Normalize, RingerRp
  #from TuningTools.CrossValid import CrossValid
  #from RingerCore.util       import include, normalizeSumRow, reshape, load, getModuleLogger

  #mainLogger = getModuleLogger(__name__)
  #mainLogger.info('Opening data...')
  #objDataFromFile                   = np.load( DatasetLocationInput )
  ##Job option configuration
  #Data                              = reshape( objDataFromFile[0] ) 
  #Target                            = reshape( objDataFromFile[1] )
  #preTool                           = Normalize( Norm='totalEnergy' )
  #Data                              = preTool( Data )
  #Cross                             = CrossValid( nSorts=50, nBoxes=10, nTrain=6, nValid=4, seed = 66)
  #OutputName                        = 'output'
  #DoMultiStop                       = True
  #ShowEvo                           = 99
  #Epochs                            = 100

  ##job configuration
  #Inits                             = 500
  #minNeuron                         = 5
  #maxNeuron                         = 5

  #del objDataFromFile
  #from TuningTools.TrainJob import TrainJob
  #trainjob = TrainJob()

  #for neuron in range( minNeuron, maxNeuron+1):
  #  trainjob( Data, Target, Cross, 
  #                  neuron=neuron, 
  #                  sort=1,
  #                  initBounds=Inits, 
  #                  epochs=Epochs,
  #                  showEvo=ShowEvo, 
  #                  #doMultiStop=DoMultiStop,
  #                  prepTools=[],
  #                  #doPerf=False,
  #                  #seed=0,
  #                  )

end = timer()
print(end - start)      
