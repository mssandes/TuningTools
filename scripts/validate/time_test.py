#!/usr/bin/env python

from timeit import default_timer as timer

start = timer()

DatasetLocationInput = '/afs/cern.ch/work/j/jodafons/public/validate_tuningtool/mc14_13TeV.147406.129160.sgn.offLikelihood.bkg.truth.trig.e24_lhmedium_L1EM20VH_etBin_0_etaBin_0.npz'

#try:
from RingerCore.Logger import Logger, LoggingLevel
mainLogger = Logger.getModuleLogger(__name__)
mainLogger.info("Entering main job.")

from TuningTools.TuningJob import TuningJob
tuningJob = TuningJob()

from TuningTools.PreProc import *

tuningJob( DatasetLocationInput, 
           neuronBoundsCol = [5, 5], 
           sortBoundsCol = [0, 2],
           initBoundsCol = 2, 
           epochs = 5,
           showEvo = 50,
           algorithmName= 'rprop',
           #doMultiStop = True,
           #doPerf = True,
           #seed = 0,
           ppCol = PreProcCollection( PreProcChain( MapStd() ) ),
           crossValidSeed = 66,
           level = LoggingLevel.DEBUG )

mainLogger.info("Finished.")

end = timer()

print 'execution time is: ', (end - start)      
