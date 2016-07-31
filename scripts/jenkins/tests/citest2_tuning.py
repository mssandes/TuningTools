#!/usr/bin/env python
from timeit import default_timer as timer
from RingerCore import Logger, LoggingLevel
from TuningTools import TuningJob
from TuningTools.TuningJob import BatchSizeMethod
from TuningTools.PreProc import *
import logging

start = timer()
DatasetLocationInput = 'data/tuningData_citest1.npz'
tuningJob = TuningJob()
tuningJob( DatasetLocationInput, 
           #neuronBoundsCol = [5, 5], 
           #sortBoundsCol = [0, 1],
           #initBoundsCol = 10, 
           etBins = 0,
           etaBins = 0,
           confFileList = 'data/config_citest0.pic.gz',
           crossValidFile = 'data/crossValid_citest0.pic.gz',
           epochs = 5000,
           showEvo = 0,
           doMultiStop = True,
           maxFail = 100,
           #batchSize = 10,
           #batchMethod = BatchSizeMethod.OneSample,
           #seed = 0,
           ppFile = 'data/ppFile_citest0.pic.gz',
           #crossValidSeed = 66,
           level = 20
           )

end = timer()

print 'execution time is: ', (end - start)     
import sys,os
sys.exit(os.EX_OK) # code 0, all ok
