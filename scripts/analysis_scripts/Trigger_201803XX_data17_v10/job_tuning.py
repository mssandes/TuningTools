#/usr/bin/env python
from timeit import default_timer as timer
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.TuningJob import TuningJob
from TuningTools.PreProc import *
from TuningTools.TuningJob import fixPPCol
from TuningTools.coreDef      import coreConf, TuningToolCores
from TuningTools.TuningJob    import ReferenceBenchmark,   ReferenceBenchmarkCollection, BatchSizeMethod
from RingerCore.Configure import Development
import logging
import argparse


start = timer()
Development.set( True )
coreConf.set(TuningToolCores.keras)



mainLogger = Logger.getModuleLogger("job")
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

parser.add_argument('-d','--data', action='store', 
    dest='data', required = True,
    help = "The input tuning files.")





import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
args = parser.parse_args()

d = args.data
d=d.replace('.npz','')
d=d.split('_')
end=len(d)-1
et = int(d[end-1].replace('et',''))
eta = int(d[end].replace('eta',''))
print 'Et = ',et, ' Eta = ',eta

tuningJob = TuningJob()
tuningJob( args.data, 
           epochs = 1,
           batchSize= 1024*4,
           showEvo = 1,
           level = 9,
           etBins = et,
           etaBins = eta,
           doMultiStop=False,
           confFileList = 'data_cern/files/cnn_config/job.hn0001.sl0000.su0009.i0000.pic.gz',
           refFile='data_cern/files/data17_13TeV.allPeriods.medium_effs.GRL_v97.npz',
           )
end = timer()

print 'execution time is: ', (end - start)      
