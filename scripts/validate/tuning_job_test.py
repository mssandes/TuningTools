#!/usr/bin/env python

from timeit import default_timer as timer

from FastNetTool.Logger import Logger, LoggingLevel
mainLogger = Logger.getModuleLogger(__name__)
mainLogger.info("Entering main job.")

start = timer()

DatasetLocationInput = '/afs/cern.ch/work/j/jodafons/public/mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH.npy'

from FastNetTool.TuningJob import TuningJob

tuningJob = TuningJob()

tuningJob( DatasetLocationInput, 
           confFileList = '$HOME/public/TrigCaloRingerAnalysisPackages/root/FastNetTool/scripts/standalone/jobConfig/job.hn0005.s0000.il0000.iu0001.pic.tgz,~/public/TrigCaloRingerAnalysisPackages/root/FastNetTool/scripts/standalone/jobConfig/job.hn0005.s0001.il0000.iu0001.pic.tgz',
           ppFileList = '$HOME/public/TrigCaloRingerAnalysisPackages/root/FastNetTool/scripts/standalone/ppFile_pp_Norm1.pic.tgz',
           crossValidFile = '$HOME/public/TrigCaloRingerAnalysisPackages/root/FastNetTool/scripts/standalone/crossValid.pic.tgz',
           epochs = 1000,
           showEvo = 25, 
           doMultiStop = True,
           doPerf = True,
           level = LoggingLevel.DEBUG )

mainLogger.info("Finished.")

end = timer()

print(end - start)      
