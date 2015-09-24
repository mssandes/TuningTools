#!/usr/bin/env python

from timeit import default_timer as timer

from RingerCore.Logger import Logger, LoggingLevel
mainLogger = Logger.getModuleLogger(__name__)
mainLogger.info("Entering main job.")

start = timer()

DatasetLocationInput = '/afs/cern.ch/work/j/jodafons/public/mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH.npy'

from TuningTools.TuningJob import TuningJob

tuningJob = TuningJob()

tuningJob( DatasetLocationInput, 
           confFileList = '$WORK/private/jobConfig/job.hn0016.s0000.il0000.iu0004.pic.gz',
           ppFileList = '$WORK/public/user.wsfreund.nn_hn16_sorts50_1by1_inits100_5by5_Sort_Seed0_Norm1_ppFile.41686299/user.wsfreund.6419093._000001.ppFileXYZ.tgz',
           crossValidFile = '$WORK/public/user.wsfreund.nn_hn16_sorts50_1by1_inits100_5by5_Sort_Seed0_Norm1_CrossValid.41686298/user.wsfreund.6419093._000001.crossValidXYZ.tgz',
           epochs = 10,
           showEvo = 25, 
           doMultiStop = True,
           doPerf = True,
           compress = False,
           level = LoggingLevel.DEBUG )

mainLogger.info("Finished.")

end = timer()

print(end - start)      
