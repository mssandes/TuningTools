#!/usr/bin/env python

from timeit import default_timer as timer

from RingerCore.Logger import Logger, LoggingLevel
mainLogger = Logger.getModuleLogger(__name__)
mainLogger.info("Entering main job.")

start = timer()

basepath = '/afs/cern.ch/work/j/jodafons/public'
DatasetLocationInput = basepath+'/mc14_13TeV.147406.129160.sgn.offLH.bkg.truth.trig.l1cluscut_20.l2etcut_19.e24_medium_L1EM18VH.npz'

from TuningTools.TuningJob import TuningJob

tuningJob = TuningJob()

tuningJob( DatasetLocationInput, 
           confFileList = basepath+'/user.wsfreund.config.nn5to20_sorts50_1by1_inits100_100by100/job.hn0015.s0040.il0000.iu0099.pic.gz',
           ppFileList = basepath+'/user.wsfreund.Norm1/ppFile_pp_Norm1.pic.gz',
           crossValidFile = basepath+'/user.wsfreund.CrossValid.50Sorts.seed_0/crossValid.pic.gz',
           epochs = 10,
           showEvo = 25, 
           doMultiStop = True,
           doPerf = True,
           compress = False,
           level = LoggingLevel.DEBUG )

mainLogger.info("Finished.")

end = timer()

print(end - start)      
