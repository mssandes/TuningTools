#/usr/bin/env python
from timeit import default_timer as timer
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.TuningJob import TuningJob
from TuningTools.PreProc import *
import logging

start = timer()
DatasetLocationInput = 'data/files/data17_13TeV_patterns/data17_13TeV.allPeriods.sgn.probes_EGAM1.bkg.vetoProbes_EGAM7_et0_eta2.npz'

ppCol = PreProcChain( RingerEtaMu() ) 
#ppCol = PreProcChain( RingerRp(alpha=0.5,beta=0.5) ) 
from TuningTools.TuningJob import fixPPCol
#ppCol = fixPPCol(ppCol)


tuningJob = TuningJob()
tuningJob( DatasetLocationInput, 
           neuronBoundsCol = [5, 5], 
           sortBoundsCol = [0, 10],
           initBoundsCol = 10, 
           epochs = 5000,
           showEvo = 10,
           doMultiStop = True,
           maxFail = 100,
           #ppCol = ppCol,
           level = 10,
           etBins = 0,
           etaBins = 2,
           #crossValidFile='crossValid.pic.gz',
           #ppFile='ppFile_norm1.pic.gz',
           #confFileList='config.n5to20.jackKnife.inits_100by100/job.hn0009.s0000.il0000.iu0099.pic.gz',
           refFile='data/files/data17_13TeV_effs/data17_13TeV.allPeriods.medium_effs.npz',
           )


end = timer()

print 'execution time is: ', (end - start)      
