#/usr/bin/env python
from timeit import default_timer as timer
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.TuningJob import TuningJob
from TuningTools.PreProc import *
import logging

start = timer()
DatasetLocationInput = 'data/files/data17_13TeV.AllPeriods.sgn.probes_EGAM1.bkg.vetoProbes_EGAM7_et4_eta0.npz'

#ppCol = PreProcChain( RingerEtaMu(pileupThreshold=100, etamin=0.0, etamax=0.8) ) 
ppCol = PreProcChain( Norm1() ) 
#ppCol = PreProcChain( RingerRp(alpha=0.5,beta=0.5) ) 
from TuningTools.TuningJob import fixPPCol
#ppCol = fixPPCol(ppCol)
from TuningTools.coreDef      import coreConf, TuningToolCores
coreConf.conf = TuningToolCores.FastNet
from TuningTools.TuningJob    import ReferenceBenchmark,   ReferenceBenchmarkCollection, BatchSizeMethod



tuningJob = TuningJob()
tuningJob( DatasetLocationInput, 
           neuronBoundsCol = [6, 8], 
           sortBoundsCol = [0, 10],
           initBoundsCol =10, 
           epochs = 2000,
           batchSize=20000,
           #batchMethod=BatchSizeMethod.HalfSizeSignalClass,
           showEvo = 100,
           doMultiStop = True,
           maxFail = 100,
           ppCol = ppCol,
           level = 10,
           etBins = 4,
           etaBins = 0,
           crossValidFile= '/home/jodafons/Public/ringer/root/TuningTools/scripts/analysis_scripts/Trigger_201711XX_data17_v7/data/files/crossValid.pic.gz',
           #ppFile='ppFile.pic.gz',
           #confFileList='config.n5to20.jackKnife.inits_100by100/job.hn0009.s0000.il0000.iu0099.pic.gz',
           refFile='data/files/data17_13TeV.allPeriods.tight_effs.npz',
           )


end = timer()

print 'execution time is: ', (end - start)      
