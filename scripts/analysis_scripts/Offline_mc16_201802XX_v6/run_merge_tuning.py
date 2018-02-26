#/usr/bin/env python
from timeit import default_timer as timer
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.TuningJob import TuningJob
from TuningTools.PreProc import *
import logging

start = timer()
DatasetLocationInput = [
                  '/home/jodafons/Public/ringer/root/TuningTools/scripts/analysis_scripts/Offline_mc16_201802XX_v6/data/files/mc16calo_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.calo.wdatadrivenlh_et2_eta0.npz',
                  '/home/jodafons/Public/ringer/root/TuningTools/scripts/analysis_scripts/Offline_mc16_201802XX_v6/data/files/mc16track_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.track.wdatadrivenlh_et2_eta0.npz',
                  ]

expertPaths = [
                'data/crossval/test/crossValStat.calo_et2_eta0.pic.gz',
                'data/crossval/test/crossValStat.track_et2_eta0.pic.gz',
                ]

from TuningTools.TuningJob import fixPPCol
from TuningTools.coreDef      import coreConf, TuningToolCores
coreConf.conf = TuningToolCores.FastNet
from TuningTools.TuningJob    import ReferenceBenchmark,   ReferenceBenchmarkCollection, BatchSizeMethod
from RingerCore.Configure import Development
Development.set( True )


tuningJob = TuningJob()
tuningJob( DatasetLocationInput, 
           expertPaths = expertPaths,
           merge = True,
           neuronBoundsCol = [10, 10], 
           sortBoundsCol = [0, 10],
           initBoundsCol =2, 
           epochs = 200,
           showEvo = 10,
           doMultiStop = True,
           maxFail = 100,
           etBins = 2,
           etaBins = 0,
           operationPoint = 'Offline_LH_DataDriven2016_Rel21_Medium',
           crossValidFile= 'data/files/user.jodafons.crossValid.10sorts.pic.gz/crossValid.10sorts.pic.gz',
           refFile = 'data/files/mc16calo_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.calo.wdatadrivenlh_eff.npz',
           ppFile= 'data/files/ppFile.expertTrackSimpleNorm.pic.gz',
           )


end = timer()

print 'execution time is: ', (end - start)      
