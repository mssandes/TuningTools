#/usr/bin/env python
from timeit import default_timer as timer
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.TuningJob import TuningJob
from TuningTools.PreProc import *
import logging

start = timer()
DatasetLocationInput = [
                  '/home/jodafons/Public/ringer/root/TuningTools/scripts/analysis_scripts/Offline_mc16_201802XX_v6/data/files/mc16calo_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.calo.wdatadrivenlh_et3_eta8.npz',
                  '/home/jodafons/Public/ringer/root/TuningTools/scripts/analysis_scripts/Offline_mc16_201802XX_v6/data/files/mc16calostd_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.calostd.wdatadrivenlh_et3_eta8.npz',
                  '/home/jodafons/Public/ringer/root/TuningTools/scripts/analysis_scripts/Offline_mc16_201802XX_v6/data/files/mc16track_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.track.wdatadrivenlh_et3_eta8.npz',
                  ]

expertPaths = [
                'data/precrossval/mc16a.zee.20M.jf17.20M.offline.binned.calo.wdatadrivenlh.v6.crossValStat/mc16a.zee.20M.jf17.20M.offline.binned.calo.wdatadrivenlh.v6.crossValStat_et3_eta8.pic.gz',
                'data/precrossval/mc16a.zee.20M.jf17.20M.offline.binned.calostd.wdatadrivenlh.v6.crossValStat/mc16a.zee.20M.jf17.20M.offline.binned.calostd.wdatadrivenlh.v6.crossValStat_et3_eta3.pic.gz',
                'data/precrossval/mc16a.zee.20M.jf17.20M.offline.binned.track.wdatadrivenlh.v6.crossValStat/mc16a.zee.20M.jf17.20M.offline.binned.track.wdatadrivenlh.v6.crossValStat_et3_eta8.pic.gz',
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
           neuronBoundsCol = [10, 10], 
           sortBoundsCol = [0, 10],
           initBoundsCol =2, 
           epochs = 5000,
           showEvo = 10,
           doMultiStop = True,
           maxFail = 100,
           etBins = 3,
           etaBins = 8,
           operationPoint = 'Offline_LH_DataDriven2016_Rel21_Medium',
           crossValidFile= 'data/files/user.jodafons.crossValid.10sorts.pic.gz/crossValid.10sorts.pic.gz',
           #refFile = 'data/files/mc16calo_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.calo.wdatadrivenlh_eff.npz',
           refFile = 'data/files/mc16calo_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.calo.wdatadrivenlh_eff.npz',
           #ppFile= '/home/jodafons/Public/ringer/root/TuningTools/scripts/analysis_scripts/Offline_mc16_201802XX_v6/data/files/user.jodafons.ppFile.ExpertNetworksSimpleNorm.pic.gz/ppFile.ExpertNetworksSimpleNorm.pic.gz',
           #ppFile= '/home/jodafons/Public/ringer/root/TuningTools/scripts/analysis_scripts/Offline_mc16_201802XX_v6/data/files/user.jodafons.ppFile.ExpertNetworksShowerShapeSimpleNorm.pic.gz/ppFile.ExpertNetworksShowerShapeSimpleNorm.pic.gz',
           ppFile= 'data/files/user.jodafons.ppFile.ExpertNetworksShowerShapeAndTrackSimpleNorm.pic.gz/ppFile.ExpertNetworksShowerShapeAndTrackSimpleNorm.pic.gz',
           )


end = timer()

print 'execution time is: ', (end - start)      
