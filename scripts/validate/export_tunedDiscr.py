#!/usr/bin/env python

from RingerCore.Logger import LoggingLevel
from RingerCore.FileIO import load
from TuningTools.CrossValidStat  import CrossValidStatAnalysis, \
                                        ReferenceBenchmark
from TuningTools.FilterEvents import RingerOperation


#path='/tmp/jodafons/news/tunedDiscr.mc14.sgn.offLH.bkg.truth.trig.l1cluscut_20.l2etcut_19.e24_medium/'+\
#'user.jodafons.nn.mc14_13TeV.147406.sgn.Off_LH.129160.bkg.truth.l1_20.l2_19.e24_medium_L1EM18VH.indep_eta_et.t0007_tunedDiscrXYZ.tgz/'

path =\
'/tmp/jodafons/news/tunedDiscr.mc14.sgn.offLH.bkg.truth.trig.l1cluscut_20.l2etcut_19.e24_medium.eta.et.dep/'+\
'tuned.mc14.sgn.offLH.bkg.truth.trig.l1cluscut_20.l2etcut_19.e24_medium_etaBin_3_etBin_2/'

stat = CrossValidStatAnalysis( path, level = LoggingLevel.DEBUG )

path = '/afs/cern.ch/work/j/jodafons/news/CrossValStat/'+\
    'crossValStat_etaBin_3_etBin_2.pic.gz'

print 'opening...'
crossValSummary = load(path)

# Add them to a list:
refBenchmarkNameList = [
                    'Medium_LH_L2Calo_Pd', 
                    'Medium_LH_EFCalo_Pd', 
                    'Medium_MaxSP', 
                    'Medium_LH_L2Calo_Pf',
                    'Medium_LH_EFCalo_Pf',
                    ]
CrossValidStatAnalysis.exportDiscrFiles(crossValSummary, RingerOperation.L2,
              baseName ='ringerTunedDiscr_e24_lhmdium_L1EM20VH_eta3_et2',
                                        refBenchmarkNameList=refBenchmarkNameList,
                                        configList=[5,5,5,5,5])




