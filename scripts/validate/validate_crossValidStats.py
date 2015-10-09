#!/usr/bin/env python

from RingerCore.Logger import LoggingLevel
from TuningTools.CrossValidStat  import *

#path = '/afs/cern.ch/work/j/jodafons/public/user_jodafons_nn_mc14_13TeV_147406_129160_sgn_Off_CitID_bkg_truth_e24_medium_L1EM20VH_t0008_fastnet_tunedXYZ'
path = '/afs/cern.ch/work/j/jodafons/news/user.jodafons.nn.mc14_13TeV.147406.sgn.Off_LH.129160.bkg.truth.l1_20.l2_19.e24_medium_L1EM18VH.indep_eta_et.t0007_tunedDiscrXYZ.tgz'
stat = CrossValidStatAnalysis( path, level = LoggingLevel.DEBUG )

cutId_loose  = ReferenceBenchmark( "e24_medium_L1EM20VH_L2Calo_loose",  "Pd", refVal = 0.981 )
cutId_tight  = ReferenceBenchmark( "e24_medium_L1EM20VH_L2Calo_tight",  "Pf", refVal = 0.137 )
lh_loose     = ReferenceBenchmark( "e24_lhmedium_L1EM20VH_L2EFCalo_loose","Pd", refVal = 0.9506 )
lh_tight     = ReferenceBenchmark( "e24_lhmedium_L1EM20VH_L2EFCalo_tight","Pf", refVal = 0.058 )
medium       = ReferenceBenchmark( "medium", "SP" )

refBenchmarkList = [
                    cutId_loose, 
                    cutId_tight,
                    lh_loose,
                    lh_tight,
                    medium, 
                    ]


#stat( refBenchmarkList )
stat.exportBestDiscriminator(refBenchmarkList, configList=[13,13,13,18,18])


#stat.save_network('sp',5,46,75,-0.735000,'network.tight.n100_5_1.'+data+'.pic')
#stat.save_network('sp',18,15,45,-0.030001,'network.medium.n100_18_1.'+data+'.pic')
#stat.save_network('sp',7,39,22,0.449999,'network.loose.n100_7_1.'+data+'.pic')
