#!/usr/bin/env python

from RingerCore.Logger import LoggingLevel
from TuningTools.CrossValidStat  import *

path = '/afs/cern.ch/work/j/jodafons/public/user_jodafons_nn_mc14_13TeV_147406_129160_sgn_Off_CitID_bkg_truth_e24_medium_L1EM20VH_t0008_fastnet_tunedXYZ'
stat = CrossValidStatAnalysis( path, level = LoggingLevel.DEBUG )

loose  = ReferenceBenchmark( "loose",  "Pd", refVal = 0.9816 )
medium = ReferenceBenchmark( "medium", "SP" )
tight  = ReferenceBenchmark( "tight",  "Pf", refVal = 0.1269 )
refBenchmarkList = [loose, medium, tight]

stat( refBenchmarkList )
stat.exportBestDiscriminator(refBenchmarkList, configList=[7,11,5])


#stat.save_network('sp',5,46,75,-0.735000,'network.tight.n100_5_1.'+data+'.pic')
#stat.save_network('sp',18,15,45,-0.030001,'network.medium.n100_18_1.'+data+'.pic')
#stat.save_network('sp',7,39,22,0.449999,'network.loose.n100_7_1.'+data+'.pic')
