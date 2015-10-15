#!/usr/bin/env python

from RingerCore.Logger import LoggingLevel
from TuningTools.CrossValidStat  import CrossValidStatAnalysis, \
                                        ReferenceBenchmark

path = '/afs/cern.ch/work/w/wsfreund/private/user.wsfreund.tuned.mc14_13TeV.147406.129160.sgn.truth.bkg.truth.offline_rings_bugfix.eta.indep.et.indep_t0005_tunedDiscrXYZ.tgz.44720370'
stat = CrossValidStatAnalysis( path, level = LoggingLevel.DEBUG )

# Define the reference benchmarks
Loose_LH_Pd  = ReferenceBenchmark( "Loose_LH_Pd",  "Pd", refVal = 0.91292190 )
Loose_LH_Pf  = ReferenceBenchmark( "Loose_LH_Pf",  "Pf", refVal = 0.01121407 )
Medium_LH_Pd = ReferenceBenchmark( "Medium_LH_Pd", "Pd", refVal = 0.86599006 )
Medium_MaxSP = ReferenceBenchmark( "Medium_MaxSP", "SP"                      )
Medium_LH_Pf = ReferenceBenchmark( "Medium_LH_Pf", "Pf", refVal = 0.00397370 )
Tight_LH_Pd  = ReferenceBenchmark( "Tight_LH_Pd",  "Pd", refVal = 0.78181517 )
Tight_LH_Pf  = ReferenceBenchmark( "Tight_LH_Pf",  "Pf", refVal = 0.00206794 )

# Add them to a list:
refBenchmarkList = [Loose_LH_Pd,  Loose_LH_Pf,
                    Medium_LH_Pd, Medium_MaxSP, Medium_LH_Pf,
                    Tight_LH_Pd,  Tight_LH_Pf]

# Run the cross-validation analysis:
stat( refBenchmarkList )

#stat.save_network('sp',5,46,75,-0.735000,'network.tight.n100_5_1.'+data+'.pic')
#stat.save_network('sp',18,15,45,-0.030001,'network.medium.n100_18_1.'+data+'.pic')
#stat.save_network('sp',7,39,22,0.449999,'network.loose.n100_7_1.'+data+'.pic')
