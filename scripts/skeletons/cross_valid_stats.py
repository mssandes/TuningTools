#!/usr/bin/env python

from RingerCore.Logger import LoggingLevel
from TuningTools.CrossValidStat import CrossValidStatAnalysis, GridJobFilter

stat = CrossValidStatAnalysis( 
    '/afs/cern.ch/work/w/wsfreund/public/Online/20.7.3.6_fixET/user.nbullacr.nn.mc14_13TeV.147406.129160.sgn.offLH.bkg.truth.trig.ef.e24_lhmedium_nod0_binned_t0001_tunedDiscrXYZ.tgz',
    #'/afs/cern.ch/work/w/wsfreund/public/Online/20.7.3.6_fixET/test',
    binFilters = GridJobFilter,
    level = LoggingLevel.DEBUG,
    )

from TuningTools.CreateData import TuningDataArchieve
TDArchieve = TuningDataArchieve('/afs/cern.ch/work/j/jodafons/public/Tuning2016/TuningConfig/mc14_13TeV.147406.129160.sgn.offLikelihood.bkg.truth.trig.e24_lhmedium_nod0_l1etcut20_l2etcut19_efetcut24_binned.pic.npz')
nEtBins = TDArchieve.nEtBins()
nEtaBins = TDArchieve.nEtaBins()
refBenchmarkList = []
from itertools import product
from TuningTools.TuningJob import ReferenceBenchmark
with TDArchieve as data:
  for etBin, etaBin in product( range( nEtBins if nEtBins is not None else 1 ),
                                range( nEtaBins if nEtaBins is not None else 1 )):
    if etBin == 0 and etaBin == 0: continue
    benchmarks = (data['signal_efficiencies'], data['background_efficiencies'])
    #cross_benchmarks = (TDArchieve['signal_cross_efficiencies'], TDArchieve['background_cross_efficiencies'])
    Medium_LH_EFCalo_Pd = ReferenceBenchmark( "Medium_LH_EFCalo_Pd", "Pd", refVal = benchmarks[0]['EFCaloAccept'][etBin][etaBin].efficiency() / 100. )
    Medium_MaxSP        = ReferenceBenchmark( "Medium_MaxSP", "SP"                                                               )
    Medium_LH_EFCalo_Pf = ReferenceBenchmark( "Medium_LH_EFCalo_Pf", "Pf", refVal = benchmarks[1]['EFCaloAccept'][etBin][etaBin].efficiency() / 100. )
    references =  [ Medium_LH_EFCalo_Pd,
                    Medium_MaxSP,
                    Medium_LH_EFCalo_Pf ] 
    print ('Et:',etBin, 'eta:', etaBin), [ref.refVal for ref in references]
    refBenchmarkList.append( references)
del data

stat( refBenchmarkList , outputName = 'FixET_Norm1_20.7.3.6' )
