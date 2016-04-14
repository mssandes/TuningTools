#!/usr/bin/env python

from TuningTools.parsers import argparse, loggerParser, LoggerNamespace, tuningJobParser

parser = argparse.ArgumentParser(add_help = False, 
                                 description = 'Tune discriminators using input data.',
                                 parents = [tuningJobParser, loggerParser])

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

# Retrieve parser args:
args = parser.parse_args( namespace = LoggerNamespace() )

from TuningTools import CrossValidStatAnalysis, GridJobFilter, TuningDataArchieve, \
                        ReferenceBenchmark

if not args.ref_name:
  raise ValueError("Attempted to run a job without reference name")

# Retrieve reference benchmark:
call_kw = {}
if args.perfFile is not None:
  refBenchmarkList = []
  # If user has specified a reference performance file:
  from itertools import product
  with TDArchieve as data:
    for etBin, etaBin in product( range( nEtBins if nEtBins is not None else 1 ),
                                  range( nEtaBins if nEtaBins is not None else 1 )):
      benchmarks = (data['signal_efficiencies'], data['background_efficiencies'])
      #cross_benchmarks = (TDArchieve['signal_cross_efficiencies'], TDArchieve['background_cross_efficiencies'])
      sigEff = data['signal_efficiencies']['EFCaloAccept'][etBin][etaBin]
      bkgEff = data['background_efficiencies']['EFCaloAccept'][etBin][etaBin]
      try:
        sigCrossEff = data['signal_cross_efficiencies']['EFCaloAccept'][etBin][etaBin]
        bkgCrossEff = data['background_cross_efficiencies']['EFCaloAccept'][etBin][etaBin]
      except KeyError:
        sigCrossEff = None; bkgCrossEff = None
      args = (sigEff, bkgEff, sigCrossEff, bkgCrossEff)
      Medium_LH_EFCalo_Pd = ReferenceBenchmark( args.ref_name + "_" + refLabel + "_Pd", "Pd", *args )
      Medium_MaxSP        = ReferenceBenchmark( args.ref_name + "_MaxSP", "SP", *args )
      Medium_LH_EFCalo_Pf = ReferenceBenchmark( args.ref_name + "_" + refLabel + "_Pf", "Pf", *args )
      references =  [ Medium_LH_EFCalo_Pd,
                      Medium_MaxSP,
                      Medium_LH_EFCalo_Pf ] 
      print ('Et:',etBin, 'eta:', etaBin), [ref.refVal for ref in references]
      refBenchmarkList.append( references )
  del data
  call_kw['refBenchmarkList'] += 


stat = CrossValidStatAnalysis( 
    '',
    binFilters = GridJobFilter,
    level = LoggingLevel.DEBUG,
    )

TDArchieve = TuningDataArchieve(args.perfFile)
nEtBins = TDArchieve.nEtBins()
nEtaBins = TDArchieve.nEtaBins()

stat( refBenchmarkList , outputName = 'FixET_Norm1_20.7.3.6' )

