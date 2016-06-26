#!/usr/bin/env python

from RingerCore import csvStr2List, str_to_class, NotSet, BooleanStr

from TuningTools.parsers import argparse, loggerParser, \
                                crossValStatsJobParser, CrossValidStatNamespace

from TuningTools import CrossValidStatAnalysis, GridJobFilter, TuningDataArchieve, \
                        ReferenceBenchmark, ReferenceBenchmarkCollection

parser = argparse.ArgumentParser(add_help = False, 
                                 description = 'Retrieve performance information from the Cross-Validation method.',
                                 parents = [crossValStatsJobParser, loggerParser])

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

## Retrieve parser args:
args = parser.parse_args( namespace = CrossValidStatNamespace() )
## Treat special arguments
# Check if binFilters is a class
if args.binFilters is not NotSet:
  try:
    args.binFilters = str_to_class( "TuningTools.CrossValidStat", args.binFilters)
  except TypeError:
    args.binFilters = csvStr2List( args.binFilters )

# Retrieve reference benchmark:
call_kw = {}
if args.perfFile is not None:
  # If user has specified a reference performance file:
  TDArchieve = TuningDataArchieve(args.perfFile)
  nEtBins = TDArchieve.nEtBins()
  nEtaBins = TDArchieve.nEtaBins()
  refBenchmarkCol = ReferenceBenchmarkCollection([])
  from itertools import product
  with TDArchieve as data:
    if args.operation is None:
      args.operation = data['operation']
    from TuningTools.ReadData import RingerOperation
    args.operation = RingerOperation.retrieve(args.operation)
    refLabel = RingerOperation.branchName(args.operation)
    for etBin, etaBin in product( range( nEtBins if nEtBins is not None else 1 ),
                                  range( nEtaBins if nEtaBins is not None else 1 )):
      # Make sure that operation is valid:
      benchmarks = (data['signal_efficiencies'][refLabel][etBin][etaBin], 
                    data['background_efficiencies'][refLabel][etBin][etaBin])
      try:
        cross_benchmarks = (data['signal_cross_efficiencies'][refLabel][etBin][etaBin], 
                            data['background_cross_efficiencies'][refLabel][etBin][etaBin])
      except KeyError:
        cross_benchmarks = (None, None)
      # Add the signal efficiency and background efficiency as goals to the
      # tuning wrapper:
      opRefs = [ReferenceBenchmark.SP, ReferenceBenchmark.Pd, ReferenceBenchmark.Pf]
      if benchmarks is None:
        raise RuntimeError("Couldn't access the benchmarks on efficiency file.")
      refBenchmarkList = ReferenceBenchmarkCollection([])
      for ref in opRefs: 
        refArgs = []
        refArgs.extend( benchmarks )
        if cross_benchmarks is not None:
          refArgs.extend( cross_benchmarks )
        refBenchmarkList.append( ReferenceBenchmark( "OperationPoint_" + refLabel.replace('Accept','') + "_" 
                                                     + ReferenceBenchmark.tostring( ref ), 
                                                     ref, *refArgs ) )
      refBenchmarkCol.append( refBenchmarkList )
  del data
  call_kw['refBenchmarkList'] = refBenchmarkCol


stat = CrossValidStatAnalysis( 
    args.discrFiles
    , binFilters = args.binFilters
    , binFilterIdxs = args.binFilterIdx
    , monitoringFileName = args.monitoringFileName
    , level = args.output_level
    )

stat(
    outputName = args.outputFileBase
    , toMatlab = args.doMatlab
    , debug    = args.debug
    , **call_kw
    )
