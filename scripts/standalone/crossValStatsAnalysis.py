#!/usr/bin/env python

from RingerCore import csvStr2List

from TuningTools.parsers import argparse, loggerParser, \
                                crossValStatsJobParser, CrossValidStatNamespace

from TuningTools import CrossValidStatAnalysis, GridJobFilter, TuningDataArchieve, \
                        ReferenceBenchmark

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
try:
  args.binFilters = str_to_class(args.binFilters)
except TypeError:
  args.binFilters = csvStr2List(args.binFilters)
# Check boolean arguments
args.doMonitoring = BooleanStr.retrieve( args.doMonitoring )
args.doMatlab = BooleanStr.retrieve( args.doMatlab )

# Retrieve reference benchmark:
call_kw = {}
if args.perfFile is not None:
  TDArchieve = TuningDataArchieve(args.perfFile)
  nEtBins = TDArchieve.nEtBins()
  nEtaBins = TDArchieve.nEtaBins()

  if not args.ref_name:
    raise ValueError("Attempted to run a job without reference name")

  refBenchmarkCol = ReferenceBenchmarkCollection([])
  # If user has specified a reference performance file:
  from itertools import product
  with TDArchieve as data:
    for etBin, etaBin in product( range( nEtBins if nEtBins is not None else 1 ),
                                  range( nEtaBins if nEtaBins is not None else 1 )):
      try:
        from TuningTools.FilterEvents import RingerOperation
        if args.operation is None:
          args.operation = TDArchieve['operation']
        # Make sure that operation is valid:
        args.operation = RingerOperation.tostring( RingerOperation.retrieve(args.operation) )
        refLabel = RingerOperation.branchName(args.operation)
        benchmarks = (TDArchieve['signal_efficiencies'][refLabel][etBin][etaBin], 
                      TDArchieve['background_efficiencies'][refLabel][etBin][etaBin])
        try:
          cross_benchmarks = (TDArchieve['signal_cross_efficiencies'][refLabel][etBin][etaBin], 
                              TDArchieve['background_cross_efficiencies'][refLabel][etBin][etaBin])
        except KeyError:
          cross_benchmarks = None
      except KeyError as e:
        operation = None
        benchmarks = None
        cross_benchmarks = None
      # Add the signal efficiency and background efficiency as goals to the
      # tuning wrapper:
      opRefs = [ReferenceBenchmark.SP, ReferenceBenchmark.Pd, ReferenceBenchmark.Pf]
      if benchmarks is None:
        raise RuntimeError("Couldn't access the benchmarks on efficiency file.")
      refBenchmarkList = ReferenceBenchmarkCollection([])
      for ref in opRefs: 
        args = []
        args.extend( benchmarks )
        if cross_benchmarks is not None:
          args.extend( cross_benchmarks )
        refBenchmarkList.append( ReferenceBenchmark( "OperationPoint_" + refLabel.replace('Accept','') + "_" 
                                                     + ReferenceBenchmark.tostring( ref ), 
                                                     ref, *args ) )
      refBenchmarkCol += refBenchmarkList
  del data
  call_kw['refBenchmarkCol'] += refBenchmarkCol

stat = CrossValidStatAnalysis( 
    args.discrFiles,
    binFilters = args.binFilters,
    monitoringFileName = args.monitoringFileName,
    )

stat(
    outputName = args.outputFileBase
    , toMatlab = args.doMatlab
    , **call_kw
    )

