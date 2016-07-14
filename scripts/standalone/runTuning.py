#!/usr/bin/env python

from TuningTools.parsers import argparse, loggerParser, LoggerNamespace, tuningJobParser
parser = argparse.ArgumentParser(description = 'Tune discriminators using input data.',
                                 parents = [tuningJobParser, loggerParser])

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args( namespace = LoggerNamespace() )

## Treating special args:
# Configuration
conf_kw = {}
if args.neuronBounds is not None: conf_kw['neuronBoundsCol'] = args.neuronBounds
if args.sortBounds   is not None: conf_kw['sortBoundsCol']   = args.sortBounds
if args.initBounds   is not None: conf_kw['initBoundsCol']   = args.initBounds
if args.confFileList is not None: conf_kw['confFileList']    = args.confFileList
# Binning
from RingerCore import printArgs, NotSet, Logger, LoggingLevel
if not(args.et_bins is NotSet) and len(args.et_bins)  == 1: args.et_bins  = args.et_bins[0]
if not(args.eta_bins is NotSet) and len(args.eta_bins) == 1: args.eta_bins = args.eta_bins[0]

logger = Logger.getModuleLogger( __name__, args.output_level )

printArgs( args, logger.debug )

compress = False if args.no_compress else True

# Submit job:
from TuningTools import TuningJob
tuningJob = TuningJob()
tuningJob( 
           args.data, 
           level          = args.output_level,
					 compress       = compress,
					 outputFileBase = args.outputFileBase,
           # Cross validation args
					 crossValidFile = args.crossFile,
           # Pre Processing
           ppFile         = args.ppFile,
           # Binning configuration
           etBins         = args.et_bins,
           etaBins        = args.eta_bins,
					 # Tuning CORE args
           showEvo        = args.show_evo,
           maxFail        = args.max_fail,
           epochs         = args.epochs,
           doPerf         = args.do_perf,
           batchSize      = args.batch_size,
           # ExMachina CORE args
           algorithmName  = args.algorithm_name,
           networkArch    = args.network_arch,
           costFunction   = args.cost_function,
           shuffle        = args.shuffle,
           seed           = args.seed,
           doMultiStop    = args.do_multi_stop,
					 # Looping configuration args
           **conf_kw
				 )
