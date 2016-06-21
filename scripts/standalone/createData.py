#!/usr/bin/env python

from TuningTools.parsers import argparse, createDataParser, loggerParser, CreateDataNamespace
parser = argparse.ArgumentParser(add_help = False, 
                                 description = 'Create TuningTool data from PhysVal.',
                                 parents = [createDataParser, loggerParser])

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args( namespace = CreateDataNamespace() )
# Treat special argument
if len(args.reference) > 2:
  raise ValueError("--reference set to multiple values: %r", args.reference)
if len(args.reference) is 1:
  args.reference.append( args.reference[0] )
from RingerCore import Logger, LoggingLevel, printArgs, NotSet
logger = Logger.getModuleLogger( __name__, args.output_level )
if args.operation != 'Offline' and not args.treePath:
  ValueError("If operation is not set to Offline, it is needed to set the TreePath manually.")

printArgs( args, logger.debug )

crossVal = NotSet
if args.crossFile not in (None, NotSet):
  from TuningTools import CrossValidArchieve
  with CrossValidArchieve( args.crossFile ) as CVArchieve:
    crossVal = CVArchieve
  del CVArchieve

from TuningTools import createData
createData( args.sgnInputFiles, 
            args.bkgInputFiles,
            ringerOperation       = args.operation,
            referenceSgn          = args.reference[0],
            referenceBkg          = args.reference[1],
            treePath              = args.treePath,
            output                = args.output,
            l1EmClusCut           = args.l1EmClusCut,
            l2EtCut               = args.l2EtCut,
            offEtCut              = args.offEtCut,
            level                 = args.output_level,
            nClusters             = args.nClusters,
            getRatesOnly          = args.getRatesOnly,
            etBins                = args.etBins,
            etaBins               = args.etaBins,
            ringConfig            = args.ringConfig,
            extractDet            = args.extractDet,
            standardCaloVariables = args.standardCaloVariables,
            useTRT                = args.useTRT,
            crossVal              = crossVal,
          )

