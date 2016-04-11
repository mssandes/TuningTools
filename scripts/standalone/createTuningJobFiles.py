#!/usr/bin/env python

from RingerCore import save
from TuningTools.parsers import argparse, loggerParser, LoggerNamespace, \
                                tuningJobFileParser, JobFileTypeCreation

parser = argparse.ArgumentParser(description = 'Generate input file for TuningTool on GRID',
                                 parents = [tuningJobFileParser, loggerParser],
                                 conflict_handler = 'resolve')

## Now the job really starts
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args( namespace = LoggerNamespace() )


# Transform fileType to the enumeration type from the string:
args.fileType = [JobFileTypeCreation.fromstring(conf) for conf in args.fileType]

# Make sure that the user didn't specify all with other file creations:
if JobFileTypeCreation.all in args.fileType and len(args.fileType) > 1:
  raise ValueError(("Chosen to create all file types and also defined another"
    " option."))

from RingerCore import printArgs, Logger
logger = Logger.getModuleLogger(__name__, args.output_level )
printArgs( args, logger.debug )

################################################################################
# Check if it is required to create the configuration files:
if JobFileTypeCreation.all in args.fileType or \
    JobFileTypeCreation.ConfigFiles in args.fileType:
  logger.info('Creating configuration files at folder %s', 
              args.jobConfiFilesOutputFolder )
  from TuningTools import createTuningJobFiles
  createTuningJobFiles( outputFolder   = args.jobConfiFilesOutputFolder,
                        neuronBounds   = args.neuronBounds,
                        sortBounds     = args.sortBounds,
                        nInits         = args.nInits,
                        nNeuronsPerJob = args.nNeuronsPerJob,
                        nInitsPerJob   = args.nInitsPerJob,
                        nSortsPerJob   = args.nSortsPerJob,
                        level          = args.output_level,
                        compress       = args.compress )

################################################################################
# Check if it is required to create the cross validation file:
if JobFileTypeCreation.all in args.fileType or \
    JobFileTypeCreation.CrossValidFile in args.fileType:
  from TuningTools import CrossValid, CrossValidArchieve
  crossValid = CrossValid(nSorts=args.nSorts,
                          nBoxes=args.nBoxes,
                          nTrain=args.nTrain, 
                          nValid=args.nValid,
                          nTest=args.nTest,
                          seed=args.seed,
                          level=args.output_level)
  place = CrossValidArchieve( args.crossValidOutputFile,
                              crossValid = crossValid ).save( args.compress )
  logger.info('Created cross-validation file at path %s', place )

################################################################################
# Check if it is required to create the ppFile:
if JobFileTypeCreation.all in args.fileType or \
    JobFileTypeCreation.ppFile in args.fileType:
  from TuningTools.PreProc import *
  ppCol = list()
  eval('ppCol.extend(%s)' % args.ppCol)
  ppCol = PreProcCollection( [PreProcChain(obj) for obj in ppCol] )
  for ppChain in ppCol:
    ppFile = '%s_%s' % ( args.preProcOutputFile, str(ppChain) )
    place = PreProcArchieve( ppFile, ppChain = ppChain ).save( args.compress )
    logger.info('Created pre-processing file at path %s', place )

logger.info('Finished creating tuning job files.')
