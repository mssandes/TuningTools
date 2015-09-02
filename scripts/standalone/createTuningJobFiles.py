#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

from FastNetTool.Parser import tuningJobFileParser, loggerParser, LoggerNamespace, JobFileTypeCreation
from FastNetTool.FileIO import save

parser = argparse.ArgumentParser(description = 'Generate input file for FastNet on GRID',
                                 parents = [tuningJobFileParser, loggerParser],
                                 conflict_handler = 'resolve')

args = parser.parse_args( namespace = LoggerNamespace() )
# Treat seed value to be set as an unsigned:
import ctypes
if not args.seed is None:
  args.seed = ctypes.c_uint( args.seed )

# Transform fileType to the enumeration type from the string:
args.fileType = [JobFileTypeCreation.fromstring(conf) for conf in args.fileType]

# Make sure that the user didn't specify all with other file creations:
if JobFileTypeCreation.all in args.fileType and len(args.fileType) > 1:
  raise ValueError(("Chosen to create all file types and also defined another"
    " option."))

from FastNetTool.util import printArgs
from FastNetTool.Logger import Logger
logger = Logger.getModuleLogger(__name__, args.output_level )
printArgs( args, logger.debug )

################################################################################
# Check if it is required to create the configuration files:
if JobFileTypeCreation.all in args.fileType or \
    JobFileTypeCreation.ConfigFiles in args.fileType:
  logger.info('Creating configuration files at folder %s', 
              args.jobConfiFilesOutputFolder )
  from FastNetTool.CreateTuningJobFiles import createTuningJobFiles
  createTuningJobFiles( outputFolder   = args.jobConfiFilesOutputFolder,
                        neuronBounds   = args.neuronBounds,
                        sortBounds     = args.sortBounds,
                        nInits         = args.nInits,
                        nNeuronsPerJob = args.nNeuronsPerJob,
                        nInitsPerJob   = args.nInitsPerJob,
                        nSortsPerJob   = args.nSortsPerJob,
                        level          = args.output_level)

################################################################################
# Check if it is required to create the cross validation file:
if JobFileTypeCreation.all in args.fileType or \
    JobFileTypeCreation.CrossValidFile in args.fileType:
  from FastNetTool.CrossValid import CrossValid
  crossValid = CrossValid(nSorts=args.nSorts,
                          nBoxes=args.nBoxes,
                          nTrain=args.nTrain, 
                          nValid=args.nValid,
                          nTest=args.nTest,
                          seed=args.seed,
                          level=args.output_level)
  crossFileData = {'version': 1,
                   'type' : 'CrossValidFile',
                   'crossValid' : crossValid }
  place = save( crossFileData, args.crossValidOutputFile )
  logger.info('Created cross-validation file at path %s', place )

################################################################################
# Check if it is required to create the ppFile:
if JobFileTypeCreation.all in args.fileType or \
    JobFileTypeCreation.ppFile in args.fileType:
  from FastNetTool.PreProc import *
  ppCol = list()
  eval('ppCol.extend(%s)' % args.ppCol)
  ppCol = PreProcCollection( [PreProcChain(obj) for obj in ppCol] )
  for ppChain in ppCol:
    ppFile = '%s_%s' % ( args.preProcOutputFile, str(ppChain) )
    logger.info('Creating pre-processing file at path %s', ppFile)
    ppFileData = {'version' : 1,
                  'type' : 'PreProcFile',
                  'ppChain' : ppChain }
    save( ppFileData, ppFile )

logger.info('Finished creating tuning job files.')
