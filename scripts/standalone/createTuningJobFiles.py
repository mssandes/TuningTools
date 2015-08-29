#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

from FastNetTool.util import EnumStringification
from FastNetTool.FileIO import save

class JobFileTypeCreation( EnumStringification ):
  """
    The possible file creation options
  """
  all = 0,
  ConfigFiles = 1,
  CrossValidFile = 2,
  ppFile = 3

################################################################################
parser = argparse.ArgumentParser( \
    description = 'Create files used by TuningJob.' )

parser.add_argument('fileType', choices = ['all', 
                                          'ConfigFiles', 
                                          'CrossValidFile',
                                          'ppFile'],
                     nargs='+',
                     help = """Which kind of files to create. You can choose one
                     or more of the available choices, just don't use all with
                     the other available choices.""")
from FastNetTool.Logger import Logger, LoggingLevel
parser.add_argument('--output-level', 
    default = LoggingLevel.tostring( LoggingLevel.INFO ), 
    type=str, help = "The logging output level.")

################################################################################
jobConfig = parser.add_argument_group( "JobConfig Files Creation Options", 
                                       """Change configuration for
                                       job config files creation.""")
jobConfig.add_argument('-outJobConfig', '--jobConfiFilesOutputFolder', 
                       default = 'jobConfig', 
                       help = "The job config files output folder.")
jobConfig.add_argument('--neuronBounds', nargs='+', type=int, default = [5,20],  
                        help = """
                            Input a sequential bounded list to be used as the
                            neuron job range, the arguments should have the
                            same format from the seq unix command or as the
                            Matlab format. If not specified, the range will
                            start from 1.  I.e 5 2 9 leads to [5 7 9] and 50
                            leads to 1:50
                               """)
jobConfig.add_argument('--sortBounds', nargs='+', type=int, default = [50],  
                       help = """
                          Input a sequential bounded list using seq format to
                          be used as the sort job range, but the last bound
                          will be opened just as happens when using python
                          range function. If not specified, the range will
                          start from 0.  I.e. 5 2 9 leads to [5 7] and 50 leads
                          to range(50)
                              """)
jobConfig.add_argument('--nInits', nargs='+', type=int, default = 100,
                       help = """
                          Input a sequential bounded list using seq format to
                          be used as the inits job range, but the last bound
                          will be opened just as happens when using python
                          range function. If not specified, the range will
                          start from 0.  I.e. 5 2 9 leads to [5 7] and 50 leads
                          to range(50)
                              """)
jobConfig.add_argument('--nNeuronsPerJob', type=int, default = 1,  
                        help = "The number of hidden layer neurons per job.")
jobConfig.add_argument('--nSortsPerJob', type=int, default = 1,  
                       help = "The number of sorts per job.")
jobConfig.add_argument('--nInitsPerJob', type=int, default = 5,  
                        help = "The number of initializations per job.")

################################################################################
crossConfig = parser.add_argument_group( "CrossValid File Creation Options", 
                                         """Change configuration for CrossValid
                                         file creation.""")
crossConfig.add_argument('-outCross', '--crossValidOutputFile', 
                       default = 'crossConfig', 
                       help = "The cross validation output file.")
crossConfig.add_argument('-ns',  '--nSorts', type=int, default = 50, 
                         help = """The number of sort used by cross validation
                                configuration.""")
crossConfig.add_argument('-nb', '--nBoxes', type=int,  default = 10, 
                         help = """The number of boxes used by cross validation
                                 configuration.""")
crossConfig.add_argument('-ntr', '--nTrain', type=int, default = 6,  
                         help = """The number of train boxes used by cross
                                validation.""")
crossConfig.add_argument('-nval','--nValid', type=int, default = 4, 
                         help = """The number of valid boxes used by cross
                                validation.""")
crossConfig.add_argument('-ntst','--nTest',  type=int, default = 0, 
                         help = """The number of test boxes used by cross
                                validation.""")
crossConfig.add_argument('-seed', type=int, default=None,
                         help = "The seed value for generating CrossValid object.")
################################################################################
ppConfig = parser.add_argument_group( "PreProc File Creation Options", 
                                      """Change configuration for pre-processing 
                                      file creation. These options will only
                                      be taken into account if job fileType is
                                      set to "ppFile" or "all".""")
ppConfig.add_argument('-outPP','--preProcOutputFile',
                      default = 'ppFile',
                      help = "The pre-processing validation output file")
ppConfig.add_argument('-ppCol', type=str,
                      default = '[[Norm1()]]',
                      help = """The pre-processing collection to apply. The
                             string will be parsed by python and created using
                             the available pre-processings on
                             FastNetTool.PreProc.py file""")

args = parser.parse_args()

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

args.output_level = LoggingLevel.fromstring( args.output_level )

from FastNetTool.util import printArgs
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
