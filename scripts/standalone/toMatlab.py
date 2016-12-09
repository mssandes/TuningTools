#!/usr/bin/env python

from RingerCore import csvStr2List, \
                       expandFolders, Logger, \
                       progressbar, LoggingLevel

from TuningTools.parsers import argparse, loggerParser, LoggerNamespace

mainParser = argparse.ArgumentParser(description = 'Merge files into unique file.',
                                     add_help = False)
mainMergeParser = mainParser.add_argument_group( "Required arguments", "")
mainMergeParser.add_argument('-i','--inputFiles', action='store', 
    metavar='InputFiles', required = True, nargs='+',
    help = "The input files that will be used to generate a matlab file")
mainLogger = Logger.getModuleLogger(__name__)
parser = argparse.ArgumentParser(description = 'Save files on matlab format.',
                                 parents = [mainParser, loggerParser],
                                 conflict_handler = 'resolve')

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

## Retrieve parser args:
args = parser.parse_args( namespace = LoggerNamespace() )
if mainLogger.isEnabledFor( LoggingLevel.DEBUG ):
  pprint(args.inputFiles)
## Treat special arguments
if len( args.inputFiles ) == 1:
  args.inputFiles = csvStr2List( args.inputFiles[0] )
args.inputFiles = expandFolders( args.inputFiles )
mainLogger.verbose("All input files are:")
if mainLogger.isEnabledFor( LoggingLevel.VERBOSE ):
  pprint(args.inputFiles)

for inFile in progressbar(args.inputFiles, len(args.inputFiles),
                          logger = mainLogger, prefix = "Processing files "):
  # Treat output file name:
  from RingerCore import checkExtension, changeExtension, load, save
  if checkExtension( inFile, "tgz|tar.gz|pic" ):
    cOutputName = changeExtension( inFile, '.mat' )
    data = load( inFile, useHighLevelObj = False )
    from scipy.io import savemat
    try:
      savemat( cOutputName, data )
    except ImportError:
      self._logger.fatal(("Cannot save matlab file, it seems that scipy is not "
          "available."), ImportError)
  else:
    mainLogger.error("Cannot transform files '%s' to matlab." % inFile)
  mainLogger.info("Successfully created matlab file: %s", cOutputName)
# end of (for fileCollection)

