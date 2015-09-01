#!/usr/bin/env python

from FastNetTool.Parser import createDataParser, LoggerNamespace
parser = argparse.ArgumentParser(add_help = False, 
                                 description = 'Create FastNet data from PhysVal.',
                                 parents = [createDataParser, loggerParser])

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args( LoggerNamespace() )
# Treat special argument
if len(args.reference) > 2:
  raise ValueError("--reference set to multiple values: %r", args.reference)
if len(args.reference) is 1:
  args.reference.append( args.reference[0] )
from FastNetTool.Logger import Logger, LoggingLevel
logger = Logger.getModuleLogger( __name__, args.output_level )
if args.operation != 'Offline' and not args.treePath:
  ValueError("If operation is not set to Offline, it is needed to set the TreePath manually.")

from FastNetTool.util import printArgs
printArgs( args, logger.debug )

from FastNetTool.CreateData import createData
createData( sgnFileList     = args.sgnInputFiles, 
            bkgFileList     = args.bkgInputFiles,
            ringerOperation = args.operation,
            referenceSgn    = args.reference[0],
            referenceBkg    = args.reference[1],
            treePath        = args.treePath,
            output          = args.output,
            l1EmClusCut     = args.l1EmClusCut,
            level           = args.output_level,
            nClusters       = args.numberOfClusters )

