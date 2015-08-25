#!/usr/bin/env python

#try:
#  parseOpts
#except NameError,e:
#  parseOpts = False
#if __name__ == "__main__" or parseOpts:

import logging
try:
  import argparse
except ImportError:
  from FastNetTool import argparse

parser = argparse.ArgumentParser(description = 'Create FastNet data from PhysVal.')
parser.add_argument('-s','--sgnInputFiles', action='store', 
    metavar='SignalInputFiles', required = True, nargs='+',
    help = "The signal files that will be used to tune the discriminators")
parser.add_argument('-b','--bkgInputFiles', action='store', 
    metavar='BackgroundInputFiles', required = True, nargs='+',
    help = "The background files that will be used to tune the discriminators")
parser.add_argument('-op','--operation', action='store', required = True, 
    help = "The operation environment for the algorithm")
parser.add_argument('-o','--output', default = 'fastnetData', 
    help = "The pickle intermediate file that will be used to train the datasets.")
parser.add_argument('--reference', action='store', nargs='+',
    metavar='(BOTH | SGN BKG)_REFERENCE', default = ['Truth'], 
    choices = ('Truth','Off_CutID','Off_Likelihood'),
    help = """
      The reference used for filtering datasets. It needs to be set
      to a value on the Reference enumeration on FilterEvents file.
      You can set only one value to be used for both datasets, or one
      value first for the Signal dataset and the second for the Background
      dataset.
          """)
parser.add_argument('--output-level', default = logging.INFO, 
    help = "The output level for the main logger")
parser.add_argument('-t','--treePath', metavar='TreePath', action = 'store', 
    default = None, type=str,
    help = "The Tree path to be filtered on the files.")
parser.add_argument('-l1','--l1EmClusCut', default = None, 
    type=int, help = "The L1 cut threshold")
parser.add_argument('-nClusters','--numberOfClusters', 
    default = None, type=int,
    help = "Maximum number of events to add to each dataset.")

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()
# Treat special argument
if len(args.reference) > 2:
  raise ValueError("--reference set to multiple values: %r", args.reference)
if len(args.reference) is 1:
  args.reference.append( args.reference[0] )
from FastNetTool.Logger import Logger, LoggingLevel
args.output_level = LoggingLevel.fromstring(args.output_level)
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

