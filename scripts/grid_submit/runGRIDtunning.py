#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

parser = argparse.ArgumentParser(description = '')
parser.add_argument('-s','--inDS-SGN', action='store', 
    metavar='SignalInputDataset', required = True, nargs='+',
    help = "The signal files that will be used to tune the discriminators")
parser.add_argument('-b','--inDS-BKG', action='store', 
    metavar='BackgroundInputDataset', required = True, nargs='+',
    help = "The background files that will be used to tune the discriminators")
parser.add_argument('-o','--outDS', action='store', 
    metavar='BackgroundInputDataset', required = True,
    help = "The background files that will be used to tune the discriminators")
parser.add_argument('-op','--operation', action='store', required = True, 
    help = "The operation environment for the algorithm")
parser.add_argument('--reference', action='store', nargs='+',
    metavar='(BOTH | SGN BKG)_REFERENCE', default = ['Truth'], choices = ('Truth','Off_CutID','Off_Likelihood'),
    help = """
      The reference used for filtering datasets. It needs to be set
      to a value on the Reference enumeration on FilterEvents file.
      You can set only one value to be used for both datasets, or one
      value first for the Signal dataset and the second for the Background
      dataset.
          """)
import logging
parser.add_argument('--output-level', default = logging.INFO, 
    help = "The output level for the main logger")
parser.add_argument('-t','--treePath', metavar='TreePath', action = 'store', 
    default = None, type=str,
    help = """The Tree path to be filtered on the files.
        This argument is required if operation isn't set to Offline!!
      """)
parser.add_argument('--test', default = '--express --', 
    help = "The output level for the main logger")
# Retrieve parser args:
args = parser.parse_args()
# Treat special argument
if len(args.reference) > 2:
  raise ValueError("--reference set to multiple values: %r", args.reference)
if len(args.reference) is 1:
  args.reference.append( args.reference[0] )
logger = logging.getLogger(__name__)
logger.setLevel(args.output_level)

try:
  from pprint import pprint
  args_dict = vars(args)
  logger.info('Retrieved the following configuration:')
  pprint([(key, args_dict[key]) for key in sorted(args_dict.keys())])
except ImportError:
  logger.info('Retrieved the following configuration: \n %r', vars(args))

import os
os.system("""
            prun --exec "gridCreateData --sgnInputFiles %IN2
                                        --bkgInputFiles %IN3" 
                 --secondaryDSs=IN2:{inDS_SGN},IN3:{inDS_BKG} 
                 --outDS={outDS}
                 --inTarBall={inTarBall}
                 --useRootCore
                 --disableAutoRetry
                 --noCompile
                 %{extraFlags}
          """.format(inDS_SGN=args.inDS_SGN,
                     inDS_BKG=args.inDS_BKG,
                     outDS=args.outDS,
                     inTarBall=args.inTarBall,
                     extraFlags=args.test))
createData( sgnFileList = args.sgnInputFiles, 
            bkgFileList = args.bkgInputFiles,
            ringerOperation = args.operation,
            referenceSgn = args.reference[0],
            referenceBkg = args.reference[1],
            treePath = args.treePath,
            output = args.output)

