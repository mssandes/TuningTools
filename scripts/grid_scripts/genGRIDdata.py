#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

from FastNetTool.Parser import ioGridParser, loggerParser, FastNetGridWorkspace
from FastNetTool.util   import get_attributes
from FastNetTool.FilterEvents import Reference, RingerOperation

### Create our parser
parser = argparse.ArgumentParser(description = 'Generate input file for FastNet on GRID',
                                 parents = [ioGridParser, loggerParser],
                                 conflict_handler = 'resolve')
## Add its options:
parser.add_argument('-s','--inDS-SGN', action='store', 
    metavar='SignalInputDataset', required = True, nargs='+', dest = 'grid_inDS', 
    help = "The signal files that will be used to tune the discriminators")
parser.add_argument('-b','--inDS-BKG', action='store', 
    metavar='BackgroundInputDataset', required = True, nargs='+',
    help = "The background files that will be used to tune the discriminators")
parser.add_argument('-op','--operation', action='store', required = True, 
    choices = get_attributes(RingerOperation, onlyVars = True),
    help = "The operation environment for the algorithm")
parser.add_argument('--reference', action='store', nargs='+',
    metavar='(BOTH | SGN BKG)_REFERENCE', default = ['Truth'], 
    choices = get_attributes(Reference, onlyVars = True),
    help = """
      The reference used for filtering datasets. It needs to be set
      to a value on the Reference enumeration on FilterEvents file.
      You can set only one value to be used for both datasets, or one
      value first for the Signal dataset and the second for the Background
      dataset.
          """)
parser.add_argument('-t','--treePath', metavar='TreePath', action = 'store', 
    default = None, type=str,
    help = """The Tree path to be filtered on the files.
        This argument is required if operation isn't set to Offline!!
      """)
## Change parent options
# Hide forceStaged and make it always be true
parser.add_argument('--forceStaged', action='store_const',
    required = False,  dest = '--forceStaged', default = True, 
    const = True, help = argparse.SUPPRESS)
# Hide forceStagedSecondary and make it always be true
parser.add_argument('--forceStagedSecondary', action='store_const',
    required = False, dest = '--forceStagedSecondary', default = True,
    const = True, help = argparse.SUPPRESS)
# Make inDS point to inDS-SGN if used
parser.add_argument('--inDS','-i', action='store', nargs='?',
    required = False, default = False,  dest = 'grid_inDS',
    help = argparse.SUPPRESS)
# Make nFiles not usable by user
parser.add_argument('--nFiles', action='store_const',
    required = False, default = 0, const = 0, dest = 'grid_nFiles',
    help = argparse.SUPPRESS)
# Make nFilesPerJob not usable by user
parser.add_argument('--nFilesPerJob', nargs='?', type=int,
    required = False, dest = 'grid_nFilesPerJob',
    help = argparse.SUPPRESS)
# Make nJobs not usable by user
parser.add_argument('--nJobs', action='store_const',
    required = False, default = 1, const = 1, dest = 'grid_nJobs',
    help = argparse.SUPPRESS)
# Make extFile not usable by user
parser.add_argument('--extFile', action='store_const', 
    required = False, dest = 'grid_extFile', default = '', const = '',
    help = argparse.SUPPRESS)
# Make secondary datasets not usable by user
parser.add_argument('--secondaryDSs', action='store_const',
    required = False, default = '', const = '', dest = 'grid_secondaryDS',
    help = argparse.SUPPRESS )
# Make outputs not usable by user
parser.add_argument('--outputs', action='store_const',
    required = False, default = '"tuningData*"', const = '"tuningData*"', 
    dest = 'grid_outputs',
    help = argparse.SUPPRESS )
# Force secondary to be reusable:
parser.add_argument('--reusableSecondary', action='store_const',
    required = False, default = 'BKG', const = 'BKG', dest = 'grid_reusableSecondary',
    help = """Allow reuse secondary dataset.""")

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

# Retrieve parser args:
args = parser.parse_args( namespace = FastNetGridWorkspace('prun') )
from FastNetTool.Logger import Logger
mainLogger = Logger.getModuleLogger(__name__)
# Treat special argument
if len(args.reference) > 2:
  raise ValueError("--reference set to multiple values: %r" % args.reference)
if len(args.reference) is 1:
  args.reference.append( args.reference[0] )
if args.operation != 'Offline' and not args.treePath:
  raise ValueError("If operation is not set to Offline, it is needed to set the TreePath manually.")

if ( len(args.inDS_BKG) > 1 or len(args.grid_inDS) > 1):
  raise NotImplementedError("Cannot use multiple datasets in this version.")

from subprocess import check_output

# TODO For now we only support one container for signal and one for background.
# We need to change it so that we can input multiple containers. This can be
# simply done by adding them as secondary datasets, copying the idea as done
# below (and looping through the input arguments:
def getNFiles( ds ):
  if ds[-1] != '/': ds += '/'
  mainLogger.info("Retriving \"%s\" number of files...", ds)
  output=check_output('dq2-ls -n %s | cut -f1' % ds, shell=True)
  mainLogger.info("Retrieved command output: %s", output[:-1])
  try:
    nFiles=int(output)
    if nFiles < 1:
      raise RuntimeError(("Couldn't retrieve the number of files on the output "
        "dataset %s. The output retrieven was: %s.") % ( ds , output ) )
    if args.gridExpand_debug and nFiles > 3:
      nFiles = 3
    return nFiles
  except ValueError:
    raise RuntimeError(("Seems that grid environment is not set. Try again after "
        "setting grid environment."))

# Fix primary dataset number of files:
nSgnFiles = getNFiles( args.grid_inDS[0] )
args.grid_nFiles = nSgnFiles
args.grid_nFilesPerJob = nSgnFiles

# Now work with secondary dataset:
nBkgFiles = getNFiles( args.inDS_BKG[0] )

# Fix secondaryDSs string:
args.grid_secondaryDS="BKG:%d:%s" % (nBkgFiles, args.inDS_BKG[0])

from FastNetTool.util import printArgs, conditionalOption
printArgs( args, mainLogger.info )

# Prepare to run
args.setExec("""source ./setrootcore.sh; 
               {gridCreateData}
                 --sgnInputFiles %IN
                 --bkgInputFiles %BKG
                 --operation {operation}
                 --output tuningData
                 --reference {referenceSgn} {referenceBkg}
                 {treePath};
             """.format( gridCreateData = "\$ROOTCOREBIN/user_scripts/FastNetTool/run_on_grid/gridCreateData.py",
                         operation=args.operation,
                         referenceSgn=args.reference[0],
                         referenceBkg=args.reference[1],
                         treePath = conditionalOption('--treePath ', args.treePath),
                         ) 
            )

# And run
args.run_cmd()
