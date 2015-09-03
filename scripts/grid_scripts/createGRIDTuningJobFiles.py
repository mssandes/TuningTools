#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from RingerCore import argparse

from RingerCore.Parser import outGridParser, loggerParser
from FastNetTool.Parser import tuningJobFileParser, FastNetGridNamespace
from RingerCore.util   import get_attributes

## The main parser
parser = argparse.ArgumentParser(description = 'Generate input file for FastNet on GRID',
                                 parents = [tuningJobFileParser, outGridParser, loggerParser],
                                 conflict_handler = 'resolve')
## Change parent options
# Make outputs not usable by user
parser.add_argument('--outputs', action='store_const',
    required = False, default = '', const = '', 
    dest = 'grid_outputs',
    help = argparse.SUPPRESS )
# Make nJobs not usable by user
parser.add_argument('--nJobs', action='store_const',
    required = False, dest = 'grid_nJobs', const = 1, default = 1,
    help = argparse.SUPPRESS)

## Now the job really starts
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)


args = parser.parse_args( namespace = FastNetGridNamespace('prun') )
from RingerCore.Logger import Logger
mainLogger = Logger.getModuleLogger( __name__, args.output_level )

# Retrieve outputs containers
outputs = []
if any(val in args.fileType for val in ("all","ConfigFiles")):
  outputs.append('configFolder/job.*')

if any(val in args.fileType for val in ("all","CrossValidFile")):
  outputs.append('crossValid*')

if any(val in args.fileType for val in ("all","ppFile")):
  outputs.append('ppFile*')
# Merge it to the grid arguments:
args.outputs = ','.join(outputs)

from RingerCore.util import printArgs, conditionalOption
printArgs( args, mainLogger.debug )

# Prepare to run
args.setExec(r"""source ./setrootcore.sh; 
               {gridCreateTuningFiles}
                 {fileType}
                 --jobConfiFilesOutputFolder=\"configFolder\"
                 --neuronBounds={neuronBounds}
                 --sortBounds={sortBounds}
                 --nInits={nInits}
                 --nNeuronsPerJob={nNeuronsPerJob}
                 --nSortsPerJob={nSortsPerJob}
                 --nInitsPerJob={nInitsPerJob}
                 --crossValidOutputFile="crossValid"
                 --nSorts={nSorts}
                 --nBoxes={nBoxes}
                 --nTrain={nTrain}
                 --nValid={nValid}
                 --nTest={nTest}
                 --preProcOutputFile=\"ppFile\"
                 -ppCol=\"{ppCol}\"
             """.format( gridCreateTuningFiles = "\$ROOTCOREBIN/user_scripts/FastNetTool/standalone/createTuningJobFiles.py",
                         fileType=' '.join(args.fileType),
                         neuronBounds=' '.join([str(i) for i in args.neuronBounds]),
                         sortBounds=' '.join([str(i) for i in args.sortBounds]),
                         nInits=args.nInits,
                         nNeuronsPerJob=args.nNeuronsPerJob,
                         nSortsPerJob=args.nSortsPerJob,
                         nInitsPerJob=args.nInitsPerJob,
                         nSorts=args.nSorts,
                         nBoxes=args.nBoxes,
                         nTrain=args.nTrain,
                         nValid=args.nValid,
                         nTest=args.nTest,
                         ppCol=args.ppCol,
                       ) 
            )

# And run
args.run_cmd()

