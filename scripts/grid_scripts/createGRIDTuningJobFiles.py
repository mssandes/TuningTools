#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from RingerCore import argparse

from RingerCore.Parser import outGridParser, loggerParser
from TuningTools.Parser import tuningJobFileParser, TuningToolGridNamespace
from RingerCore.util   import get_attributes

## The main parser
parser = argparse.ArgumentParser(description = 'Generate input file for TuningTool on GRID',
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
parser.add_argument('--compress', action='store_const', 
    default = 0, const = 0, required = False, 
    help = argparse.SUPPRESS)

## Now the job really starts
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)


args = parser.parse_args( namespace = TuningToolGridNamespace('prun') )
from RingerCore.Logger import Logger
mainLogger = Logger.getModuleLogger( __name__, args.output_level )

# Retrieve outputs containers
outputs = []
if any(val in args.fileType for val in ("all","ConfigFiles")):
  outputs.append('Config:job.*')

if any(val in args.fileType for val in ("all","CrossValidFile")):
  outputs.append('CrossValid:crossValid*')

if any(val in args.fileType for val in ("all","ppFile")):
  outputs.append('ppFile:ppFile*')
# Merge it to the grid arguments:
args.grid_outputs = '"' + ','.join(outputs) + '"'

from RingerCore.util import printArgs, conditionalOption
printArgs( args, mainLogger.debug )

# Prepare to run
args.setExec(r"""source ./setrootcore.sh; 
               {gridCreateTuningFiles}
                 {fileType}
                 --jobConfiFilesOutputFolder=\".\"
                 --neuronBounds {neuronBounds}
                 --sortBounds {sortBounds}
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
                 --compress={compress}
                 -ppCol=\"{ppCol}\"
             """.format( gridCreateTuningFiles = "\$ROOTCOREBIN/user_scripts/TuningTools/standalone/createTuningJobFiles.py",
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
                         compress=args.compress,
                       ) 
            )

# And run
args.run_cmd()

