#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from RingerCore import argparse

from RingerCore.Parser import ioGridParser, loggerParser
from TuningTools.Parser import createDataParser, TuningToolGridNamespace

## Create our paser
# Add base parser options (this is just a wrapper so that we can have this as
# the first options to show, as they are important options)
parentParser = argparse.ArgumentParser(add_help = False, 
                                          description = 'Tune a discriminator for data.')
parentParser.add_argument('-d','--dataDS', required = True, metavar='DATA',
    action='store', nargs='+',
    help = "The dataset with the data for discriminator tuning.")
parentParser.add_argument('-c','--configFileDS', metavar='Config_DS', 
    required = True, action='store', nargs='+', dest = 'grid_inDS',
    help = """Input dataset to loop upon files to retrieve configuration. There
              will be one job for each file on this container.""")
parentParser.add_argument('-pp','--ppFileDS', 
    metavar='PP_DS', required = True, action='store', nargs='+',
    help = """The pre-processing files container.""")
parentParser.add_argument('-x','--crossValidDS', 
    metavar='CrossValid_DS', required = True, action='store', nargs='+',
    help = """The cross-validation files container.""")
## The main parser
parser = argparse.ArgumentParser(description = 'Run tuning job on grid',
                                 parents = [parentParser, ioGridParser, loggerParser],
                                 conflict_handler = 'resolve')
# Force secondary to be reusable:
parser.add_argument('--reusableSecondary', action='store_const',
    required = False, default = 'DATA,PP,CROSSVAL', const = 'DATA,CONFIG,PP,CROSSVAL', 
    dest = 'grid_reusableSecondary',
    help = argparse.SUPPRESS)
# Make inDS point to inDS-SGN if used
parser.add_argument('--inDS','-i', action='store', nargs='?',
    required = False, default = False,  dest = 'grid_inDS',
    help = argparse.SUPPRESS)
# Make outputs not usable by user
parser.add_argument('--outputs', action='store_const',
    required = False, default = '"tunedDiscr*"', const = '"tunedDiscr*"', 
    dest = 'grid_outputs',
    help = argparse.SUPPRESS )
# Make nFiles not usable by user
parser.add_argument('--nFiles', action='store_const',
    required = False, default = False, const = False, dest = 'grid_nFiles',
    help = argparse.SUPPRESS)
# Make nFilesPerJob not usable by user
parser.add_argument('--nFilesPerJob', action='store_const',
    required = False, default = 1, const = 1, dest = 'grid_nFilesPerJob',
    help = argparse.SUPPRESS)
# Make nJobs not usable by user
parser.add_argument('--nJobs', action='store_const',
    required = False, default = None, const = None, dest = 'grid_nJobs',
    help = argparse.SUPPRESS)
# Hide forceStaged and make it always be true
parser.add_argument('--forceStaged', action='store_const',
    required = False,  dest = 'grid_forceStaged', default = True, 
    const = True, help = argparse.SUPPRESS)
# Hide forceStagedSecondary and make it always be true
parser.add_argument('--forceStagedSecondary', action='store_const',
    required = False, dest = 'grid_forceStagedSecondary', default = True,
    const = True, help = argparse.SUPPRESS)
parser.add_argument('--long', action='store_const',
    required = False, dest = 'grid_long', default = True,
    const = True, help = argparse.SUPPRESS)
parser.add_argument('--compress', action='store_const', 
    default = 0, const = 0, required = False, 
    help = argparse.SUPPRESS)

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

# Retrieve parser args:
args = parser.parse_args( namespace = TuningToolGridNamespace('prun') )

if args.gridExpand_debug != '--skipScout':
  args.grid_nFiles = 1

# Fix secondaryDSs string:
args.grid_secondaryDS = "DATA:1:%s,PP:1:%s,CROSSVAL:1:%s" % (args.dataDS[0], 
                                                             args.ppFileDS[0],
                                                             args.crossValidDS[0])


from RingerCore.util import printArgs
from RingerCore.Logger import Logger
mainLogger = Logger.getModuleLogger( __name__, args.output_level )
printArgs( args, mainLogger.debug )

# Prepare to run
args.setExec("""source ./setrootcore.sh;
                export OMP_NUM_THREADS=1; export ROOTCORE_NCPUS=1;
                {tuningJob} %DATA %IN %PP %CROSSVAL tunedDiscr {compress}
             """.format( tuningJob = "\$ROOTCOREBIN/user_scripts/TuningTools/run_on_grid/BSUB_tuningJob.py",
                         compress = args.compress
                       ) 
            )

# And run
args.run_cmd()
