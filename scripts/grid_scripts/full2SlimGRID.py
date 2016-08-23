#!/usr/bin/env python
from RingerCore import csvStr2List, str_to_class, NotSet, BooleanStr, WriteMethod, \
                       get_attributes, expandFolders, Logger, getFilters, select, \
                       appendToFileName, ensureExtension, progressbar, LoggingLevel, \
                       printArgs, conditionalOption

from TuningTools.parsers import argparse, ioGridParser, loggerParser, \
                                TuningToolGridNamespace

parser = argparse.ArgumentParser(description = 'Extract ntuple files into unique file on the GRID.',
                                 parents = [ioGridParser, loggerParser],
                                 conflict_handler = 'resolve')

parser.add_argument('-t','--trigger' , action='store', 
   required = True , 
   help = "Trigger required to filter.")
# outputs 
parser.add_argument('--outputs', action='store',
   required = False, default = '"NTUPLE.*.root"', 
   dest = 'grid_outputs',
   help = argparse.SUPPRESS )
# Hide forceStaged and make it always be true
parser.add_argument('--forceStaged', action='store_const',
   required = False,  dest = 'grid_forceStaged', default = True,
   const = True, help = argparse.SUPPRESS)
# Hide forceStagedSecondary and make it always be true
parser.add_argument('--forceStagedSecondary', action='store_const',
   required = False, dest = 'grid_forceStagedSecondary', default = True,
   const = True, help = argparse.SUPPRESS)

mainLogger = Logger.getModuleLogger(__name__)

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args( namespace = TuningToolGridNamespace('prun') )
mainLogger = Logger.getModuleLogger( __name__, args.output_level )
printArgs( args, mainLogger.debug )

import os.path
user_scope = 'user.%s' % os.path.expandvars('$USER')
# hold the output
args.grid_outputs=args.grid_outputs.replace('*',args.trigger.replace('HLT_',''))
mainLogger.info( ( 'Hold the output with name %s')%(args.grid_outputs) )
args.setExec("""source ./setrootcore.sh --grid;
                {full2SlimJob} 
                --inputFiles %IN
                {TRIGGER_LIST}
                {OUTPUT_FILE}
             """.format( full2SlimJob     = "\$ROOTCOREBIN/user_scripts/TuningTools/standalone/full2Slim.py",
                         TRIGGER_LIST     = conditionalOption("--trigger",       args.trigger       ),
                         OUTPUT_FILE      = conditionalOption("-o",       args.grid_outputs         ))
            )
# And run
args.run_cmd()








