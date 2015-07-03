#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

parser = argparse.ArgumentParser(description = 'Run tuning job on grid')
parser.add_argument('-d','--dataDS', required = True, metavar='DATA',
    help = "The dataset with the data for discriminator tuning.")
parser.add_argument('-o','--outDS', required = True, metavar='OUT',
    help = "The output dataset name.")
parser.add_argument('-c','--configFileDS', metavar='CONFIG', required = True,
    help = "Input dataset to loop upon files to retrieve configuration. There will be one job for each file on this container.")
parser.add_argument('--debug', action='store_const',
    const = '--nFiles=1 --express --debugMode --allowTaskDuplication',
    help = "Set debug options and only run 1 job.")
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()

from FastNetTool.util import printArgs, getModuleLogger, start_after, conditionalOption
logger = getModuleLogger(__name__)
printArgs( args, logger.info )


import os
workDir=os.path.expandvars("$ROOTCOREBIN/user_scripts/FastNetTool/run_on_grid/")
os.chdir(workDir)
exec_str = """\
            prun --exec
                    "source $ROOTCOREBIN/../setrootcore.sh; \\
                    {tuningJob} \\ 
                      %DATA \\
                      %IN \\
                      fastnet.tuned" \\
                 --inDS={configFileDS} \\
                 --secondaryDSs=DATA:1:{data}  \\
                 --outDS={outDS} \\
                 --workDir={workDir} \\
                 --useRootCore \\
                 --outputs="fastnet.tuned*.pic" \\
                 --disableAutoRetry \\
                 --tmpDir=/tmp \\
                 {extraFlags}
          """.format(tuningJob="$ROOTCOREBIN/user_scripts/FastNetTool/run_on_grid/tuningJob.py",
                     configFileDS=args.configFileDS,
                     data=args.dataDS,
                     outDS=args.outDS,
                     workDir=workDir,
                     extraFlags = args.debug if args.debug else '--skipScout',
                     )
logger.info("Executing following command:\n%s", exec_str)
import re
exec_str = re.sub(' +',' ',exec_str)
exec_str = re.sub('\\\\','',exec_str) # FIXME We should be abble to do this only in one line...
exec_str = re.sub('\n','',exec_str)
#logger.info("Command without spaces:\n%s", exec_str)
os.system(exec_str)
