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
    const = '--nFiles=1 --debugMode --allowTaskDuplication',
    help = "Set debug options and only run 1 job.")
parser.add_argument('-s','--site',default = 'AUTO',
    help = "The site location where the job should run.")
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()

from FastNetTool.util import printArgs, getModuleLogger, start_after, conditionalOption
logger = getModuleLogger(__name__)
printArgs( args, logger.info )

# We need this to avoid being banned from grid:
import os
if not os.path.isfile(os.path.expandvars("$ROOTCOREBIN/../FastNetTool/cmt/boost_1_58_0.tar.gz")):
  logger.info('Downloading boost to avoid doing it on server side.')
  import urllib
  urllib.urlretrieve("http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz", 
                     filename=os.path.expandvars("$ROOTCOREBIN/../FastNetTool/cmt/boost_1_58_0.tar.gz"))
else:
  logger.info('Boost already downloaded.')

workDir=os.path.expandvars("$ROOTCOREBIN/..")
os.chdir(workDir) # We need to cd to this dir so that prun accepts the submission
exec_str = """\
            prun --bexec "source ./buildthis.sh" \\
                 --exec \\
                    "source ./setrootcore.sh; \\
                    {tuningJob} \\ 
                      %DATA \\
                      %IN \\
                      fastnet.tuned" \\
                 --inDS={configFileDS} \\
                 --secondaryDSs=DATA:1:{data}  \\
                 --reusableSecondary=DATA \\
                 --outDS={outDS} \\
                 --workDir={workDir} \\
                 --nFilesPerJob=1 \\
                 --outputs="fastnet.tuned*.pic" \\
                 --forceStaged \\
                 --forceStagedSecondary \\
                 --excludeFile "*.o,*.so,*.a,*.gch" \\
                 --extFile "FastNetTool/cmt/boost_1_58_0.tar.gz" \\
                 --tmpDir=/tmp \\
                 --long \\
                 {site} \\
                 {extraFlags}
          """.format(tuningJob="\$ROOTCOREBIN/user_scripts/FastNetTool/run_on_grid/tuningJob.py",
                     configFileDS=args.configFileDS,
                     data=args.dataDS,
                     outDS=args.outDS,
                     workDir=workDir,
                     site = '--site=' + args.site,
                     extraFlags = args.debug if args.debug else '',
                     )
logger.info("Executing following command:\n%s", exec_str)
import re
exec_str = re.sub('\\\\ *\n','', exec_str )
exec_str = re.sub(' +',' ', exec_str)
#logger.info("Command without spaces:\n%s", exec_str)
os.system(exec_str)
