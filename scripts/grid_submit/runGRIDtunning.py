#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

parser = argparse.ArgumentParser(description = 'Run training job on grid')
parser.add_argument('-id','--data', required = True,
    help = "The input dataset with the data for discriminator tunning")
parser.add_argument('-o','--outDS', required = True,
    help = "The output dataset name.")
parser.add_argument('-i','--inDS', 
    metavar = 'InputDataset', 
    help = "Input dataset to loop upon files to retrieve configuration. There will be one job for each file on this container")
parser.add_argument('--debug',  
    const = '--nFiles=1 --express --debugMode --allowTaskDuplication'
    help = "Set debug options and only run 1 job.")
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()

from FastNetTool.util import printArgs, getModuleLogger
logger = getModuleLogger(__name__)
printArgs( args, logger.info )

import os
#os.system('rcSetup -u')
exec_str = """\
      i bsub -q {queue} -u \"\" -J pyTrain \\
        {bsub_script} \\ 
          --jobConfig {jobFile} \\
          --datasetPlace {data} \\
          --output {output} \\
          --outputPlace {outputPlace} \\
    """.format(bsub_script = os.path.expandvars("$ROOTCOREBIN/user_scripts/FastNetTool/grid_submit/bsub_script.sh"),
               queue = args.queue,
               data = args.data,
               jobFile = f,
               output = args.output,
               outputPlace = args.outputPlace,
               )

exec_str = """\
            prun --exec "./ --sgnInputFiles %IN \\
                                          --bkgInputFiles %BKG \\
                                          --operation {operation} \\
                                          --output fastnet.pic \\
                                          --reference {referenceSgn} {referenceBkg} \\
                                          {treePath} " \\
                 --inDS={inDS_SGN} \\
                 --nFilesPerJob={nSgnFiles} \\
                 --nFiles={nSgnFiles} \\
                 --nJobs=1 \\
                 --secondaryDSs=BKG:{nBkgFiles}:{inDS_BKG}  \\
                 --outDS={outDS} \\
                 {tarBallArg} \\
                 --site=ANALY_BNL_SHORT,ANALY_BNL_LONG \\
                 --useRootCore \\
                 --disableAutoRetry \\
                 --outputs=fastnet.pic \\
                 --maxNFilesPerJob={nInputFiles} \\
                 --nGBPerJob=10000 \\
                 {extraFlags}
          """.format(inDS_SGN='%s' % ' '.join(args.inDS_SGN),
                     inDS_BKG='%s' % ' '.join(args.inDS_BKG),
                     operation=args.operation,
                     outDS=args.outDS,
                     nSgnFiles=nSgnFiles, # it seems that setting nJobs=1 doesn't work... but why?
                     nBkgFiles=nBkgFiles,
                     nInputFiles=nSgnFiles+nBkgFiles,
                     tarBallArg=conditionalOption('--inTarBall=', args.inTarBall) + conditionalOption('--outTarBall=', args.outTarBall),
                     referenceSgn=args.reference[0],
                     referenceBkg=args.reference[1],
                     treePath = conditionalOption('--treePath ', args.treePath),
                     extraFlags = args.debug if args.debug else '--skipScout',
                     )
logger.info("Executing following command:\n%s", exec_str)
import re
exec_str = re.sub(' +',' ',exec_str)
exec_str = re.sub('\\\\','',exec_str) # FIXME We should be abble to do this only in one line...
exec_str = re.sub('\n','',exec_str)
#logger.info("Command without spaces:\n%s", exec_str)
os.system(exec_str)
