#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

parser = argparse.ArgumentParser(description = 'Run training job on grid')
parser.add_argument('-d','--dataDS', required = True,
    help = "The dataset with the data for discriminator tunning.")
parser.add_argument('-o','--outDS', required = True,
    help = "The output dataset name.")
parser.add_argument('-c','--configFileDS', 
    metavar = 'InputDataset', 
    help = "Input dataset to loop upon files to retrieve configuration. There will be one job for each file on this container.")
parser.add_argument('--debug',  
    const = '--nFiles=1 --express --debugMode --allowTaskDuplication',
    help = "Set debug options and only run 1 job.")
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()

from FastNetTool.util import printArgs, getModuleLogger, trunc_at
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
            prun --exec
                    "source $ROOTCOREBIN/../setrootcore.sh; \\
                    {tunningJob} \\ 
                      %DATA \\
                      %IN \\
                      {outputFileName}" \\
                 --inDS={configFileDS} \\
                 --secondaryDSs=DATA:{data}  \\
                 --outDS={outDS} \\
                 {tarBallArg} \\
                 --useRootCore \\
                 --outputs="{outputFileName}*.pic" \\
                 --disableAutoRetry \\
                 {extraFlags}
          """.format(configFileDS=args.inDS,
                     data=args.data,
                     outDS=args.outDS,
                     outputFileName=trunc_at(outDS,'.',2),
                     tarBallArg=conditionalOption('--inTarBall=', args.inTarBall) + conditionalOption('--outTarBall=', args.outTarBall),
                     extraFlags = args.debug if args.debug else '--skipScout',
                     )
logger.info("Executing following command:\n%s", exec_str)
import re
exec_str = re.sub(' +',' ',exec_str)
exec_str = re.sub('\\\\','',exec_str) # FIXME We should be abble to do this only in one line...
exec_str = re.sub('\n','',exec_str)
#logger.info("Command without spaces:\n%s", exec_str)
os.system(exec_str)
