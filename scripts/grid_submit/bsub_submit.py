#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

parser = argparse.ArgumentParser(description = 'Run training job on grid')
parser.add_argument('-d','--data', action='store', 
    required = True,
    help = "The file containing data for discriminator tunning")
parser.add_argument('-o','--output', action='store', 
    required = True,
    help = "The output base string for the discriminator file.")
parser.add_argument('-op','--outputPlace', action='store', 
    required = True,
    help = "The output place to a lxplus tmp.")
parser.add_argument('-n','--neurons', action='store', 
    required = True, nargs='+', 
    help = "Input a sequential list, the arguments should have the same format from the seq unix command.")
parser.add_argument('-s','--nSorts', action='store', 
    default = 50, nargs='+',
    help = "Number of cross validation sorts (kFold) ")
parser.add_argument('-s','--queue', action='store', 
    default = "1nw", 
    help = "The bjob queue. Use bqueues for a list of available queues.")
parser.add_argument('-i','--inits', action='store', 
    default = 100, 
    help = "The bjob queue. Use bqueues for a list of available queues.")
parser.add_argument('--debug',  
    action='store_true',
    help = "The output level for the main logger")
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()
if len(neurons) == 1:
  neurons[1] = neurons[0]
  neurons[0] = 0
elif len(neurons) == 2:
  neurons[1] = neurons[1] + 1
elif len(neurons) == 3:
  neurons[1] = neurons[1] + 1
else:
  raise ValueError("Neurons param has size greater than 3!")

from FastNetTool.util import printArgs, getModuleLogger
logger = getModuleLogger(__name__)
printArgs( args, logger.debug )

# FIXME It is needed to set environment on bsub_script so that it sources the
# new path data
# FIXME Debug this.
for neuron in range(*neurons):
  for sort in range(nSorts): 
    exec_bsub_str = """\
        bsub {bsub_script}\\ 
          -q {queue} -u \"\"\\
          --datasetPlace {data} \\
          --neuron {neuron} \\
          --sort {sort} \\
          --inits {inits} \\
          --output {output} \\
          --outputPlace {outputPlace}
        """.format(bsub_script = os.path.expandvars("$ROOTCOREBIN/user_scripts/FastNetTool/grid_submit/bsub_script.sh"),
                   queue = args.queue,
                   datasetPlace = args.data,
                   neuron = neuron,
                   sort = sort,
                   inits = args.nInits,
                   output = args.output,
                   outputPlace = args.outputPlace,
                   )
    logger.info("Executing following command:\n%s", exec_str)
    import re
    exec_str = re.sub(' +',' ',exec_str)
    exec_str = re.sub('\\\\','',exec_str) # FIXME We should be abble to do this only in one line...
    exec_str = re.sub('\n','',exec_str)
    logger.info("Command without spaces:\n%s", exec_str)
    import os
    os.system(exec_str)
