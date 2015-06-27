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
    required = True, nargs='+', type=int,
    help = "Input a sequential list, the arguments should have the same format from the seq unix command.")
parser.add_argument('-s','--nSorts', action='store', 
    default = 50, type=int,
    help = "Number of cross validation sorts (kFold) ")
parser.add_argument('-q','--queue', action='store', 
    default = "1nw", 
    help = "The bjob queue. Use bqueues for a list of available queues (ignored if using --debug).")
parser.add_argument('-i','--nInits', action='store', 
    default = 100, 
    help = "The number of initializations for each sort.")
parser.add_argument('--debug',  
    action='store_true',
    help = "Set queue to 8nm")
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()
if len(args.neurons) == 1:
  args.neurons[1] = args.neurons[0]
  args.neurons[0] = 0
elif len(args.neurons) == 2:
  args.neurons[1] = args.neurons[1] + 1
elif len(args.neurons) == 3:
  tmp = args.neurons[1]
  if tmp > 0:
    args.neurons[1] = args.neurons[2] + 1
  else:
    args.neurons[1] = args.neurons[2] - 1
  args.neurons[2] = tmp
else:
  raise ValueError("Neurons param has wrong size!")

if args.debug:
  args.queue = '8nm'

from FastNetTool.util import printArgs, getModuleLogger
logger = getModuleLogger(__name__)
printArgs( args, logger.info )

import os
for neuron in range(*args.neurons):
  for sort in range(args.nSorts): 
    exec_str = """\
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
                   data = args.data,
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
    #logger.info("Command without spaces:\n%s", exec_str)
    os.system(exec_str)
