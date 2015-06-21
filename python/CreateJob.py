#!/usr/bin/env python
import logging

class CreateJob():

  def __init__( self, logger = None ):
    import logging
    self._logger = logger or logging.getLogger(__name__)

  def __call__(self, inputDataName,  **kw):
    """
      Creates a pickle file ntuple with...

    """
    from FastNetTool.CrosValid import CrossValid
    import numpy as np
    import pickle

    #Cross validation configuration
    nSort        = kw.pop('nSort', 10 )
    nBoxes       = kw.pop('nBoxes', 10 )
    nTrain       = kw.pop('nTrain', 5 )
    nValid       = kw.pop('nValid', 3 )
    nTest        = kw.pop('nTest', 2 )  
    #Output ocnfiguration
    nMaxLayer    = kw.pop('nMaxLayer', 20)
    inits        = kw.pop('inits', 100 )
    nSortPerJob  = kw.pop('nSortsPerJob', 1 )
    nMaxLayers   = kw.pop('nMaxLayers',20)
    del kw

    print 'Opening dataset'
    objLoad_target = np.load( inputDataName )[1]
    self._logger.info('Extracted target rings with size: %r',objLoad_target.shape])

    cross = CrossValid(objLoad_target, nSort=nSort, nBoxes=nBoxes,nTrain=nTrain, nValid=nValid )

    for neuron in range(nMaxLayers):
      sort = 0
      while sort < nSort:
        jobName = 'jobTrainConfig.neuron_'+str(h)+'.s'+str(s)+'.pic'
        print 'save job option with name:  ', jobName
        sortMin = sort
        sortMax = sortMin+nSortPerJob
        sort = sortMax+1
        print 'job configuration are: [ neuron=',neuron+1,'[sortMin=',sortMin,', sortMax=',sortMax, ', inits=', inits, ', crossObj]'
        objSave = [neuron+1, [sortMin, sortMax], inits, cross]
        pickle.dump( objSave, open( jobName, "wb" ) )
        self._logger.info('Save job option configuration with name: %r',jobName)

createJob = CreateJob()

try:
  parseOpts
except NameError,e:
  parseOpts = False

if __name__ == "__main__" or parseOpts:
  try:
    import argparse
  except ImportError:
    from FastNetTool import argparse

  parser = argparse.ArgumentParser(description = '')
  parser.add_argument('-inDS','--inputFile', action='store',help = "The input file that will be used to tune the discriminators")
  parser.add_argument('-ns',  '--nSort',  default = 50, help = "The number of sort used by cross validation configuration.")
  parser.add_argument('-nb',  '--nBoxes', default = 10, help = "The number of boxes used by cross validation configuration.")
  parser.add_argument('-ntr', '--nTrain', default = 5,  help = "The number of train boxes used by cross validation.")
  parser.add_argument('-nval','--nValid', default = 3,  help = "The number of valid boxes used by cross validation.")
  parser.add_argument('-ntst','--nTest',  default = 2,  help = "The number of test boxes used by cross validation.")
  parser.add_argument('--nSortsPerJob', default = 5,  help = "The number of sorts per job.")
  parser.add_argument('--inits',  default = 100,  help = "The number of initialization per train.")
  parser.add_argument('--nMaxLayers',  default = 20,  help = "The number of neurons into hidden layer, this will be: 1 neuron until maxLayer.")
  
  import sys
  if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
  # Retrieve parser args:
  args = parser.parse_args()
  # Treat special argument
  if len(args.reference) > 2:
    raise ValueError("--reference set to multiple values: %r", args.reference)
  logger = logging.getLogger(__name__)
  logger.setLevel( logging.DEBUG ) # args.output_level)

  from FastNetTool.util import printArgs
  printArgs( args, logger.debug )

  createJob( args.inputFile,
             nSort      = args.nSort,
             nBoxes     = args.nBoxes,
             nTrain     = args.nTrain,
             nValid     = args.nValid,
             nTest      = args.nTest,
             inits      = args.inits,
             nSortsPerJob = args.nSortsPerJob,
             nMaxLayers   = args.nMaxLayers)


