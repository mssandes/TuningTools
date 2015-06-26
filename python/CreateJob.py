#!/usr/bin/env python

from FastNetTool.Logger import Logger

class CreateJob(Logger):

  def __init__( self, logger = None ):
    Logger.__init__( self, logger = logger )

  def __call__(self, inputDataName,  **kw):
    """
      Creates a pickle file ntuple with...
    """
    from FastNetTool.CrossValid import CrossValid
    from FastNetTool.util import reshape
    import numpy as np
    import pickle

    #Cross validation configuration
    nSorts       = kw.pop('nSorts', 10 )
    nBoxes       = kw.pop('nBoxes', 10 )
    nTrain       = kw.pop('nTrain', 6  )
    nValid       = kw.pop('nValid', 4  )
    nTest        = kw.pop('nTest',  0  )  
    #Output ocnfiguration
    nMaxLayer    = kw.pop('nMaxLayer',  20  )
    inits        = kw.pop('inits',      100 )
    nSortPerJob  = kw.pop('nSortsPerJob', 1 )
    nMaxLayers   = kw.pop('nMaxLayers', 20  )
    # and delete it to avoid mistakes:
    from FastNetTool.util import checkForUnusedVars
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    self._logger.info('Opening dataset')
    objLoad_target = reshape( np.load( inputDataName )[1])
    self._logger.info('Extracted target rings with size: %r',objLoad_target.shape)
    cross = CrossValid(objLoad_target,  nSorts=nSorts,
                                        nBoxes=10,
                                        nTrain=nTrain, 
                                        nValid=nValid,
                                        )
    for neuron in range(2,nMaxLayers+1):
      sort = 0
      while sort < nSorts:

        #print 'save job option with name:  ', jobName
        sortMin = sort
        sortMax = sortMin+nSortPerJob-1
        if sortMax > nSorts: sortMax = nSorts
        sort = sortMax+1
        self._logger.info('Retrieved following job configuration : [ neuron=%d, sortMin=%d, sortMax=%d, inits=%d, crossValidObj %r]',
                          neuron, sortMin, sortMax, inits, cross)
        jobName = 'jobTrainConfig.neuron_%04d.s%04ds.s%04.pic' % (neuron, sortMin, sortMax)
        objSave = [neuron, [sortMin, sortMax], inits, cross]
        filehandler = open( jobName, 'w')
        pickle.dump( objSave, filehandler, protocol=2 )
        self._logger.info('Save job option configuration with name: %r',jobName)

createJob = CreateJob()

try:
  parseOpts
except NameError,e:
  parseOpts = False

if __name__ == "__main__" or parseOpts:
  import logging
  try:
    import argparse
  except ImportError:
    from FastNetTool import argparse

  parser = argparse.ArgumentParser(description = '')
  parser.add_argument('-inDS','--inputFile', action='store',help = "The input file that will be used to tune the discriminators")
  parser.add_argument('-ns',  '--nSorts',  type=int,default = 50, help = "The number of sort used by cross validation configuration.")
  parser.add_argument('-nb',  '--nBoxes', type=int,default = 10, help = "The number of boxes used by cross validation configuration.")
  parser.add_argument('-ntr', '--nTrain', type=int,default = 5,  help = "The number of train boxes used by cross validation.")
  parser.add_argument('-nval','--nValid', type=int,default = 3,  help = "The number of valid boxes used by cross validation.")
  parser.add_argument('-ntst','--nTest',  type=int,default = 2,  help = "The number of test boxes used by cross validation.")
  parser.add_argument('--nSortsPerJob', type=int,default = 5,  help = "The number of sorts per job.")
  parser.add_argument('--inits',type=int,  default = 100,  help = "The number of initialization per train.")
  parser.add_argument('--nMaxLayers', type=int, default = 20,  help = "The number of neurons into hidden layer, this will be: 1 neuron until maxLayer.")
  
  import sys
  if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
  # Retrieve parser args:
  args = parser.parse_args()
  # Treat special argument
  '''
  if len(args.reference) > 2:
    raise ValueError("--reference set to multiple values: %r", args.reference)
  if len(args.reference) is 1:
    args.reference.append( args.reference[0] )
  ''' 

  logger = logging.getLogger(__name__)
  logger.setLevel( logging.DEBUG ) # args.output_level)

  from FastNetTool.util import printArgs
  printArgs( args, logger.debug )

  createJob( args.inputFile,
             nSorts      = args.nSorts,
             nBoxes     = args.nBoxes,
             nTrain     = args.nTrain,
             nValid     = args.nValid,
             nTest      = args.nTest,
             inits      = args.inits,
             nSortsPerJob = args.nSortsPerJob,
             nMaxLayers   = args.nMaxLayers)


