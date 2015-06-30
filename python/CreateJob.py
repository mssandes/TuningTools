#!/usr/bin/env python

from FastNetTool.Logger import Logger

class CreateJob(Logger):

  def __init__( self, logger = None ):
    Logger.__init__( self, logger = logger )

  def __call__(self, **kw):
    """
      Creates a pickle file ntuple with...
    """
    from FastNetTool.CrossValid import CrossValid
    import numpy as np
    import pickle

    #Cross validation configuration
    outputFolder = kw.pop('outputFolder','jobConfig')
    nSorts       = kw.pop('nSorts', 10 )
    nBoxes       = kw.pop('nBoxes', 10 )
    nTrain       = kw.pop('nTrain', 6  )
    nValid       = kw.pop('nValid', 4  )
    nTest        = kw.pop('nTest',  0  )  
    #Output ocnfiguration
    inits        = kw.pop('inits',      100          )
    nSortPerJob  = kw.pop('nSortsPerJob', 1          )
    neurons      = kw.pop('neurons',    range(5,21)  )
    # and delete it to avoid mistakes:
    from FastNetTool.util import checkForUnusedVars
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    self._logger.info('run cross validation algorithm...')
    cross = CrossValid(nSorts=nSorts,
                       nBoxes=10,
                       nTrain=nTrain, 
                       nValid=nValid)

    self._logger.info("Created the following CrossValid object:\n%s", cross) 

    from FastNetTool.util import mkdir_p
    mkdir_p(outputFolder)

    for neuron in range(*neurons):
      sort = 0
      while sort < nSorts:
        sortMin = sort
        sortMax = sortMin+nSortPerJob-1
        if sortMax > nSorts: sortMax = nSorts
        sort = sortMax+1
        self._logger.info('Retrieved following job configuration : [ neuron=%-3d, sortMin=%-3d, sortMax=%-3d, inits=%-3d]',
                          neuron, sortMin, sortMax, inits)
        jobName = '%s/job.n%04d.i%04d.s%04d.s%04d.pic' % (outputFolder, neuron,inits, sortMin, sortMax)
        objSave = [neuron, [sortMin, sortMax], inits, cross]
        filehandler = open( jobName, 'w')
        pickle.dump( objSave, filehandler, protocol=2 )
        self._logger.info('Saved job option configuration with name: %r',jobName)

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
  parser.add_argument('-out', '--outputFolder', default = 'newJob', help = "The name of the job configuration.")
  parser.add_argument('-ns',  '--nSorts', type=int,default = 50, help = "The number of sort used by cross validation configuration.")
  parser.add_argument('-nb',  '--nBoxes', type=int,default = 10, help = "The number of boxes used by cross validation configuration.")
  parser.add_argument('-ntr', '--nTrain', type=int,default = 6,  help = "The number of train boxes used by cross validation.")
  parser.add_argument('-nval','--nValid', type=int,default = 4,  help = "The number of valid boxes used by cross validation.")
  parser.add_argument('-ntst','--nTest',  type=int,default = 0,  help = "The number of test boxes used by cross validation.")
  parser.add_argument('--nSortsPerJob', type=int, default = 1,  help = "The number of sorts per job.")
  parser.add_argument('--inits',type=int,  default = 100,  help = "The number of initialization per train.")
  parser.add_argument('--neurons', nargs='+', type=int, default = [5,20],  
      help = "Input a sequential list, the arguments should have the same format from the seq unix command.")
  
  import sys
  #if len(sys.argv)==1:
  #  parser.print_help()
  #  sys.exit(1)
  # Retrieve parser args:
  args = parser.parse_args()
  # Treat specials arguments
  if len(args.neurons) == 1:
    args.neurons.append( args.neurons[0] + 1 )
  elif len(args.neurons) == 2:
    args.neurons[1] = args.neurons[1] + 1
  elif len(args.neurons) == 3:
    tmp = args.neurons[1]
    if tmp > 0:
      args.neurons[1] = args.neurons[2] + 1
    else:
      args.neurons[1] = args.neurons[2] - 1
    args.neurons[2] = tmp

  from FastNetTool.util import printArgs, getModuleLogger
  logger = getModuleLogger(__name__)
  printArgs( args, logger.info )

  createJob( outputFolder = args.outputFolder,
             nSorts       = args.nSorts,
             nBoxes       = args.nBoxes,
             nTrain       = args.nTrain,
             nValid       = args.nValid,
             nTest        = args.nTest,
             inits        = args.inits,
             nSortsPerJob = args.nSortsPerJob,
             neurons      = args.neurons)


