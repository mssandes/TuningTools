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

    from FastNetTool.util import checkForUnusedVars, treatRangeVec
    #Cross validation configuration
    outputFolder = kw.pop('outputFolder','jobConfig' )
    nSorts       = kw.pop('nSorts', 10               )
    nInits       = kw.pop('nInits', 100              )
    nBoxes       = kw.pop('nBoxes', 10               )
    nTrain       = kw.pop('nTrain', 6                )
    nValid       = kw.pop('nValid', 4                )
    nTest        = kw.pop('nTest',  0                )  
    #Output ocnfiguration
    nSortsPerJob  = kw.pop('nSortsPerJob', 1         )
    nInitsPerJob  = kw.pop('nInitsPerJob', 1         )
    neurons      =  treatRangeVec( kw.pop('neurons',    [5, 20] ) )
    # and delete it to avoid mistakes:
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
        sortMax = sortMin+nSortsPerJob-1
        if sortMax >= nSorts: sortMax = nSorts-1
        sort = sortMax+1
        init = 0
        while init < nInits:
          initMin = init
          initMax = initMin+nInitsPerJob-1
          if initMax >= nInits: initMax = nInits-1
          init = initMax+1
          self._logger.info('Retrieved following job configuration : [ neuron=%-4d, sortMin=%-4d, sortMax=%-4d, initMin=%-4d, initMax=%-4d]',
                            neuron, sortMin, sortMax, initMin, initMax)
          jobName = '%s/job.n%04d.sl%04d.su%04d.il%04d.iu%04d.pic' % (outputFolder, neuron, sortMin, sortMax, initMin, initMax )
          objSave = [neuron, [sortMin, sortMax], [initMin, initMax], cross]
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
  parser.add_argument('-ni', '--nInits', type=int,  default = 100,  help = "The total number of initializations.")
  parser.add_argument('-nb', '--nBoxes', type=int,default = 10, help = "The number of boxes used by cross validation configuration.")
  parser.add_argument('-ntr', '--nTrain', type=int,default = 6,  help = "The number of train boxes used by cross validation.")
  parser.add_argument('-nval','--nValid', type=int,default = 4,  help = "The number of valid boxes used by cross validation.")
  parser.add_argument('-ntst','--nTest',  type=int,default = 0,  help = "The number of test boxes used by cross validation.")
  parser.add_argument('--nSortsPerJob', type=int, default = 1,  help = "The number of sorts per job.")
  parser.add_argument('--nInitsPerJob', type=int, default = 1,  help = "The number of initializations per job.")
  parser.add_argument('--neurons', nargs='+', type=int, default = [5,20],  
      help = "Input a sequential list, the arguments should have the same format from the seq unix command.")
  
  import sys
  #if len(sys.argv)==1:
  #  parser.print_help()
  #  sys.exit(1)
  # Retrieve parser args:
  args = parser.parse_args()

  from FastNetTool.util import printArgs, getModuleLogger
  logger = getModuleLogger(__name__)
  printArgs( args, logger.info )

  createJob( outputFolder = args.outputFolder,
             nSorts       = args.nSorts,
             nInits       = args.nInits,
             nBoxes       = args.nBoxes,
             nTrain       = args.nTrain,
             nValid       = args.nValid,
             nTest        = args.nTest,
             nSortsPerJob = args.nSortsPerJob,
             nInitsPerJob = args.nInitsPerJob,
             neurons      = args.neurons)


