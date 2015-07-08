#!/usr/bin/env python

from FastNetTool.Logger import Logger

class TrainJob(Logger):
  def __init__(self, logger = None ):
    Logger.__init__( self, logger = logger )
    self._fastnet = None
    self._fastnetLock = False

  def __call__(self, data, target, cross, **kw ):

    import pickle
    import numpy as np
    from FastNetTool.FastNet import FastNet
    from FastNetTool.Neural import Neural
    from FastNetTool.util import checkForUnusedVars
 
    neuron      = kw.pop('neuron',2)
    sort        = kw.pop('sort',0)
    initBounds  = kw.pop('initBounds', [0, 99] )
    doMultiStop = kw.pop('doMultiStop', False)
    showEvo     = kw.pop('showEvo', 5)
    epochs      = kw.pop('epochs',1000)
    doPerf      = kw.pop('doPerf', False)
    level       = kw.pop('level', 1)
    output      = kw.pop('output','train')
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    # Make bounds from 0 until unique value:
    if not isinstance(initBounds, list):
      initBounds = [initBounds]
    if len(initBounds) == 1:
      initBounds.append( initBounds[0] )
      initBounds[1] -= 1
      initBounds[0] = 0

    initBounds[1] += 1

    nInputs     = data.shape[1]
    split = cross( data, target, sort )
    del data, cross
    self._logger.info('Extracted cross validation sort')

    batchSize=0
    trnData=split[0]
    valData=split[1]
    if len(split) > 2:  tstData=split[2]
    
    del split
    batchSize = trnData[1].shape[0] if trnData[0].shape[0] > trnData[1].shape[0] else \
                trnData[0].shape[0]


    if not self._fastnetLock:
      self.fastnetLock=True
      self._logger.info('parse to fastnet python class')
      self._fastnet = FastNet( trnData, valData,
                               tstData=tstData,
                               doMultiStop=doMultiStop,
                               doPerf=doPerf,
                               epochs=epochs,
                               showEvo=showEvo,
                               batchSize=batchSize )
      
    train = []
    for init in range( *initBounds ):
      self._logger.info('train: neuron = %d, sort = %d, init = %d', neuron, sort, init)
      self._fastnet.new_ff([nInputs, neuron, 1], ['tansig', 'tansig'])
      nets = self._fastnet.train_ff()
      train.append( nets )
    
    # return it to initial form:
    initBounds[1] -= 1
    fullOutput = '%s.n%04d.s%04d.id%04d.iu%04d.pic' % ( output, neuron, sort, initBounds[0], initBounds[1] )
    self._logger.info('Saving file named %s...', fullOutput)
    objSave = [neuron, sort, initBounds, train]
    filehandler = open(fullOutput, 'w')
    pickle.dump(objSave, filehandler, protocol = 2 )
    self._logger.info('File "%s" saved!', fullOutput)
    filehandler.close()



