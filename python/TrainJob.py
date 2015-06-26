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
 
    neuron      = kw.pop('neuron',2)
    sort        = kw.pop('sort',0)
    inits       = kw.pop('inits',100)
    doMultiStop = kw.pop('doMultiStop', False)
    showEvo     = kw.pop('showEvo', 5)
    epochs      = kw.pop('epochs',1000)
    doPerf      = kw.pop('doPerf', False)
    level       = kw.pop('level', 1)
    output      = kw.pop('output','train')
    from FastNetTool.util import checkForUnusedVars
    checkForUnusedVars( kw, self._logger.warning )
    del kw

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
                               batchSize=batchSize)
      
    fullOutput = '%s.n%04d.s%04d.pic' % ( output, neuron, sort )
    train = []
    for init in range( inits ):
      self._logger.info('train: neuron = %d, sort = %d, init = %d', neuron, sort, init)
      self._fastnet.new_ff([nInputs, neuron, 1], ['tansig', 'tansig'])
      nets = self._fastnet.train_ff()
      train.append( nets )
    
    self._logger.info('Saving file named %s...', fullOutput)
    objSave = [neuron, sort, inits, train]
    filehandler = open(fullOutput, 'w')
    pickle.dump(objSave, filehandler, protocol = 2 )
    self._logger.info('File "%s" saved!', fullOutput)
    filehandler.close()
    



