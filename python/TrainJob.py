#!/usr/bin/env python

class TrainJob():
  def __init__(self, logger = None ):
    import logging

    self._logger = logger or logging.getLogger(__name__)
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
    del kw

    nInputs     = data.shape[1]
    split = cross.getSort( data, sort )
    self._logger.info('Extracted cross validation sort')
    trnData = [split[0][0].tolist(), split[0][1].tolist()]
    batchSize = len( trnData[1] )
    del data

    valData = [split[1][0].tolist(), split[1][1].tolist()]
    try:
      tstData   = [split[2][0].tolist(), split[2][1].tolist()]
    except: tstData = None


    if not self._fastnetLock:
      self.fastnetLock=True
      self._fastnet = FastNet( trnData, valData,
                               tstData=tstData,
                               doMultiStop=doMultiStop,
                               doPerf=doPerf,
                               epochs=epochs,
                               showEvo=showEvo,
                               batchSize=batchSize)
      
    fullOutput = output+'.n00'+str(neuron)+'.s00'+str(sort)+'.pic'
    train = []
    for init in range( inits ):
      self._logger.info('train < neuron = %s, sort = %s >', neuron, sort)
      self._fastnet.new_ff([nInputs, neuron, 1], ['tansig', 'tansig'])
      nets = self._fastnet.train_ff()
      train.append( nets )
    
    self._logger.info('object saving...')
    objSave = [neuron, sort, inits, train]
    filehandler = open(fullOutput, 'w')
    pickle.dump(objSave, filehandler, protocol = 2 )
    self._logger.info('object saved!')
    



