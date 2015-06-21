#!/usr/bin/env python
import logging


class TrainJob():
  def __init__(self, logger = None ):
    import logging
    from FastNetTool.FastNet import FastNet
    self._logger = logger or logging.getLogger(__name__)
    self._fastnet = FastNet(2)
    self.lookParse = False


  def __call__(self, data, target, **kw ):

    import pickle
    import numpy as np
    from FastNetTool.Neural import Neural

    inputJob= kw.pop('inputJob', None)
    if inputJob:
      filehandler = open(inputJob, 'r')
      objLoad = pickle.load(filehandle, protocol=2)
      neuron = objLoad[0]
      sort = objLoad[1]
      inits = objLoad[2]
      cross = objLoad[3]
      del objLoad
    else:
      cross = kw.pop('cross', None)
      neuron = kw.pop('neuron',2)
      sort = kw.pop('sort',0)
      inits = kw.pop('inits',100)

    doMultiStop = kw.pop('doMultiStop', False)
    showEvo = kw.pop('showEvo', 5)
    epochs = kw.pop('epochs',1000)
    doPerf = kw.pop('doPerf', False)
    level = kw.pop('level', 1)
    output = kw.pop('output','train')
    del kw

    split = cross.getSort( data, sort )
    self._logger.info('Extracted cross validation sort')
    trainData = [split[0][0].tolist(), split[0][1].tolist()]
    valData   = [split[1][0].tolist(), split[1][1].tolist()]
    simData   = [np.vstack((split[0][0],split[1][0])).tolist(), np.vstack((split[0][1], split[1][1])).tolist()]

    if not self.lookParse:  self._fastnet.setData( trainData, valData, [], simData )
    self.lookParse = True
    self._fastnet.epochs        = epochs
    self._fastnet.show          = showEvo
    self._fastnet.doPerformance = doPerf
    self._fastnet.top           = neuron
    self._fastnet.batchSize     = len(trainData[1]) 

    if doMultiStop:  self._fastnet.useAll()
    else: self._fastnet.useSP()

    train = []
    fullOutput = output+'.n00'+str(neuron)+'.s00'+str(sort)+'.pic'

    for init in range( inits ):

      self._logger.info('train < neuron = %s, sort = %s >', neuron, sort)
      self._fastnet.initialize()
      self._fastnet.execute()
      train.append( self._fastnet.getNeuralObjectsList() )
    
    print 'object saving...'
    objSave = [neuron, sort, inits, train]
    filehandler = open(fullOutput, 'w')
    pickle.dump(objSave, filehandler, protocol = 2 )
    print 'object saved!'
    

trainJob = TrainJob()


