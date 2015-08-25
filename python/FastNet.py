#!/usr/bin/env python
'''
  Author: Joao Victor da Fonseca Pinto
  Email: jodafons@cern.ch 
  Description:
       FastNet: This class is used to connect the python interface and
       the c++ fastnet core. The class FastnetPyWrapper have some methods
       thad can be used to set some train param. Please check the list 
       of methods below:
      
       - neuron: the number of neurons into the hidden layer.
       - batchSise:
       - doPerf: do performance analisys (default = False)
       - trnData: the train data list
       - valData: the validate data list
       - testData: (optional) The test data list
       - doMultiStop: do sp, fa and pd stop criteria (default = False).

'''
import numpy as np
from libFastNetTool     import FastnetPyWrapper
from FastNetTool.Neural import Neural
from FastNetTool.Logger import Logger

class FastNet(FastnetPyWrapper, Logger):
  """
    FastNet is the higher level representation of the FastnetPyWrapper class.
  """

  def __init__( self, **kw ):
    Logger.__init__( self, kw )
    FastnetPyWrapper.__init__(self, self.level/10)
    from FastNetTool.util import checkForUnusedVars
    self.seed                = kw.pop('seed',        None    )
    self.batchSize           = kw.pop('batchSize',    100    )
    self.trainFcn            = kw.pop('trainFcn',  'trainrp' )
    self.doPerf              = kw.pop('doPerf',      False   )
    self.showEvo             = kw.pop('showEvo',       5     )
    self.epochs              = kw.pop('epochs',      1000    )
    self.doMultiStop         = kw.pop('doMultiStop', False   ) 
    checkForUnusedVars(kw, self._logger.warning )
    del kw

  def getMultiStop(self):
    return self._doMultiStop

  def setDoMultistop(self, value):
    """
      doMultiStop: Sets FastnetPyWrapper self.useAll() if set to true,
        otherwise sets to self.useSP()
    """
    if value: 
      self._doMultiStop = True
      self.useAll()
    else: 
      self._doMultiStop = False
      self.useSP()

  doMultiStop = property( getMultiStop, setDoMultistop )

  def getSeed(self):
    return FastnetPyWrapper.getSeed(self)

  def setSeed(self, value):
    """
      Set seed value
    """
    import ctypes
    if not value is None: 
      return FastnetPyWrapper.setSeed(self,value)

  seed = property( getSeed, setSeed )

  def setTrainData(self, trnData):
    """
      Overloads FastnetPyWrapper setTrainData to change numpy array to its
      ctypes representation.
    """
    if trnData:
      self._logger.debug("Setting trainData to new representation.")
    else:
      self._logger.debug("Emptying trainData.")
    FastnetPyWrapper.setTrainData(self, trnData)

  def setValData(self, valData):
    """
      Overloads FastnetPyWrapper setValData to change numpy array to its
      ctypes representation.
    """
    if valData:
      self._logger.debug("Setting valData to new representation.")
    else:
      self._logger.debug("Emptying valData.")
    FastnetPyWrapper.setValData(self, valData)

  def setTestData(self, testData):
    """
      Overloads FastnetPyWrapper setTstData to change numpy array to its
      ctypes representation.
    """
    if testData:
      self._logger.debug("Setting testData to new representation.")
    else:
      self._logger.debug("Emptying testData.")
    FastnetPyWrapper.setTestData(self, testData)

  def newff(self, nodes, funcTrans = ['tansig', 'tansig']):
    """
      Creates new feedforward neural network
    """
    self._logger.info('Initalizing newff...')
    FastnetPyWrapper.newff(self, nodes, funcTrans, self.trainFcn)

  def train_c(self):
    """
      Train feedforward neural network
    """
    from FastNetTool.util import Roc
    netList = []
    [DiscriminatorPyWrapperList , TrainDataPyWrapperList] = \
        FastnetPyWrapper.train_c(self)
    self._logger.debug('Successfully exited C++ training.')
    for netPyWrapper in DiscriminatorPyWrapperList:
      tstPerf = None
      opPerf  = None
      if self.doPerf:
        self._logger.debug('Calling valid_c to retrieve performance.')
        perfList = self.valid_c(netPyWrapper)
        opPerf   = Roc( perfList[1], 'operation' )
        self._logger.info('Operation: sp = %f, det = %f and fa = %f', \
            opPerf.sp, opPerf.det, opPerf.fa)
        tstPerf  = Roc( perfList[0] , 'test')
        self._logger.info('Test: sp = %f, det = %f and fa = %f', \
            tstPerf.sp, tstPerf.det, tstPerf.fa)
        self._logger.debug('Finished valid_c on python side.')
      netList.append( [Neural(netPyWrapper, train=TrainDataPyWrapperList), \
          tstPerf, opPerf] )
    self._logger.debug("Finished train_c on python side.")
    return netList


