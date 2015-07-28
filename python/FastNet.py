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
       - tstData: (optional) The test data list
       - doMultiStop: do sp, fa and pd stop criteria (default = False).

'''
import sys
import os
from FastNetTool.util import sourceEnvFile, reshape_to_array, genRoc, Roc
#sourceEnvFile()
import numpy as np
from libFastNetTool     import FastnetPyWrapper
from FastNetTool.Neural import Neural
from FastNetTool.Logger import Logger

class FastNet(FastnetPyWrapper, Logger):

  def __init__( self, trnData, valData,  **kw ):
    Logger.__init__( self, **kw )
    from FastNetTool.util import checkForUnusedVars
    self.seed                = kw.pop('seed', None)
    if not self.seed is None:
      FastnetPyWrapper.__init__(self, self._level/10, self.seed)
    else:
      FastnetPyWrapper.__init__(self, self._level/10)
    self.batchSize           = kw.pop('batchSize',100)
    self.trainFcn            = kw.pop('trainFcn','trainrp')
    self.doPerf              = kw.pop('doPerf', False)
    self.show                = kw.pop('showEvo', 5)
    self.epochs              = kw.pop('epochs', 1000)
    doMultiStop              = kw.pop('doMultiStop', False) 
    tstData                  = kw.pop('tstData', None)
    self._inputSize          = trnData[0].shape[1]
    checkForUnusedVars(kw)
    del kw
    trnData = self.__to_array(trnData)
    valData = self.__to_array(valData)
    self.setTrainData( [trnData[0].tolist(), trnData[1].tolist()], self._inputSize )
    del trnData
    self.setValData(   [valData[0].tolist(), valData[1].tolist()], self._inputSize)

    if tstData: 
      tstData = self.__to_array(tstData)
      self.setTestData( [ tstData[0].tolist(), tstData[1].tolist() ], self._inputSize )
      del tstData

    del valData

    if doMultiStop: self.useAll()
    else: self.useSP()
    self._logger.info('Created FastNet class!')

  def __to_array(self, data):
    return (reshape_to_array(data[0]), reshape_to_array(data[1]))

  def new_ff(self, nodes, funcTrans = ['tansig', 'tansig']):
    self.newff(nodes, funcTrans, self.trainFcn)
    self._logger.info('Initalized newff!')

  def train_ff(self):
    netList = []
    [DiscriminatorPyWrapperList , TrainDataPyWrapperList] = self.train_c()
    for netPyWrapper in DiscriminatorPyWrapperList:
      tstPerf = None
      opPerf  = None
      if self.doPerf:
        perfList = self.valid_c(netPyWrapper)
        opPerf   = Roc( perfList[1], 'operation' )
        self._logger.info('Operation: sp = %f, det = %f and fa = %f',opPerf.sp,opPerf.det,opPerf.fa)
        tstPerf  = Roc( perfList[0] , 'test')
        self._logger.info('Test: sp = %f, det = %f and fa = %f',tstPerf.sp,tstPerf.det,tstPerf.fa)
      
      netList.append( [Neural(netPyWrapper, train=TrainDataPyWrapperList), tstPerf, opPerf] )

    return netList


  
  
  
