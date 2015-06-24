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
import numpy as np
from libFastNetTool     import FastnetPyWrapper
from FastNetTool.Neural import Neural

class FastNet(FastnetPyWrapper):

  def __init__( self, trnData, valData,  **kw ):

    #retrieve python logger
    import logging
    self._level = kw.pop('level', logging.INFO)
    logging.basicConfig( level=self._level )

    self._logger = logging.getLogger(__name__)
    FastnetPyWrapper.__init__(self, self._level/10)
    self.batchSize           = kw.pop('batchSize',100)
    self.trainFcn            = kw.pop('trainFcn','trainrp')
    self.doPerf              = kw.pop('doPerf', False)
    self.simData             = kw.pop('simData', None)
    doMultiStop              = kw.pop('doMultiStop', False) 
    tstData                  = kw.pop('tstData', None)
    
    self.setTrainData( [trnData[0].tolist(), trnData[1].tolist()] )
    del trnData
    self.setValData(   [valData[0].tolist(), valData[1].tolist()] )
    del valData

    if tstData: self.setTestData( [ tstData[0].tolist(), tstData[1].tolist() ] )
    else: self.setTestData( [] )
    
    if doMultiStop: self.useAll()
    else: self.useSP()
    self._logger.info('create class fastnet')

  def new_ff(self, nodes, funcTrans = ['tansig', 'tansig']):
    self.newff(nodes, funcTrans, self.trainFcn)
    self._logger.info('init newff')

  def train_ff(self):
    netList = []
    [DiscriminatorPyWrapperList , TrainDataPyWrapperList] = self.train()

    for netPyWrapper in DiscriminatorPyWrapperList:
      net = Neural( netPyWrapper, train=TrainDataPyWrapperList,  level=self._level ) 
      if self.doPerf:
        out_sim  = [self.sim(netPyWrapper, self.simData[0]), self.sim( netPyWrapper, self.simData[1])]
        out_tst  = [self.sim(netPyWrapper, self.tstData[0]), self.sim( netPyWrapper, self.tstData[1])]
        [spVec, cutVec, detVec, faVec] = self.__genRoc( out_sim[0], out_sim[1], 1000 )
        net.setSimPerformance( spVec,detVec,faVec,cutVec )
        [spVec, cutVec, detVec, faVec] = self.__genRoc( out_tst[0], out_tst[1], 1000 )
        net.setTstPerformance(spVec,detVec,faVec,cutVec)
      
      netList.append( net )
    return netList


  '''
    Private methods
  '''
  def __geomean(nums):
    return (reduce(lambda x, y: x*y, nums))**(1.0/len(nums))
  
  def __mean(nums):
    return (sum(nums)/len(nums))


  def __getEff( self, outSignal, outNoise, cut ):
    '''
      [detEff, faEff] = getEff(outSignal, outNoise, cut)
      Returns the detection and false alarm probabilities for a given input
      vector of detector's perf for signal events(outSignal) and for noise 
      events (outNoise), using a decision threshold 'cut'. The result in whithin
      [0,1].
    '''
    detEff = np.where(outSignal >= cut)[0].shape[0]/ float(outSignal.shape[0]) 
    faEff  = np.where(outNoise >= cut)[0].shape[0]/ float(outNoise.shape[0])
    return [detEff, faEff]
  
  def __calcSP( self, pd, pf ):
    '''
      ret  = calcSP(x,y) - Calculates the normalized [0,1] SP value.
      effic is a vector containing the detection efficiency [0,1] of each
      discriminating pattern.  
    '''
    return math.sqrt(self.__geomean([pd,pf]) * self.__mean([pd,pf]))
  
  def __genRoc( self, outSignal, outNoise, numPts = 1000 ):
    '''
      [spVec, cutVec, detVec, faVec] = genROC(out_signal, out_noise, numPts, doNorm)
      Plots the RoC curve for a given dataset.
      Input Parameters are:
         out_signal     -> The perf generated by your detection system when
                           electrons were applied to it.
         out_noise      -> The perf generated by your detection system when
                           jets were applied to it
         numPts         -> (opt) The number of points to generate your ROC.
      
      If any perf parameters is specified, then the ROC is plot. Otherwise,
      the sp values, cut values, the detection efficiency and false alarm rate 
      are returned (in that order).
    '''
    cutVec = np.arange( -1,1,2 /float(numPts))
    cutVec = cutVec[0:numPts - 1]
    detVec = np.array(cutVec.shape[0]*[float(0)])
    faVec  = np.array(cutVec.shape[0]*[float(0)])
    spVec  = np.array(cutVec.shape[0]*[float(0)])
    for i in range(cutVec.shape[0]):
      [detVec[i],faVec[i]] = getEff( np.array(outSignal), np.array(outNoise),  cutVec[i] ) 
      spVec[i] = calcSP(detVec[i],1-faVec[i])
    return [spVec.tolist(), cutVec.tolist(), detVec.tolist(), faVec.tolist()]
  
  
  
