#!/usr/bin/env python
"""
  Class: Neural
  author: Joao Victor da Fonseca Pinto
  email: joao.victor.da.fonseca.pinto@cern.ch
  
  Neural Netowrk single class using to export the weghts and bias from 
  FastNet c++ core to python object. This can be uses to save into a file

  Constructor:
      Neural( DiscriminatorPyWrapper, TrainDataPyWrapper )
  Methods:
      - output operator()( input ): This method is more generic and can be used
        to calculate the netrk output for a list of events.
      - singleOutput = propagateInput ( singleInput ): This calculate the net
        output for only one event
"""

from FastNetTool.Logger import Logger

import math
import string
"""
  Class TrainDataEvolution is a sub class. This hold the train evolution into a
  list. Basically this is like a c++ struct.
"""
class DataTrainEvolution:
  def __init__(self, train):
    
    #Train evolution information
    self.epoch          = []
    self.mse_trn        = []
    self.mse_val        = []
    self.sp_val         = []
    self.det_val        = []
    self.fa_val         = []
    self.mse_tst        = []
    self.sp_tst         = []
    self.det_tst        = []
    self.fa_tst         = []
    self.is_best_mse    = []
    self.is_best_sp     = []
    self.is_best_det    = []
    self.is_best_fa     = []
    self.num_fails_mse  = []
    self.num_fails_sp   = []
    self.num_fails_det  = []
    self.num_fails_fa   = []
    self.stop_mse       = []
    self.stop_sp        = []   
    self.stop_det       = []   
    self.stop_fa        = []   
    
    #Get train evolution information from TrainDatapyWrapper
    for i in range(len(train)):

      self.epoch.append(train[i].epoch)
      self.mse_trn.append(train[i].mseTrn)
      self.mse_val.append(train[i].mseVal)
      self.sp_val.append(train[i].spVal)
      self.det_val.append(train[i].detVal)
      self.fa_val.append(train[i].faVal)
      self.mse_tst.append(train[i].mseTst)
      self.sp_tst.append(train[i].spTst)
      self.det_tst.append(train[i].detTst)
      self.fa_tst.append(train[i].faTst)
      self.is_best_mse.append(train[i].isBestMse)
      self.is_best_sp.append(train[i].isBestSP)
      self.is_best_det.append(train[i].isBestDet)
      self.is_best_fa.append(train[i].isBestFa)
      self.num_fails_mse.append(train[i].numFailsMse)
      self.num_fails_sp.append(train[i].numFailsSP)
      self.num_fails_det.append(train[i].numFailsDet)
      self.num_fails_fa.append(train[i].numFailsFa)
      self.stop_mse.append(train[i].stopMse)
      self.stop_sp.append(train[i].stopSP)
      self.stop_det.append(train[i].stopDet)
      self.stop_fa.append(train[i].stopFa)

    self.epoch_best_sp  = self.__lastIndex(self.is_best_sp,  True)
    self.epoch_best_det = self.__lastIndex(self.is_best_det, True)
    self.epoch_best_fa  = self.__lastIndex(self.is_best_fa,  True)

  def __lastIndex(self,  l, value ):
    l.reverse()
    return len(l)  -1 - l.index(value)


"""
  Class Neural will hold the weights and bias information that came
  from fastnet core format
"""
class Neural( Logger ):
  def __init__(self, net, **kw):
    Logger.__init__( self, **kw)

    train = kw.pop('train',None)
    from FastNetTool.util import checkForUnusedVars
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    #Extract the information from c++ wrapper code
    self.nNodes         = []        
    self.numberOfLayers = net.getNumLayers()
    self.w              = []
    self.b              = []
    self.trfFunc        = []
    self.layerOutput    = []
    self.dataTrain      = None
    #Hold the train evolution information
    if train:
      self.dataTrain = DataTrainEvolution(train)

    #Get nodes information  
    for l in range(self.numberOfLayers):
      self.nNodes.append( net.getNumNodes(l) )

    self.layerOutput.append([])
    for l in range(len(self.nNodes) - 1):
      self.trfFunc.append( net.getTrfFuncName(l) )
      #alloc space for weight
      self.w.append( self.__alloc_space(self.nNodes[l+1], self.nNodes[l]) )
      #alloc space for bias          
      self.b.append( [0]*self.nNodes[l+1] )
      self.layerOutput.append( [0]*self.nNodes[l+1] )

    #Population matrix from DiscriminatorpyWrapper
    for l in range( len(self.nNodes) - 1 ):
      for n in range( self.nNodes[l+1] ):
        for k in range( self.nNodes[l] ):
          self.w[l][n][k] = net.getWeight(l,n,k)
        self.b[l][n] = net.getBias(l,n)

    self._logger.debug('The Neural object was created.')

  '''
    This method can be used like this:
      outputVector = net( inputVector )
    where net is a Neural object intance and outputVector
    is a list with the same length of the input
  '''
  def __call__(self, input):
    [numEvents, numInputs] = self.__size(input)
    if numEvents == 1:
      return self.__propagateInput( input )
    else:
      outputVec = []
      for event in input:
        outputVec.append( self.__propagateInput( input ) )
      return outputVec      


  def __sigmoid(self, x):
      return math.tanh(x)
  
  def __call_trf_func(self, input, type):
      return sigmoid(input)


  def __alloc_space(self, i, j, fill=0.0):
      n = []
      for m in range(i):
          n.append([fill]*j)
      return n
  
  def __size(self, l):
    try:
      row = len(l)
      col = len(l[0])
    except:
      row = 1
      col = len(l)
    return [row, col]    
  
  def __propagateInput(self, input):

    self.layerOutput[0] = input 
    for l in range( len(self.nNodes) - 1 ):
      for n in range( self.nNodes[l+1] ):
        self.layerOutput[l+1][n] = self.b[l][n]
        for k in range( self.nNodes[l] ):
          self.layerOutput[l+1][n] = self.layerOutput[l+1][n] + self.layerOutput[l][k]*self.w[l][n][k]
        self.layerOutput[l+1][n] = self.__call_trf_func(self.layerOutput[l+1][n],  self.trfFunc[l])

    if(self.nNodes[len(self.nNodes)-1]) == 1: 
      return self.layerOutput[len(self.nNodes)-1][0]
    else:  
      return self.layerOutput[len(self.nNodes)-1]




