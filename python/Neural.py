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
from FastNetTool.util import checkForUnusedVars, Roc
import numpy as np
import math
import string

class DataTrainEvolution:
  """
    Class TrainDataEvolution is a sub class. This hold the train evolution into a
    list. Basically this is like a c++ struct.
  """
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



class Layer(Logger):
  def __init__(self, w, b, **kw):
    Logger.__init__( self, kw)
    self.layer = kw.pop('Layer',0)
    self.func  = kw.pop('Func' ,'tansig')
    checkForUnusedVars( kw, self._logger.warning )
    del kw
    self.W = np.matrix(w)
    self.b = np.transpose( np.matrix(b) )

  def __call_func(self, Y):
    if self.func == 'tansig':  return self.__tansig(Y)
    if self.func == 'sigmoid': return self.__sigmoid(Y)
 
  def __sigmoid(self, x):
    return (1 / (1 + np.exp(-x)))

  def __tansig(self, x):
    return (2 / (1 + np.exp(-2*x)))-1

  def __call__(self, X):
    B = self.b * np.ones((1, X.shape[1]))
    Y = np.dot(self.W,X)+B
    return self.__call_func(Y)
 
  def get_w_array(self):
    return np.array(np.reshape(self.W, (1,self.W.shape[0]*self.W.shape[1])))[0]
 
  def get_b_array(self):
    return np.array(np.reshape(self.b, (1,self.b.shape[0]*self.b.shape[1])))[0]

  def showInfo(self):
    self._logger.info('Layer: %d , function: %s, neurons: %d and inputs: %d',\
                      self.layer,self.func,self.W.shape[0],self.W.shape[1])

class Neural( Logger ):
  """
    Class Neural will hold the weights and bias information that came
    from fastnet core format
  """

  def __init__(self, net, **kw):
    Logger.__init__( self, kw )

    from FastNetTool.util import checkForUnusedVars
    train = kw.pop('train',None)
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    #Extract the information from c++ wrapper code
    self.nNodes         = []        
    self.numberOfLayers = net.getNumLayers()
    
    self.dataTrain      = None
    #Hold the train evolution information
    if train: self.dataTrain = DataTrainEvolution(train)
    self.layers = self.__retrieve(net)
    
    self._logger.debug('The Neural object was created.')

  '''
    This method can be used like this:
      outputVector = net( inputVector )
    where net is a Neural object intance and outputVector
    is a list with the same length of the input
  '''
  def __call__(self, input):
    Y = []
    for l in range(len(self.nNodes) - 1): 
      if l == 0: Y = self.layers[l](input)
      else: Y = self.layers[l](Y)
    return Y

  def showInfo(self):

    self._logger.info('The Neural configuration:')
    self._logger.info('input  layer: %d', self.nNodes[0])
    self._logger.info('hidden layer: %d', self.nNodes[1])
    self._logger.info('output layer: %d', self.nNodes[2])
    self._logger.info('The layers configuration:') 
    for l in range(len(self.nNodes) - 1):
      self.layers[l].showInfo()

  def get_w_array(self):
    w = np.array([])
    for l in range(len(self.nNodes) - 1):
      w = np.concatenate((w,self.layers[l].get_w_array()),axis=0)
    return w

  def get_b_array(self):
    b = np.array([])
    for l in range(len(self.nNodes) - 1):
      b = np.concatenate((b,self.layers[l].get_b_array()),axis=0)
    return b

  def __alloc_space(self, i, j, fill=0.0):
      n = []
      for m in range(i):
          n.append([fill]*j)
      return n
 
  def __retrieve(self, net):
    layers    = []
    w         = [] 
    b         = []
    layers    = []
    func      = []
    #Get nodes information  
    for l in range(self.numberOfLayers): 
      self.nNodes.append( net.getNumNodes(l) )

    for l in range(len(self.nNodes) - 1):
      func.append( net.getTrfFuncName(l) )
      w.append( self.__alloc_space(self.nNodes[l+1], self.nNodes[l]) )
      b.append( [0]*self.nNodes[l+1] )

    # Populate our matrxi from DiscriminatorpyWrapper
    for l in range( len(self.nNodes) - 1 ):
      for n in range( self.nNodes[l+1] ):
        for k in range( self.nNodes[l] ):
          w[l][n][k] = net.getWeight(l,n,k)
        b[l][n] = net.getBias(l,n)
      layers.append( Layer( w[l], b[l], Layer=l, Func=func[l] ) )
    return layers

from FastNetTool.LimitedTypeList import LimitedTypeList
NeuralCollection = LimitedTypeList('NeuralCollection',(),{'_acceptedTypes':(Neural,)})
