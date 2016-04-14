__all__ = ['DataTrainEvolution', 'Layer', 'Neural', 'NeuralCollection']

import numpy as np
from RingerCore import LimitedTypeList, checkForUnusedVars, Logger

class DataTrainEvolution:

  """
    Class TrainDataEvolution is a sub class. This hold the train evolution into a
    list. Basically this is like a c++ struct.
  """
  def __init__(self, train=None, full_data=False):
    #Slim data
    self.mse_trn        = list()  
    self.mse_val        = list()  
    self.sp_val         = list()  
    self.det_val        = list()  
    self.fa_val         = list()  
    self.mse_tst        = list()  
    self.sp_tst         = list()  
    self.det_tst        = list()  
    self.fa_tst         = list()  
    self.det_fitted     = list()  
    self.fa_fitted      = list()  

    #Train evolution information
    is_best_mse    = list() 
    is_best_sp     = list() 
    is_best_det    = list() 
    is_best_fa     = list()  
 
    if full_data:
      self.num_fails_mse  = list() 
      self.num_fails_sp   = list() 
      self.num_fails_det  = list() 
      self.num_fails_fa   = list() 
      self.stop_mse       = list() 
      self.stop_sp        = list()    
      self.stop_det       = list() 
      self.stop_fa        = list() 

    #Get train evolution information from TrainDatapyWrapper
    if train is not None:
      self.maxEpoch = len(train)
      for i in range(len(train)):
        self.mse_trn.append(train[i].mseTrn)
        self.mse_val.append(train[i].mseVal)
        self.sp_val.append(train[i].spVal)
        self.det_val.append(train[i].detVal)
        self.fa_val.append(train[i].faVal)
        self.mse_tst.append(train[i].mseTst)
        self.sp_tst.append(train[i].spTst)
        self.det_tst.append(train[i].detTst)
        self.fa_tst.append(train[i].faTst)
        self.det_fitted.append(train[i].detFitted)
        self.fa_fitted.append(train[i].faFitted)
        
        is_best_mse.append(train[i].isBestMse)
        is_best_sp.append(train[i].isBestSP)
        is_best_det.append(train[i].isBestDet)
        is_best_fa.append(train[i].isBestFa)

        if full_data:
          self.num_fails_mse.append(train[i].numFailsMse)
          self.num_fails_sp.append(train[i].numFailsSP)
          self.num_fails_det.append(train[i].numFailsDet)
          self.num_fails_fa.append(train[i].numFailsFa)
          self.stop_mse.append(train[i].stopMse)
          self.stop_sp.append(train[i].stopSP)
          self.stop_det.append(train[i].stopDet)
          self.stop_fa.append(train[i].stopFa)

      self.epoch_best_mse = self.__lastIndex(is_best_mse,  True)
      self.epoch_best_sp  = self.__lastIndex(is_best_sp ,  True)
      self.epoch_best_det = self.__lastIndex(is_best_det,  True)
      self.epoch_best_fa  = self.__lastIndex(is_best_fa ,  True)

  def toRawObj(self):
    "Return a raw dict object from itself"
    from copy import copy # Every complicated object shall be changed to a rawCopyObj
    raw = copy(self.__dict__)
    return raw

  def __lastIndex(self,  l, value ):
    try:
      l.reverse()
      return len(l) - 1 - l.index(value)
    except ValueError:
      return len(l) - 1 

class Layer:
  def __init__(self, w, b, **kw):
    self.layer = kw.pop('Layer',0)
    self.func  = kw.pop('Func' ,'tansig')
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
    print ('Layer: %d , function: %s, neurons: %d and inputs: %d')%\
          (self.layer,self.func,self.W.shape[0],self.W.shape[1])



class Neural:
  """
    Class Neural will hold the weights and bias information that came
    from tuningtool core format
  """

  def __init__(self, net, **kw):

    train = kw.pop('train',None)
    del kw

    #Extract the information from c++ wrapper code
    self.nNodes         = []        
    self.layers         = []
    self.numberOfLayers = 0
    self.dataTrain      = None

    #Hold the train evolution information
    if net: 
      self.numberOfLayers = net.getNumLayers()
      self.layers = self.__retrieve(net)

  def __call__(self, input):
    '''
      This method can be used like this:
        outputVector = net( inputVector )
      where net is a Neural object intance and outputVector
      is a list with the same length of the input
    '''
    Y = []
    for l in range(len(self.nNodes) - 1): 
      if l == 0: Y = self.layers[l](input)
      else: Y = self.layers[l](Y)
    return Y


  def rawDiscrDict(self):
    return {
             'nodes' : self.nNodes,
             'weights' : self.get_w_array(),
             'bias' : self.get_b_array(),
           }

  def showInfo(self):
    print  'The Neural configuration:'
    print ('input  layer: %d') % (self.nNodes[0])
    print ('hidden layer: %d') % (self.nNodes[1])
    print ('output layer: %d') % (self.nNodes[2])
    print 'The layers configuration:'
 
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

NeuralCollection = LimitedTypeList('NeuralCollection',(),{'_acceptedTypes':(Neural,)})

######################################################################################
######################################################################################
#
#                             Neural version decrepted
#
######################################################################################
######################################################################################

class OldLayer(Logger):
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

class OldNeural( Logger ):
  """
    Class Neural will hold the weights and bias information that came
    from tuningtool core format
  """

  def __init__(self, net, **kw):
    Logger.__init__( self, kw )

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

