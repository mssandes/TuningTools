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
    self.mse_tst        = list()  
    
    self.bestsp_point_sp_val     = list()  
    self.bestsp_point_det_val    = list()  
    self.bestsp_point_fa_val     = list()  
    self.bestsp_point_sp_tst     = list()  
    self.bestsp_point_det_tst    = list()  
    self.bestsp_point_fa_tst     = list()  
    self.det_point_sp_val        = list()  
    self.det_point_det_val       = list()  
    self.det_point_fa_val        = list()  
    self.det_point_sp_tst        = list()  
    self.det_point_det_tst       = list()  
    self.det_point_fa_tst        = list()  
    self.fa_point_sp_val         = list()  
    self.fa_point_det_val        = list()  
    self.fa_point_fa_val         = list()  
    self.fa_point_sp_tst         = list()  
    self.fa_point_det_tst        = list()  
    self.fa_point_fa_tst         = list()  

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
        
        self.mse_trn.append              ( train[i].mseTrn               )
        self.mse_val.append              ( train[i].mseVal               )
        self.mse_tst.append              ( train[i].mseTst               )

        self.bestsp_point_sp_val.append  ( train[i].bestsp_point_sp_val  )
        self.bestsp_point_det_val.append ( train[i].bestsp_point_det_val )
        self.bestsp_point_fa_val.append  ( train[i].bestsp_point_fa_val  )
        self.bestsp_point_sp_tst.append  ( train[i].bestsp_point_sp_tst  )
        self.bestsp_point_det_tst.append ( train[i].bestsp_point_det_tst )
        self.bestsp_point_fa_tst.append  ( train[i].bestsp_point_fa_tst  )
 
        self.det_point_sp_val.append     ( train[i].det_point_sp_val     )
        self.det_point_det_val.append    ( train[i].det_point_det_val    )
        self.det_point_fa_val.append     ( train[i].det_point_fa_val     )
        self.det_point_sp_tst.append     ( train[i].det_point_sp_tst     )
        self.det_point_det_tst.append    ( train[i].det_point_det_tst    )
        self.det_point_fa_tst.append     ( train[i].det_point_fa_tst     )
 
        self.fa_point_sp_val.append      ( train[i].fa_point_sp_val      )
        self.fa_point_det_val.append     ( train[i].fa_point_det_val     )
        self.fa_point_fa_val.append      ( train[i].fa_point_fa_val      )
        self.fa_point_sp_tst.append      ( train[i].fa_point_sp_tst      )
        self.fa_point_det_tst.append     ( train[i].fa_point_det_tst     )
        self.fa_point_fa_tst.append      ( train[i].fa_point_fa_tst      )
        
        is_best_mse.append               ( train[i].isBestMse            )
        is_best_sp.append                ( train[i].isBestSP             )
        is_best_det.append               ( train[i].isBestDet            )
        is_best_fa.append                ( train[i].isBestFa             )

        if full_data:
          self.num_fails_mse.append ( train[i].numFailsMse )
          self.num_fails_sp.append  ( train[i].numFailsSP  )
          self.num_fails_det.append ( train[i].numFailsDet )
          self.num_fails_fa.append  ( train[i].numFailsFa  )
          self.stop_mse.append      ( train[i].stopMse     )
          self.stop_sp.append       ( train[i].stopSP      )
          self.stop_det.append      ( train[i].stopDet     )
          self.stop_fa.append       ( train[i].stopFa      )

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

  def __str__(self):
    return ('Layer: %d , function: %s, neurons: %d and inputs: %d')%\
           (self.layer,self.func,self.W.shape[0],self.W.shape[1])


class Neural:
  """
    Class Neural will hold the weights and bias information that came
    from tuningtool core format
  """
  def __init__(self, name):
    self._name    = name
    self._nodes   = list()        
    self._layers  = list()
    self._nLayers = 0


  def __call__(self, input):
    '''
      This method can be used like this:
        outputVector = net( inputVector )
      where net is a Neural object intance and outputVector
      is a list with the same length of the input
    '''
    Y = []
    for l in range(len(self._nodes) - 1): 
      if l == 0: Y = self._layers[l](input)
      else: Y = self._layers[l](Y)
    return Y


  def rawDiscrDict(self):
    return {
             'nodes' : self._nodes,
             'weights' : self.get_w_array(),
             'bias' : self.get_b_array(),
           }

  def show(self):
    print  'The Neural configuration:'
    print ('input  layer: %d') % (self._nodes[0])
    print ('hidden layer: %d') % (self._nodes[1])
    print ('output layer: %d') % (self._nodes[2])
    print 'The layers configuration:'
    for layer in range(self._nLayers):
      print self._layers[layer]

  def get_w_array(self):
    w = np.array([])
    for l in range(len(self._nodes) - 1):
      w = np.concatenate((w,self._layers[l].get_w_array()),axis=0)
    return w

  def get_b_array(self):
    b = np.array([])
    for l in range(len(self._nodes) - 1):
      b = np.concatenate((b,self._layers[l].get_b_array()),axis=0)
    return b

  def nodes(self):
    return self._nodes

  def layers(self):
    self._layers

  def set_from_dict(self, rawDict):
    #Retrieve nodes information
    self._nodes = rawDict['nodes']
    self._nLayers = len(self._nodes)-1
    w = rawDict['weights'].tolist()
    b = rawDict['bias'].tolist()
    self._layers=[]
    #Loop over layers
    for layer in range(self._nLayers):
      W=[]; B=[]
      for count in range(self._nodes[layer]*self._nodes[layer+1]):
         W.append(w.pop(0))
      W=np.array(W).reshape(self._nodes[layer+1], self._nodes[layer]).tolist()
      for count in range(self._nodes[layer+1]):
        B.append(b.pop(0))
      self._layers.append( Layer(W,B,Layer=layer) )

  def set_from_fastnet(self, fastnetObj):
    w = []; b = [];
    #Reset layers
    self._layers    = []

    #Retrieve nodes information
    self._nLayers = fastnetObj.getNumLayers()
    for layer in range(self._nLayers): 
      self._nodes.append( fastnetObj.getNumNodes(layer) )
    #Alloc zeros
    for layer in range(len(self._nodes) - 1):
      #func.append( net.getTrfFuncName(layer) )
      w.append( [ [0]*self._nodes[layer] for d in range( self._nodes[layer+1])]  )
      b.append( [0]*self._nodes[layer+1] )
    #Populate our matrix from DiscriminatorpyWrapper
    for layer in range( len(self._nodes) - 1 ):
      for n in range( self._nodes[layer+1] ):
        for k in range( self._nodes[layer] ):
          w[layer][n][k] = fastnetObj.getWeight(layer,n,k)
        b[layer][n] = fastnetObj.getBias(layer,n)
      self._layers.append( Layer( w[layer], b[layer], Layer=layer) )

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

