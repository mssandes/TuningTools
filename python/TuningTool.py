'''
  Author: Joao Victor da Fonseca Pinto
  Email: jodafons@cern.ch 
  Description:
       TuningTool: This class is used to connect the python interface and
       the c++ ExMachina core. Please check the list 
       of methods below:
      
       - neuron: the number of neurons into the hidden layer.
       - batchSise:
       - doPerf: do performance analisys (default = False)
       - trnData: the train data list
       - valData: the validate data list
       - testData: (optional) The test data list

'''
import numpy as np
import exmachina
from RingerCore.Logger  import Logger, LoggingLevel
from RingerCore.util    import genRoc, Roc


class TuningTool(Logger):
  """
    TuningTool is the higher level representation of the TuningToolPyWrapper class.
  """
  trainOptions=dict()
  _trnData  =None
  _valData  =None
  _tstData  =None
  _trnTarget=None
  _valTrget =None
  _tstTarget=None

  def __init__( self, **kw ):
    Logger.__init__( self, kw )
    from RingerCore.util import checkForUnusedVars
    self.trainOptions['batchSize']     = kw.pop('batchSize'     ,  100            )
    self.trainOptions['networkArch']   = kw.pop('networkArch'   ,  'feedforward'  )
    self.trainOptions['algorithmName'] = kw.pop('algorithmName' ,  'rprop'        )
    self.trainOptions['contFunction']  = kw.pop('costFunction'  ,  'sp'           )
    self.trainOptions['print']         = kw.pop('print'         ,  False          )
    self.trainOptions['nEpochs']       = kw.pop('nEpochs'       ,  1000           )
    self.trainOptions['nFails']        = kw.pop('nFails'        ,  50             )
    
    self.trainOptions['shuffle']       = True
    checkForUnusedVars(kw, self._logger.warning )
    del kw

  def setTrainData(self, trnData, trnTarget):
    """
      Overloads setTrainData to change numpy array to its
      ctypes representation.
    """
    if not np.isfortran(trnData):
      raise TypeError('[train] data numpy order is not fortran!')
    elif not np.isfortran(trnTarget):
      raise TypeError('[train] target numpy order is not fortran!')
    else:
      self._trnData = exmachina.DataHandler(trnData,trnTarget)
      self._trnTarget = trnTarget


  def setValData(self, valData, valTarget):
    """
      Overloads setTrainData to change numpy array to its
      ctypes representation.
    """
    if not np.isfortran(valData):
      raise TypeError('[validation] data numpy order is not fortran!')
    elif not np.isfortran(valTarget):
      raise TypeError('[validation] target numpy order is not fortran!')
    else:
      self._valData = exmachina.DataHandler(valData,valTarget)
      self._valTarget = valTarget


  def setTestData(self, tstData, tstTarget):
    """
      Overloads setTrainData to change numpy array to its
      ctypes representation.
    """
    if not np.isfortran(tstData):
      raise TypeError('[test] data numpy order is not fortran!')
    elif not np.isfortran(tstTarget):
      raise TypeError('[test] target numpy order is not fortran!')
    else:
      self._tstData = exmachina.DataHandler(tstData,tstTarget)
      self._tstTarget = tstTarget


  def newff(self, nodes, funcTrans = ['tanh', 'tanh']):
    """
      Creates new feedforward neural network
    """
    self._logger.info('Initalizing newff...')
    self._net = exmachina.FeedForward(nodes,funcTrans,'nw')

  def train_c(self):
    """
      Train feedforward neural network
    """
    self._logger.info('Initalizing train_c')

    try:
      trainer = exmachina.NeuralNetworkTrainer(self._net,
        [self._trnData,self._valData, self._tstData] if self._tstData\
        else [self._trnData, self._valData, self._valData],
        self.trainOptions)
    except:
      raise RuntimeError('Can not initialize the trainer. Abort!')

    self._logger.info('execute train_c')
    try:
      trainer.train()
    except:
      raise RuntimeError('Can not execute the trainer. Abort!')

    self._logger.debug('Successfully exited C++ training.')

    tunedDiscrData= self.__neural_to_dict( self._net )
    trnOutput = self._net.propagateDataset(self._trnData)[0]
    valOutput = self._net.propagateDataset(self._valData)[0]
   
    if self._tstData: tstOutput = self._net.propagateDataset(self._tstData)

    allOutput = np.concatenate([trnOutput,valOutput,tstOutput] if self._tstData\
                               else [trnOutput,valOutput],axis=1)
    allTarget = np.concatenate([self._trnTarget,self._valTarget, self._tstTarget] if self._tstData\
                               else [self._trnTarget, self._valTarget],axis=1)

    operationReceiveOperationCurve = Roc( self.__generateReceiveOperationCurve(allOutput,allTarget),
                                                            'operation')
    if self._tstData:
      testReceiveOperationCurve = Roc( self.__generateReceiveOperationCurve(tstOutput,self._tstTarget),
                                                            'test')
    else:
      testReceiveOperationCurve = Roc( self.__generateReceiveOperationCurve(valOutput,self._valTarget),
                                                            'val')

    self._logger.info('Operation: sp = %f, det = %f and fa = %f', \
                      operationReceiveOperationCurve.sp, 
                      operationReceiveOperationCurve.det, 
                      operationReceiveOperationCurve.fa)

    self._logger.info('Test: sp = %f, det = %f and fa = %f', \
                      testReceiveOperationCurve.sp, 
                      testReceiveOperationCurve.det, 
                      testReceiveOperationCurve.fa)

    tunedDiscrData['summaryInfo'] = dict()
    tunedDiscrData['summaryInfo']['roc_operation'] = operationReceiveOperationCurve
    tunedDiscrData['summaryInfo']['roc_test'] = testReceiveOperationCurve

    del trnOutput, valOutput,  allOutput, allTarget 
    if self._tstData:  del tstOutput

    self._logger.debug("Finished train_c on python side.")
    return (tunedDiscrData)


  def __neural_to_dict(self, net):
    obj = dict()
    obj['trainEvolution'] = dict()
    obj['trainEvolution']['epoch']   = list()
    obj['trainEvolution']['mse_trn'] = list()
    obj['trainEvolution']['mse_val'] = list()
    obj['trainEvolution']['mse_tst'] = list()
    obj['trainEvolution']['sp_val']  = list()
    obj['trainEvolution']['sp_tst']  = list()
    obj['trainEvolution']['det_val'] = list()
    obj['trainEvolution']['det_tst'] = list()
    obj['trainEvolution']['fa_val']  = list()
    obj['trainEvolution']['fa_tst']  = list()
    obj['network'] = dict()
    obj['network']['nodes']   = net.layers
    obj['network']['weights'] = net.weights
    obj['network']['bias']    = net.bias
    self._logger.debug('neural to dictionary.')

    return obj


  def __generateReceiveOperationCurve( self, output, target ):
    if len(np.unique(target)) != 2:
      raise RuntimeError('The number of patterns > 2. Abort generateReceiveOperationCurve method.')
    sgn = output[np.where(target ==  1)[1]].T
    noise = output[np.where(target == -1)[1]].T
    return genRoc(sgn,noise)

  def concatenate_patterns(self, patterns):

    if type(patterns) is list:
      if len(patterns) == 2: tgt = [1,-1]
      else: tgt=range(len(patterns)) 
      data=None; target=None; idx=0
      for cl in patterns:
        if idx==0:
          data = cl.T
          target = tgt[idx]*np.ones((1,len(cl)), order='F',dtype='double')
        else:
          data = np.concatenate((data,cl.T),axis=1)
          target = np.concatenate( (target,tgt[idx]*np.ones((1,len(cl)),dtype='double',order='F')), axis=1)
        idx+=1
      self._logger.debug('data shape is %s and target shape is %s',data.shape[1],target.shape[1])
      #FIXME: There is some problem into concatenate numpy method. This doest return a
      #vector with fortran ordem.
      return data, np.array(target,order='F',dtype='double')
    else:
      raise RuntimeError('Can not concatenate patterns, error type from constructor')

  def separate_patterns(self, data, target):
    try: 
      patterns = list()
      targets  = np.unique(target).tolist(); idx=0
      if len(targets) == 2: targets = [1, -1]

      for tgt in targets:
        patterns.append(data[:,np.where(target==tgt)[1]].T)
        self._logger.debug('pattern %i shape is [%d,%d]',idx, patterns[idx].shape[0],patterns[idx].shape[1])
        idx+=1
      self._logger.debug('Number of patterns is: %d',len(patterns))
      return patterns
    except:
      raise RuntimeError('Can not separate patterns. Abort!')






