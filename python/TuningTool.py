'''
  Author: Joao Victor da Fonseca Pinto
  Email: jodafons@cern.ch 
  Description:
       TuningTool: This class is used to connect the python interface and
       the c++ tuningtool core. The class TuningToolPyWrapper have some methods
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
import exmachina
from RingerCore.Logger  import Logger, LoggingLevel
from RingerCore.util    import genRoc, Roc


class TuningTool(Logger):
  """
    TuningTool is the higher level representation of the TuningToolPyWrapper class.
  """
  trainOptions=dict()

  def __init__( self, **kw ):
    Logger.__init__( self, kw )
    from RingerCore.util import checkForUnusedVars
    self.trainOptions['batchSize']     = kw.pop('batchSize'     ,  100            )
    self.trainOptions['networkArch']   = kw.pop('networkArch'   ,  'feedfoward'   )
    self.trainOptions['algorithmName'] = kw.pop('algorithmName' ,  'rprop'           )
    self.trainOptions['constFunction'] = kw.pop('constFunction' ,  'sp'           )
    self.trainOptions['print']         = kw.pop('print'         ,  False          )
    self.trainOptions['nEpochs']       = kw.pop('nEpochs'       ,  1000           )
    self.trainOptions['shuffle']       = True

    checkForUnusedVars(kw, self._logger.warning )
    del kw

  def setTrainData(self, trnData, trnTarget):
    """
      Overloads setTrainData to change numpy array to its
      ctypes representation.
    """
    self._trnData = exmachina.DataHandler(trnData,trnTarget)
    self._trnTarget = trnTarget



  def setValData(self, valData, valTarget):
    """
      Overloads setTrainData to change numpy array to its
      ctypes representation.
    """
    self._valData = exmachina.DataHandler(valData,valTarget)
    self._valTarget = valTarget


  def setTestData(self, tstData, tstTarget):
    """
      Overloads setTrainData to change numpy array to its
      ctypes representation.
    """
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
    self._trainer = exmachina.NeuralNetworkTrainer(self._net,
        [self._trnData,self._valData, self._tstData],
        self.trainOptions)

    self._logger.debug('Successfully exited C++ training.')

    tunedDiscData= self.__neural_to_dict( net )
    trnOutput = net.propagateDataset(self._trnData)
    valOutput = net.propagateDataset(self._valData)
    tstOutput = net.propagateDataset(self._tstData)

    allOutput = np.concatenate([trnOutput,valOutput,tstOutput])
    allTarget = np.concatenate([self._trnTarget,self._valTarget,self._tstTarget])
    
    operationReceiveOperationCurve = Roc( self.__generateOperationReceiveCurve(allOutput,allTarget),
                                                            'operation')
    testReceiveOperationCurve = Roc( self.__generateOperationReceiveCurve(tstOutput,self._tstTarget),
                                                            'test')

    self._logger.info('Operation: sp = %f, det = %f and fa = %f', \
                      operationReceiveOperationCurve.sp, 
                      operationReceiveOperationCurve.det, 
                      operationReceiveOperationCurve.fa)

    self._logger.info('Test: sp = %f, det = %f and fa = %f', \
                      testReceiveOperationCurve.sp, 
                      testReceiveOperationCurve.det, 
                      testReceiveOperationCurve.fa)

    tunedDiscrData['summaryInfo']=dict()
    tunedDiscrData['summaryInfo']['roc_operation'] = operatioReceiveOperationCurve
    tunedDiscrData['summaryInfo']['roc_test'] = testReceiveOperationCurve

    del trnOutput, valOutput, tstOutput, allOutput, allTarget 

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
    return obj


  def __generateReceiveOperationCurve( self, output, target ):
    if len(np.unique(target)) != 2:
      raise RuntimeError('The number of patterns > 2. Abort generateReceiveOperationCurve method.')

    sgn = output[:,np.where(target ==  1)]
    noise = output[:,np.where(target == -1)]
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
          data = np.concatenate((cl.T, data),axis=1)
          target = np.concatenate((tgt[idx]*np.ones((1,len(cl)), order='F',dtype='double'), target),axis=1)
        idx+=1

      self._logger.debug('data shape is %s',data.shape)
      return data, target
    else:
      raise RuntimeError('Can not concatenate patterns, error type from constructor')

  def separate_patterns(self, data, target):
    if type(data) is np.array and type(target) is np.array:
      patterns = list()
      targets  = np.unique(target).tolist()
      for tgt in targets:
        patterns.append(data[:,np.where(target==tgt)].T)
      return patterns
    else:
      raise RuntimeError('Can not separate patterns, error type from constructor')






