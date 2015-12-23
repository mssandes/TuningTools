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
from TuningTools.Neural import Neural


class TuningTool(TuningToolPyWrapper, Logger):
  """
    TuningTool is the higher level representation of the TuningToolPyWrapper class.
  """
  _trainOptions=dict()

  def __init__( self, **kw ):
    Logger.__init__( self, kw )
    TuningToolPyWrapper.__init__(self, LoggingLevel.toC(self.level))
    from RingerCore.util import checkForUnusedVars
    self.seed                = kw.pop('seed',        None    )
    self._trainOptions['batchSize']     = kw.pop('batchSize'     ,  100            )
    self._trainOptions['networkArch']   = kw.pop('networkArch'   ,  'feedfoward'   )
    self._trainOptions['algorithmName'] = kw.pop('algorithmName' ,  'lm'           )
    self._trainOptions['constFunction'] = kw.pop('constFunction' ,  'sp'           )
    self._trainOptions['print']         = kw.pop('print'         ,  False          )
    self._trainOptions['nEpochs']       = kw.pop('nEpochs'       ,  1000           )

    self._trainOptions['shuffle'] = True
    self.doPerf                       = kw.pop('doPerf',      False   )
    checkForUnusedVars(kw, self._logger.warning )
    del kw

  def setTrainData(self, trnData, trnTarget):
    """
      Overloads setTrainData to change numpy array to its
      ctypes representation.
    """
    self._trnData = exmachina.DataHandler(trnData,trnTarget)


  def setValData(self, valData, valTarget):
    """
      Overloads setTrainData to change numpy array to its
      ctypes representation.
    """
    self._valData = exmachina.DataHandler(valData,valTarget)


  def setTestData(self, tstData, tstTarget):
    """
      Overloads setTrainData to change numpy array to its
      ctypes representation.
    """
    self._tstData = exmachina.DataHandler(tstData,tstTarget)


  def newff(self, nodes, funcTrans = ['tansig', 'tansig']):
    """
      Creates new feedforward neural network
    """
    self._logger.info('Initalizing newff...')
    self._net = exmachina.FeedForward(nodes,funcTrans,'nw')


  def train_c(self):
    """
      Train feedforward neural network
    """
    self._trainer = exmachina.NeuralNetworkTrainer(self._net,
        [self._trnData,self._valData, self._tstData],
        self._trainOptions)

    self._logger.debug('Successfully exited C++ training.')
    
    netList = []
    '''
    from RingerCore.util import Roc

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
    '''
    return netList


  def concatenate_patterns(self, patterns):

    if type(patterns) is list:
      if len(patterns) == 2: tgt = [1,-1]
      else: tgt=range(len(patterns)) 
     
      data = np.array([],dtype='float32',order='F')
      target=np.array([],dtype='int',order='F')
      idx=0
      for cl in patterns:
        np.concatenate((data,cl.T),axis=1)
        np.concatenate((target, tgt[idx]*np.ones(1,len(cl), order='F',dtype='int')))
        idx+=1
      return data, target
    else:
      raise RuntimeError('Can not concatenate patterns, error type from constructor')

  def separate_patterns(self, data, target):
    if type(data) is np.array and type(target) is np.array:
      patterns = list()
      targets  = np.unique(target).tolist()
      for tgt in targets:
        patterns.append(data(:,np.where(target==tgt)).T)
    else:
      raise RuntimeError('Can not separate patterns, error type from constructor')






