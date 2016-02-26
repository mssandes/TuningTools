import numpy as np
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.npdef import retrieve_npConstants
npCurrent = retrieve_npConstants()

def _checkData(data,target=None):
  if not npCurrent.check_order(data):
    raise TypeError('order of numpy data is not fortran!')
  if target is not None and not npCurrent.check_order(target):
    raise TypeError('order of numpy target is not fortran!')

class TuningWrapper(Logger):
  """
    TuningTool is the higher level representation of the TuningToolPyWrapper class.
  """

  # FIXME Create a dict with default options for FastNet and for ExMachina

  def __init__( self, **kw ):
    Logger.__init__( self, kw )
    from RingerCore.util import checkForUnusedVars
    import exmachina
    self._core = exmachina
    self.trainOptions = dict()
    self.trainOptions['batchSize']     = kw.pop('batchSize'     ,  100            )
    self.trainOptions['networkArch']   = kw.pop('networkArch'   ,  'feedforward'  )
    self.trainOptions['algorithmName'] = kw.pop('algorithmName' ,  'rprop'        )
    self.trainOptions['contFunction']  = kw.pop('costFunction'  ,  'sp'           )
    self.trainOptions['print']         = kw.pop('print'         ,  False          )
    self.trainOptions['nEpochs']       = kw.pop('nEpochs'       ,  1000           )
    self.trainOptions['nFails']        = kw.pop('nFails'        ,  50             )
    self.trainOptions['shuffle']       = True
    checkForUnusedVars(kw, self._logger.warning )
    self._trnData    = npCurrent.fp_array([])
    self._valData    = npCurrent.fp_array([])
    self._tstData    = npCurrent.fp_array([])
    self._trnHandler = None
    self._valHandler = None
    self._tstHandler = None
    self._trnTarget  = npCurrent.fp_array([[]]).reshape( 
            npCurrent.access( pidx=1,
                              oidx=0 ) )
    self._valTrget   = npCurrent.fp_array([[]]).reshape( 
            npCurrent.access( pidx=1,
                              oidx=0 ) )
    self._tstTarget  = npCurrent.fp_array([[]]).reshape( 
            npCurrent.access( pidx=1,
                              oidx=0 ) )
    del kw

  def trnData(self, release = False):
    ret =  self.separate_patterns(self._trnData,self._trnTarget) if self._core.__name__ == 'exmachina' \
      else self._trnData
    if release: 
      self._trnData = npCurrent.fp_array([])
      self._trnHandler = None
      self._trnTarget = npCurrent.fp_array([[]]).reshape( 
            npCurrent.access( pidx=1,
                              oidx=0 ) )
    return ret


  def setTrainData(self, data, target=None):
    """
      Set train dataset of the tuning method.
    """
    if target is None:
      data, target = self.concatenate_patterns(data)
    _checkData(data, target)
    self._trnData = data
    self._trnTarget = target
    self._trnHandler = self._core.DataHandler(data,target)

  def valData(self, release = False):
    ret =  self.separate_patterns(self._valData,self._valTarget) if self._core.__name__ == 'exmachina' \
      else self._valData
    if release: 
      self._valData = npCurrent.fp_array([])
      self._valHandler = None
      self._valTarget = npCurrent.fp_array([[]]).reshape( 
            npCurrent.access( pidx=1,
                              oidx=0 ) )
    return ret

  def setValData(self, data, target=None):
    """
      Set validation dataset of the tuning method.
    """
    if target is None:
      data, target = self.concatenate_patterns(data)
    _checkData(data, target)
    self._valData = data
    self._valTarget = target
    self._valHandler = self._core.DataHandler(data,target)

  def testData(self, release = False):
    ret =  self.separate_patterns(self._tstData,self._tstTarget) if self._core.__name__ == 'exmachina' \
      else self._tstData
    if release: 
      self._tstData = npCurrent.fp_array([])
      self._tstHandler = None
      self._tstTarget = npCurrent.fp_array([[]]).reshape( 
            npCurrent.access( pidx=1,
                              oidx=0 ) )
    return ret


  def setTestData(self, data, target=None):
    """
      Set test dataset of the tuning method.
    """
    if target is None:
      data, target = self.concatenate_patterns(data)
    _checkData(data, target)
    self._tstData = data
    self._tstTarget = target
    self._tstHandler = self._core.DataHandler(data,target)

  def newff(self, nodes, funcTrans = ['tanh', 'tanh']):
    """
      Creates new feedforward neural network
    """
    self._logger.info('Initalizing newff...')
    self._net = self._core.FeedForward(nodes,funcTrans,'nw')

  def train_c(self):
    """
      Train feedforward neural network
    """
    from RingerCore.util   import Roc
    self._logger.info('Initalizing train_c')
    try:
      trainer = self._core.NeuralNetworkTrainer(self._net,
        [self._trnHandler, 
         self._valHandler, 
         (self._tstHandler if self._tstHandler else self._valHandler),
        ],
        self.trainOptions)
    except Exception, e:
      raise RuntimeError('Couldn''t initialize the trainer. Reason: %s' % str(e))

    self._logger.info('execute train_c')
    try:
      trainer.train()
    except Exception, e:
      raise RuntimeError('Couldn''t tune. Reason: %s' % str(e))
    self._logger.debug('Successfully exited C++ training.')

    # Retrieve raw network
    tunedDiscrData = self.__neural_to_dict( self._net )

    # Retrieve outputs:
    trnOutput = self._net.propagateDataset(self._trnHandler)[0]
    valOutput = self._net.propagateDataset(self._valHandler)[0]
    tstOutput = self._net.propagateDataset(self._tstHandler)[0] if self._tstHandler else npCurrent.fp_array([])
    allOutput = np.concatenate([trnOutput,valOutput,tstOutput], axis=npCurrent.odim )
    allTarget = np.concatenate([self._trnTarget,self._valTarget, self._tstTarget], axis=npCurrent.odim )

    # Retrieve Rocs:
    operationReceiveOperationCurve = Roc( allOutput, allTarget, 'operation', npConst = npCurrent)
    if self._tstHandler:
      testReceiveOperationCurve = Roc( tstOutput, self._tstTarget, 'test', npConst = npCurrent)
    else:
      testReceiveOperationCurve = Roc( valOutput, self._valTarget, 'val', npConst = npCurrent)

    # Print information:
    self._logger.info('Operation: sp = %f, det = %f, fa = %f, cut = %f', \
                      operationReceiveOperationCurve.sp, 
                      operationReceiveOperationCurve.det, 
                      operationReceiveOperationCurve.fa,
                      operationReceiveOperationCurve.cut)
    self._logger.info('Test: sp = %f, det = %f, fa = %f, cut = %f', \
                      testReceiveOperationCurve.sp, 
                      testReceiveOperationCurve.det, 
                      testReceiveOperationCurve.fa,
                      testReceiveOperationCurve.cut)

    # Add rocs to output information
    tunedDiscrData['summaryInfo'] = dict()
    tunedDiscrData['summaryInfo']['roc_operation'] = operationReceiveOperationCurve
    tunedDiscrData['summaryInfo']['roc_test'] = testReceiveOperationCurve

    self._logger.debug("Finished train_c on python side.")

    return tunedDiscrData
  # end of train_c

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
    self._logger.debug('Extracted discriminator to raw dictionary.')
    return obj

  def concatenate_patterns(self, patterns):
    if type(patterns) not in (list,tuple):
      raise RuntimeError('Input must be a tuple or list')
    pSize = [pat.shape[npCurrent.odim] for pat in patterns]
    target = npCurrent.fp_ones(npCurrent.shape(npat=1,nobs=np.sum(pSize)))
    target[npCurrent.access(pidx=0,oidx=slice(pSize[0],None))] = -1.
    data = npCurrent.fix_fp_array( np.concatenate(patterns,axis=npCurrent.odim) )
    return data, target

  def separate_patterns(self, data, target):
    patterns = list()
    classTargets = [1., -1.] # np.unique(target).tolist()
    for idx, classTarget in enumerate(classTargets):
      patterns.append( data[ npCurrent.access( pidx=':', oidx=np.where(target==classTarget)[1] ) ] )
      self._logger.debug('Separated pattern %d shape is %r', idx, patterns[idx].shape)
    return patterns


