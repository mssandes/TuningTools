import numpy as np
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.coreDef import retrieve_npConstants, TuningToolCores, retrieve_core
npCurrent, _ = retrieve_npConstants()
from RingerCore.util import NotSet

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
    from RingerCore.util import checkForUnusedVars, retrieve_kw
    self.level = retrieve_kw( kw, 'level',  LoggingLevel.DEBUG   )
    self.doPerf = retrieve_kw( kw, 'doPerf',    True )
    batchSize   = retrieve_kw( kw, 'batchSize', 100  )
    epochs      = retrieve_kw( kw, 'epochs',    1000 )
    maxFail     = retrieve_kw( kw, 'maxFail',   50   )
    self._core, self._coreEnum = retrieve_core()
    if self._coreEnum is TuningToolCores.ExMachina:
      self.trainOptions = dict()
      self.trainOptions['algorithmName'] = retrieve_kw( kw, 'algorithmName', 'rprop'       )
      self.trainOptions['print']         = retrieve_kw( kw, 'showEvo',       True          )
      self.trainOptions['networkArch']   = retrieve_kw( kw, 'networkArch',   'feedforward' )
      self.trainOptions['costFunction']  = retrieve_kw( kw, 'costFunction',  'sp'          )
      self.trainOptions['shuffle']       = retrieve_kw( kw, 'shuffle',       True          )
      self.trainOptions['batchSize']     = batchSize
      self.trainOptions['nEpochs']       = epochs
      self.trainOptions['nFails']        = maxFail
    elif self._coreEnum is TuningToolCores.FastNet:
      seed = retrieve_kw( kw, 'seed', None )
      self._core = self._core( LoggingLevel.toC(self.level), seed )
      self._core.trainFcn    = retrieve_kw( kw, 'algorithmName', 'trainrp' )
      self._core.showEvo     = retrieve_kw( kw, 'showEvo',       5         )
      self._core.multiStop   = retrieve_kw( kw, 'doMultiStop',   True      )
      self._core.batchSize   = batchSize
      self._core.epochs      = epochs
      self._core.maxFail     = maxFail
    else:
      raise RuntimeError("TuningWrapper not implemented for %s" % TuningToolCores.tostring(self._coreEnum))
    checkForUnusedVars(kw, self._logger.debug )
    del kw
    # Set default empty values:
    if self._coreEnum is TuningToolCores.ExMachina:
      self._emptyData  = npCurrent.fp_array([])
    elif self._coreEnum is TuningToolCores.FastNet:
      self._emptyData = list()
    self._emptyHandler = None
    if self._coreEnum is TuningToolCores.ExMachina:
      self._emptyTarget = npCurrent.fp_array([[]]).reshape( 
              npCurrent.access( pidx=1,
                                oidx=0 ) )
    elif self._coreEnum is TuningToolCores.FastNet:
      self._emptyTarget = None
    # Set holders:
    self._trnData    = self._emptyData
    self._valData    = self._emptyData
    self._tstData    = self._emptyData
    self._trnHandler = self._emptyHandler
    self._valHandler = self._emptyHandler
    self._tstHandler = self._emptyHandler
    self._trnTarget  = self._emptyTarget
    self._valTarget  = self._emptyTarget
    self._tstTarget  = self._emptyTarget
  # TuningWrapper.__init__

  def release(self):
    """
    Release holden data, targets and handlers.
    """
    self._trnData = self._emptyData
    self._trnHandler = self._emptyHandler
    self._trnTarget = self._emptyTarget

  @property
  def batchSize(self):
    """
    External access to batchSize
    """
    if self._coreEnum is TuningToolCores.ExMachina:
      return self.trainOptions['batchSize']
    elif self._coreEnum is TuningToolCores.FastNet:
      return self._core.batchSize

  @batchSize.setter
  def batchSize(self, val):
    """
    External access to batchSize
    """
    if self._coreEnum is TuningToolCores.ExMachina:
      self.trainOptions['batchSize'] = val
    elif self._coreEnum is TuningToolCores.FastNet:
      self._core.batchSize   = val
    self._logger.debug('Set batchSize to %d', val )


  def trnData(self, release = False):
    ret =  self.__separate_patterns(self._trnData,self._trnTarget) if self._coreEnum is TuningToolCores.ExMachina \
      else self._trnData
    if release: self.release()
    return ret

  def setTrainData(self, data, target=None):
    """
      Set train dataset of the tuning method.
    """
    if self._coreEnum is TuningToolCores.ExMachina:
      if target is None:
        data, target = self.__concatenate_patterns(data)
      _checkData(data, target)
      self._trnData = data
      self._trnTarget = target
      self._trnHandler = self._core.DataHandler(data,target)
    elif self._coreEnum is TuningToolCores.FastNet:
      self._trnData = data
      self._core.setTrainData( data )


  def valData(self, release = False):
    ret =  self.__separate_patterns(self._valData,self._valTarget) if self._coreEnum is TuningToolCores.ExMachina \
      else self._valData
    if release: self.release()
    return ret

  def setValData(self, data, target=None):
    """
      Set validation dataset of the tuning method.
    """
    if self._coreEnum is TuningToolCores.ExMachina:
      if target is None:
        data, target = self.__concatenate_patterns(data)
      _checkData(data, target)
      self._valData = data
      self._valTarget = target
      self._valHandler = self._core.DataHandler(data,target)
    elif self._coreEnum is TuningToolCores.FastNet:
      self._valData = data
      self._core.setValData( data )

  def testData(self, release = False):
    ret =  self.__separate_patterns(self._tstData,self._tstTarget) if self._coreEnum is TuningToolCores.ExMachina \
      else self._tstData
    if release: self.release()
    return ret


  def setTestData(self, data, target=None):
    """
      Set test dataset of the tuning method.
    """
    if self._coreEnum is TuningToolCores.ExMachina:
      if target is None:
        data, target = self.__concatenate_patterns(data)
      _checkData(data, target)
      self._tstData = data
      self._tstTarget = target
      self._tstHandler = self._core.DataHandler(data,target)
    elif self._coreEnum is TuningToolCores.FastNet:
      self._tstData = data
      self._core.setValData( data )

  def newff(self, nodes, funcTrans = NotSet):
    """
      Creates new feedforward neural network
    """
    self._logger.debug('Initalizing newff...')
    if self._coreEnum is TuningToolCores.ExMachina:
      if funcTrans is NotSet: funcTrans = ['tanh', 'tanh']
      self._net = self._core.FeedForward(nodes, funcTrans, 'nw')
    elif self._coreEnum is TuningToolCores.FastNet:
      if funcTrans is NotSet: funcTrans = ['tansig', 'tansig']
      if not self._core.newff(nodes, funcTrans, self._core.trainFcn):
        raise RuntimeError("Couldn't allocate new feed-forward!")

  def train_c(self):
    """
      Train feedforward neural network
    """
    from RingerCore.util   import Roc
    if self._coreEnum is TuningToolCores.ExMachina:
      self._logger.debug('Initalizing train_c')
      try:
        trainer = self._core.NeuralNetworkTrainer(self._net,
          [self._trnHandler, 
           self._valHandler, 
           (self._tstHandler if self._tstHandler else self._valHandler),
          ],
          self.trainOptions)
      except Exception, e:
        raise RuntimeError('Couldn''t initialize the trainer. Reason:\n %s' % str(e))

      self._logger.debug('executing train_c')
      try:
        trainer.train()
      except Exception, e:
        raise RuntimeError('Couldn''t tune. Reason:\n %s' % str(e))
      self._logger.debug('finished train_c')

      tunedDiscrDataList = []
      # Retrieve raw network
      tunedDiscrDataList.append( self.__discr_to_dict( self._net ) )

    elif self._coreEnum is TuningToolCores.FastNet:
      self._logger.debug('executing train_c')
      [discriminatorPyWrapperList, trainDataPyWrapperList] = self._core.train_c()
      self._logger.debug('finished train_c')
      # Transform net tolist of  dict
      tunedDiscrDataList = []
      for discr in discriminatorPyWrapperList:
        tunedDiscrDataList.append( self.__discr_to_dict( discr, trainDataPyWrapperList ) )
    # cores

    # Retrieve performance:
    for idx, tunedDiscr in enumerate(tunedDiscrDataList):
      opROC = None
      testROC = None
      if self.doPerf:
        self._logger.debug('Retrieving performance.')
        if self._coreEnum is TuningToolCores.ExMachina:
          # FIXME Hardcoded. If ExMachina starts solving issue for multiple
          # benchmarks, then this will need to be changed...
          # Retrieve outputs:
          trnOutput = self._net.propagateDataset(self._trnHandler)[0]
          valOutput = self._net.propagateDataset(self._valHandler)[0]
          tstOutput = self._net.propagateDataset(self._tstHandler)[0] if self._tstHandler else npCurrent.fp_array([])
          allOutput = np.concatenate([trnOutput,valOutput,tstOutput], axis=npCurrent.odim )
          allTarget = np.concatenate([self._trnTarget,self._valTarget, self._tstTarget], axis=npCurrent.odim )
          # Retrieve Rocs:
          opROC = Roc( 'operation', allOutput, allTarget, npConst = npCurrent)
          if self._tstHandler:
            testROC = Roc( 'test', tstOutput, self._tstTarget, npConst = npCurrent)
          else:
            testROC = Roc( 'val', valOutput, self._valTarget, npConst = npCurrent)
        elif self._coreEnum is TuningToolCores.FastNet:
          perfList = self._core.valid_c( discriminatorPyWrapperList[idx] )
          opROC    = Roc( 'operation', perfList[1], npConst = npCurrent )
          testROC  = Roc( 'test',  perfList[0], npConst = npCurrent )
        # Print information:
        self._logger.info('Operation: sp = %f, det = %f, fa = %f, cut = %f', \
                          opROC.sp, opROC.det, opROC.fa, opROC.cut)
        self._logger.info('Test: sp = %f, det = %f, fa = %f, cut = %f', \
                          testROC.sp, testROC.det, testROC.fa, testROC.cut)
      # Add rocs to output information
      tunedDiscr['summaryInfo'] = { 'roc_operation' : opROC,
                                    'roc_test' : testROC }

    self._logger.debug("Finished train_c on python side.")

    return tunedDiscrDataList
  # end of train_c

  def __discr_to_dict(self, net, tuningData = None):
    """
    Transform higher level objects to dictionary
    """
    if self._coreEnum is TuningToolCores.ExMachina:
      discrDict = {
                    'nodes' : net.layers,
                    'weights' : net.weights,
                    'bias' : net.bias,
                  }
      trainEvoDict = dict()
    elif self._coreEnum is TuningToolCores.FastNet:
      from TuningTools.Neural import Neural
      holder = Neural(net, train=tuningData)
      discrDict = holder.rawDiscrDict()
      trainEvoDict = holder.rawEvoDict()
    #
    rawDict = { 'discriminator' : discrDict,
                'trainEvolution' : trainEvoDict }
    self._logger.debug('Extracted discriminator to raw dictionary.')
    return rawDict

  def __concatenate_patterns(self, patterns):
    if type(patterns) not in (list,tuple):
      raise RuntimeError('Input must be a tuple or list')
    pSize = [pat.shape[npCurrent.odim] for pat in patterns]
    target = npCurrent.fp_ones(npCurrent.shape(npat=1,nobs=np.sum(pSize)))
    target[npCurrent.access(pidx=0,oidx=slice(pSize[0],None))] = -1.
    data = npCurrent.fix_fp_array( np.concatenate(patterns,axis=npCurrent.odim) )
    return data, target

  def __separate_patterns(self, data, target):
    patterns = list()
    classTargets = [1., -1.] # np.unique(target).tolist()
    for idx, classTarget in enumerate(classTargets):
      patterns.append( data[ npCurrent.access( pidx=':', oidx=np.where(target==classTarget)[1] ) ] )
      self._logger.debug('Separated pattern %d shape is %r', idx, patterns[idx].shape)
    return patterns


