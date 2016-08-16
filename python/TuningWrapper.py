__all__ = ['TuningWrapper']

import numpy as np
from RingerCore import Logger, LoggingLevel, NotSet, checkForUnusedVars, \
                       retrieve_kw, Roc
from TuningTools.coreDef      import retrieve_npConstants, TuningToolCores,              retrieve_core
from TuningTools.TuningJob    import ReferenceBenchmark,   ReferenceBenchmarkCollection, BatchSizeMethod
from TuningTools.ReadData     import Dataset
npCurrent, _ = retrieve_npConstants()

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
    self.references = ReferenceBenchmarkCollection( [] )
    self.doPerf                = retrieve_kw( kw, 'doPerf',                True                   )
    self.batchMethod           = BatchSizeMethod.retrieve(
                               retrieve_kw( kw, 'batchMethod', BatchSizeMethod.MinClassSize \
        if not 'batchSize' in kw or kw['batchSize'] is NotSet else BatchSizeMethod.Manual         ) 
                                 )
    self.batchSize             = retrieve_kw( kw, 'batchSize',             NotSet                 )
    epochs                     = retrieve_kw( kw, 'epochs',                10000                  )
    maxFail                    = retrieve_kw( kw, 'maxFail',               50                     )
    self.useTstEfficiencyAsRef = retrieve_kw( kw, 'useTstEfficiencyAsRef', False                  )
    self._core, self._coreEnum = retrieve_core()
    self.sortIdx = None
    if self._coreEnum is TuningToolCores.ExMachina:
      self.trainOptions = dict()
      self.trainOptions['algorithmName'] = retrieve_kw( kw, 'algorithmName', 'rprop'       )
      self.trainOptions['print']         = retrieve_kw( kw, 'showEvo',       True          )
      self.trainOptions['networkArch']   = retrieve_kw( kw, 'networkArch',   'feedforward' )
      self.trainOptions['costFunction']  = retrieve_kw( kw, 'costFunction',  'sp'          )
      self.trainOptions['shuffle']       = retrieve_kw( kw, 'shuffle',       True          )
      self.trainOptions['nEpochs']       = epochs
      self.trainOptions['nFails']        = maxFail
      self.doMultiStop                   = False
    elif self._coreEnum is TuningToolCores.FastNet:
      seed = retrieve_kw( kw, 'seed', None )
      self._core = self._core( level = LoggingLevel.toC(self.level), seed = seed )
      self._core.trainFcn    = retrieve_kw( kw, 'algorithmName', 'trainrp' )
      self._core.showEvo     = retrieve_kw( kw, 'showEvo',       50        )
      self._core.multiStop   = retrieve_kw( kw, 'doMultiStop',   True      )
      self._core.epochs      = epochs
      self._core.maxFail     = maxFail
    else:
      self._logger.fatal("TuningWrapper not implemented for %s" % TuningToolCores.tostring(self._coreEnum))
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
    if val is not NotSet:
      self.batchMethod = BatchSizeMethod.Manual
      if self._coreEnum is TuningToolCores.ExMachina:
        self.trainOptions['batchSize'] = val
      elif self._coreEnum is TuningToolCores.FastNet:
        self._core.batchSize   = val
      self._logger.debug('Set batchSize to %d', val )

  def __batchSize(self, val):
    """
    Internal access to batchSize
    """
    if self._coreEnum is TuningToolCores.ExMachina:
      self.trainOptions['batchSize'] = val
    elif self._coreEnum is TuningToolCores.FastNet:
      self._core.batchSize   = val
    self._logger.debug('Set batchSize to %d', val )

  @property
  def doMultiStop(self):
    """
    External access to doMultiStop
    """
    if self._coreEnum is TuningToolCores.ExMachina:
      return False
    elif self._coreEnum is TuningToolCores.FastNet:
      return self._core.multiStop

  def setReferences(self, references):
    # Make sure that the references are a collection of ReferenceBenchmark
    references = ReferenceBenchmarkCollection(references)
    if len(references) == 0:
      self._logger.fatal("Reference collection must be not empty!", ValueError)
    if self._coreEnum is TuningToolCores.ExMachina:
      self._logger.info("Setting reference target to MSE.")
      if len(references) != 1:
        self._logger.error("Ignoring other references as ExMachina currently works with MSE.")
        references = references[:1]
      self.references = references
      ref = self.references[0]
      if ref.reference != ReferenceBenchmark.MSE:
        self._logger.fatal("Tuning using MSE and reference is not MSE!")
    elif self._coreEnum is TuningToolCores.FastNet:
      if self.doMultiStop:
        self.references = ReferenceBenchmarkCollection( [None] * 3 )
        # This is done like this for now, to prevent adding multiple 
        # references. However, this will be removed when the FastNet is 
        # made compatible with multiple references
        retrievedSP = False
        retrievedPD = False
        retrievedPF = False
        for ref in references:
          if ref.reference is ReferenceBenchmark.SP:
            if not retrievedSP:
              retrievedSP = True
              self.references[0] = ref
            else:
              self._logger.warning("Ignoring multiple Reference object: %s", ref)
          elif ref.reference is ReferenceBenchmark.Pd:
            if not retrievedPD:
              retrievedPD = True
              self.references[1] = ref
              self._core.det = self.references[1].getReference()
            else:
              self._logger.warning("Ignoring multiple Reference object: %s", ref)
          elif ref.reference is ReferenceBenchmark.Pf:
            if not retrievedPF:
              retrievedPF = True
              self.references[2] = ref
              self._core.fa = self.references[2].getReference()
            else:
              self._logger.warning("Ignoring multiple Reference object: %s", ref)
        self._logger.info('Set multiStop target [Sig_Eff(%%) = %r, Bkg_Eff(%%) = %r].', 
                          self._core.det * 100.,
                          self._core.fa * 100.  )
      else:
        self._logger.info("Setting reference target to MSE.")
        if len(references) != 1:
          self._logger.warning("Ignoring other references when using FastNet with MSE.")
          references = references[:1]
        self.references = references
        ref = self.references[0]
        if ref.reference != ReferenceBenchmark.MSE:
          self._logger.fatal("Tuning using MSE and reference is not MSE!")

  def setSortIdx(self, sort):
    if self._coreEnum is TuningToolCores.FastNet:
      if self.doMultiStop and self.useTstEfficiencyAsRef:
        if not len(self.references) == 3 or  \
            not self.references[0].reference == ReferenceBenchmark.SP or \
            not self.references[1].reference == ReferenceBenchmark.Pd or \
            not self.references[2].reference == ReferenceBenchmark.Pf:
          self._logger.fatal("The tuning wrapper references are not correct!")
        self.sortIdx = sort
        self._core.det = self.references[1].getReference( ds = Dataset.Validation, sort = sort )
        self._core.fa = self.references[2].getReference( ds = Dataset.Validation, sort = sort )
        self._logger.info('Set multiStop target [sort:%d | Sig_Eff(%%) = %r, Bkg_Eff(%%) = %r].', 
                          sort,
                          self._core.det * 100.,
                          self._core.fa * 100.  )

  def trnData(self, release = False):
    ret =  self.__separate_patterns(self._trnData,self._trnTarget) if self._coreEnum is TuningToolCores.ExMachina \
      else self._trnData
    if release: self.release()
    return ret

  def setTrainData(self, data, target=None):
    """
      Set train dataset of the tuning method.
    """
    self._sgnSize = data[0].shape[npCurrent.odim]
    self._bkgSize = data[1].shape[npCurrent.odim]
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
        self._logger.fatal("Couldn't allocate new feed-forward!")

  def train_c(self):
    """
      Train feedforward neural network
    """
    from copy import deepcopy
    # Holder of the discriminators:
    tunedDiscrList = []
    tuningInfo = {}

    # Set batch size:
    if self.batchMethod is BatchSizeMethod.MinClassSize:
      self.__batchSize( self._bkgSize if self._sgnSize > self._bkgSize else self._sgnSize )
    elif self.batchMethod is BatchSizeMethod.HalfSizeSignalClass:
      self.__batchSize( self._sgnSize // 2 )
    elif self.batchMethod is BatchSizeMethod.OneSample:
      self.__batchSize( 1 )

    rawDictTempl = { 'discriminator' : None,
                     'benchmark' : None }
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
        self._logger.fatal('Couldn''t initialize the trainer. Reason:\n %s' % str(e))
      self._logger.debug('executing train_c')
      try:
        trainer.train()
      except Exception, e:
        self._logger.fatal('Couldn''t tune. Reason:\n %s' % str(e))
      self._logger.debug('finished train_c')

      # Retrieve raw network
      rawDictTempl['discriminator'] = self.__discr_to_dict( self._net ) 
      rawDictTempl['benchmark'] = self.references[0]
      tunedDiscrList.append( deepcopy( rawDictTempl ) )

    elif self._coreEnum is TuningToolCores.FastNet:
      self._logger.debug('executing train_c')
      [discriminatorPyWrapperList, trainDataPyWrapperList] = self._core.train_c()
      self._logger.debug('finished train_c')
      # Transform net tolist of  dict

      if self.doMultiStop:
        for idx, discr in enumerate( discriminatorPyWrapperList ):
          rawDictTempl['discriminator'] = self.__discr_to_dict( discr ) 
          rawDictTempl['benchmark'] = self.references[idx]
          # FIXME This will need to be improved if set to tune for multiple
          # Pd and Pf values.
          tunedDiscrList.append( deepcopy( rawDictTempl ) )
      else:
        rawDictTempl['discriminator'] = self.__discr_to_dict( discriminatorPyWrapperList[0] ) 
        rawDictTempl['benchmark'] = self.references[0]
        if self.useTstEfficiencyAsRef and self.sortIdx is not None:
          rawDictTempl['sortIdx'] = self.sortIdx
        tunedDiscrList.append( deepcopy( rawDictTempl ) )
      from TuningTools.Neural import DataTrainEvolution
      tuningInfo = DataTrainEvolution( trainDataPyWrapperList ).toRawObj()
    # cores

    # Retrieve performance:
    for idx, tunedDiscrDict in enumerate(tunedDiscrList):
      discr = tunedDiscrDict['discriminator']
      ref = tunedDiscrDict['benchmark']
      opROC = None
      testROC = None
      if self.doPerf:
        self._logger.debug('Retrieving performance.')
        if self._coreEnum is TuningToolCores.ExMachina:
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
        opData = [ opROC.spVec, opROC.detVec, opROC.faVec ]
        testData = [ testROC.spVec, testROC.detVec, testROC.faVec ]
        # Add rocs to output information
        tunedDiscrDict['summaryInfo'] = { 'roc_operation' : opROC,
                                          'roc_test' : testROC }
        from TuningTools import PerfHolder
        perfHolder = PerfHolder( 
                                 tunedDiscrDict, 
                                 tuningInfo, 
                               )
        opSP,  opPd,  opPf,  opCut,  _ = perfHolder.getOperatingBenchmarks( ref, ds = Dataset.Operation )
        tstSP, tstPd, tstPf, tstCut, _ = perfHolder.getOperatingBenchmarks( ref, ds = Dataset.Test,
                                                         useTstEfficiencyAsRef = self.useTstEfficiencyAsRef )
        # Print information:
        self._logger.info(
                          'Operation (%s): sp = %f, det = %f, fa = %f, cut = %f', \
                          ref.name,
                          opSP, 
                          opPd, 
                          opPf,
                          opCut,
                         )
        self._logger.info(
                          'Test (%s): sp = %f, det = %f, fa = %f, cut = %f', \
                          ref.name,
                          tstSP,
                          tstPd,
                          tstPf,
                          tstCut,
                         )

    self._logger.debug("Finished train_c on python side.")

    return tunedDiscrList, tuningInfo
  # end of train_c

  def __discr_to_dict(self, net):
    """
    Transform discriminators to dictionary
    """
    if self._coreEnum is TuningToolCores.ExMachina:
      discrDict = {
                    'nodes' : net.layers,
                    'weights' : net.weights,
                    'bias' : net.bias,
                  }
    elif self._coreEnum is TuningToolCores.FastNet:
      from TuningTools.Neural import Neural
      holder = Neural('NeuralNetwork')
      holder.set_from_fastnet(net)
      discrDict = holder.rawDiscrDict()
    #
    self._logger.debug('Extracted discriminator to raw dictionary.')
    return discrDict



  def __concatenate_patterns(self, patterns):
    if type(patterns) not in (list,tuple):
      self._logger.fatal('Input must be a tuple or list')
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


