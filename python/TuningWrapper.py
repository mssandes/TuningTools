__all__ = ['TuningWrapper']

import numpy as np
from RingerCore import ( Logger, LoggingLevel, NotSet, checkForUnusedVars
                       , retrieve_kw )
from TuningTools.coreDef      import coreConf, npCurrent, TuningToolCores
from TuningTools.TuningJob    import ReferenceBenchmark,   ReferenceBenchmarkCollection, BatchSizeMethod
from TuningTools.dataframe.EnumCollection     import Dataset
from TuningTools.Neural import Neural, DataTrainEvolution, Roc

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
    coreframe = coreConf.core_framework()
    self.doPerf                = retrieve_kw( kw, 'doPerf',                True                   )
    self.batchMethod           = BatchSizeMethod.retrieve(
                               retrieve_kw( kw, 'batchMethod', BatchSizeMethod.MinClassSize \
        if not 'batchSize' in kw or kw['batchSize'] is NotSet else BatchSizeMethod.Manual         ) 
                                 )

    epochs                     = retrieve_kw( kw, 'epochs',                10000                  )
    maxFail                    = retrieve_kw( kw, 'maxFail',               50                     )
    self.useTstEfficiencyAsRef = retrieve_kw( kw, 'useTstEfficiencyAsRef', False                  )
    self._merged               = retrieve_kw( kw, 'merged',                False                  )
    self.networks              = retrieve_kw( kw, 'networks',              NotSet                 )
    self._saveOutputs          = retrieve_kw( kw, 'saveOutputs',           False                  )
    self.sortIdx = None
    if coreConf() is TuningToolCores.FastNet:
      seed = retrieve_kw( kw, 'seed', None )
      self._core = coreframe( level = LoggingLevel.toC(self.level), seed = seed )
      self._core.trainFcn    = retrieve_kw( kw, 'algorithmName', 'trainrp' )
      self._core.showEvo     = retrieve_kw( kw, 'showEvo',       50        )
      self._core.multiStop   = retrieve_kw( kw, 'doMultiStop',   True      )
      self._core.epochs      = epochs
      self._core.maxFail     = maxFail
      # TODO Add properties
    elif coreConf() is TuningToolCores.keras:
      self._core = coreframe
      from keras import callbacks
      from keras.optimizers import RMSprop, SGD
      from TuningTools.keras_util.callbacks import PerformanceHistory
      self.trainOptions = dict()
      self.trainOptions['optmin_alg']    = retrieve_kw( kw, 'optmin_alg',     RMSprop(lr=0.001, rho=0.9, epsilon=1e-08) )
      #self.trainOptions['optmin_alg']    = retrieve_kw( kw, 'optmin_alg',    SGD(lr=0.1, decay=1e-6, momentum=0.7)  )
      self.trainOptions['costFunction']  = retrieve_kw( kw, 'binary_crossentropy',  'mean_squared_error'  ) # 'binary_crossentropy' #'mean_squared_error' # 
      self.trainOptions['metrics']       = retrieve_kw( kw, 'metrics',       ['accuracy', ]          )
      self.trainOptions['shuffle']       = retrieve_kw( kw, 'shuffle',       True                  )
      self._multiStop                    = retrieve_kw( kw, 'doMultiStop',   True      )
      self.trainOptions['nEpochs']       = epochs
      self.trainOptions['nFails']        = maxFail
      #self._earlyStopping = callbacks.EarlyStopping( monitor='val_Tuning_L2Calo_SP_sp_value' #  val_loss, self.trainOptions['metrics'][0] FIXME This must change
      #                                             , patience=self.trainOptions['nFails']
      #                                             , verbose=0
      #                                             , mode='max')
      self._earlyStopping = callbacks.EarlyStopping( monitor='val_loss' # val_acc
                                                   , patience=self.trainOptions['nFails']
                                                   , verbose=0
                                                   , mode='auto')
      self._historyCallback = PerformanceHistory( display = retrieve_kw( kw, 'showEvo', 50 ) )
    else:
      self._fatal("TuningWrapper not implemented for %s", coreConf)

    self.batchSize             = retrieve_kw( kw, 'batchSize',             NotSet                 )
    checkForUnusedVars(kw, self._debug )
    del kw
    # Set default empty values:
    if coreConf() is TuningToolCores.keras:
      self._emptyData  = npCurrent.fp_array([])
    elif coreConf() is TuningToolCores.FastNet:
      self._emptyData = list()
    self._emptyHandler = None
    if coreConf() is TuningToolCores.keras:
      self._emptyTarget = npCurrent.fp_array([[]]).reshape( 
              npCurrent.access( pidx=1,
                                oidx=0 ) )
    elif coreConf() is TuningToolCores.FastNet:
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
    if coreConf() is TuningToolCores.keras:
      return self.trainOptions.get('batchSize', None)
    elif coreConf() is TuningToolCores.FastNet:
      return self._core.batchSize

  @batchSize.setter
  def batchSize(self, val):
    """
    External access to batchSize
    """
    if val is not NotSet:
      self.batchMethod = BatchSizeMethod.Manual
      if coreConf() is TuningToolCores.keras:
        self.trainOptions['batchSize'] = val
      elif coreConf() is TuningToolCores.FastNet:
        self._core.batchSize   = val
      self._debug('Set batchSize to %d', val )

  def __batchSize(self, val):
    """
    Internal access to batchSize
    """
    if coreConf() is TuningToolCores.keras:
      self.trainOptions['batchSize'] = val
    elif coreConf() is TuningToolCores.FastNet:
      self._core.batchSize   = val
    self._debug('Set batchSize to %d', val )

  @property
  def doMultiStop(self):
    """
    External access to doMultiStop
    """
    if coreConf() is TuningToolCores.keras:
      return self._multiStop
    elif coreConf() is TuningToolCores.FastNet:
      return self._core.multiStop

  def setReferences(self, references):
    # Make sure that the references are a collection of ReferenceBenchmark
    references = ReferenceBenchmarkCollection(references)
    if len(references) == 0:
      self._fatal("Reference collection must be not empty!", ValueError)
    if coreConf() is TuningToolCores.ExMachina:
      self._info("Setting reference target to MSE.")
      if len(references) != 1:
        self._logger.error("Ignoring other references as ExMachina currently works with MSE.")
        references = references[:1]
      self.references = references
      ref = self.references[0]
      if ref.reference != ReferenceBenchmark.MSE:
        self._fatal("Tuning using MSE and reference is not MSE!")
    elif coreConf() is TuningToolCores.FastNet:
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
              self._warning("Ignoring multiple Reference object: %s", ref)
          elif ref.reference is ReferenceBenchmark.Pd:
            if not retrievedPD:
              retrievedPD = True
              self.references[1] = ref
              self._core.det = self.references[1].getReference()
            else:
              self._warning("Ignoring multiple Reference object: %s", ref)
          elif ref.reference is ReferenceBenchmark.Pf:
            if not retrievedPF:
              retrievedPF = True
              self.references[2] = ref
              self._core.fa = self.references[2].getReference()
            else:
              self._warning("Ignoring multiple Reference object: %s", ref)
        self._info('Set multiStop target [Sig_Eff(%%) = %r, Bkg_Eff(%%) = %r].', 
                          self._core.det * 100.,
                          self._core.fa * 100.  )
      else:
        self._info("Setting reference target to MSE.")
        if len(references) != 1:
          self._warning("Ignoring other references when using FastNet with MSE.")
          references = references[:1]
        self.references = references
        ref = self.references[0]
        #if ref.reference != ReferenceBenchmark.MSE:
        #  self._fatal("Tuning using MSE and reference is not MSE!")
    elif coreConf() is TuningToolCores.keras:
      self.references = references
      self._info("keras will be using the following references:")
      #from TuningTools.keras_util.metrics import Efficiencies
      def addMetricToKeras( func, name, metrics ):
        from keras import metrics as metrics_module
        setattr( metrics_module, name, func )
        metrics.append( name )
      for idx, bench in enumerate(self.references):
        self._info("Added %s", bench)
      self._historyCallback.references = references
        #effMetric = Efficiencies( bench )
        # Append functions to module as if they were part of it:
        # NOTE: This has the limitation of only being calculated for the batch size
        # NOTE: At the end of one iteration, keras calculate the performance
        # over all data (?) but using in batches
        # For some reason, it seemed not to be calculating those metrics for validation dataset.
        #addMetricToKeras( effMetric.false_alarm_probability, bench.name + '_pf',  self.trainOptions['metrics'] )
        #addMetricToKeras( effMetric.detection_probability,   bench.name + '_pd',  self.trainOptions['metrics'] )
        #addMetricToKeras( effMetric.sp_index,                bench.name + '_sp',  self.trainOptions['metrics'] )
        #addMetricToKeras( effMetric.threshold,               bench.name + '_cut', self.trainOptions['metrics'] )
        #if idx == 0:
        #  addMetricToKeras( effMetric.auc,                     'auc',               self.trainOptions['metrics'] )

  def setSortIdx(self, sort):
    if coreConf() is TuningToolCores.FastNet:
      if self.doMultiStop and self.useTstEfficiencyAsRef:
        if not len(self.references) == 3 or  \
            not self.references[0].reference == ReferenceBenchmark.SP or \
            not self.references[1].reference == ReferenceBenchmark.Pd or \
            not self.references[2].reference == ReferenceBenchmark.Pf:
          self._fatal("The tuning wrapper references are not correct!")
        self.sortIdx = sort
        self._core.det = self.references[1].getReference( ds = Dataset.Validation, sort = sort )
        self._core.fa = self.references[2].getReference( ds = Dataset.Validation, sort = sort )
        self._info('Set multiStop target [sort:%d | Sig_Eff(%%) = %r, Bkg_Eff(%%) = %r].', 
                          sort,
                          self._core.det * 100.,
                          self._core.fa * 100.  )

  def trnData(self, release = False):
    if coreConf() is TuningToolCores.FastNet:
      ret =  self.__separate_patterns(self._trnData,self._trnTarget)
    elif coreConf() is TuningToolCores.keras: 
      ret = self._trnData
    if release: self.release()
    return ret

  def setTrainData(self, data, target=None):
    """
      Set train dataset of the tuning method.
    """
    if self._merged:
      data_calo = data[0]
      data_track = data[1]
      self._sgnSize = data_calo[0].shape[npCurrent.odim]
      self._bkgSize = data_track[1].shape[npCurrent.odim]
      if coreConf() is TuningToolCores.keras:
        if target is None:
          data_calo, target = self.__concatenate_patterns(data_calo)
          data_track, _ = self.__concatenate_patterns(data_track)
        _checkData(data_calo, target)
        _checkData(data_track, target)
        data = [data_calo, data_track]
        self._trnData = data 
        self._trnTarget = target
        self._historyCallback.trnData = (data, target)
      elif coreConf() is TuningToolCores.FastNet:
        self._fatal( "Expert Neural Networks not implemented for FastNet core" )
    else:
      self._sgnSize = data[0].shape[npCurrent.odim]
      self._bkgSize = data[1].shape[npCurrent.odim]
      if coreConf() is TuningToolCores.keras:
        if target is None:
          data, target = self.__concatenate_patterns(data)
        _checkData(data, target)
        self._trnData = data
        self._trnTarget = target
        self._historyCallback.trnData = (data, target)
      elif coreConf() is TuningToolCores.FastNet:
        self._trnData = data
        self._core.setTrainData( data )


  def valData(self, release = False):
    if coreConf() is TuningToolCores.keras:
      ret =  self.__separate_patterns(self._valData,self._valTarget)
    elif coreConf() is TuningToolCores.Keras: 
      ret = self._valData
    if release: self.release()
    return ret

  def setValData(self, data, target=None):
    """
      Set validation dataset of the tuning method.
    """
    if self._merged:
      data_calo = data[0]
      data_track = data[1]
      if coreConf() is TuningToolCores.keras:
        if target is None:
          data_calo, target = self.__concatenate_patterns(data_calo)
          data_track, _ = self.__concatenate_patterns(data_track)
        _checkData(data_calo, target)
        _checkData(data_track, target)
        data = [data_calo, data_track]
        self._valData = data 
        self._valTarget = target
        self._historyCallback.valData = (data, target)
      elif coreConf() is TuningToolCores.FastNet:
        self._fatal( "Expert Neural Networks not implemented for FastNet core" )
    else:
      if coreConf() is TuningToolCores.keras:
        if target is None:
          data, target = self.__concatenate_patterns(data)
        _checkData(data, target)
        self._valData = data
        self._valTarget = target
        self._historyCallback.valData = (data, target)
      elif coreConf() is TuningToolCores.FastNet:
        self._valData = data
        self._core.setValData( data )

  def testData(self, release = False):
    if coreConf() is TuningToolCores.keras:
      ret =  self.__separate_patterns(self._tstData,self._tstTarget)
    else: 
      ret = self._tstData
    if release: self.release()
    return ret


  def setTestData(self, data, target=None):
    """
      Set test dataset of the tuning method.
    """
    if self._merged:
      data_calo = data[0]
      data_track = data[1]
      if coreConf() is TuningToolCores.keras:
        if target is None:
          data_calo, target = self.__concatenate_patterns(data_calo)
          data_track, _ = self.__concatenate_patterns(data_track)
        _checkData(data_calo, target)
        _checkData(data_track, target)
        data = [data_calo, data_track]
        self._tstData = data 
        self._tstTarget = target
        self._historyCallback.tstData = (data, target)
      elif coreConf() is TuningToolCores.FastNet:
        self._fatal( "Expert Neural Networks not implemented for FastNet core" )
    else:
      if coreConf() is TuningToolCores.keras:
        if target is None:
          data, target = self.__concatenate_patterns(data)
        _checkData(data, target)
        self._tstData = data
        self._tstTarget = target
        self._historyCallback.tstData = (data, target)
      elif coreConf() is TuningToolCores.FastNet:
        self._tstData = data
        self._core.setValData( data )

  def newff(self, nodes, funcTrans = NotSet):
    """
      Creates new feedforward neural network
    """
    self._debug('Initalizing newff...')
    if coreConf() is TuningToolCores.ExMachina:
      if funcTrans is NotSet: funcTrans = ['tanh', 'tanh']
      self._model = self._core.FeedForward(nodes, funcTrans, 'nw')
    elif coreConf() is TuningToolCores.FastNet:
      if funcTrans is NotSet: funcTrans = ['tansig', 'tansig']
      if not self._core.newff(nodes, funcTrans, self._core.trainFcn):
        self._fatal("Couldn't allocate new feed-forward!")
    elif coreConf() is TuningToolCores.keras:
      from keras.models import Sequential
      from keras.layers.core import Dense, Dropout, Activation
      model = Sequential()
      model.add( Dense( nodes[0]
                      , input_dim=nodes[0]
                      , init='identity'
                      , trainable=False 
                      , name='dense_1' ) )
      model.add( Activation('linear') )
      model.add( Dense( nodes[1]
                      , input_dim=nodes[0]
                      , init='uniform'
                      , name='dense_2' ) )
      model.add( Activation('tanh') )
      model.add( Dense( nodes[2], init='uniform', name='dense_3' ) ) 
      model.add( Activation('tanh') )
      model.compile( loss=self.trainOptions['costFunction']
                   , optimizer = self.trainOptions['optmin_alg']
                   , metrics = self.trainOptions['metrics'] )
      #keras.callbacks.History()
      #keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
      self._model = model
      self._historyCallback.model = model

  def newExpff(self, nodes, et, eta, sort, funcTrans = NotSet):
    """
      Creates new feedfoward neural network from expert calorimeter and tracking networks.
    """
    self._debug('Initalizing newExpff...')
    models = {}
    if coreConf() is TuningToolCores.ExMachina:
      self._fatal( "Expert Neural Networks not implemented for ExMachina" ) 
    elif coreConf() is TuningToolCores.FastNet:
      self._fatal( "Expert Neural Networks not implemented for FastNet" ) 
    elif coreConf() is TuningToolCores.keras:
      from keras.models import Sequential
      from keras.layers import Merge
      from keras.layers.core import Dense, Dropout, Activation 
      references = ['Pd','Pf','SP']
      if len(self.networks[1][et][eta]) == 1:
        ref = 'SP'
        track_n = [ self.__dict_to_discr( self.networks[1][et][eta][ref]['sort_%1.3i'%(sort)],'track', pruneLastLayer=True ) ]
      else:
        track_n = {}
        for ref in references:
          track_n[ref] = self.__dict_to_discr( self.networks[1][et][eta][ref]['sort_%1.3i'%(sort)], 'track', pruneLastLayer=True )
      calo_nn = {}
      for ref in references:
        calo_nn = self.__dict_to_discr( self.networks[0][et][eta][ref]['sort_%1.3i'%(sort)], 'calo', pruneLastLayer=True )

        ## Extracting last layers
        if len(track_n) == 1: 
          from copy import deepcopy
          track_nn = deepcopy(track_n[0])
        else: track_nn = track_n[ref]

        merg_layer = Merge([calo_nn, track_nn], mode='concat',concat_axis=-1, name='merge_layer')
        
        ## Merged Model
        model = Sequential()
        model.add(merg_layer)
        names=['merge_dense_1','merge_dense_2']
        # NOTE: verify if there is nedd of a 'merge_dense_0' with identity
        model.add(Dense(nodes[1],
                        kernel_initializer='uniform',
                        name=names[0]))
        model.add(Activation('tanh'))
        model.add(Dense(nodes[2],
                        kernel_initializer='uniform',
                        input_dim=nodes[1],
                        trainable=True,
                        name=names[1]))
        model.add(Activation('tanh'))
        model.compile( loss=self.trainOptions['costFunction']
                     , optimizer = self.trainOptions['optmin_alg']
                     , metrics = self.trainOptions['metrics'] )

        models[ref] = model
    self._model = models
    # FIXME: check historycallback compatibility
    self._historyCallback.model = models


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

    if coreConf() is TuningToolCores.keras:
      history = self._model.fit( self._trnData
                               , self._trnTarget
                               , epochs          = self.trainOptions['nEpochs']
                               , batch_size      = self.batchSize
                               #, callbacks      = [self._historyCallback, self._earlyStopping]
                               , callbacks       = [self._earlyStopping]
                               , verbose         = 0
                               , validation_data = ( self._valData , self._valTarget )
                               , shuffle         = self.trainOptions['shuffle'] 
                               )
      # Retrieve raw network
      rawDictTempl['discriminator'] = self.__discr_to_dict( self._model ) 
      rawDictTempl['benchmark'] = self.references[0]
      tunedDiscrList.append( deepcopy( rawDictTempl ) )
      tuningInfo = DataTrainEvolution( history ).toRawObj()

      try:
        from sklearn.metrics import roc_curve
      except ImportError:
        # FIXME Can use previous function that we used here as an alternative
        raise ImportError("sklearn is not available, please install it.")

    elif coreConf() is TuningToolCores.FastNet:
      self._debug('executing train_c')
      [discriminatorPyWrapperList, trainDataPyWrapperList] = self._core.train_c()
      self._debug('finished train_c')
      # Transform model tolist of  dict

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
      tuningInfo = DataTrainEvolution( trainDataPyWrapperList ).toRawObj()
      # TODO
    # cores

    # Retrieve performance:
    opRoc, tstRoc, trnRoc = Roc(), Roc(), Roc()
    for idx, tunedDiscrDict in enumerate(tunedDiscrList):
      discr = tunedDiscrDict['discriminator']
      if self.doPerf:
        self._debug('Retrieving performance.')
        if coreConf() is TuningToolCores.keras:
          # propagate inputs:
          trnOutput = self._model.predict(self._trnData)
          valOutput = self._model.predict(self._valData)
          tstOutput = self._model.predict(self._tstData) if self._tstData else npCurrent.fp_array([])
          try:
            allOutput = np.concatenate([trnOutput,valOutput,tstOutput] )
            allTarget = np.concatenate([self._trnTarget,self._valTarget, self._tstTarget] )
          except ValueError:
            allOutput = np.concatenate([trnOutput,valOutput] )
            allTarget = np.concatenate([self._trnTarget,self._valTarget] )
          # Retrieve Rocs:
          opRoc( allOutput, allTarget )
          if self._tstData: tstRoc( tstOutput, self._tstTarget )
          else: tstRoc( valOutput, self._valTarget )
        elif coreConf() is TuningToolCores.FastNet:
          perfList = self._core.valid_c( discriminatorPyWrapperList[idx] )
          opRoc( perfList[1] )
          tstRoc( perfList[0] )
          #trnRoc( perfList[0] )
        # Add rocs to output information
        # TODO Change this to raw object
        tunedDiscrDict['summaryInfo'] = { 'roc_operation' : opRoc.toRawObj(),
                                          'roc_test' : tstRoc.toRawObj(),
                                          }
        if self._saveOutputs:
          tunedDiscrDict['summaryInfo']['trnOutput'] = [perfList[2],perfList[3]] if coreConf() is TuningToolCores.FastNet else trnOutput
          tunedDiscrDict['summaryInfo']['valOutput'] = [perfList[4],perfList[5]] if coreConf() is TuningToolCores.FastNet else valOutput
  

        for ref in self.references:
          if coreConf() is TuningToolCores.FastNet:
            # FastNet won't loop on this, this is just looping for keras right now
            ref = tunedDiscrDict['benchmark']

          opPoint = opRoc.retrieve( ref )
          tstPoint = tstRoc.retrieve( ref )
          # Print information:
          self._info( 'Operation (%s): sp = %f, pd = %f, pf = %f, thres = %f'
                    , ref.name
                    , opPoint.sp_value
                    , opPoint.pd_value
                    , opPoint.pf_value
                    , opPoint.thres_value )
          self._info( 'Test (%s): sp = %f, pd = %f, pf = %f, thres = %f'
                    , ref.name
                    , tstPoint.sp_value
                    , tstPoint.pd_value
                    , tstPoint.pf_value
                    , tstPoint.thres_value )

          if coreConf() is TuningToolCores.FastNet:
            break

    self._debug("Finished train_c on python side.")

    return tunedDiscrList, tuningInfo
  # end of train_c

  def trainC_Exp( self ):
    """
      Train expert feedforward neural network
    """ 
    if coreConf() is TuningToolCores.ExMachina:
      self._fatal( "Expert Neural Networks not implemented for ExMachina" ) 
    elif coreConf() is TuningToolCores.FastNet:
      self._fatal( "Expert Neural Networks not implemented for FastNet" ) 
    elif coreConf() is TuningToolCores.keras:
      from copy import deepcopy

      # Set batch size:
      if self.batchMethod is BatchSizeMethod.MinClassSize:
        self.__batchSize( self._bkgSize if self._sgnSize > self._bkgSize else self._sgnSize )
      elif self.batchMethod is BatchSizeMethod.HalfSizeSignalClass:
        self.__batchSize( self._sgnSize // 2 )
      elif self.batchMethod is BatchSizeMethod.OneSample:
        self.__batchSize( 1 )

      references = ['SP','Pd','Pf']

      # Holder of the discriminators:
      tunedDiscrList = []
      tuningInfo = {}

      for idx, ref in enumerate(references):
        rawDictTempl = { 'discriminator' : None,
                         'benchmark' : None }
        
        history = self._model[ref].fit( self._trnData
                                      , self._trnTarget
                                      , epochs          = self.trainOptions['nEpochs']
                                      , batch_size      = self.batchSize
                                      , callbacks       = [self._historyCallback, self._earlyStopping]
                                      #, callbacks       = [self._earlyStopping]
                                      , verbose         = 0
                                      , validation_data = ( self._valData , self._valTarget )
                                      , shuffle         = self.trainOptions['shuffle'] 
                                      )
        # Retrieve raw network
        rawDictTempl['discriminator'] = self.__expDiscr_to_dict( self._model[ref] ) 
        rawDictTempl['benchmark'] = self.references[idx]
        tunedDiscrList.append( deepcopy( rawDictTempl ) )
        tuningInfo[ref] = DataTrainEvolution( history ).toRawObj()

        try:
          from sklearn.metrics import roc_curve
        except ImportError:
          # FIXME Can use previous function that we used here as an alternative
          raise ImportError("sklearn is not available, please install it.")

        # Retrieve performance:
        opRoc, tstRoc = Roc(), Roc() 
        for idx, tunedDiscrDict in enumerate(tunedDiscrList):
          discr = tunedDiscrDict['discriminator']
          if self.doPerf:
            self._debug('Retrieving performance for %s networks.'%(ref))
            # propagate inputs:
            trnOutput = self._model[ref].predict(self._trnData)
            valOutput = self._model[ref].predict(self._valData)
            tstOutput = self._model[ref].predict(self._tstData) if self._tstData else npCurrent.fp_array([])
            try:
              allOutput = np.concatenate([trnOutput,valOutput,tstOutput] )
              allTarget = np.concatenate([self._trnTarget,self._valTarget, self._tstTarget] )
            except ValueError:
              allOutput = np.concatenate([trnOutput,valOutput] )
              allTarget = np.concatenate([self._trnTarget,self._valTarget] )
            # Retrieve Rocs:
            opRoc( allOutput, allTarget )
            if self._tstData: tstRoc( tstOutput, self._tstTarget )
            else: tstRoc( valOutput, self._valTarget )
            # Add rocs to output information
            # TODO Change this to raw object
            tunedDiscrDict['summaryInfo'] = { 'roc_operation' : opRoc.toRawObj(),
                                              'roc_test' : tstRoc.toRawObj() }

            for ref2 in self.references:
              opPoint = opRoc.retrieve( ref2 )
              tstPoint = tstRoc.retrieve( ref2 )
              # Print information:
              self._info( '%s NETWORKS Operation (%s): sp = %f, pd = %f, pf = %f, thres = %f'
                        , ref
                        , ref2.name
                        , opPoint.sp_value
                        , opPoint.pd_value
                        , opPoint.pf_value
                        , opPoint.thres_value )
              self._info( '%s NETWORKS Test (%s): sp = %f, pd = %f, pf = %f, thres = %f'
                        , ref
                        , ref2.name
                        , tstPoint.sp_value
                        , tstPoint.pd_value
                        , tstPoint.pf_value
                        , tstPoint.thres_value )
        self._info("Finished trainC_Exp for %s networks."%(ref))

    self._debug("Finished trainC_Exp on python side.")

    return tunedDiscrList, tuningInfo
  # end of trainC_Exp

  def __discr_to_dict(self, model):
    """
    Transform discriminators to dictionary
    """
    if coreConf() is TuningToolCores.keras:
      hw, hb = model.get_layer(name='dense_2').get_weights()
      ow, ob = model.get_layer(name='dense_3').get_weights()
      discrDict = {
                    'nodes':   npCurrent.int_array( [hw.shape[0], hw.shape[1], ow.shape[1]] ),
                    'weights': np.concatenate( [hw.reshape(-1,order='F'), ow.reshape(-1,order='F')] ),
                    'bias':    np.concatenate( [hb.reshape(-1,order='F'), ob.reshape(-1,order='F')] ),
                  }
    elif coreConf() is TuningToolCores.FastNet:
      n = []; w = []; b = [];
      for l in range( model.getNumLayers() ):
        n.append( model.getNumNodes(l) )
      for l in range( len(n) - 1 ):
        for j in range( n[l+1] ):
          for k in range( n[l] ):
            w.append( model.getWeight(l,j,k) )
          b.append( model.getBias(l,j) )
      discrDict = {
                    'nodes':   npCurrent.int_array(n),
                    'weights': npCurrent.fp_array(w),
                    'bias':    npCurrent.fp_array(b)
                  }
    self._debug('Extracted discriminator to raw dictionary.')
    return discrDict

  def __dict_to_discr( self, discrDict, appendage=None, pruneLastLayer=True ):
    """
    Transform dictionaries of networks into discriminators.
    """
    nodes = discrDict['nodes']
    weights = discrDict['weights']
    bias = discrDict['bias']
    if coreConf() is TuningToolCores.keras:
      from keras.models import Sequential
      from keras.layers.core import Dense, Dropout, Activation
      model = Sequential()
      names = [ 'dense_1','dense_2','dense_3' ]
      if appendage:
        for i in range(len(names)-1 if pruneLastLayer else len(names)):
          names[i] = '%s_%s'%(appendage,names[i])
      model.add( Dense( nodes[0]
                      , input_dim=nodes[0]
                      , kernel_initializer='identity'
                      , trainable=False 
                      , name=names[0] ) )
      model.add( Activation('linear') )
      model.add( Dense( nodes[1]
                      , input_dim=nodes[0]
                      , trainable = False
                      , kernel_initializer='uniform'
                      , name=names[1] ) )
      model.add( Activation('tanh') )
      w1 = weights[0:(nodes[0]*nodes[1])]
      w1 = w1.reshape((nodes[0],nodes[1]), order = 'F')
      b1 = bias[0:nodes[1]]
      model.get_layer(name=names[1]).set_weights( (w1, b1) )
      if not pruneLastLayer:
        model.add( Dense( nodes[2]
                        , kernel_initializer='uniform'
                        , trainable = False
                        , name=names[2] ) ) 
        model.add( Activation('tanh') )
        w2 = weights[(nodes[0]*nodes[1]):(nodes[0]*nodes[1] + nodes[1]*nodes[2])]
        w2 = w2.reshape((nodes[1],nodes[2]), order = 'F')
        b2 = bias[nodes[1]:nodes[1]+nodes[2]]
        model.layers[-2].set_weights( (w2, b2) )
      return model


  def __expDiscr_to_dict( self, model ):
    """
    Transform expert discriminators to dictionary
    """
    if coreConf() is TuningToolCores.keras:
      ow, ob = model.get_layer( name='merge_dense_2' ).get_weights()
      hw, hb = model.get_layer( name='merge_dense_1' ).get_weights()
      chw, chb = model.get_layer( name='calo_dense_2' ).get_weights()
      thw, thb = model.get_layer( name='track_dense_2' ).get_weights()
      discrDict = {
                    'nodes':   [[chw.shape[0], thw.shape[0]], [chw.shape[1], thw.shape[1]], hw.shape[0], hw.shape[1], ow.shape[1]],
                    'weights': {
                                'output_layer':         ow,
                                'merged_hidden_layer':  hw,
                                'calo_layer':           chw,
                                'track_layer':          thw
                               },
                    'bias':    {
                                'output_layer':         ob,
                                'merged_hidden_layer':  hb,
                                'calo_layer':           chb,
                                'track_layer':          thb
                               }
                  }
    self._debug('Extracted discriminator to raw dictionary.')
    return discrDict


  def __concatenate_patterns(self, patterns):
    if type(patterns) not in (list,tuple):
      self._fatal('Input must be a tuple or list')
    pSize = [pat.shape[npCurrent.odim] for pat in patterns]
    target = npCurrent.fp_ones(npCurrent.shape(npat=1,nobs=np.sum(pSize)))
    # FIXME Could I use flag_ones?
    target[npCurrent.access(pidx=0,oidx=slice(pSize[0],None))] = -1.
    data = npCurrent.fix_fp_array( np.concatenate(patterns,axis=npCurrent.odim) )
    return data, target


  # FIXME This is not work when I call the undo preproc function.
  # The target is None for some reason...
  def __separate_patterns(self, data, target):
    patterns = list()
    classTargets = [1., -1.] # np.unique(target).tolist()
    for idx, classTarget in enumerate(classTargets):
      patterns.append( data[ npCurrent.access( pidx=':', oidx=np.where(target==classTarget)[0]) ] )
      self._debug('Separated pattern %d shape is %r', idx, patterns[idx].shape)
    return patterns


