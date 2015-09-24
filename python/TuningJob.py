import os

from RingerCore.Logger        import Logger, LoggingLevel
from RingerCore.FileIO        import save, load
from RingerCore.LoopingBounds import *
from RingerCore.util          import EnumStringification, checkForUnusedVars
from TuningTools.Neural       import Neural
from TuningTools.TuningTool   import TuningTool
from TuningTools.PreProc      import *

class TunedDiscrArchieve( Logger ):
  """
  Context manager for Tuned Discriminators archives
  """

  _type = 'tunedFile'
  _version = 1
  _neuronBounds = None
  _sortBounds = None
  _initBounds = None
  _tunedDiscriminators = None

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with TunedDiscrArchieve("/path/to/file") as data:
      BLOCK

    TunedDiscrArchieve( "file/path", neuronBounds = ...,
                                     sortBounds = ...,
                                     initBounds = ...
                                     tunedDiscr = ... )
    """
    Logger.__init__(self, kw)
    self._filePath = filePath
    self.neuronBounds = kw.pop('neuronBounds', None )
    self.sortBounds = kw.pop('sortBounds', None )
    self.initBounds = kw.pop('initBounds', None )
    self.tunedDiscr = kw.pop('tunedDiscr', None )
    checkForUnusedVars( kw, self._logger.warning )

  @property
  def filePath( self ):
    return self._filePath

  @filePath.setter
  def filePath( self, val ):
    self._filePath = val


  @property
  def neuronBounds( self ):
    return self._neuronBounds

  @neuronBounds.setter
  def neuronBounds( self, val ):
    if not val is None and not isinstance(val, LoopingBounds):
      raise ValueType("Attempted to set neuronBounds to an object not of LoopingBounds type.")
    else:
      self._neuronBounds = val

  @property
  def sortBounds( self ):
    return self._sortBounds

  @sortBounds.setter
  def sortBounds( self, val ):
    if not val is None and not isinstance(val, LoopingBounds):
      raise ValueType("Attempted to set sortBounds to an object not of LoopingBounds type.")
    else:
      self._sortBounds = val

  @property
  def initBounds( self ):
    return self._initBounds

  @initBounds.setter
  def initBounds( self, val ):
    if not val is None and not isinstance(val, LoopingBounds):
      raise ValueType("Attempted to set initBounds to an object not of LoopingBounds type.")
    else:
      self._initBounds = val

  @property
  def tunedDiscr( self ):
    return self._tunedDiscr

  @tunedDiscr.setter
  def tunedDiscr( self, val ):
    self._tunedDiscr = val

  def getData( self ):
    if not self._neuronBounds or \
         not self._sortBounds or \
         not self._initBounds or \
         not self._tunedDiscr:
      raise RuntimeError("Attempted to retrieve empty data from TunedDiscrArchieve.")
    return { 'version': self._version,
                'type': self._type,
        'neuronBounds': transformToMatlabBounds( self._neuronBounds ).getOriginalVec(),
          'sortBounds': transformToPythonBounds( self._sortBounds ).getOriginalVec(),
          'initBounds': transformToPythonBounds( self._initBounds ).getOriginalVec(),
 'tunedDiscriminators': self._tunedDiscr }

  def save(self, compress = True):
    return save( self.getData(), self._filePath, compress = compress )

  def __enter__(self):
    # Open file:
    from cPickle import PickleError
    try:
      tunedData = load(self._filePath)
    except (PickleError, TypeError):
      # It failed without renaming the module, retry renaming old module
      # structure to new one.
      import TuningTools.Neural
      import sys
      sys.modules['FastNetTool.Neural'] = Neural
      tunedData = load(self._filePath)
    try:
      if type(tunedData) is dict:
        if tunedData['type'] != self._type:
          raise RuntimeError(("Input tunedData file is not from tunedData " 
              "type."))
        # Read configuration file to retrieve pre-processing, 
        if tunedData['version'] == 1:
          self._version = 1
          self.neuronBounds = MatlabLoopingBounds( tunedData['neuronBounds'] )
          self.sortBounds   = PythonLoopingBounds( tunedData['sortBounds']   )
          self.initBounds   = PythonLoopingBounds( tunedData['initBounds']   )
          self.tunedDiscr   = tunedData['tunedDiscriminators']
        else:
          raise RuntimeError("Unknown job configuration version")
      elif type(tunedData) is list: # zero version file (without versioning 
        # control):
        # Old version was saved as follows:
        #objSave = [neuron, sort, initBounds, train]
        self._version = 0
        self.neuronBounds = MatlabLoopingBounds( [tunedData[0], tunedData[0]] )
        self.sortBounds   = MatlabLoopingBounds( [tunedData[1], tunedData[1]] )
        self.initBounds   = MatlabLoopingBounds( tunedData[2] )
        self.tunedDiscr   = tunedData[3]
      else:
        raise RuntimeError("Unknown file type entered for config file.")
    except RuntimeError, e:
      raise RuntimeError(("Couldn't read configuration file '%s': Reason:"
          "\n\t %s" % (self._filePath, e)))
    return self

  def getTunedInfo( neuron, sort, init ):
    if not self._nList:
      self._nList = self.neuronBounds.list(); self._nListLen = len( nList )
      self._sList = self.sortBounds.list();   self._sListLen = len( sList )
      self._iList = self.initBounds.list();   self._iListLen = len( iList )
    try:
      # On version 0 and 1 we first loop on sort list, then on neuron bound, to
      # finally loop over the initializations:
      yield sList.index( sort ) * ( nListLen * iListLen ) + \
            nList.index( neuron ) * ( iListLen ) + \
            iList.index( init )
    except ValueError, e:
      raise ValueError(("Couldn't find one the required indexes on the job bounds. "
          "The retrieved error was: %s") % e)
    
  def __exit__(self, exc_type, exc_value, traceback):
    # Remove bounds
    self.neuronBounds = None 
    self.sortBounds = None 
    self.initBounds = None 
    self.tunedDiscr = None 
    self._nList = None; self._nListLen = None
    self._sList = None; self._sListLen = None
    self._iList = None; self._iListLen = None

class TuningJob(Logger):
  """
    This class is used to tune a classifier through the call method.
  """

  _tuningtool = TuningTool(level = LoggingLevel.INFO)

  def __init__(self, logger = None ):
    """
      Initialize the TuningJob using a log level.
    """
    Logger.__init__( self, logger = logger )
    self._tuningtool.setLevel( self.level )
    self.compress = False

  @classmethod
  def fixLoopingBoundsCol(cls, var, wantedType = LoopingBounds,
      wantedCollection = LoopingBoundsCollection ):
    """
      Helper method to correct variable to be a looping bound collection
      correctly represented by a LoopingBoundsCollection instance.
    """
    if not isinstance( var, wantedCollection ):
      if not isinstance( var, wantedType ):
        var = wantedType( var )
      var = wantedCollection( var )
    return var


  def __call__(self, dataLocation, **kw ):
    """
      Run discrimination tuning for input data created via CreateData.py
      Arguments:
        - dataLocation: A string containing a path to the data file written
          by CreateData.py
      Mutually exclusive optional arguments: Either choose the cross (x) or
        circle (o) of the following block options.
       -------
        x crossValid [CrossValid( nSorts=50, nBoxes=10, nTrain=6, nValid=4, 
                                  seed=crossValidSeed )]:
          The cross-validation sorts object. The files can be generated using a
          CreateConfFiles instance which can be accessed via command line using
          the createTuningJobFiles.py script.
        x crossValidSeed [None]: Only used when not specifying the crossValid option.
          The seed is used by the cross validation random sort generator and
          when not specified or specified as None, time is used as seed.
        o crossValidFile: The cross-validation file path, pointing to a file
          created with the create tuning job files
       -------
        x confFileList [None]: A python list or a comma separated list of the
          root files containing the configuration to run the jobs. The files can
          be generated using a CreateConfFiles instance which can be accessed via
          command line using the createTuningJobFiles.py script.
        o neuronBoundsCol [MatlabLoopingBounds(5,5)]: A LoopingBoundsCollection
          range where the the neural network should loop upon.
        o sortBoundsCol [PythonLoopingBounds(50)]: A LoopingBoundsCollection
          range for the sorts to use from the crossValid object.
        o initBoundsCol [PythonLoopingBounds(100)]: A LoopingBoundsCollection
          range for the initialization numbers to be ran on this tuning job.
          The neuronBoundsCol, sortBoundsCol, initBoundsCol will be synchronously
          looped upon, that is, all the first collection information upon those 
          variables will be used to feed the first job configuration, all of 
          those second collection information will be used to feed the second job 
          configuration, and so on...
          In the case you have only one job configuration, it can be input as
          a single LoopingBounds instance, or values that feed the LoopingBounds
          initialization. In the last case, the neuronBoundsCol will be used
          to feed a MatlabLoopingBounds, and the sortBoundsCol together with
          initBoundsCol will be used to feed a PythonLoopingBounds.
          For instance, if you use neuronBoundsCol set to [5,2,11], it will 
          loop upon the list [5,7,9,11], while if this was set to sortBoundsCol,
          it would generate [5,7,9].
       -------
        x ppFileList [None]: A python list or a comma separated list of the
          root files containing the pre-processing chain to apply into 
          input space and obtain the pattern space. The files can be generated
          using a CreateConfFiles instance which can be accessed via command
          line using the createTuningJobFiles.py script.
        o ppCol [PreProcChain( Norm1() )]: A PreProcCollection with the
          PreProcChain instances to be applied to each of the configuration
          ranges chosen by the above configurations.
       -------
      Optional arguments:
        - compress [True]: Whether to compress file or not.
        - doMultiStop (C++ TuningTool prop) [True]: Whether to optimize for SP,
            Pf, Pa for the same tuning.
        - showEvo (C++ TuningTool prop) [50]: The number of iterations where the
            performance is shown.
        - maxFail (C++ TuningTool prop) [50]: Maximum number of failures to improve
            performance over validation dataset.
        - epochs (C++ TuningTool prop) [1000]: Maximum number iterations, where
            the tuning algorithm should stop the optimization.
        - doPerf (C++ TuningTool prop) [True]: Whether we should run performance
            testing under convergence conditions, using test/validation dataset
            and estimate operation conditions.
        - level [logging.info]: The logging output level.
        - seed (C++ TuningTool prop) [None]: The seed to be used by the tuning
            algorithm.
        - maxFail (C++ TuningTool prop) [50]: Number of epochs which failed to improve
            validation efficiency to stop training.
        - outputFileBase ['nn.tuned']: The tuning outputFile starting string.
            It will also contain a custom string representing the configuration
            used to tune the discriminator.
    """
    import gc
    from RingerCore.util import fixFileList

    if 'level' in kw: 
      self.setLevel( kw.pop('level') )# log output level
    self._tuningtool.setLevel( self.level )
    self.compress                = kw.pop('compress',           True    )
    ### Retrieve configuration from input values:
    ## We start with basic information:
    self._tuningtool.doMultiStop = kw.pop('doMultiStop',        True    )
    self._tuningtool.showEvo     = kw.pop('showEvo',             50     )
    self._tuningtool.epochs      = kw.pop('epochs',             1000    )
    self._tuningtool.doPerf      = kw.pop('doPerf',             True    )
    self._tuningtool.seed        = kw.pop('seed',               None    )
    self._tuningtool.maxFail     = kw.pop('maxFail',             50     )
    outputFileBase               = kw.pop('outputFileBase',  'nn.tuned' )
    ## Now we go to parameters which need higher treating level, starting with
    ## the CrossValid object:
    # Make sure that the user didn't try to use both options:
    if 'crossValid' in kw and 'crossValidFile' in kw:
      raise ValueError("crossValid is mutually exclusive with crossValidFile, \
          either use or another terminology to specify CrossValid object.")
    crossValidFile               = kw.pop('crossValidFile', None )
    from TuningTools.CrossValid import CrossValid, CrossValidArchieve
    if not crossValidFile:
      # Cross valid was not specified, read it from crossValid:
      crossValid                 = kw.pop('crossValid', \
          CrossValid( nSorts=50, nBoxes=10, nTrain=6, nValid=4, level = self.level, \
                      seed = kw.pop('crossValidSeed', None ) ) )
    else:
      with CrossValidArchieve( crossValidFile ) as CVArchieve:
        crossValid = CVArchieve
      del CVArchieve
    ## Read configuration for job parameters:
    # Check if there is no conflict on job parameters:
    if 'confFileList' in kw and ( 'neuronBoundsCol' in kw or \
                                  'sortBoundsCol'   in kw or \
                                  'initBoundsCol'   in kw ):
      raise ValueError(("confFileList is mutually exclusive with [neuronBounds, " \
          "sortBounds and initBounds], either use one or another " \
          "terminology to specify the job configuration."))
    confFileList    = kw.pop('confFileList', None )
    # Retrieve configuration looping parameters
    if not confFileList:
      # There is no configuration file, read information from kw:
      neuronBoundsCol   = kw.pop('neuronBoundsCol', MatlabLoopingBounds(5, 5) )
      sortBoundsCol     = kw.pop('sortBoundsCol',   PythonLoopingBounds(50)   )
      initBoundsCol     = kw.pop('initBoundsCol',   PythonLoopingBounds(100)  )
    else:
      # Make sure confFileList is in the correct format
      confFileList = fixFileList( confFileList )
      # Now loop over confFiles and add to our configuration list:
      neuronBoundsCol = LoopingBoundsCollection()
      sortBoundsCol   = LoopingBoundsCollection()
      initBoundsCol   = LoopingBoundsCollection()
      from TuningTools.CreateTuningJobFiles import TuningJobConfigArchieve
      for confFile in confFileList:
        with TuningJobConfigArchieve( confFile ) as (neuronBounds, 
                                                     sortBounds,
                                                     initBounds):
          neuronBoundsCol += neuronBounds
          sortBoundsCol   += sortBounds
          initBoundsCol   += initBounds
    # Now we make sure that bounds variables are LoopingBounds objects:
    neuronBoundsCol = TuningJob.fixLoopingBoundsCol( neuronBoundsCol,
                                                     MatlabLoopingBounds )
    sortBoundsCol   = TuningJob.fixLoopingBoundsCol( sortBoundsCol,
                                                     PythonLoopingBounds )
    initBoundsCol   = TuningJob.fixLoopingBoundsCol( initBoundsCol,
                                                     PythonLoopingBounds )
    # Check if looping bounds are ok:
    for neuronBounds in neuronBoundsCol():
      if neuronBounds.lowerBound() < 1:
        raise ValueError("Neuron lower bound is not allowed, it must be at least 1.")
    for sortBounds in sortBoundsCol():
      if sortBounds.lowerBound() < 0:
        raise ValueError("Sort lower bound is not allowed, it must be at least 0.")
      if sortBounds.upperBound() >= crossValid.nSorts():
        raise ValueError(("Sort upper bound is not allowed, it is higher then the number"
            "of sorts used."))
    for initBounds in initBoundsCol():
      if initBounds.lowerBound() < 0:
        raise ValueError("Attempted to create an initialization index lower than 0.")
    ## Check ppCol or ppFileList
    if 'ppFileList' in kw and 'ppCol' in kw:
      raise ValueError(("ppFileList is mutually exclusive with ppCol, "
          "either use one or another terminology to specify the job "
          "configuration."))
    ppFileList    = kw.pop('ppFileList', None )
    if not ppFileList:
      ppCol = kw.pop( 'ppCol', PreProcChain( Norm1(level = self.level) ) )
    else:
      # Make sure confFileList is in the correct format
      ppFileList = fixFileList( ppFileList )
      # Now loop over ppFileList and add it to our pp list:
      ppCol = PreProcCollection()
      for ppFile in ppFileList:
        with PreProcArchieve(ppFile) as PPArchieve:
          ppCol += PPArchieve
      del PPArchieve
    # Make sure that our pre-processings are PreProcCollection instances:
    ppCol = TuningJob.fixLoopingBoundsCol( ppCol,
                                           PreProcChain,
                                           PreProcCollection )
    ## Finished retrieving information from kw:
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    # Load data
    self._logger.info('Opening data...')

    from TuningTools.CreateData import TuningDataArchive
    with TuningDataArchive(dataLocation) as TDArchieve:
      data = TDArchieve
    del TDArchieve

    # Retrieve some useful information and keep it on memory
    ppColSize = len(ppCol)
    nConfigs = len(neuronBoundsCol)

    # For the ppCol, we loop independently:
    for ppChainIdx, ppChain in enumerate(ppCol):
      # Apply ppChain:
      data = ppChain( data )
      # Retrieve resulting data shape
      nInputs = data[0].shape[1]
      # Hold the training records
      train = []
      # For the bounded variables, we loop them together for the collection:
      for confNum, neuronBounds, sortBounds, initBounds in \
          zip(range(nConfigs), neuronBoundsCol, sortBoundsCol, initBoundsCol):
        self._logger.info('Running configuration file number %d', confNum)
        nSorts = len(sortBounds)
        # Finally loop within the configuration bounds
        for sort in sortBounds():
          self._logger.info('Extracting cross validation sort %d', sort)
          trnData, valData, tstData = crossValid( data, sort )
          sgnSize = trnData[0].shape[0]
          bkgSize = trnData[1].shape[0]
          batchSize = bkgSize if sgnSize > bkgSize else sgnSize
          # Update tuningtool working data information:
          self._tuningtool.batchSize = batchSize
          self._logger.debug('Set batchSize to %d', self._tuningtool.batchSize )
          self._tuningtool.setTrainData(   trnData   )
          self._tuningtool.setValData  (   valData   )
          self._tuningtool.setTestData (   tstData   )
          del data
          # Garbage collect now, before entering training stage:
          gc.collect()
          # And loop over neuron configurations and initializations:
          for neuron in neuronBounds():
            for init in initBounds():
              self._logger.info('Training <Neuron = %d, sort = %d, init = %d>...', \
                  neuron, sort, init)
              self._tuningtool.newff([nInputs, neuron, 1], ['tansig', 'tansig'])
              tunedDiscr = self._tuningtool.train_c()
              self._logger.debug('Finished C++ training, appending tuned discriminators to training record...')
              # Append retrieven tuned discriminators
              train.append( tunedDiscr )
            self._logger.debug('Finished all initializations for sort %d...', sort)
          # Finished all inits for this sort, we need to undo the crossValid if
          # we are going to do a new sort, otherwise we continue
          if not ( confNum == nConfigs and sort == nSorts):
            data = crossValid.revert( trnData, valData, tstData, sort = sort )
            del trnData, valData, tstData
          self._logger.debug('Finished all hidden layer neurons for sort %d...', sort)
        self._logger.debug('Finished all sorts for configuration %d in collection...', confNum)
        # Finished retrieving all tuned discriminators for this config file for
        # this pre-processing. Now we head to save what we've done so far:

        # Define output file name:
        ppStr = str(ppChain) if (ppColSize == 1 and len(ppChain) < 2) else ('pp%04d' % ppIdx)

        fulloutput = '{outputFileBase}.{ppStr}.{neuronStr}.{sortStr}.{initStr}.pic'.format( 
                      outputFileBase = outputFileBase, 
                      ppStr = ppStr,
                      neuronStr = neuronBounds.formattedString('hn'), 
                      sortStr = sortBounds.formattedString('s'),
                      initStr = initBounds.formattedString('i') )

        self._logger.info('Saving file named %s...', fulloutput)

        savedFile = TunedDiscrArchieve( fulloutput, neuronBounds = neuronBounds, 
                                        sortBounds = sortBounds, 
                                        initBounds = initBounds,
                                        tunedDiscr = train ).save( self.compress )
        self._logger.info('File "%s" saved!', savedFile)

      # Finished all we had to do for this pre-processing
      if ppColSize > 1 and (ppChainIdx + 1) != ppColSize:
        # If we have more pre-processings to test, then we need to revert
        # previous pre-processing to obtain data in the input space once again:
        if ppChain.isRevertible():
          self._logger.debug("Reverting pre-processing chain...")
          data = ppChain(data, True) # Revert it
        else:
          # We cannot revert ppChain, reload data:
          self._logger.info('Re-opening raw data...')
          data = self._loadData( dataLocation )
      self._logger.debug('Finished all configurations for ppChain %s...', str(ppChain))
    # finished ppCol
    self._logger.info('Finished tuning job!')
  # end of __call__ member fcn

