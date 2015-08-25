from FastNetTool.Logger  import Logger, LoggingLevel
from FastNetTool.FileIO  import save, load
from FastNetTool.FastNet import FastNet
from FastNetTool.LoopingBounds import *
from FastNetTool.PreProc import *
from FastNetTool.util    import EnumStringification

class TuningJob(Logger):
  """
    This class is used to tune a classifier through the call method.
  """

  _fastnet = FastNet(level = LoggingLevel.INFO)

  def __init__(self, logger = None ):
    """
      Initialize the TuningJob using a log level.
    """
    Logger.__init__( self, logger = logger )

  @classmethod
  def __separateClasses( cls, data, target ):
    """
      Function for dealing with legacy data.
    """
    import numpy as np
    sgn = data[np.where(target==1)]
    bkg = data[np.where(target==-1)]
    return (bkg, sgn)

  def _loadData(self, filePath):
    """
      Helper method to load data from a path.
    """
    import os
    if not os.path.isfile( os.path.expandvars( filePath ) ):
      raise ValueError("Cannot reach file %s" % filePath )
    npData = load( filePath )
    try:
      if type(npData) is np.ndarray:
        # Legacy type:
        from FastNetTool.util   import reshape
        data = reshape( npData[0] ) 
        target = reshape( npData[1] ) 
        data = TuningJob.__separateClasses( data, target )
      elif type(npData) is np.lib.npyio.NpzFile:
          if npData['type'] != 'TuningData':
            raise RuntimeError("Input file is not of TuningData type!")
          if npData['version'] == 1:
            data = (npData['signal_rings'], npData['background_rings'])
          else:
            raise RuntimeError("Unknown file version!")
      else:
        raise RuntimeError("Object on file is of unkown type.")
    except RuntimeError, e:
      raise RuntimeError("Couldn't read data file. Reason:\n\t%s" % e)
    return data

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
        - doMultiStop (C++ FastNet prop) [True]: Whether to optimize for SP,
            Pf, Pa for the same tuning.
        - showEvo (C++ FastNet prop) [50]: The number of iterations where the
            performance is shown.
        - epochs (C++ FastNet prop) [1000]: Maximum number iterations, where
            the tuning algorithm should stop the optimization.
        - doPerf (C++ FastNet prop) [True]: Whether we should run performance
            testing under convergence conditions, using test/validation dataset
            and estimate operation conditions.
        - level [logging.info]: The logging output level.
        - seed (C++ FastNet prop) [None]: The seed to be used by the tuning
            algorithm.
        - outputFileBase ['nn.tuned']: The tuning outputFile starting string.
            It will also contain a 
    """
    import gc
    from FastNetTool.util import checkForUnusedVars, fixFileList

    if 'level' in kw: 
      print "Changing log level"
      self.setLevel( kw.pop('level') )# log output level
    ### Retrieve configuration from input values:
    ## We start with basic information:
    self._fastnet.doMultiStop = kw.pop('doMultiStop',        True    )
    self._fastnet.showEvo     = kw.pop('showEvo',             50     )
    self._fastnet.epochs      = kw.pop('epochs',             1000    )
    self._fastnet.doPerf      = kw.pop('doPerf',             True    )
    self._fastnet.seed        = kw.pop('seed',               None    )
    outputFileBase            = kw.pop('outputFileBase',  'nn.tuned' )
    ## Now we go to parameters which need higher treating level, starting with
    ## the CrossValid object:
    # Make sure that the user didn't try to use both options:
    if 'crossValid' in kw and 'crossValidFile' in kw:
      raise ValueError("crossValid is mutually exclusive with crossValidFile, \
          either use or another terminology to specify CrossValid object.")
    crossValidFile   = kw.pop('crossValidFile', None )
    from FastNetTool.CrossValid import CrossValid
    if not crossValidFile:
      # Cross valid was not specified, read it from crossValid:
      crossValid       = kw.pop('crossValid', \
          CrossValid( nSorts=50, nBoxes=10, nTrain=6, nValid=4, level = self.level, \
                      seed = kw.pop('crossValidSeed', None ) ) )
    else:
      # Open crossValidFilefile:
      crossValidInfo   = load(crossValidFile)
      try: 
        if crossValidInfo['type'] != 'CrossValidFile':
          raise RuntimeError(("Input crossValid file is not from PreProcFile " 
              "type."))
        if crossValidInfo['version'] == 1:
          crossValid = crossValidInfo['crossValid']
        else:
          raise RuntimeError("Unknown job configuration version.")
      except RuntimeError, e:
        raise RuntimeError(("Couldn't read configuration file '%s': Reason:"
            "\n\t %s" % e))
      del ppColInfo
      if not isinstance(crossValid, CrossValid ):
        raise ValueError(("crossValidFile \"%s\" doesnt contain a CrossValid " \
            "object!") % crossValidFile)
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
      for confFile in confFileList:
        # Open file:
        jobConfig     = load(confFile)
        try:
          if type(jobConfig) is dict:
            if jobConfig['type'] != "ConfJobFile":
              raise RuntimeError(("Input jobConfig file is not from jobConfig " 
                  "type."))
            # Read configuration file to retrieve pre-processing, 
            if jobConfig['version'] == 1:
              neuronBoundsCol += MatlabLoopingBounds( jobConfig['neuronBounds'] )
              sortBoundsCol   += MatlabLoopingBounds( jobConfig['sortBounds']   )
              initBoundsCol   += MatlabLoopingBounds( jobConfig['initBounds']   )
            else:
              raise RuntimeError("Unknown job configuration version")
          elif type(jobConfig) is list: # zero version file (without versioning 
            # control):
            neuronBoundsCol   += MatlabLoopingBounds( [jobConfig[0], jobConfig[0]] )
            sortBoundsCol     += MatlabLoopingBounds( jobConfig[1] )
            initBoundsCol     += MatlabLoopingBounds( jobConfig[2] )
          else:
            raise RuntimeError("Unknown file type entered for config file.")
        except RuntimeError, e:
          raise RuntimeError(("Couldn't read configuration file '%s': Reason:"
              "\n\t %s" % e))
        del jobConfig
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
        ppColInfo += load( ppFile )
        try: 
          if ppColInfo['type'] != 'PreProcFile':
            raise RuntimeError(("Input crossValid file is not from PreProcFile " 
                "type."))
          if ppColInfo['version'] == 1:
            ppCol = ppColInfo['ppCol']
          else:
            raise RuntimeError("Unknown job configuration version.")
        except RuntimeError, e:
          raise RuntimeError(("Couldn't read configuration file '%s': Reason:"
              "\n\t %s" % e))
        del ppColInfo
    # Make sure that our pre-processing are PreProcCollection instances:
    ppCol           = TuningJob.fixLoopingBoundsCol( ppCol,
                                                     PreProcChain,
                                                     PreProcCollection )
    ## Finished retrieving information from kw:
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    # Load data
    self._logger.info('Opening data...')
    data = self._loadData( dataLocation )

    ppColSize = len(ppCol)

    # For the ppCol, we loop independently:
    for ppChainIdx, ppChain in enumerate(ppCol):

      # Apply normalizations:
      data = ppChain( data )

      # Retrieve resulting data shape
      nInputs = data[0].shape[1]

      # Hold the training records
      train = []

      # For the bounded variables, we loop them together for the collection:
      for confNum, neuronBounds, sortBounds, initBounds in \
          zip(range(len(neuronBoundsCol)), neuronBoundsCol, sortBoundsCol, initBoundsCol):

        # Finally loop within the configuration bounds
        for neuron in neuronBounds():
          for sort in sortBounds():
            self._logger.info('Extracting cross validation sort')
            trnData, valData, tstData = crossValid( data, sort )
            bkgSize = trnData[1].shape[0]
            sgnSize = trnData[0].shape[0]
            batchSize = bkgSize if sgnSize > bkgSize else sgnSize
            # Update fastnet working data information:
            self._fastnet.batchSize = batchSize
            self._logger.debug('Set batchSize to %d', self._fastnet.batchSize )
            self._fastnet.setTrainData(   trnData   )
            self._fastnet.setValData  (   valData   )
            self._fastnet.setTestData (   tstData   )
            del data
            # Garbage collect now, before entering training stage:
            gc.collect()
            for init in initBounds():
              self._logger.info('Training <Neuron = %d, sort = %d, init = %d>...', \
                  neuron, sort, init)
              self._fastnet.newff([nInputs, neuron, 1], ['tansig', 'tansig'])
              nets = self._fastnet.train_c()
              self._logger.debug('Finished C++ training, appending nets to training record...')
              train.append( nets )
            self._logger.debug('Finished all initializations for sort %d...', sort)
          # Finished all inits for this sort, we need to undo the crossValid if
          # we are going to do a new sort, otherwise we continue
          nSorts = len(sortBounds)
          if nSorts > 1 and sort != sortBounds.upperBound():
            data = crossValid.revert( trnData, valData, tstData, sort )
            del trnData, valData, tstData
          self._logger.debug('Finished all sorts for neuron %d...', neuron)
        self._logger.debug('Finished all neurons for configuration %d in collection...', confNum)
        # finished this config file for this normalization

        # Define output file name:
        ppStr = str(ppChain) if (ppColSize == 1 and len(ppChain) < 2) else ('pp%04d' % ppIdx)

        fulloutput = '{outputFileBase}.{ppStr}.{neuronStr}.{sortStr}.{initStr}.pic'.format( 
                      outputFileBase = outputFileBase, 
                      ppStr = ppStr,
                      neuronStr = neuronBounds.formattedString('hn'), 
                      sortStr = sortBounds.formattedString('s'),
                      initStr = initBounds.formattedString('i') )

        self._logger.info('Saving file named %s...', fulloutput)

        objSave = {"version" : 1,
                   "neuronBounds" : neuronBounds.getOriginalVec(),
                   "sortBounds" : sortBounds.getOriginalVec(),
                   "initBounds" : initBounds.getOriginalVec(),
                   "tunedDiscriminators" : train }
        save( objSave, fulloutput )
        self._logger.info('File "%s" saved!', fulloutput)
      # Finished all we had to do for this normalization
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

