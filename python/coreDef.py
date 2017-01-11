__all__ = [ 'hasExmachina', 'hasFastnet', 'hasKeras', 'TuningToolCores'
          , 'AvailableTuningToolCores', 'CoreConfiguration', 'coreConf'
          , 'NumpyConfiguration', 'npCurrent'
          , 'DataframeConfiguration' , 'dataframeConf']

import os, pkgutil
# This is needed due to some keras issue with numpy import order
try:
  import keras
except ImportError:
  pass


hasExmachina = bool( pkgutil.find_loader( 'exmachina' )      )
hasFastnet   = bool( pkgutil.find_loader( 'libTuningTools' ) )
hasKeras     = bool( pkgutil.find_loader( 'keras' )          )

from RingerCore import ( EnumStringification, npConstants, Configure
                       , EnumStringificationOptionConfigure, Holder
                       , NotSet )

class TuningToolCores( EnumStringification ):
  _ignoreCase = True
  FastNet = 0
  ExMachina = 1
  keras = 2

class AvailableTuningToolCores( EnumStringification ):
  _ignoreCase = True
  if hasFastnet: FastNet = 0
  if hasExmachina: ExMachina = 1
  if hasKeras: keras = 2

  @classmethod
  def retrieve(cls, val):
    ret = TuningToolCores.retrieve( val )
    if not cls.tostring( ret ):
      raise ValueError("TuningTool core %s is not available in the current system." % TuningToolCores.tostring( ret ))
    return ret 

class _ConfigureCoreFramework( EnumStringificationOptionConfigure ):
  """
  Singleton class for configurating the core framework used tuning data

  It also specifies how the numpy data should be represented for that specified
  core.
  """

  _enumType = TuningToolCores

  core = property( EnumStringificationOptionConfigure.get, EnumStringificationOptionConfigure.set )

  def auto( self ):
    self._logger.debug("Using automatic configuration for core specification.")
    # Check whether we can retrieve from the parser.
    from TuningTools.parsers.BaseModuleParser import coreFrameworkParser
    import sys
    args, argv = coreFrameworkParser.parse_known_args()
    if args.core_framework not in (None, NotSet):
      self.core = args.core_framework
      # Consume option
      sys.argv = sys.argv[:1] + argv
    else:
			# Couldn't retrieve from the parser, retrieve default:
			self.core = self.default()

  def default( self ):
    if hasFastnet: 
      core = TuningToolCores.FastNet
    elif hasKeras:
      core = TuningToolsCores.keras
    elif hasExmachina:
      core = TuningToolCores.ExMachina
    else:
      self._logger.fatal("Couldn't define which tuning core was compiled.")
    return core

  def numpy_wrapper(self):
    """
    Returns the api instance which is to be used to read the data
    """
    import numpy as np
    if self.core is TuningToolCores.ExMachina:
      # Define the exmachina numpy constants
      kwargs = { 'useFortran' : True, 'fp_dtype' : np.float64, 'int_dtype' : np.int64 }
    elif self.core is TuningToolCores.FastNet:
      kwargs = { 'useFortran' : False, 'fp_dtype' : np.float32, 'int_dtype' : np.int32 }
    elif self.core is TuningToolCores.keras:
      from keras.backend import backend
      if backend() == "theano": # Theano copies data if input is not c-contiguous
        kwargs = { 'useFortran' : False, 'fp_dtype' : np.float32, 'int_dtype' : np.int32 }
      elif backend() == "tensorflow": # tensorflow copies data if input is not fortran-contiguous
        kwargs = { 'useFortran' : True, 'fp_dtype' : np.float32, 'int_dtype' : np.int32 }
    return npConstants( **kwargs )

  def core_framework(self):
    if self.core is TuningToolCores.FastNet:
      from libTuningTools import TuningToolPyWrapper as RawWrapper
      import sys, os
      class TuningToolPyWrapper( RawWrapper, object ): 
        def __init__( self
                    , level
                    , useColor = not(int(os.environ.get('RCM_GRID_ENV',0)) or not(sys.stdout.isatty()))
                    , seed = None):
          self._doMultiStop = False
          if seed is None:
            RawWrapper.__init__(self, level, useColor)
          else:
            RawWrapper.__init__(self, level, useColor, seed)

        @property
        def multiStop(self):
          return self._doMultiStop

        @multiStop.setter
        def multiStop(self, value):
          if value: 
            self._doMultiStop = True
            self.useAll()
          else: 
            self._doMultiStop = False
            self.useSP()

      # End of TuningToolPyWrapper
      return TuningToolPyWrapper
    elif self.core is TuningToolCores.ExMachina:
      import exmachina
      return exmachina
    elif self.core is TuningToolCores.keras:
      import keras
      return keras

# The singleton holder
CoreConfiguration = Holder( _ConfigureCoreFramework() )

# Standard core configuration object
coreConf = CoreConfiguration()

class _ConfigureNumpyWrapper( Configure ):
  """
  Wrapper for numpy module setting defaults accordingly to the core used.
  """

  wrapper = property( Configure.get, Configure.set )

  def auto( self ):
    self._logger.debug("Using automatic configuration for numpy wrapper.")
    self.wrapper = coreConf.numpy_wrapper()
    self._logger.debug("Retrieved the following numpy wrapper:\n%s", self.wrapper)

  def __getattr__(self, attr ):
    if hasattr( self.wrapper, attr ):
      return getattr( self.wrapper, attr )
    else:
      raise AttributeError( attr )

# The singleton holder
NumpyConfiguration = Holder( _ConfigureNumpyWrapper() )

# Standard numpy configuration object
npCurrent = NumpyConfiguration()

from TuningTools.dataframe.EnumCollection import Dataframe as DataframeEnum

class _ConfigureDataframe( EnumStringificationOptionConfigure ):
  """
  Singleton class for configurating the data framework used for reading the
  files and generating the tuning-data
  """

  _enumType = DataframeEnum

  dataframe = property( EnumStringificationOptionConfigure.get, EnumStringificationOptionConfigure.set )

  def auto_retrieve_testing_sample( self, sample ):
    self._sample = sample

  def auto( self ):
    self._logger.debug("Using automatic configuration for dataframe specification.")
    # Check whether we can retrieve from the parser.
    from TuningTools.parsers.BaseModuleParser import dataframeParser
    import sys
    args, argv = dataframeParser.parse_known_args()
    if args.dataframe not in (None, NotSet):
      self.dataframe = args.dataframe
      # Consume option
      sys.argv = sys.argv[:1] + argv
    if not self.configured() and not hasattr(self, '_sample'):
      self._logger.fatal("Cannot auto-configure which dataframe to use because no sample was specified via the auto_retrieve_sample() method.")
    elif not self.configured():
      if self._sample and isinstance(self._sample[0], basestring ):
        from RingerCore import csvStr2List, expandFolders
        fList = csvStr2List ( self._sample[0] )
        fList = expandFolders( fList )
        from ROOT import TFile
        for inputFile in fList:
          f  = TFile.Open(inputFile, 'read')
          if not f or f.IsZombie():
            continue
          self.dataframe = DataframeEnum.PhysVal
          for key in f.GetListOfKeys():
            if key.GetName == "ZeeCanditate":
              self.dataframe = DataframeEnum.Egamma
              break
          break
      elif isinstance(self._sample, dict):
        for key in self._sample:
          if 'elCand2_' in key:
            self.dataframe = DataframeEnum.Egamma
          else:
            self.dataframe = DataframeEnum.PhysVal
          break
    if not self.configured():
      self._logger.fatal("Couldn't auto-configure dataframe.")

  def api(self):
    """
    Returns the api instance which is to be used to read the data
    """
    if self.dataframe is DataframeEnum.PhysVal:
      from TuningTools.dataframe.ReadPhysVal import readData
    elif self.dataframe is DataframeEnum.Egamma:
      from TuningTools.dataframe.ReadEgamma import readData
    return readData

  def efficiencyBranches(self):
    from TuningTools.dataframe.EnumCollection import RingerOperation
    if self.dataframe is DataframeEnum.PhysVal:
      return { RingerOperation.L2Calo                      : 'L2CaloAccept'
             , RingerOperation.L2                          : 'L2ElAccept'
             , RingerOperation.EFCalo                      : 'EFCaloAccept'
             , RingerOperation.HLT                         : 'HLTAccept'
             , RingerOperation.Offline_LH_VeryLoose        : None
             , RingerOperation.Offline_LH_Loose            : 'LHLoose'
             , RingerOperation.Offline_LH_Medium           : 'LHMedium'
             , RingerOperation.Offline_LH_Tight            : 'LHTight'
             , RingerOperation.Offline_LH                  : ['LHLoose','LHMedium','LHTight']
             , RingerOperation.Offline_CutBased_Loose      : 'CutBasedLoose'
             , RingerOperation.Offline_CutBased_Medium     : 'CutBasedMedium'
             , RingerOperation.Offline_CutBased_Tight      : 'CutBasedTight'
             , RingerOperation.Offline_CutBased            : ['CutBasedLoose','CutBasedMedium','CutBasedTight']
             }
    elif self.dataframe is DataframeEnum.Egamma:
      return { RingerOperation.L2Calo                  : None
             , RingerOperation.L2                      : None
             , RingerOperation.EFCalo                  : None
             , RingerOperation.HLT                     : None
             , RingerOperation.Offline_LH_VeryLoose    : 'elCand2_isVeryLooseLLH_Smooth_v11' # isVeryLooseLL2016_v11
             , RingerOperation.Offline_LH_Loose        : 'elCand2_isLooseLLH_Smooth_v11'
             , RingerOperation.Offline_LH_Medium       : 'elCand2_isMediumLLH_Smooth_v11'
             , RingerOperation.Offline_LH_Tight        : 'elCand2_isTightLLH_Smooth_v11'
             , RingerOperation.Offline_LH              : ['elCand2_isVeryLooseLLH_Smooth_v11'
                                                         ,'elCand2_isLooseLLH_Smooth_v11'
                                                         ,'elCand2_isMediumLLH_Smooth_v11'
                                                         ,'elCand2_isTightLLH_Smooth_v11']
             , RingerOperation.Offline_CutBased_Loose  : 'elCand2_isEMLoose2015'
             , RingerOperation.Offline_CutBased_Medium : 'elCand2_isEMMedium2015'
             , RingerOperation.Offline_CutBased_Tight  : 'elCand2_isEMTight2015'
             , RingerOperation.Offline_CutBased        : ['elCand2_isEMLoose2015'
                                                         ,'elCand2_isEMMedium2015'
                                                         ,'elCand2_isEMTight2015']
             }

# The singleton holder
DataframeConfiguration = Holder( _ConfigureDataframe() )

# Standard dataframe configuration object
dataframeConf = DataframeConfiguration()
