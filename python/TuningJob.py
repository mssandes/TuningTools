__all__ = ['TunedDiscrArchieve', 'TunedDiscrArchieveCol', 'ReferenceBenchmark', 
           'ReferenceBenchmarkCollection', 'TuningJob', 'fixPPCol', 
           'fixLoopingBoundsCol',]
import numpy as np

from RingerCore               import Logger, LoggerStreamable, LoggingLevel, \
                                     RawDictCnv, LoggerRawDictStreamer, LoggerLimitedTypeListRDS, RawDictStreamer, \
                                     save, load, \
                                     EnumStringification, \
                                     checkForUnusedVars, NotSet, csvStr2List, retrieve_kw, \
                                     traverse, LimitedTypeList, RawDictStreamable, \
                                     LimitedTypeStreamableList
from RingerCore.LoopingBounds import *

from TuningTools.PreProc      import *
from TuningTools.ReadData     import Dataset
from TuningTools.coreDef      import retrieve_npConstants

npCurrent, _ = retrieve_npConstants()

class TunedDiscrArchieveRDS( LoggerRawDictStreamer ):
  """
  The TunedDiscrArchieve RawDict Streamer
  """
  def __init__(self, **kw):
    LoggerRawDictStreamer.__init__( self, 
        transientAttrs = {'_readVersion',},
        toPublicAttrs = {'_neuronBounds','_sortBounds','_initBounds',
                         '_etaBin', '_etBin', 
                         '_etaBinIdx', '_etBinIdx', 
                         '_tuningInfo', '_tunedDiscr', '_tunedPP'} )

  def treatDict(self, obj, raw):
    """
    Method dedicated to modifications on raw dictionary
    """
    if not obj.neuronBounds or \
       not obj.sortBounds   or \
       not obj.initBounds   or \
       not obj.tunedDiscr   or \
       not obj.tuningInfo   or \
       not obj.tunedPP:
      raise RuntimeError("Attempted to retrieve empty data from TunedDiscrArchieve.")
    # Treat looping bounds:
    raw['neuronBounds'] = transformToMatlabBounds( raw['neuronBounds'] ).getOriginalVec()
    raw['sortBounds']   = transformToPythonBounds( raw['sortBounds'] ).getOriginalVec()
    raw['initBounds']   = transformToPythonBounds( raw['initBounds'] ).getOriginalVec()
    # Treat raw discriminator:
    def transformToRawDiscr(tunedDiscr):
      for obj in traverse( tunedDiscr, simple_ret = True ):
        obj['benchmark'] = obj['benchmark'].toRawObj()
      return tunedDiscr
    raw['tunedDiscr']   = transformToRawDiscr( raw['tunedDiscr'] )
    return LoggerRawDictStreamer.treatDict(self, obj, raw)

class TunedDiscrArchieveRDC( RawDictCnv ):
  """
  The TunedDiscrArchieve RawDict Converter
  """

  def __init__(self, **kw):
    RawDictCnv.__init__( self, 
                         ignoreAttrs = {'type','version',
                                        # We add old version parameters here:
                                        'tuningInformation', 'trainEvolution', 'tunedDiscriminators',
                                        'tunedPPCollection'}, 
                         toProtectedAttrs = {'_neuronBounds','_sortBounds','_initBounds',
                                             '_etaBin', '_etBin', 
                                             '_etaBinIdx', '_etBinIdx',
                                             '_tuningInfo','_tunedDiscr', '_tunedPP'}, 
                         ignoreRawChildren = True,
                         **kw )

  def treatObj( self, obj, d ):
    if 'version' in d:
      # Treat 1->6 read versions:
      obj._readVersion = d['version']
    # Treat looping bounds:
    obj._neuronBounds = MatlabLoopingBounds( d['neuronBounds'] )
    obj._sortBounds   = PythonLoopingBounds( d['sortBounds']   )
    obj._initBounds   = PythonLoopingBounds( d['initBounds']   )
    def retrieveRawDiscr(tunedDiscr):
      for obj in traverse( tunedDiscr, simple_ret = True ):
        if type(obj['benchmark']) is dict:
          obj['benchmark'] = ReferenceBenchmark.fromRawObj( obj['benchmark'] )
      return tunedDiscr
    # end of local function retrieveRawDiscr
    if obj._readVersion >= 7:
      obj._tunedDiscr   = retrieveRawDiscr( d['tunedDiscr'] )
    else:
      # Read tuning information
      if obj._readVersion in (5,6,):
        obj._tuningInfo = d['tuningInformation']
      elif obj._readVersion >= 1:
        obj._tuningInfo = [tData[0]['trainEvolution'] for tData in d['tunedDiscriminators']]
      else:
        obj._logger.warning(("This TunedDiscrArchieve version still needs to have "
                             "implemented the access to the the tuning information."))
        obj._tuningInfo = None
      obj._tunedDiscr   = retrieveRawDiscr( d['tunedDiscriminators'] )
      if obj._readVersion <= 4:
        # Before version 4 we didn't save the benchmarks:
        def ffilt(tData): 
          for idx, discr in enumerate(tData):
            if idx == 0:
              discr['benchmark'] = ReferenceBenchmark( 'Tuning_EFCalo_SP', 'SP' )
            elif idx == 1:
              discr['benchmark'] = ReferenceBenchmark( 'Tuning_EFCalo_SP_Pd', 'SP' )
            elif idx == 2:
              discr['benchmark'] = ReferenceBenchmark( 'Tuning_EFCalo_SP_Pf', 'SP' )
        for tData in obj._tunedDiscr:
          ffilt(tData)
      if obj._readVersion == 3:
        # On version 3 we saved only the binning index:
        obj._etaBinIdx    = d['etaBin']
        obj._etBinIdx     = d['etBin']
        obj._etaBin       = npCurrent.array([0.,0.8,1.37,1.54,2.5])
        obj._etaBin       = obj._etaBin[obj._etaBinIdx:obj._etaBinIdx+2]
        obj._etBin        = npCurrent.array([0,30.,40.,50.,20000.])*1e3
        obj._etBin        = obj._etBin[obj._etBinIdx:obj._etBinIdx+2]
      if obj._readVersion <= 1:
        # On first version we didn't save the pre-processing, but we used only Norm1:
        obj._tunedPP      = PreProcCollection( [ PreProcChain( Norm1() ) for i in range(len(obj._sortBounds)) ] )
      elif obj._readVersion < 6:
        # From version 2 to 5 we used non-raw PreProcCollection with key "tunedPPCollection"
        obj._tunedPP      = PreProcCollection( d['tunedPPCollection'] )
      elif obj._readVersion < 7:
        # On version 6 we used raw PreProcCollection with key "tunedPPCollection"
        obj._tunedPP      = PreProcCollection.fromRawObj( d['tunedPPCollection'] )
    return obj


class TunedDiscrArchieve( LoggerStreamable ):
  """
  Manager for Tuned Discriminators archives

  Version 7: - Uses same save class attributes as dict keys and streamable
               infrastructure but makes profit of RDS and RDC functionality.
  Version 6: - Saves raw object from PreProcCollection
               Saves raw reference object
  Version 5: - added tuning benchmarks. 
             - separated tuning information from the tuned discriminators
               (tunedDiscr) variable.
  Version 4: - added eta/et bin limits
  Version 3: - added eta/et bin compatibility (only indexes)
  Version 2: - added pre-processing collection
  Version 1: - started using MatlabLoopingBounds and PythonLoopingBounds to save 
               the objects.
  Version 0: - save pickle file with a list containing the neuron/sort/init
               bounds in the same object
  """

  _streamerObj  = TunedDiscrArchieveRDS(transientAttrs = {'_tarMember', '_filePath'})
  _cnvObj       = TunedDiscrArchieveRDC()
  _version      = 7

  def __init__(self, **kw):
    Logger.__init__(self, kw)
    self._neuronBounds = kw.pop('neuronBounds', None                )
    self._sortBounds   = kw.pop('sortBounds',   None                )
    self._initBounds   = kw.pop('initBounds',   None                )
    self._tunedDiscr   = kw.pop('tunedDiscr',   None                )
    self._tuningInfo   = kw.pop('tuningInfo',   None                )
    self._tunedPP      = kw.pop('tunedPP',      None                )
    self._etaBinIdx    = kw.pop('etaBinIdx',    -1                  )
    self._etBinIdx     = kw.pop('etBinIdx',     -1                  )
    self._etaBin       = kw.pop('etaBin',       npCurrent.array([]) )
    self._etBin        = kw.pop('etBin',        npCurrent.array([]) )
    self._tarMember    = kw.pop('tarMember',    None                )
    self._filePath     = kw.pop('filePath',     None                )
    checkForUnusedVars( kw, self._logger.warning )

  @property
  def filePath( self ):
    return self._filePath

  @property
  def tarMember( self ):
    return self._tarMember

  @property
  def neuronBounds( self ):
    return self._neuronBounds

  @property
  def sortBounds( self ):
    return self._sortBounds

  @property
  def initBounds( self ):
    return self._initBounds

  @property
  def tunedDiscr( self ):
    return self._tunedDiscr

  @property
  def tunedPP( self ):
    return self._tunedPP

  @property
  def tuningInfo( self ):
    return self._tuningInfo

  @property
  def etaBinIdx( self ):
    return self._etaBinIdx

  @property
  def etBinIdx( self ):
    return self._etBinIdx

  @property
  def etaBin( self ):
    return self._etaBin

  @property
  def etBin( self ):
    return self._etBin

  def save(self, filePath, compress = True):
    """
    Save the TunedDiscrArchieve object to disk.
    """
    return save( self.toRawObj(), filePath, compress = compress )

  @classmethod
  def load(cls, filePath, useGenerator = False, tarMember = None):
    """
    Load a TunedDiscrArchieve object from disk and return it.
    """
    # Open file:
    from cPickle import PickleError
    try:
      import sys, inspect
      import TuningTools.ReadData as FilterEvents
      sys.modules['TuningTools.FilterEvents'] = inspect.getmodule(FilterEvents)
      rawObjCol = load(filePath, useHighLevelObj = False, 
                       useGenerator = useGenerator,
                       tarMember = tarMember)
    except (PickleError, TypeError, ImportError) as e: # TypeError due to add object inheritance on Logger
      # It failed without renaming the module, retry renaming old module
      # structure to new one.
      lLogger = Logger.getModuleLogger(cls.__name__)
      import traceback
      lLogger.warning("Couldn't load file due to:\n %s.\n Attempting to read on legacy mode...", traceback.format_exc())
      import TuningTools.Neural
      cNeural = TuningTools.Neural.Neural
      cLayer = TuningTools.Neural.Layer
      TuningTools.Neural.Layer = TuningTools.Neural.OldLayer
      TuningTools.Neural.Neural = TuningTools.Neural.OldNeural
      import sys
      import RingerCore.util
      sys.modules['FastNetTool.util'] = RingerCore.util
      sys.modules['FastNetTool.Neural'] = TuningTools.Neural
      rawObj = load(filePath, useHighLevelObj = False )
      TuningTools.Neural.Layer = cLayer
      TuningTools.Neural.Neural = cNeural
    if not useGenerator:
      if type(rawObjCol) is not list:
        rawObjCol = [rawObjCol]
    def __objRead(rawObjCol):
      i = 0
      for rawObj in rawObjCol:
        if type(rawObj) is list: # zero version file (without versioning 
          # control):
          # Old version was saved as follows:
          #objSave = [neuron, sort, initBounds, train]
          tunedList = rawObj; rawObj = dict()
          rawObj['__version']    = 0
          rawObj['neuronBounds'] = MatlabLoopingBounds( [tunedList[0], tunedList[0]] )
          rawObj['sortBounds']   = MatlabLoopingBounds( [tunedList[1], tunedList[1]] )
          rawObj['initBounds']   = MatlabLoopingBounds( tunedList[2] )
          rawObj['tunedDiscr']   = tunedList[3]
        # Finally, create instance from raw object
        if useGenerator: # load returns a tuple with the object/member path in
          # tarfile when using generator
          obj = cls.fromRawObj( rawObj[0] )
          obj._tarMember = rawObj[1]
          obj._filePath = filePath
          yield obj
        else:
          yield cls.fromRawObj( rawObj )
        i += 1
      # end of (for rawObj)
    # end of (__objRead)
    o = __objRead(rawObjCol)
    if not useGenerator:
      o = list(o)
      for obj in o: obj._filePath = filePath
      if len(o) == 1: o = o[0]
    return o
  # end of (load)

  def __str__(self):
    """
    Return string representation of object
    """
    ppStr = 'pp-' + self.tunedPP[0].shortName()[:12] # Truncate on 12th char
    neuronStr = self.neuronBounds.formattedString('hn')
    sortStr = self.sortBounds.formattedString('s')
    initStr = self.initBounds.formattedString('i')
    return 'TunedDiscrArchieve<%s.%s.%s.%s>' % (ppStr, neuronStr, sortStr, initStr)

  def getTunedInfo( self, neuron, sort, init ):
    """
    Retrieve tuned information within this archieve using neuron/sort/init indexes.
    """
    nList = self._neuronBounds.list(); nListLen = len( nList )
    sList = self._sortBounds.list();   sListLen = len( sList )
    iList = self._initBounds.list();   iListLen = len( iList )
    try:
      if self._readVersion >= 0:
        # For now, we first loop on sort list, then on neuron bound, to
        # finally loop over the initializations:
        idx = sList.index( sort ) * ( nListLen * iListLen ) + \
              nList.index( neuron ) * ( iListLen ) + \
              iList.index( init )
        sortIdx = sList.index( sort )
      return { 'tunedDiscr' : self._tunedDiscr[ idx ], \
               'tunedPP' : self._tunedPP[ sortIdx ], \
               'tuningInfo' : self._tuningInfo[ idx ] }
    except ValueError, e:
      raise ValueError(("Couldn't find one the required indexes on the job bounds. "
          "The retrieved error was: %s") % e)
  # getTunedInfo

  def exportDiscr( self, neuron, sort, init, operation, rawBenchmark ):
    """
      Export discriminator to be used by athena.
    """
    # FIXME refBenchmark.reference shall change to a user defined reference?
    # Maybe we should use the reference closiest to the wanted benchmark...
    from CrossValidStat import ReferenceBenchmark
    # FIXME This will need to be a method outside here, as we need to have
    # access to every bin discriminator in order to export it.
    tunedInfo = self.getTunedInfo(neuron, sort, init) \
                                 [ReferenceBenchmark.fromstring(rawBenchmark['reference'])]
    tunedDiscr = tunedInfo['tunedDiscr']
    from ReadData import RingerOperation
    if operation is RingerOperation.Offline:
      keep_lifespan_list = []
      # Load reflection information
      try:
        import cppyy
      except ImportError:
        import PyCintex as cppyy
      try:
        cppyy.loadDict('RingerSelectorTools_Reflex')
      except RuntimeError:
        raise RuntimeError("Couldn't load RingerSelectorTools_Reflex dictionary.")
      ## Import reflection information
      from ROOT import std # Import C++ STL
      from ROOT.std import vector # Import C++ STL
      from ROOT import Ringer
      from ROOT import MsgStream
      from ROOT import MSG
      from ROOT.Ringer import Discrimination
      from ROOT.Ringer import PreProcessing
      from ROOT.Ringer.PreProcessing      import Norm
      from ROOT.Ringer.Discrimination     import NNFeedForwardVarDep
      from ROOT.Ringer.PreProcessing.Norm import Norm1VarDep
      from ROOT.Ringer import IPreProcWrapperCollection
      from ROOT.Ringer import RingerProcedureWrapper
      ## Instantiate the templates:
      RingerNNIndepWrapper = RingerProcedureWrapper("Ringer::Discrimination::NNFeedForwardVarDep",
                                                    "Ringer::EtaIndependent",
                                                    "Ringer::EtIndependent",
                                                    "Ringer::NoSegmentation")
      RingerNorm1IndepWrapper = RingerProcedureWrapper("Ringer::PreProcessing::Norm::Norm1VarDep",
                                                       "Ringer::EtaIndependent",
                                                       "Ringer::EtIndependent",
                                                       "Ringer::NoSegmentation")
      ## Retrieve the pre-processing chain:
      BaseVec = vector("Ringer::PreProcessing::Norm::Norm1VarDep*")
      vec = BaseVec() # We are not using eta dependency
      norm1VarDep = Norm1VarDep()
      keep_lifespan_list.append( norm1VarDep )# We need to tell python to keep this object alive!
      vec.push_back( norm1VarDep )
      vecvec = vector(BaseVec)() # We are not using et dependency
      vecvec.push_back(vec)
      norm1Vec = vector( vector( BaseVec ) )() # We are not using longitudinal segmentation
      norm1Vec.push_back(vecvec)
      ## Create pre-processing wrapper:
      self._logger.debug('Initiazing norm1Wrapper:')
      norm1Wrapper = RingerNorm1IndepWrapper(norm1Vec)
      keep_lifespan_list.append( norm1Wrapper )
      ## Add it to the pre-processing collection chain
      self._logger.debug('Creating PP-Chain')
      ringerPPCollection = IPreProcWrapperCollection()
      ringerPPCollection.push_back(norm1Wrapper)
      keep_lifespan_list.append( ringerPPCollection )
      ## Retrieve the discriminator collection:
      BaseVec = vector("Ringer::Discrimination::NNFeedForwardVarDep*")
      vec = BaseVec() # We are not using eta dependency
      ## Export discriminator to the RingerSelectorTools format:
      nodes = std.vector("unsigned int")(); nodes += tunedDiscr.nNodes
      weights = std.vector("float")(); weights += tunedDiscr.get_w_array()
      bias = vector("float")(); bias += tunedDiscr.get_b_array()
      ringerDiscr = NNFeedForwardVarDep(nodes, weights, bias)
      keep_lifespan_list.append( ringerDiscr ) # Python has to keep this in memory as well.
      # Print information discriminator information:
      msg = MsgStream('ExportedNeuralNetwork')
      msg.setLevel(LoggingLevel.toC(self.level))
      keep_lifespan_list.append( msg )
      ringerDiscr.setMsgStream(msg)
      getattr(ringerDiscr,'print')(MSG.DEBUG)
      ## Add it to Discriminator collection
      vec.push_back(ringerDiscr)
      vecvec = vector( BaseVec )() # We are not using et dependency
      vecvec.push_back( vec )
      ringerNNVec = vector( vector( BaseVec ) )() # We are not using longitudinal segmentation
      ringerNNVec.push_back( vecvec )
      keep_lifespan_list.append( ringerNNVec )
      ## Create the discrimination wrapper:
      self._logger.debug('Creating RingerNNIndepWrapper:')
      nnWrapper = RingerNNIndepWrapper( ringerPPCollection, ringerNNVec )
      self._logger.debug('Returning information...')
      return nnWrapper, keep_lifespan_list
    # if operation
  # exportDiscr

class ReferenceBenchmark(EnumStringification, LoggerStreamable):
  """
  Reference benchmark to set discriminator operation point.

    - SP: Use the SUM-PRODUCT coeficient as an optimization target. 
    - Pd: Aims at operating with signal detection probability as close as
      possible from reference value meanwhile minimazing the false
      alarm probability.
    - Pf: Aims at operating with false alarm probability as close as
      possible from reference value meanwhile maximazing the detection
      probability.
    - MSE: Aims at reducing as much as possible the mean-squared error.
      If MSE is used to retrieve the outermost performance, it will return
      the outermost SP-index.
  """
  SP = 0
  Pd = 1
  Pf = 2
  MSE = 3

  _def_eps = .002

  #_streamerObj  = RawDictStreamer()
  #_cnvObj       = RawDictCnv()
  #_version      = 1

  def __init__(self, name = "", reference = SP, 
               signal_efficiency = None, background_efficiency = None,
               signal_cross_efficiency = None, background_cross_efficiency = None, 
               **kw):
    """
    ref = ReferenceBenchmark(name, reference, signal_efficiency = None, background_efficiency = None, 
                                   signal_cross_efficiency = None, background_cross_efficiency = None,
                                   [, removeOLs = False])

      * name: The name for this reference benchmark;
      * reference: The reference benchmark type. It must one of ReferenceBenchmark enumerations.
      * signal_efficiency: The reference benchmark signal efficiency.
      * background_efficiency: The reference benchmark background efficiency.
      * signal_cross_efficiency: The reference benchmark signal efficiency measured with the Cross-Validation sets.
      * background_cross_efficiency: The reference benchmark background efficiency with the Cross-Validation sets.
      * removeOLs [False]: Whether to remove outliers from operation.
      * allowLargeDeltas [True]: When set to true and no value is within the operation bounds,
       then it will use operation closer to the reference.
    """
    LoggerStreamable.__init__(self, kw)
    self.signal_efficiency           = signal_efficiency
    self.signal_cross_efficiency     = signal_cross_efficiency
    self.background_efficiency       = background_efficiency
    self.background_cross_efficiency = background_cross_efficiency
    self.removeOLs                   = kw.pop('removeOLs', False)
    self.allowLargeDeltas            = kw.pop('allowLargeDeltas', True)
    checkForUnusedVars( kw, self._logger.warning )
    del kw
    if not (type(name) is str):
      raise TypeError("Name must be a string.")
    self.name = name
    self.reference = ReferenceBenchmark.retrieve(reference)
    if not( self.reference  in (ReferenceBenchmark.SP, ReferenceBenchmark.MSE)) and \
       (self.signal_efficiency is None or self.background_efficiency is None):
      raise RuntimeError("Cannot create Pd/Pf object without signal/background efficiency")
  # __init__

  @property
  def refVal(self):
    if self.reference is ReferenceBenchmark.Pd:
      return self.signal_efficiency.efficiency()/100.
    elif self.reference == ReferenceBenchmark.Pf:
      return self.background_efficiency.efficiency()/100.
    else:
      return None

  @property
  def etaBinIdx(self):
    if self.signal_efficiency is not None:
      return self.signal_efficiency.etaBin
    else:
      return None

  @property
  def etBinIdx(self):
    if self.signal_efficiency is not None:
      return self.signal_efficiency.etBin
    else:
      return None

  def checkEtaBinIdx(self, val):
    if self.signal_efficiency is not None:
      return self.signal_efficiency.etaBin == val
    else:
      return False

  def checkEtBinIdx(self, val):
    if self.signal_efficiency is not None:
      return self.signal_efficiency.etBin == val
    else:
      return False

  def getReference(self, ds = Dataset.Operation, sort = None):
    """
    Get reference value. If sort is not specified, return the operation value.

    Otherwise, return the efficiency value over the test (or validation if test
    if not available).
    """
    if sort is not None:
      if self.reference is ReferenceBenchmark.Pd:
        if self.signal_cross_efficiency is not None:
          return self.signal_cross_efficiency.efficiency(ds, sort)/100.
        else:
          self._logger.warning("Cross-validation efficiency couldn't be retrieved. Using operation efficiency.")
          return self.refVal
      elif self.reference == ReferenceBenchmark.Pf:
        if self.background_cross_efficiency is not None:
          return self.background_cross_efficiency.efficiency(ds, sort)/100.
        else:
          self._logger.warning("Cross-validation efficiency couldn't be retrieved. Using operation efficiency.")
          return self.refVal
    else:
      return self.refVal

  def rawInfo(self):
         """
         Return raw benchmark information. Used by CrossValidStats, cannot be recovered.
         """
         from TuningTools.ReadData import BranchEffCollector, BranchCrossEffCollector
         return { 'reference': ReferenceBenchmark.tostring(self.reference),
                     'refVal': (self.refVal if not self.refVal is None else -999),
          'signal_efficiency': self.signal_efficiency.toRawObj() \
                               if self.signal_efficiency is not None else \
                               BranchEffCollector().toRawObj(),
    'signal_cross_efficiency': self.signal_cross_efficiency.toRawObj(noChildren=True) \
                               if self.signal_cross_efficiency is not None else \
                               BranchCrossEffCollector().toRawObj(noChildren=True),
      'background_efficiency': self.background_efficiency.toRawObj()
                               if self.background_efficiency is not None else \
                               BranchEffCollector().toRawObj(),
'background_cross_efficiency': self.background_cross_efficiency.toRawObj(noChildren=True) \
                               if self.background_cross_efficiency is not None else \
                               BranchCrossEffCollector().toRawObj(noChildren=True),
                  'removeOLs': self.removeOLs }

  def getOutermostPerf(self, data, **kw):
    """
    Get outermost performance for the tuned discriminator performances on data. 
    idx = refBMark.getOutermostPerf( data [, eps = .002 ][, cmpType = 1])

     * data: A list with following struction:
        data[0] : SP
        data[1] : Pd
        data[2] : Pf

     * eps [.005] is used for softening. The larger it is, more candidates will
      be possible to be considered, but farther the returned operation may be from
      the reference. The default is _def_eps deviation from the reference value.
     * cmpType [+1.] is used to change the comparison type. Use +1 for best
      performance, and -1 for worst performance.
    """
    # Retrieve optional arguments
    eps     = retrieve_kw( kw, 'eps',     self._def_eps )
    cmpType = retrieve_kw( kw, 'cmpType', 1.            )
    sortIdx = retrieve_kw( kw, 'sortIdx', None          )
    ds      = retrieve_kw( kw, 'ds',      Dataset.Test  )
    # We will transform data into np array, as it will be easier to work with
    npData = []
    for aData, label in zip(data, ['SP', 'Pd', 'Pf']):
      npArray = np.array(aData, dtype='float_')
      npData.append( npArray )
      #self._logger.verbose('%s performances are:\n%r', label, npArray)
    # Retrieve reference and benchmark arrays
    if self.reference is ReferenceBenchmark.Pf:
      refVec = npData[2]
      benchmark = (cmpType) * npData[1]
      # FIXME benchmark = cmpType * npData[0]
    elif self.reference is ReferenceBenchmark.Pd:
      refVec = npData[1] 
      benchmark = (-1. * cmpType) * npData[2]
      # FIXME benchmark = cmpType * npData[0]
    elif self.reference in (ReferenceBenchmark.SP, ReferenceBenchmark.MSE):
      benchmark = (cmpType) * npData[0]
    else:
      raise ValueError("Unknown reference %d" % self.reference)
    # Retrieve the allowed indexes from benchmark which are not outliers
    if self.removeOLs:
      q1=percentile(benchmark,25.0)
      q3=percentile(benchmark,75.0)
      outlier_higher = q3 + 1.5*(q3-q1)
      outlier_lower  = q1 + 1.5*(q1-q3)
      allowedIdxs = np.all([benchmark > q3, benchmark < q1], axis=0).nonzero()[0]
    lRefVal = self.getReference( ds = ds, sort = sortIdx )
    #import pdb; pdb.set_trace()
    # Finally, return the index:
    if self.reference in (ReferenceBenchmark.SP, ReferenceBenchmark.MSE): 
      if self.removeOLs:
        idx = np.argmax( cmpType * benchmark[allowedIdxs] )
        return allowedIdx[ idx ]
      else:
        return np.argmax( benchmark )
    else:
      if self.removeOLs:
        refAllowedIdxs = ( np.abs( refVec[allowedIdxs] - lRefVal ) < eps ).nonzero()[0]
        if not refAllowedIdxs.size:
          if not self.allowLargeDeltas:
            # We don't have any candidate, raise:
            raise RuntimeError("eps is too low, no indexes passed constraint! Reference is %r | RefVec is: \n%r" %
                (lRefVal, refVec))
          else:
            # We can search for the closest candidate available:
            return allowedIdxs[ np.argmin( np.abs(refVec[allowedIdxs] - lRefVal ) ) ]
        # Otherwise we return best benchmark for the allowed indexes:
        return refAllowedIdxs[ np.argmax( ( benchmark[allowedIdxs] )[ refAllowedIdxs ] ) ]
      else:
        refAllowedIdxs = ( np.abs( refVec - lRefVal) < eps ).nonzero()[0]
        if not refAllowedIdxs.size:
          if not self.allowLargeDeltas:
            # We don't have any candidate, raise:
            raise RuntimeError("eps is too low, no indexes passed constraint! Reference is %r | RefVec is: \n%r" %
                (lRefVal, refVec))
          else:
            # FIXME We need to protect it from choosing 0% and 100% references.
            distances = np.abs( refVec - lRefVal )
            minDistanceIdx = np.argmin( distances )
            # We can search for the closest candidate available:
            self._logger.warning("No indexes passed eps constraint (%r%%) for reference value (%s:%r) where refVec is: \n%r",
                                 eps*100., ReferenceBenchmark.tostring(self.reference), lRefVal, refVec)
            # This is the new minimal distance:
            lRefVal = refVec[minDistanceIdx]
            # and the other indexes which correspond to this value
            refAllowedIdxs = ( np.abs(refVec - lRefVal) == 0. ).nonzero()[0]
            self._logger.verbose("Found %d points with minimum available distance of %r%% to original. They are: %r", 
                              len(refAllowedIdxs), distances[minDistanceIdx]*100., refAllowedIdxs )
        else:
          if len(refAllowedIdxs) != len(refVec):
            self._logger.info("Found %d points within %r%% distance from benchmark.", 
                              len(refAllowedIdxs), eps*100. )
        # Otherwise we return best benchmark for the allowed indexes:
        return refAllowedIdxs[ np.argmax( benchmark[ refAllowedIdxs ] ) ]
  # end of getOutermostPerf

  def getEps(self, eps = NotSet ):
    """
      Retrieve eps value replacing it to a custom value if input parameter is
      not NotSet
    """
    return self._def_eps if eps is NotSet else eps

  def __str__(self):
    str_ =  self.name + '(' + ReferenceBenchmark.tostring(self.reference) 
    if self.refVal is not None: str_ += ':' + str(self.refVal)
    str_ += ')'
    return str_

ReferenceBenchmarkCollection = LimitedTypeList('ReferenceBenchmarkCollection',(),
                                               {'_acceptedTypes':(ReferenceBenchmark,type(None),)})
ReferenceBenchmarkCollection._acceptedTypes = ReferenceBenchmarkCollection._acceptedTypes + (ReferenceBenchmarkCollection,)

def fixLoopingBoundsCol( var, 
    wantedType = LoopingBounds,
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

def fixPPCol( var, nSorts = 1, nEta = 1, nEt = 1, level = None ):
  """
    Helper method to correct variable to be a looping bound collection
    correctly represented by a LoopingBoundsCollection instance.
  """
  tree_types = (PreProcCollection, PreProcChain, list, tuple )
  try: 
    # Retrieve collection maximum depth
    _, _, _, _, depth = traverse(var, tree_types = tree_types).next()
  except GeneratorExit:
    depth = 0
  if depth < 5:
    if depth == 0:
      var = [[[[var]]]]
    elif depth == 1:
      var = [[[var]]]
    elif depth == 2:
      var = [[var]]
    elif depth == 3:
      var = [var]
    # We also want to be sure that they are in correct type and correct size:
    from RingerCore import inspect_list_attrs
    var = inspect_list_attrs(var, 3, PreProcChain,      tree_types = tree_types,                                level = level   )
    var = inspect_list_attrs(var, 2, PreProcCollection, tree_types = tree_types, dim = nSorts, name = "nSorts",                 )
    var = inspect_list_attrs(var, 1, PreProcCollection, tree_types = tree_types, dim = nEta,   name = "nEta",                   )
    var = inspect_list_attrs(var, 0, PreProcCollection, tree_types = tree_types, dim = nEt,    name = "nEt",    deepcopy = True )
  else:
    raise ValueError("Pre-processing dimensions size is larger than 5.")

  return var

class BatchSizeMethod( EnumStringification ):
  _ignoreCase = True
  Manual = 0
  MinClassSize = 1
  OneSample = 2
  HalfSizeSignalClass = 3

class TuningJob(Logger):
  """
    This class is used to create and tune a classifier through the call method.
  """

  def __init__(self, logger = None ):
    """
      Initialize the TuningJob using a log level.
    """
    Logger.__init__( self, logger = logger )
    self.compress = False

  def __call__(self, dataLocation, **kw ):
    """
      Run discriminatior tuning for input data created with CreateData.py
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
        x ppFile [None]: The file containing the pre-processing collection to apply into 
          input space and obtain the pattern space. The files can be generated
          using a CreateConfFiles instance which is accessed via command
          line using the createTuningJobFiles.py script.
        o ppCol PreProcCollection( [ PreProcCollection( [ PreProcCollection( [ PreProcChain( Norm1() ) ] ) ] ) ] ): 
          A PreProcCollection with the PreProcChain instances to be applied to
          each sort and eta/et bin.
       -------
      Optional arguments:
        - operation [NotSet]: The discriminator operation level(s). When
            NotSet, the operation level will be retrieved from the tuning data
            file. This is important only when using the MultiStop criterea,
            where all operation points will be optimized together using the
            signal and background efficiency from the operation.
        - etBins [None]: The et bins to use within this job. 
            When not specified, all bins available on the file will be tuned
            separately.
            If specified as a integer or float, it is assumed that the user
            wants to run the job only for the specified bin index.
            In case a list is specified, it is transformed into a
            MatlabLoopingBounds, read its documentation on:
              http://nbviewer.jupyter.org/github/wsfreund/RingerCore/blob/master/readme.ipynb#LoopingBounds
            for more details.
        - etaBins [None]: The eta bins to use within this job. Check etBins
          help for more information.
        - compress [True]: Whether to compress file or not.
        - level [loggingLevel.INFO]: The logging output level.
        - outputFileBase ['nn.tuned']: The tuning outputFile starting string.
            It will also contain a custom string representing the configuration
            used to tune the discriminator.
        - showEvo (TuningWrapper prop) [50]: The number of iterations wher
            performance is shown (used as a boolean on ExMachina).
        - maxFail (TuningWrapper prop) [50]: Maximum number of failures
            tolerated failing to improve performance over validation dataset.
        - epochs (TuningWrapper prop) [10000]: Number of iterations where
            the tuning algorithm can run the optimization.
        - doPerf (TuningWrapper prop) [True]: Whether we should run performance
            testing under convergence conditions, using test/validation dataset
            and also estimate operation condition.
        - maxFail (TuningWrapper prop) [50]: Number of epochs which failed to improve
            validation efficiency. When reached, the tuning process is stopped.
        - batchSize (TuningWrapper prop) [number of observations of the class
            with the less observations]: Set the batch size used during tuning.
        - algorithmName (TuningWrapper prop) [resilient back-propgation]: The
            tuning method to use.
        - batchMethod (TuningWrapper prop) [MinClassSize]: The method to choose 
            the batching size. Use one of those decribed by BatchSizeMethod
            EnumStringification.
       -------
      ExMachina props
        - networkArch (ExMachina prop) ['feedforward']: the neural network
            architeture to use.
        - costFunction (ExMachina prop) ['sp']: the cost function used by ExMachina
        - shuffle (ExMachina prop) [True]: Whether to shuffle datasets while
          training.
       -------
      FastNet props
        - seed (FastNet prop) [None]: The seed to be used by the tuning
            algorithm.
        - doMultiStop (FastNet prop) [True]: Tune classifier using P_D, P_F and
          SP when set to True. Uses only SP when set to False.
    """
    import gc
    from copy import deepcopy
    ### Retrieve configuration from input values:
    ## We start with basic information:
    self.level          = retrieve_kw(kw, 'level',          LoggingLevel.INFO                   )
    self.compress       = retrieve_kw(kw, 'compress',       True                                )
    self.operationPoint = retrieve_kw(kw, 'operationPoint', None                                )
    outputFileBase      = retrieve_kw(kw, 'outputFileBase', 'nn.tuned'                          )
    ## Now we go to parameters which need higher treating level, starting with
    ## the CrossValid object:
    # Make sure that the user didn't try to use both options:
    if 'crossValid' in kw and 'crossValidFile' in kw:
      raise ValueError("crossValid is mutually exclusive with crossValidFile, \
          either use or another terminology to specify CrossValid object.")
    crossValidFile      = retrieve_kw( kw, 'crossValidFile', None )
    from TuningTools.CrossValid import CrossValid, CrossValidArchieve
    if not crossValidFile:
      # Cross valid was not specified, read it from crossValid:
      crossValid                 = kw.pop('crossValid', \
          CrossValid( level = self.level, \
                      seed = retrieve_kw(kw, 'crossValidSeed' ) ) )
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
      neuronBoundsCol   = retrieve_kw( kw, 'neuronBoundsCol', MatlabLoopingBounds(5, 5) )
      sortBoundsCol     = retrieve_kw( kw, 'sortBoundsCol',   PythonLoopingBounds(50)   )
      initBoundsCol     = retrieve_kw( kw, 'initBoundsCol',   PythonLoopingBounds(100)  )
    else:
      # Make sure confFileList is in the correct format
      confFileList = csvStr2List( confFileList )
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
    neuronBoundsCol = fixLoopingBoundsCol( neuronBoundsCol,
                                           MatlabLoopingBounds )
    sortBoundsCol   = fixLoopingBoundsCol( sortBoundsCol,
                                           PythonLoopingBounds )
    initBoundsCol   = fixLoopingBoundsCol( initBoundsCol,
                                           PythonLoopingBounds )
    # Check if looping bounds are ok:
    for neuronBounds in neuronBoundsCol():
      if neuronBounds.lowerBound() < 1:
        raise ValueError("Neuron lower bound is not allowed, it must be at least 1.")
    for sortBounds in sortBoundsCol():
      if sortBounds.lowerBound() < 0:
        raise ValueError("Sort lower bound is not allowed, it must be at least 0.")
      if sortBounds.upperBound() >= crossValid.nSorts():
        raise ValueError(("Sort upper bound is not allowed, it is higher then the number "
            "of sorts used."))
    for initBounds in initBoundsCol():
      if initBounds.lowerBound() < 0:
        raise ValueError("Attempted to create an initialization index lower than 0.")
    nSortsVal = crossValid.nSorts()
    ## Retrieve binning information: 
    etBins  = retrieve_kw(kw, 'etBins',  None )
    etaBins = retrieve_kw(kw, 'etaBins', None )
    # Check binning information
    if type(etBins) in (int,float):
      etBins = [etBins, etBins]
    if type(etaBins) in (int,float):
      etaBins = [etaBins, etaBins]
    if etBins is not None:
      etBins = MatlabLoopingBounds(etBins)
    if etaBins is not None:
      etaBins = MatlabLoopingBounds(etaBins)
    ## Retrieve the Tuning Data Archieve
    from TuningTools.CreateData import TuningDataArchieve
    tdArchieve = TuningDataArchieve(dataLocation)
    nEtBins = tdArchieve.nEtBins()
    self._logger.debug("Total number of et bins: %d" , nEtBins if nEtBins is not None else 0)
    nEtaBins = tdArchieve.nEtaBins()
    self._logger.debug("Total number of eta bins: %d" , nEtaBins if nEtaBins is not None else 0)
    # Check if use requested bins are ok:
    if etBins is not None:
      if nEtBins is None:
        raise ValueError("Requested to run for specific et bins, but no et bins are available.")
      if etBins.lowerBound() < 0 or etBins.upperBound() >= nEtBins:
        raise ValueError("etBins (%r) bins out-of-range. Total number of et bins: %d" % (etBins.list(), nEtBins) )
      if nEtaBins is None:
        raise ValueError("Requested to run for specific eta bins, but no eta bins are available.")
      if etaBins.lowerBound() < 0 or etaBins.upperBound() >= nEtaBins:
        raise ValueError("etaBins (%r) bins out-of-range. Total number of eta bins: %d" % (etaBins.list(), nEtaBins) )

    ## Check ppCol or ppFile
    if 'ppFile' in kw and 'ppCol' in kw:
      raise ValueError(("ppFile is mutually exclusive with ppCol, "
          "either use one or another terminology to specify the job "
          "configuration."))
    ppFile    = retrieve_kw(kw, 'ppFile', None )
    if not ppFile:
      ppCol = kw.pop( 'ppCol', PreProcChain( Norm1(level = self.level) ) )
    else:
      # Now loop over ppFile and add it to our pp list:
      with PreProcArchieve(ppFile) as ppCol: pass
    # Make sure that our pre-processings are PreProcCollection instances and matches
    # the number of sorts, eta and et bins.
    ppCol = fixPPCol( ppCol,
                      nSortsVal,
                      nEtaBins,
                      nEtBins,
                      level = self.level )
    # Retrieve some useful information and keep it on memory
    nConfigs = len( neuronBoundsCol )
    ## Now create the tuning wrapper:
    from TuningTools.TuningWrapper import TuningWrapper
                                   # Wrapper confs:
    tuningWrapper = TuningWrapper( level                 = self.level
                                 , doPerf                = retrieve_kw( kw, 'doPerf',                NotSet)
                                   # All core confs:
                                 , maxFail               = retrieve_kw( kw, 'maxFail',               NotSet)
                                 , algorithmName         = retrieve_kw( kw, 'algorithmName',         NotSet)
                                 , epochs                = retrieve_kw( kw, 'epochs',                NotSet)
                                 , batchSize             = retrieve_kw( kw, 'batchSize',             NotSet)
                                 , batchMethod           = retrieve_kw( kw, 'batchMethod',           NotSet)
                                 , showEvo               = retrieve_kw( kw, 'showEvo',               NotSet)
                                 , useTstEfficiencyAsRef = retrieve_kw( kw, 'useTstEfficiencyAsRef', NotSet)
                                   # ExMachina confs:
                                 , networkArch           = retrieve_kw( kw, 'networkArch',           NotSet)
                                 , costFunction          = retrieve_kw( kw, 'costFunction',          NotSet)
                                 , shuffle               = retrieve_kw( kw, 'shuffle',               NotSet)
                                   # FastNet confs:
                                 , seed                  = retrieve_kw( kw, 'seed',                  NotSet)
                                 , doMultiStop           = retrieve_kw( kw, 'doMultiStop',           NotSet)
                                 )
    ## Finished retrieving information from kw:
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    from itertools import product
    for etBinIdx, etaBinIdx in product( range( nEtBins if nEtBins is not None else 1 ) if etBins is None \
                                   else etBins(), 
                                  range( nEtaBins if nEtaBins is not None else 1 ) if etaBins is None \
                                   else etaBins() ):
      binStr = '' 
      saveBinStr = 'no-bin'
      if nEtBins is not None or nEtaBins is not None:
        binStr = ' (etBinIdx=%d,etaBinIdx=%d) ' % (etBinIdx, etaBinIdx)
        saveBinStr = 'et%04d.eta%04d' % (etBinIdx, etaBinIdx)
      self._logger.info('Opening data%s...', binStr)
      # Load data bin
      with TuningDataArchieve(dataLocation, et_bin = etBinIdx if nEtBins is not None else None,
                                            eta_bin = etaBinIdx if nEtaBins is not None else None) as tdArchieve:
        patterns = (tdArchieve['signal_patterns'], tdArchieve['background_patterns'])
        try:
          from TuningTools.ReadData import RingerOperation
          if self.operationPoint is None:
            operation = tdArchieve['operation']
          # Make sure that operation is valid:
          RingerOperation.retrieve(operation)
          refLabel = RingerOperation.branchName(operation)
          benchmarks = (tdArchieve['signal_efficiencies'][refLabel], 
                        tdArchieve['background_efficiencies'][refLabel])
          try:
            cross_benchmarks = (tdArchieve['signal_cross_efficiencies'][refLabel], 
                                tdArchieve['background_cross_efficiencies'][refLabel])
          except KeyError:
            cross_benchmarks = None
        except KeyError as e:
          operation = None
          benchmarks = None
          cross_benchmarks = None
        if nEtBins is not None:
          etBin = tdArchieve['et_bins']
          self._logger.info('Tuning Et bin: %r', tdArchieve['et_bins'])
        if nEtaBins is not None:
          etaBin = tdArchieve['eta_bins']
          self._logger.info('Tuning eta bin: %r', tdArchieve['eta_bins'])
        # Add the signal efficiency and background efficiency as goals to the
        # tuning wrapper:
        if tuningWrapper.doMultiStop:
          opRefs = [ReferenceBenchmark.SP, ReferenceBenchmark.Pd, ReferenceBenchmark.Pf]
        else:
          opRefs = [ReferenceBenchmark.SP] # FIXME is it?
        if benchmarks is None:
          raise RuntimeError("Couldn't access the benchmarks on efficiency file and MultiStop was requested.")
        references = ReferenceBenchmarkCollection([])
        for ref in opRefs: 
          args = []
          args.extend( benchmarks )
          if cross_benchmarks is not None:
            args.extend( cross_benchmarks )
          references.append( ReferenceBenchmark( "Tuning_" + refLabel.replace('Accept','') + "_" 
                                                 + ReferenceBenchmark.tostring( ref ), 
                                                 ref, *args ) )
        tuningWrapper.setReferences( references )
      del tdArchieve
      # For the bounded variables, we loop them together for the collection:
      for confNum, neuronBounds, sortBounds, initBounds in \
          zip(range(nConfigs), neuronBoundsCol, sortBoundsCol, initBoundsCol ):
        self._logger.info('Running configuration file number %d%s', confNum, binStr)
        tunedDiscr = []
        tuningInfo = []
        nSorts = len(sortBounds)
        # Finally loop within the configuration bounds
        for sort in sortBounds():
          self._logger.info('Extracting cross validation sort %d%s', sort, binStr)
          trnData, valData, tstData = crossValid( patterns, sort )
          del patterns # Keep only one data representation
          # Take ppChain parameters on training data:
          ppChain = ppCol[etBinIdx][etaBinIdx][sort]
          self._logger.info('Tuning pre-processing chain (%s)...', ppChain)
          ppChain.takeParams( trnData )
          self._logger.debug('Done tuning pre-processing chain!')
          self._logger.info('Applying pre-processing chain to all sets...')
          # Apply ppChain:
          self._logger.debug('Applying pp chain to train dataset...')
          trnData = ppChain( trnData )
          self._logger.debug('Applying pp chain to validation dataset...')
          valData = ppChain( valData ) 
          self._logger.debug('Applying pp chain to test dataset...')
          tstData = ppChain( tstData )
          self._logger.debug('Done applying the pre-processing chain to all sets!')
          # Retrieve resulting data shape
          nInputs = trnData[0].shape[npCurrent.pdim]
          # Update tuningtool working data information:
          tuningWrapper.setTrainData( trnData ); del trnData
          tuningWrapper.setValData  ( valData ); del valData
          if len(tstData) > 0:
            tuningWrapper.setTestData( tstData ); del tstData
          else:
            self._logger.debug('Using validation dataset as test dataset.')
          tuningWrapper.setSortIdx(sort)
          # Garbage collect now, before entering training stage:
          gc.collect()
          # And loop over neuron configurations and initializations:
          for neuron in neuronBounds():
            for init in initBounds():
              self._logger.info('Training <Neuron = %d, sort = %d, init = %d>%s...', \
                  neuron, sort, init, binStr)
              tuningWrapper.newff([nInputs, neuron, 1])
              cTunedDiscr, cTuningInfo = tuningWrapper.train_c()
              self._logger.debug('Finished C++ tuning, appending tuned discriminators to tuning record...')
              # Append retrieved tuned discriminators and its tuning information
              tunedDiscr.append( cTunedDiscr )
              tuningInfo.append( cTuningInfo )
            self._logger.debug('Finished all initializations for neuron %d...', neuron)
          self._logger.debug('Finished all neurons for sort %d...', sort)
          # Finished all inits for this sort, we need to undo the crossValid if
          # we are going to do a new sort, otherwise we continue
          if not ( (confNum+1) == nConfigs and sort == sortBounds.endBound()):
            if ppChain.isRevertible():
              trnData = tuningWrapper.trnData(release = True)
              valData = tuningWrapper.valData(release = True)
              tstData = tuningWrapper.testData(release = True)
              patterns = crossValid.revert( trnData, valData, tstData, sort = sort )
              del trnData, valData, tstData
              patterns = ppChain( patterns , revert = True )
            else:
              # We cannot revert ppChain, reload data:
              self._logger.info('Re-opening raw data...')
              with TuningDataArchieve(dataLocation, et_bin = etBinIdx if nEtBins is not None else None,
                                                    eta_bin = etaBinIdx if nEtaBins is not None else None) as tdArchieve:
                patterns = (tdArchieve['signal_patterns'], tdArchieve['background_patterns'])
              del tdArchieve
          self._logger.debug('Finished all hidden layer neurons for sort %d...', sort)
        self._logger.debug('Finished all sorts for configuration %d in collection...', confNum)
        ## Finished retrieving all tuned discriminators for this config file for
        ## this pre-processing. Now we head to save what we've done so far:
        # This pre-processing was tuned during this tuning configuration:
        tunedPP = PreProcCollection( [ ppCol[etBinIdx][etaBinIdx][sort] for sort in sortBounds() ] )
        # Define output file name:
        fulloutput = '{outputFileBase}.{ppStr}.{neuronStr}.{sortStr}.{initStr}.{saveBinStr}.pic'.format( 
                      outputFileBase = outputFileBase, 
                      ppStr = 'pp-' + ppChain.shortName()[:12], # Truncate on 12th char
                      neuronStr = neuronBounds.formattedString('hn'), 
                      sortStr = sortBounds.formattedString('s'),
                      initStr = initBounds.formattedString('i'),
                      saveBinStr = saveBinStr )

        self._logger.info('Saving file named %s...', fulloutput)
        extraKw = {}
        if nEtBins is not None:
          extraKw['etBinIdx'] = etBinIdx
          extraKw['etBin'] = etBin
        if nEtaBins is not None:
          extraKw['etaBinIdx'] = etaBinIdx
          extraKw['etaBin'] = etaBin
        savedFile = TunedDiscrArchieve( neuronBounds = neuronBounds, 
                                        sortBounds = sortBounds, 
                                        initBounds = initBounds,
                                        tunedDiscr = tunedDiscr,
                                        tuningInfo = tuningInfo,
                                        tunedPP = tunedPP,
                                        **extraKw
                                      ).save( fulloutput, self.compress )
        self._logger.info('File "%s" saved!', savedFile)
      # Finished all configurations we had to do
      self._logger.info('Finished tuning job!')

  # end of __call__ member fcn

class TunedDiscrArchieveCol( Logger ):
  """
    The TunedDiscrArchieveCol holds a collection of TunedDiscrArchieve. It is
    used by the file merger method to merge the TunedDiscrArchieve files into a
    unique file.

    Deprecated: Decided not to work with this solution, as this would be
    extremely slow. However code is kept for future reference.
  """

  # Use class factory
  __metaclass__ = LimitedTypeStreamableList
  _streamerObj  = LoggerLimitedTypeListRDS
  #_cnvObj       = LimitedTypeListRDC( level = LoggingLevel.VERBOSE )

  # These are the list (LimitedTypeList) accepted objects:
  _acceptedTypes = (TunedDiscrArchieve, str)

  def __init__( self, *args, **kw ):
    Logger.__init__(self, kw)
    from RingerCore.LimitedTypeList import _LimitedTypeList____init__
    _LimitedTypeList____init__(self, *args)

  def toRawObj(self):
    from RingerCore.RawDictStreamable import _RawDictStreamable__toRawObj
    rawDict = _RawDictStreamable__toRawObj(self)
    # Expand items to be files:
    for idx, item in enumerate(rawDict['items']):
      # FIXME If item is a string, expand it to have the correct format
      rawDict['file_' + str(idx)] = item
    rawDict.pop('items')
    return rawDict

  def save(self, filePath):
    """
    Save the TunedDiscrArchieveCol object to disk.
    """
    return save( self.toRawObj(), filePath, protocol = 'savez_compressed' )

  @classmethod
  def load( cls, filePath ):
    """
    Load a TunedDiscrArchieveCol object from disk and return it.
    """
    rawObj = load( filePath, useHighLevelObj = False )
    # TODO Work with the numpy file
    #return cls.fromRawObj( rawObj )
    return rawObj
