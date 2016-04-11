__all__ = ['TunedDiscrArchieve', 'ReferenceBenchmark', 'TuningJob',
           'fixPPCol', 'fixLoopingBoundsCol',]
import os

from RingerCore               import Logger, LoggingLevel, save, load, EnumStringification, \
                                     checkForUnusedVars, NotSet, fixFileList, retrieve_kw
from RingerCore.LoopingBounds import *
from TuningTools.Neural       import Neural
from TuningTools.PreProc      import *

class TunedDiscrArchieve( Logger ):
  """
  Context manager for Tuned Discriminators archives
  """

  _type = 'tunedFile'
  _version = 4

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with TunedDiscrArchieve("/path/to/file") as data:
      BLOCK

    TunedDiscrArchieve( "file/path", neuronBounds = ...,
                                     sortBounds = ...,
                                     initBounds = ...
                                     tunedDiscr = ...,
                                     etaBin = ...,
                                     etBin = ...)
    """
    Logger.__init__(self, kw)
    self._filePath = filePath
    self._neuronBounds = kw.pop('neuronBounds', None                )
    self._sortBounds   = kw.pop('sortBounds',   None                )
    self._initBounds   = kw.pop('initBounds',   None                )
    self._tunedDiscr   = kw.pop('tunedDiscr',   None                )
    self._tunedPP      = kw.pop('tunedPP',      None                )
    self._etaBinIdx    = kw.pop('etaBinIdx',    -1                  )
    self._etBinIdx     = kw.pop('etBinIdx',     -1                  )
    self._etaBin       = kw.pop('etaBin',       npCurrent.array([]) )
    self._etBin        = kw.pop('etBin',        npCurrent.array([]) )
    self._nList = None; self._nListLen = None
    self._sList = None; self._sListLen = None
    self._iList = None; self._iListLen = None

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

  def getData( self ):
    if not self._neuronBounds or \
       not self._sortBounds   or \
       not self._initBounds   or \
       not self._tunedDiscr   or \
       not self._tunedPP:
      raise RuntimeError("Attempted to retrieve empty data from TunedDiscrArchieve.")
    return { 'version': self._version,
                'type': self._type,
        'neuronBounds': transformToMatlabBounds( self._neuronBounds ).getOriginalVec(),
          'sortBounds': transformToPythonBounds( self._sortBounds ).getOriginalVec(),
          'initBounds': transformToPythonBounds( self._initBounds ).getOriginalVec(),
 'tunedDiscriminators': self._tunedDiscr,
   'tunedPPCollection': list(self._tunedPP),
           'etaBinIdx': self._etaBinIdx,
            'etBinIdx': self._etBinIdx,
              'etaBin': self._etaBin,
               'etBin': self._etBin,}
  # getData

  def save(self, compress = True):
    return save( self.getData(), self._filePath, compress = compress )

  def __enter__(self):
    # Open file:
    from cPickle import PickleError
    try:
      tunedData = load(self._filePath)
    except (PickleError, TypeError, ImportError): # TypeError due to add object inheritance on Logger
      # It failed without renaming the module, retry renaming old module
      # structure to new one.
      import TuningTools.Neural
      cNeural = TuningTools.Neural.Neural
      cLayer = TuningTools.Neural.Layer
      TuningTools.Neural.Layer = TuningTools.Neural.OldLayer
      TuningTools.Neural.Neural = TuningTools.Neural.OldNeural
      import sys
      import RingerCore.util
      sys.modules['FastNetTool.util'] = RingerCore.util
      sys.modules['FastNetTool.Neural'] = TuningTools.Neural
      tunedData = load(self._filePath)
      TuningTools.Neural.Layer = cLayer
      TuningTools.Neural.Neural = cNeural
    try:
      if type(tunedData) is dict:
        if tunedData['type'] != self._type:
          raise RuntimeError(("Input tunedData file is not from tunedData " 
              "type."))
        self.readVersion = tunedData['version']
        # Read configuration file to retrieve pre-processing, 
        if tunedData['version'] == 4:
          self._neuronBounds = MatlabLoopingBounds( tunedData['neuronBounds'] )
          self._sortBounds   = PythonLoopingBounds( tunedData['sortBounds']   )
          self._initBounds   = PythonLoopingBounds( tunedData['initBounds']   )
          self._tunedDiscr   = tunedData['tunedDiscriminators']
          self._tunedPP      = PreProcCollection( tunedData['tunedPPCollection'] )
          self._etaBinIdx    = tunedData['etaBinIdx']
          self._etBinIdx     = tunedData['etBinIdx']
          self._etaBin       = tunedData['etaBin']
          self._etBin        = tunedData['etBin']
        elif tunedData['version'] == 3:
          self._neuronBounds = MatlabLoopingBounds( tunedData['neuronBounds'] )
          self._sortBounds   = PythonLoopingBounds( tunedData['sortBounds']   )
          self._initBounds   = PythonLoopingBounds( tunedData['initBounds']   )
          self._tunedDiscr   = tunedData['tunedDiscriminators']
          self._tunedPP      = PreProcCollection( tunedData['tunedPPCollection'] )
          self._etaBinIdx    = tunedData['etaBin']
          self._etBinIdx     = tunedData['etBin']
          self._etaBin       = npCurrent.array([0.,0.8,1.37,1.54,2.5])
          self._etaBin       = self._etaBin[self._etaBinIdx:self._etaBinIdx+2]
          self._etBin        = npCurrent.array([0,30.,40.,50.,20000.])*1e3
          self._etBin        = self._etBin[self._etBinIdx:self._etBinIdx+2]
        elif tunedData['version'] == 2:
          self._neuronBounds = MatlabLoopingBounds( tunedData['neuronBounds'] )
          self._sortBounds   = PythonLoopingBounds( tunedData['sortBounds']   )
          self._initBounds   = PythonLoopingBounds( tunedData['initBounds']   )
          self._tunedDiscr   = tunedData['tunedDiscriminators']
          self._tunedPP      = PreProcCollection( tunedData['tunedPPCollection'] )
        elif tunedData['version'] == 1:
          self._neuronBounds = MatlabLoopingBounds( tunedData['neuronBounds'] )
          self._sortBounds   = PythonLoopingBounds( tunedData['sortBounds']   )
          self._initBounds   = PythonLoopingBounds( tunedData['initBounds']   )
          self._tunedDiscr   = tunedData['tunedDiscriminators']
          self._tunedPP      = PreProcCollection( [ PreProcChain( Norm1() ) for i in range(len(self._sortBounds)) ] )
        else:
          raise RuntimeError("Unknown job configuration version")
      elif type(tunedData) is list: # zero version file (without versioning 
        # control):
        # Old version was saved as follows:
        #objSave = [neuron, sort, initBounds, train]
        self.readVersion = 0
        self._neuronBounds = MatlabLoopingBounds( [tunedData[0], tunedData[0]] )
        self._sortBounds   = MatlabLoopingBounds( [tunedData[1], tunedData[1]] )
        self._initBounds   = MatlabLoopingBounds( tunedData[2] )
        self._tunedDiscr   = tunedData[3]
        self._tunedPP      = PreProcCollection( [ PreProcChain( Norm1() ) for i in range(len(self._sortBounds)) ] )
      else:
        raise RuntimeError("Unknown file type entered for config file.")
    except RuntimeError, e:
      raise RuntimeError(("Couldn't read configuration file '%s': Reason:"
          "\n\t %s" % (self._filePath, e)))
    return self
  # __enter__

  def getTunedInfo( self, neuron, sort, init ):
    if not self._nList:
      self._nList = self._neuronBounds.list(); self._nListLen = len( self._nList )
      self._sList = self._sortBounds.list();   self._sListLen = len( self._sList )
      self._iList = self._initBounds.list();   self._iListLen = len( self._iList )
    try:
      # On version 0 and 1 we first loop on sort list, then on neuron bound, to
      # finally loop over the initializations:

      return self._tunedDiscr[
               self._sList.index( sort ) * ( self._nListLen * self._iListLen ) + \
               self._nList.index( neuron ) * ( self._iListLen ) + \
               self._iList.index( init ) ], self._tunedPP[self._sList.index( sort )]
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
    # FIXME Index [0] is the discriminator, [1] is the normalization. This should be more organized.
    tunedDiscr = self.getTunedInfo(neuron, sort, init) \
                                  [ReferenceBenchmark.fromstring(rawBenchmark['reference'])] \
                                  [0]
    from FilterEvents import RingerOperation
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
    # if ringerOperation
  # exportDiscr
        
  def __exit__(self, exc_type, exc_value, traceback):
    # Remove bounds
    self.neuronBounds = None 
    self.sortBounds = None 
    self.initBounds = None 
    self.tunedDiscr = None 
    self._nList = None; self._nListLen = None
    self._sList = None; self._sListLen = None
    self._iList = None; self._iListLen = None
  # __exit__

class ReferenceBenchmark(EnumStringification):
  """
  Reference benchmark to set discriminator operation point.

    - SP: Use the SUM-PRODUCT coeficient as an optimization target. 
    - Pd: Aim to operate with signal detection probability as close as
      possible from reference value meanwhile minimazing the false
      alarm probability.
    - Pf: Aim to operate with false alarm probability as close as
      possible from reference value meanwhile maximazing the detection
      probability.
  """
  SP = 0
  Pd = 1
  Pf = 2

  def __init__(self, name, reference, signal_efficiency, background_efficiency,
                                      signal_cross_efficiency = None,
                                      background_cross_efficiency = None, **kw):
    """
    ref = ReferenceBenchmark(name, reference, signal_efficiency, background_efficiency, 
                                   signal_cross_efficiency, background_cross_efficiency,
                                   [, removeOLs = False])

      * name: The name for this reference benchmark;
      * reference: The reference benchmark type. It must one of
          ReferenceBenchmark enumerations.
      * signal_efficiency: The reference benchmark signal efficiency.
      * background_efficiency: The reference benchmark background efficiency.
      * signal_cross_efficiency: The reference benchmark signal efficiency measured with the Cross-Validation sets.
      * background_cross_efficiency: The reference benchmark background efficiency with the Cross-Validation sets.
      * removeOLs [False]: Whether to remove outliers from operation.
      * allowLargeDeltas [True]: When set to true and no value is within the operation bounds,
       then it will use operation closer to the reference.
    """
    self.signal_efficiency = signal_efficiency
    self.signal_cross_efficiency = signal_cross_efficiency
    self.background_efficiency = background_efficiency
    self.background_cross_efficiency = background_cross_efficiency
    self.removeOLs = kw.pop('removeOLs', False)
    self.allowLargeDeltas = kw.pop('allowLargeDeltas', True)
    if not (type(name) is str):
      raise TypeError("Name must be a string.")
    self.name = name
    self.reference = ReferenceBenchmark.retrieve(reference)
    self.refVal = None
    if self.reference is ReferenceBenchmark.Pd:
      self.refVal = self.signal_efficiency.efficiency()/100.
    elif self.reference == ReferenceBenchmark.Pf:
      self.refVal = self.background_efficiency.efficiency()/100.
  # __init__

  def rawInfo(self):
         """
         Return raw benchmark information
         """
         return { 'reference': ReferenceBenchmark.tostring(self.reference),
                     'refVal': (self.refVal if not self.refVal is None else -999),
          'signal_efficiency': self.signal_efficiency.toRawObj(),
    'signal_cross_efficiency': self.signal_cross_efficiency.toRawObj(noChildren=True) if self.signal_cross_efficiency is not None else '',
      'background_efficiency': self.background_efficiency.toRawObj(),
'background_cross_efficiency': self.background_cross_efficiency.toRawObj(noChildren=True) if self.background_cross_efficiency is not None else '',
                  'removeOLs': self.removeOLs }

  def getOutermostPerf(self, data, **kw):
    """
    Get outermost performance for the tuned discriminator performances on data. 
    idx = refBMark.getOutermostPerf( data [, eps = .001 ][, cmpType = 1])

     * data: A list with following struction:
        data[0] : SP
        data[1] : Pd
        data[2] : Pf

     * eps [.001] is used for softening. The larger it is, more candidates will
      be possible to be considered, but farther the returned operation may be from
      the reference. The default is 0.1% deviation from the reference value.
     * cmpType [+1.] is used to change the comparison type. Use +1 for best
      performance, and -1 for worst performance.
    """
    # Retrieve optional arguments
    eps = kw.pop('eps', 0.001 )
    cmpType = kw.pop('cmpType', 1.)
    # We will transform data into np array, as it will be easier to work with
    npData = []
    for aData in data:
      npData.append( np.array(aData, dtype='float_') )
    # Retrieve reference and benchmark arrays
    if self.reference is ReferenceBenchmark.Pf:
      refVec = npData[2]
      benchmark = (cmpType) * npData[1]
    elif self.reference is ReferenceBenchmark.Pd:
      refVec = npData[1] 
      benchmark = (-1. * cmpType) * npData[2]
    elif self.reference is ReferenceBenchmark.SP:
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
    # Finally, return the index:
    if self.reference is ReferenceBenchmark.SP: 
      if self.removeOLs:
        idx = np.argmax( cmpType * benchmark[allowedIdxs] )
        return allowedIdx[ idx ]
      else:
        return np.argmax( cmpType * benchmark )
    else:
      if self.removeOLs:
        refAllowedIdxs = ( np.abs( refVec[allowedIdxs] - self.refVal ) < eps ).nonzero()[0]
        if not refAllowedIdxs.size:
          if not self.allowLargeDeltas:
            # We don't have any candidate, raise:
            raise RuntimeError("eps is too low, no indexes passed constraint! Reference is %r | RefVec is: \n%r" %
                (self.refVal, refVec))
          else:
            # We can search for the closest candidate available:
            return allowedIdxs[ np.argmin( np.abs(refVec[allowedIdxs] - self.refVal ) ) ]
        # Otherwise we return best benchmark for the allowed indexes:
        return refAllowedIdxs[ np.argmax( ( benchmark[allowedIdxs] )[ refAllowedIdxs ] ) ]
      else:
        refAllowedIdxs = ( np.abs( refVec - self.refVal) < eps ).nonzero()[0]
        if not refAllowedIdxs.size:
          if not self.allowLargeDeltas:
            # We don't have any candidate, raise:
            raise RuntimeError("eps is too low, no indexes passed constraint! Reference is %r | RefVec is: \n%r" %
                (self.refVal, refVec))
          else:
            # We can search for the closest candidate available:
            return np.argmin( np.abs(refVec - self.refVal) )
        # Otherwise we return best benchmark for the allowed indexes:
        return refAllowedIdxs[ np.argmax( benchmark[ refAllowedIdxs ] ) ]

  def __str__(self):
    str_ =  self.name + '(' + ReferenceBenchmark.tostring(self.reference) 
    if self.refVal: str_ += ':' + str(self.refVal)
    str_ += ')'
    return str_

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

def fixPPCol( var, nSorts = 1, nEta = 1, nEt = 1 ):
  """
    Helper method to correct variable to be a looping bound collection
    correctly represented by a LoopingBoundsCollection instance.
  """
  try: 
    for _, _, _, _, level in traverse(var,tree_types = (PreProcCollection, PreProcChain, list, tuple )): pass
  except TypeError:
    level = 0
  # Make sure we have a structure of type PreProcCollection( PreProcChain( pp1, pp2, pp3 ) )
  if level < 3:
    var = fixLoopingBoundsCol( var, 
                               PreProcChain,
                               PreProcCollection )
    # Span collection:
    if len(var) == 1:
      var = PreProcCollection( var * nSorts )
    var = PreProcCollection( [var] * nEta )
    var = PreProcCollection( [var] * nEt  )
  elif level < 5:
    if level == 3:
      var = [var]
    # We still need to make sure that it is a pre-processing collection of
    # pre-processing chains:
    for obj, idx, parent, depth_dist, level in traverse(var, 
                                                        tree_types = (PreProcCollection, PreProcChain, list, tuple ), 
                                                        max_depth = 3,
                                                       ):
      parent[idx] = PreProcChain(obj)
    # We also want to be sure that 
    try:
      for obj, idx, parent, depth_dist, level in traverse(var, 
                                                          tree_types = (PreProcCollection, PreProcChain, list, tuple ), 
                                                          max_depth = 2,
                                                         ):
        parent[idx] = PreProcCollection(obj)
        if len(parent[idx]) == 1:
          parent[idx] = parent[idx] * nSorts
    except TypeError:
      var = PreProcCollection( PreProcCollection( PreProcCollection(var) ) * nSorts )
    try:
      for obj, idx, parent, depth_dist, level in traverse(var, 
                                                          tree_types = (PreProcCollection, PreProcChain, list, tuple ), 
                                                          max_depth = 1,
                                                         ):
        parent[idx] = PreProcCollection(obj)
        if len(parent[idx]) == 1:
          parent[idx] = parent[idx] * nEat
    except TypeError:
      var = PreProcCollection( PreProcCollection( PreProcCollection(var) ) * nEta )
    # Make sure that var itself is a PreProcCollection (not a list or tuple):
    var = PreProcCollection( var )
    # And that its size spans over eta:
    if len(var) == 1:
      var = var * nEt
  else:
    raise ValueError("Pre-processing dimension is larger than 4.")

  if len(var) != nEt:
    raise ValueError("Pre-processing does not match with number of et-bins.")
  for obj in traverse(var, 
                      tree_types = (PreProcCollection,), 
                      max_depth = 1,
                      simple_ret = True,
                     ):
    if len(obj) != nEta:
      raise ValueError("Pre-processing does not match with number of eta-bins.")
  for obj in traverse(var, 
                      tree_types = (PreProcCollection,), 
                      max_depth = 2,
                      simple_ret = True,
                     ):
    if len(obj) != nSorts:
      raise ValueError("Pre-processing does not match with number of sorts.")
  return var

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
        x ppFile [None]: The file containing the pre-processing collection to apply into 
          input space and obtain the pattern space. The files can be generated
          using a CreateConfFiles instance which is accessed via command
          line using the createTuningJobFiles.py script.
        o ppCol PreProcCollection( [ PreProcCollection( [ PreProcCollection( [ PreProcChain( Norm1() ) ] ) ] ) ] ): 
          A PreProcCollection with the PreProcChain instances to be applied to
          each sort and eta/et bin.
       -------
      Optional arguments:
        - operationLevel [None]: The discriminator operation level. When set to
            None, the operation level will be retrieved from the tuning data
            file. For now, this is only used to set the default operation targets
            on Loose and Tight tunings.
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
        - tuneOperationTargets [['Loose', 'Pd' , #looseBenchmarkRef],
                                ['Medium', 'SP'],
                                ['Tight', 'Pf' , #tightBenchmarkRef]]
            The tune operation targets which should be used for this tuning
            job. The strings inputs must be part of the ReferenceBenchmark
            enumeration.
            Instead of an enumeration string (or the enumeration itself),
            you can set it directly to a value, e.g.: 
              [['Loose97', 'Pd', .97,],['Tight005','Pf',.005]]
            This can also be set using a string, e.g.:
              [['Loose97','Pd' : '.97'],['Tight005','Pf','.005']]
            , which may contain a percentage symbol:
              [['Loose97','Pd' : '97%'],['Tight005','Pf','0.5%']]
            When set to None, the Pd and Pf will be set to the value of the
            benchmark correspondent to the operation level set.
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
        - networkArch (ExMachina prop) ['feedforward']: the neural network
            architeture to use.
        - costFunction (ExMachina prop) ['sp']: the cost function used by ExMachina
        - shuffle (ExMachina prop) [True]: Whether to shuffle datasets while
          training.
        - seed (FastNet prop) [None]: The seed to be used by the tuning
            algorithm.
        - doMultiStop (FastNet prop) [True]: Tune classifier using P_D, P_F and
          SP when set to True. Uses only SP when set to False.
    """
    import gc
    from copy import deepcopy
    ### Retrieve configuration from input values:
    ## We start with basic information:
    self.level     = retrieve_kw(kw, 'level',          LoggingLevel.INFO )
    self.compress  = retrieve_kw(kw, 'compress',       True              )
    outputFileBase = retrieve_kw(kw, 'outputFileBase', 'nn.tuned'        )
    ## Now we go to parameters which need higher treating level, starting with
    ## the CrossValid object:
    # Make sure that the user didn't try to use both options:
    if 'crossValid' in kw and 'crossValidFile' in kw:
      raise ValueError("crossValid is mutually exclusive with crossValidFile, \
          either use or another terminology to specify CrossValid object.")
    crossValidFile               = retrieve_kw( kw, 'crossValidFile', None )
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
      neuronBoundsCol   = retrieve_kw( kw, 'neuronBoundsCol', MatlabLoopingBounds(5, 5) )
      sortBoundsCol     = retrieve_kw( kw, 'sortBoundsCol',   PythonLoopingBounds(50)   )
      initBoundsCol     = retrieve_kw( kw, 'initBoundsCol',   PythonLoopingBounds(100)  )
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
    TDArchieve = TuningDataArchieve(dataLocation)
    nEtBins = TDArchieve.nEtBins()
    self._logger.debug("Total number of et bins: %d" , nEtBins if nEtBins is not None else 0)
    nEtaBins = TDArchieve.nEtaBins()
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
                      nEtBins )
    # Retrieve some useful information and keep it on memory
    nConfigs = len( neuronBoundsCol )
    ## Now create the tuning wrapper:
    from TuningTools.TuningWrapper import TuningWrapper
    tunningWrapper = TuningWrapper( #Wrapper confs:
                                    level         = self.level,
                                    doPerf        = retrieve_kw( kw, 'doPerf',        NotSet),
                                    # All core confs:
                                    maxFail       = retrieve_kw( kw, 'maxFail',       NotSet),
                                    algorithmName = retrieve_kw( kw, 'algorithmName', NotSet),
                                    epochs        = retrieve_kw( kw, 'epochs',        NotSet),
                                    batchSize     = retrieve_kw( kw, 'batchSize',     NotSet),
                                    showEvo       = retrieve_kw( kw, 'showEvo',       NotSet),
                                    # ExMachina confs:
                                    networkArch   = retrieve_kw( kw, 'networkArch',   NotSet),
                                    costFunction  = retrieve_kw( kw, 'costFunction',  NotSet),
                                    shuffle       = retrieve_kw( kw, 'shuffle',       NotSet),
                                    # FastNet confs:
                                    seed          = retrieve_kw( kw, 'seed',          NotSet),
                                    doMultiStop   = retrieve_kw( kw, 'doMultiStop',   NotSet),
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
                                            eta_bin = etaBinIdx if nEtaBins is not None else None) as TDArchieve:
        patterns = (TDArchieve['signal_rings'], TDArchieve['background_rings'])
        try:
          benchmarks = (TDArchieve['signal_efficiencies'], TDArchieve['background_efficiencies'])
          try:
            cross_benchmarks = (TDArchieve['signal_cross_efficiencies'], TDArchieve['background_cross_efficiencies'])
          except KeyError:
            cross_benchmarks = None
        except KeyError:
          benchmarks = None
          cross_benchmarks = None
        if nEtBins is not None:
          etBin = TDArchieve['et_bins']
          self._logger.info('Tuning Et bin: %r', TDArchieve['et_bins'])
        if nEtaBins is not None:
          etaBin = TDArchieve['eta_bins']
          self._logger.info('Tuning eta bin: %r', TDArchieve['eta_bins'])
      del TDArchieve
      # For the bounded variables, we loop them together for the collection:
      for confNum, neuronBounds, sortBounds, initBounds in \
          zip(range(nConfigs), neuronBoundsCol, sortBoundsCol, initBoundsCol ):
        self._logger.info('Running configuration file number %d%s', confNum, binStr)
        tunedDiscr = []
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
          self._logger.info('Applying pre-processing chain...')
          # Apply ppChain:
          trnData = ppChain( trnData )
          valData = ppChain( valData ) 
          tstData = ppChain( tstData )
          self._logger.debug('Done applying the pre-processing chain!')
          # Retrieve resulting data shape
          nInputs = trnData[0].shape[npCurrent.pdim]
          # Hold the training records
          sgnSize = trnData[0].shape[npCurrent.odim]
          bkgSize = trnData[1].shape[npCurrent.odim]
          batchSize = bkgSize if sgnSize > bkgSize else sgnSize
          # Update tuningtool working data information:
          tunningWrapper.batchSize = batchSize
          tunningWrapper.setTrainData( trnData ); del trnData
          tunningWrapper.setValData  ( valData ); del valData
          if len(tstData) > 0:
            tunningWrapper.setTestData( tstData ); del tstData
          else:
            self._logger.debug('Using validation dataset as test dataset.')
          # Garbage collect now, before entering training stage:
          gc.collect()
          # And loop over neuron configurations and initializations:
          for neuron in neuronBounds():
            for init in initBounds():
              self._logger.info('Training <Neuron = %d, sort = %d, init = %d>%s...', \
                  neuron, sort, init, binStr)
              tunningWrapper.newff([nInputs, neuron, 1])
              cTunedDiscr = tunningWrapper.train_c()
              self._logger.debug('Finished C++ tuning, appending tuned discriminators to tuning record...')
              # Append retrieved tuned discriminators
              tunedDiscr.append( cTunedDiscr )
            self._logger.debug('Finished all initializations for sort %d...', sort)
          # Finished all inits for this sort, we need to undo the crossValid if
          # we are going to do a new sort, otherwise we continue
          if not ( (confNum+1) == nConfigs and (sort+1) == nSorts):
            if ppChain.isRevertible():
              trnData = tunningWrapper.trnData(release = True)
              valData = tunningWrapper.valData(release = True)
              tstData = tunningWrapper.testData(release = True)
              patterns = crossValid.revert( trnData, valData, tstData, sort = sort )
              del trnData, valData, tstData
              patterns = ppChain( patterns , revert = True )
            else:
              # We cannot revert ppChain, reload data:
              self._logger.info('Re-opening raw data...')
              with TuningDataArchieve(dataLocation, et_bin = etBinIdx if nEtBins is not None else None,
                                                    eta_bin = etaBinIdx if nEtaBins is not None else None) as TDArchieve:
                patterns = (TDArchieve['signal_rings'], TDArchieve['background_rings'])
              del TDArchieve
          self._logger.debug('Finished all hidden layer neurons for sort %d...', sort)
        self._logger.debug('Finished all sorts for configuration %d in collection...', confNum)
        ## Finished retrieving all tuned discriminators for this config file for
        ## this pre-processing. Now we head to save what we've done so far:
        # This pre-processing were tuned during this tuning configuration:
        tunedPP = ppCol[etBinIdx][etaBinIdx]
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
        savedFile = TunedDiscrArchieve( fulloutput, neuronBounds = neuronBounds, 
                                        sortBounds = sortBounds, 
                                        initBounds = initBounds,
                                        tunedDiscr = tunedDiscr,
                                        tunedPP = tunedPP,
                                        **extraKw
                                      ).save( self.compress )
        self._logger.info('File "%s" saved!', savedFile)
      # Finished all configurations we had to do
      self._logger.info('Finished tuning job!')

  # end of __call__ member fcn

