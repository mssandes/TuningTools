
__all__ = ["SubsetGeneratorArchieve", "SubsetGeneratorPatterns", "SubsetGeneratorCollection",\
    "Cluster", "GMMCluster","fixSubsetCol"]

from RingerCore import Logger, LoggerStreamable, checkForUnusedVars, save, load, printArgs, traverse, \
                       retrieve_kw, EnumStringification, RawDictCnv, LoggerRawDictStreamer, LimitedTypeStreamableList

from abc import ABCMeta, abstractmethod
from TuningTools.PreProc import PreProcChain, PrepObj

# Retrieve numpy
from TuningTools.coreDef import retrieve_npConstants
npCurrent, _ = retrieve_npConstants()
import numpy as np
import gc

# Base class
class Subset(LoggerStreamable):

  # There is only need to change version if a property is added
  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_ppChain'})
  _cnvObj      = RawDictCnv(toProtectedAttrs         = {'_ppChain'})

  def __init__(self, d={}, **kw):
    d.update( kw )
    self._ppChain    = d.pop('ppChain', PreProcChain(PrepObj()) )
    self._range      = d.pop('binRange'  , None)
    self._patternIdx = d.pop('pattern'   , 0)
    LoggerStreamable.__init__(self, d)

  def __call__(self, data):
    return self._apply(data)

  @abstractmethod
  def _apply(self, data):
    """
      Overload this method to apply the pre-processing
    """
    return self._ppChain.takeParams(data)

  def isRevertible(self):
    # Not possible to return after this
    return False

  def getBin(self):
    return self._range

  def setPatternIndex(self, idx):
    self._patternIdx=idx

  def checkPatternIndex(self,idx):
    if idx==self._patternIdx:
      return True
    else:
      return False

  def getPatternIndex(self):
    return self._patternIdx

class SubsetGeneratorArchieve( Logger ):
  """
  Context manager for SubsetGenerator archives
  Version 0: Saving raw dict and rebuilding object  when loading.
  """

  _type      = 'SubsetGeneratorCollectionFile'
  _version   = 0
  _subsetCol = None

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it
    with SubsetGeneratorArchieve("/path/to/file") as data:
      BLOCK
    SubsetGeneratorArchieve( "file/path", subsetCol= SubsetGeneratorPatternCollection([...]) )
    """
    Logger.__init__(self, kw)
    self._filePath = filePath
    self._subsetCol = kw.pop( 'subsetCol', None )
    checkForUnusedVars( kw, self._logger.warning )

  @property
  def filePath( self ):
    return self._filePath

  @filePath.setter
  def filePath( self, val ):
    self._filePath = val

  @property
  def subsetCol( self ):
    return self._subsetCol

  @subsetCol.setter
  def subsetCol( self, val ):
    if not val is None and not isinstance(val, SubsetGeneratorCollection):
      self._logger.fatal("Attempted to set subsetCol to an object not of SubsetGeneratorCollection type.")
    else:
      self._subsetCol = val

  def getData( self ):
    if not self._subsetCol:
       self._logger.fatal("Attempted to retrieve empty data from SubsetGeneratorArchieve.")
    return {'type' : self._type,
            'version' : self._version,
            'subsetCol' : self._subsetCol.toRawObj() }

  def save(self, compress = True):
    return save( self.getData(), self._filePath, compress = compress )

  def __enter__(self):
    
    subsetColInfo   = load( self._filePath )
    # Open crossValidPreprocFile:
    try: 
      if isinstance(subsetColInfo, dict):
        if subsetColInfo['type'] != 'SubsetGeneratorCollectionFile':
          self._logger.fatal(("Input subsetCol file is not from SubsetGeneratorCollectionFile type."))
        if subsetColInfo['version'] == 0:
          self._subsetCol = SubsetGeneratorCollection.fromRawObj( subsetColInfo['subsetCol'] )
        else:
          self._logger.fatal("Unknown job configuration version.")
      else:
        self._logger.fatal("Invalid SubsetGeneratorCollectionFile contents.")
    except RuntimeError, e:
      self._logger.fatal(("Couldn't read cross preproc validation file file '%s': Reason:"
          "\n\t %s") % (self._filePath, e))

    return self._subsetCol
    
  def __exit__(self, exc_type, exc_value, traceback):
    # Remove bound
    self._subsetCol=None





class SubsetGeneratorPatterns ( Logger ):

  # Use class factory
  __metaclass__ = LimitedTypeStreamableList
  #_streamerObj  = LoggerLimitedTypeListRDS( level = LoggingLevel.VERBOSE )
  #_cnvObj       = LimitedTypeListRDC( level = LoggingLevel.VERBOSE )

  # These are the list (LimitedTypeList) accepted objects:
  _acceptedTypes = (Subset,)

  def __init__(self, *args, **kw):
    from RingerCore.LimitedTypeList import _LimitedTypeList____init__
    _LimitedTypeList____init__(self, *args)
    Logger.__init__(self, kw)
    self._dependentPatterns = []
    self._lowestEvtPerCluster=1

  def __call__(self, data, patternIdx):
    """
      Apply subset selection.
    """
    if not self:
      self._logger.warning("No subset generator available for this pattern")
      return

    for cluster in self:
      # check if the cluster match with the pattern index
      if cluster.checkPatternIndex(patternIdx):
        if not cluster.getBin() is None:
          # dependent case
          if self._dependentPatterns is None:
            self._logger.fatal("This cluster is Dependent of a value but the dependentPattern is None!")
          return self._treatSubset(cluster(data[np.where(\
            np.logical_and(self._dependentPatterns[patternIdx] >= cluster.getBin()[0], \
            self._dependentPatterns[patternIdx] < cluster.getBin()[1]))[0]][:]))
        else:
          return self._treatSubset(cluster(data))
      else:
        self._logger.debug("This cluster does not match with the pattern.")


  def isRevertible(self):
    """
      Check whether the Resample is revertible
    """
    return False

  def isDependent(self):
    """
      Check if we have an dependent object inside of the list
    """
    for s in self:
      if not s.getBin() is None:
        return True
    return False
  
  def setDependentPatterns( self, dpatterns ):
    self._dependentPatterns = dpatterns

  def setLowestNumberOfEvertPerCluster(self, n):
    self._lowestEvtPerCluster=n

  def _treatSubset( self, subsets ):
    # case 1: If cluster with zero events, we need to remove this from the list
    # case 2: If cluster with events minus than boxes, we need to split this into other clusters 
    subsetList = []
    remainderSubsetList = []
    for cluster in subsets:
      if len(cluster) > self._lowestEvtPerCluster:
        subsetList.append(cluster)
      elif len(cluster)>0:
        remainderSubsetList.append(cluster)
        self._logger.warning("Too few events for dividing data into this cluster. Try to recovery...")
      else:
        self._logger.warning("No events into this cluster")
    # Distribut events into other clusters...
    for cluster in remainderSubsetList:
      self._logger.warning("Recovering events...")
      count=0 # circle counter to split each event into each cluster
      for evt in cluster:
        # Append the current event into this cluster
        subsetList[count] = np.concatenate( (subsetList[count],np.reshape(evt, (1,evt.shape[0]) )) )
        if count == len(subsetList)-1:
          count=0
        else:
          count=count+1
    # Return the fixed cluster list with out zeros and clusters with events >= boxes
    return subsetList

  def setLevel(self, value):
    """
      Override Logger setLevel method so that we can be sure that every
      pre-processing will have same logging level than that was set for the
      ResampleChain instance.
    """
    for pp in self:
      pp.level = LoggingLevel.retrieve( value )
    self._level = LoggingLevel.retrieve( value )
    self._logger.setLevel(self._level)

  level = property( Logger.getLevel, setLevel )





class SubsetGeneratorCollection( object ):
  """
    The SubsetGeneratorCollection will hold a series of SubsetGeneratorPatterns objects to be
    tested. The CrossValid will apply them one by one, looping over the testing
    configurations for each case.
  """

  # Use class factory
  __metaclass__ = LimitedTypeStreamableList
  #_streamerObj  = LimitedTypeListRDS( level = LoggingLevel.VERBOSE )
  #_cnvObj       = LimitedTypeListRDC( level = LoggingLevel.VERBOSE )

  # These are the list (LimitedTypeList) accepted objects:
  _acceptedTypes = (SubsetGeneratorPatterns,)

# The ResampleCollection can hold a collection of itself:
SubsetGeneratorCollection._acceptedTypes = SubsetGeneratorCollection._acceptedTypes + (SubsetGeneratorCollection,)




def fixSubsetCol( var, nSorts = 1, nEta = 1, nEt = 1, level = None ):
  """
    Helper method to correct variable to be a looping bound collection
    correctly represented by a LoopingBoundsCollection instance.
  """
  tree_types = (SubsetGeneratorCollection, SubsetGeneratorPatterns, list, tuple )
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
    var = inspect_list_attrs(var, 3, SubsetGeneratorPatterns  , tree_types = tree_types,                                level = level   )
    var = inspect_list_attrs(var, 2, SubsetGeneratorCollection, tree_types = tree_types, dim = nSorts, name = "nSorts",                 )
    var = inspect_list_attrs(var, 1, SubsetGeneratorCollection, tree_types = tree_types, dim = nEta,   name = "nEta",                   )
    var = inspect_list_attrs(var, 0, SubsetGeneratorCollection, tree_types = tree_types, dim = nEt,    name = "nEt",    deepcopy = True )
  else:
    raise ValueError("subset generator dimensions size is larger than 5.")

  return var






# Simple cluster finder using euclidian distance
class Cluster( Subset ):

  # There is only need to change version if a property is added
  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_code_book','_w'})
  _cnvObj      = RawDictCnv(toProtectedAttrs         = {'_code_book','_w'})

  def __init__(self, d={}, **kw):
    """
      Cluster finder class base on three parameters:
        code_book: centroids of the cluster given by any algorithm (e.g: kmeans)
        w        : weights, this will multipli the size of the cluster depends of the factor
                   e.g: the cluster was found 100 events and the w factor is 2. In the end we
                   will duplicate the events into the cluster to 200.
        matrix   : projection apply on the centroids.
    """
    d.update( kw ); del kw
    Subset.__init__(self,d) 

    self._code_book = d.pop('code_book', [])
    self._w         = d.pop('w'  , 1   )
    checkForUnusedVars(d, self._logger.warning )  
    del d
    # Some protections before start
    if type(self._code_book) is list:
      self._code_book = npCurrent.array(self._code_book)
    # If weigth factor is an integer, transform to an array of factors with the 
    # same size of the centroids
    if type(self._w) is int:
      self._w = npCurrent.array([self._w for i in range(self._code_book.shape[0])], dtype=int)
    # transform to np.array if needed
    if type(self._w) is list:
      self._w = npCurrent.array(self._w,dtype=int)
    # In case to pass a list of weights, we need to check if weights and centroids has the same length.
    if self._w.shape[0] != self._code_book.shape[0]:
      raise ValueError("Weight factor must be an int, list or np.array with the same size than the code book param")
  #__init__ end


  def __call__(self, data):
    return self._apply(data)
  
  def _apply(self,data):
    """
    This function is slower than the C version but works for
    all input types.  If the inputs have the wrong types for the
    C versions of the function, this one is called as a last resort.

    It is about 20 times slower than the C version.
    """
    # Take param and apply pre-processing
    # hold the unprocess data
    self._ppChain.takeParams(data)
    tdata = self._ppChain(data)

    # n = number of observations
    # d = number of features
    if np.ndim(tdata) == 1:
      if not np.ndim(tdata) == np.ndim(self._code_book):
        raise ValueError("Observation and code_book should have the same rank")
    else:
      (n, d) = tdata.shape
    # code books and observations should have same number of features and same shape
    if not np.ndim(tdata) == np.ndim(self._code_book):
      raise ValueError("Observation and code_book should have the same rank")
    elif not d == self._code_book.shape[1]:
      raise ValueError("Code book(%d) and obs(%d) should have the same "
                       "number of features (eg columns)""" %
                       (self._code_book.shape[1], d))

    # see here: http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc
    code = np.argmin(np.sqrt(np.sum(np.power(tdata[:,np.newaxis]-self._code_book ,2),axis=-1)),axis=1)
    # Release memory
    del tdata
    gc.collect()
    # Join all clusters into a list of clusters
    cpattern=[]
    for target in range(self._code_book.shape[0]):
      cpattern.append(data[np.where(code==target)[0],:])
    
    # Resize the cluster
    for i, c in enumerate(cpattern):
      cpattern[i] = np.repeat(cpattern[i],self._w[i],axis=0)  
      self._logger.info('Cluster %d and factor %d with %d events and %d features',\
                        i,self._w[i],cpattern[i].shape[0],cpattern[i].shape[1])
    return cpattern


# Simple cluster finder using Gaussian Mixtures Model
class GMMCluster( Cluster ):
  # There is only need to change version if a property is added
  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_sigma'})
  _cnvObj      = RawDictCnv(toProtectedAttrs         = {'_sigma'})

  def __init__(self,  d={}, **kw):
    """
      Cluster finder class base on three parameters:
        code_book: centroids of the cluster given by any algorithm (e.g: kmeans)
        w        : weights, this will multipli the size of the cluster depends of the factor
                   e.g: the cluster was found 100 events and the w factor is 2. In the end we
                   will duplicate the events into the cluster to 200.
        matrix   : projection apply on the centroids.
        sigma    : variance param of the gaussian, this algorithm will calculate the likelihood 
                   value using: lh[i] = np.exp(np.power((data-centroid[i])/sigma[i],2))
    """
    d.update( kw ); del kw
    self._sigma = d.pop('sigma' , npCurrent.array([])   )
    Cluster.__init__(self, d) 
    del d

    # Checking the sigma type
    if type(self._sigma) is list:
      self._sigma = npCurrent.array(self._sigma)
    if not self._sigma.shape == self._code_book.shape:
      raise ValueError("Code book and sigma matrix should have the same shape")
    #__init__ end


  def _apply(self,data):
    """
    This function is slower than the C version but works for
    all input types.  If the inputs have the wrong types for the
    C versions of the function, this one is called as a last resort.

    It is about 20 times slower than the C version.
    """
    # Take param and apply pre-processing
    # hold the unprocess data
    self._ppChain.takeParams(data)
    tdata = self._ppChain(data)
    # n = number of observations
    # d = number of features
    if np.ndim(tdata) == 1:
      if not np.ndim(tdata) == np.ndim(self._code_book):
        raise ValueError("Observation and code_book should have the same rank")
    else:
      (n, d) = tdata.shape
    # code books and observations should have same number of features and same shape
    if not np.ndim(tdata) == np.ndim(self._code_book):
      raise ValueError("Observation and code_book should have the same rank")
    elif not d == self._code_book.shape[1]:
      raise ValueError("Code book(%d) and obs(%d) should have the same "
                       "number of features (eg columns)""" %
                       (self._code_book.shape[1], d))
    # Prob finder equation is:
    # tdata     is n X d
    # code_book is m X d where m is the number of clusters
    # Sigma     is m X d
    # Prob = exp()
    # see here: http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc
    code = np.argmax(np.sum(np.exp(np.power((tdata[:,np.newaxis]-self._code_book),2)/self._sigma[np.newaxis,:]),axis=-1),axis=1)
    del tdata
    gc.collect()
    
    # Join all clusters into a list of clusters
    cpattern=[]
    for target in range(self._code_book.shape[0]):
      cpattern.append(data[np.where(code==target)[0],:])
    
    # Resize the cluster
    for i, c in enumerate(cpattern):
      cpattern[i] = np.repeat(cpattern[i],self._w[i],axis=0)  
      self._logger.info('Cluster %d and factor %d with %d events and %d features',\
                        i,self._w[i],cpattern[i].shape[0],cpattern[i].shape[1])
    return cpattern



