from RingerCore.Logger import Logger
from RingerCore.util import checkForUnusedVars
from RingerCore.FileIO import save, load
import numpy as np

from abc import ABCMeta, abstractmethod

class PreProcArchieve( Logger ):
  """
  Context manager for Pre-Processing archives
  """

  _type = 'PreProcFile'
  _version = 1
  _ppChain = None

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with PreProcArchieve("/path/to/file") as data:
      BLOCK

    PreProcArchieve( "file/path", ppChain = Norm1() )
    """
    Logger.__init__(self, kw)
    self._filePath = filePath
    self.ppChain = kw.pop( 'ppChain', None )
    checkForUnusedVars( kw, self._logger.warning )

  @property
  def filePath( self ):
    return self._filePath

  def filePath( self, val ):
    self._filePath = val

  @property
  def ppChain( self ):
    return self._ppChain

  @ppChain.setter
  def ppChain( self, val ):
    if not val is None and not isinstance(val, PreProcChain):
      raise ValueError("Attempted to set ppChain to an object not of PreProcChain type.")
    else:
      self._ppChain = val

  def getData( self ):
    if not self._ppChain:
       raise RuntimeError("Attempted to retrieve empty data from PreProcArchieve.")
    return {'type' : self._type,
            'version' : self._version,
            'ppChain' : self._ppChain }

  def save(self, compress = True):
    return save( self.getData(), self._filePath, compress = compress )

  def __enter__(self):
    from cPickle import PickleError
    try:
      ppChainInfo = load( self._filePath )
    except PickleError:
      # It failed without renaming the module, retry renaming old module
      # structure to new one.
      import sys
      sys.modules['FastNetTool.PreProc'] = sys.modules[__name__]
      ppChainInfo = load( self._filePath )
    try: 
      if ppChainInfo['type'] != self._type:
        raise RuntimeError(("Input crossValid file is not from PreProcFile " 
            "type."))
      if ppChainInfo['version'] == 1:
        ppChain = ppChainInfo['ppChain']
      else:
        raise RuntimeError("Unknown job configuration version.")
    except RuntimeError, e:
      raise RuntimeError(("Couldn't read PreProcArchieve('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))
    return ppChain
    
  def __exit__(self, exc_type, exc_value, traceback):
    # Remove bound
    self.ppChain = None 


class UndoPreProcError(RuntimeError):
  """
    Raised when it is not possible to undo pre-processing.
  """
  def __init__( self, *args ):
    RuntimeError.__init__( self, *args )

# TODO List:
#
# - Add remove_constant_rows
# - Check for Inf, NaNs and so on

class PrepObj(Logger):
  """
    This is the base class of all pre-processing objects.
  """
  def __init__(self, d = {}, **kw):
    d.update( kw )
    Logger.__init__(self, d)

  def __call__(self, data, revert = False):
    """
      The revert should be used to undo the pre-processing.
    """
    if revert:
      try:
        self._logger.debug('Reverting %s...', self.__class__.__name__)
        data = self._undo(data)
      except AttributeError:
        raise RuntimeError("It is impossible to revert PreProc %s" % \
            self.__name__)
    else:
      self._logger.debug('Applying %s...', self.__class__.__name__)
      data = self._apply(data)
    return data

  def takeParams(self, data):
    """
      Calculate pre-processing parameters.
    """
    self._logger.debug("No need to retrieve any parameters from data.")
    pass

  def release(self):
    """
      Release calculated pre-proessing parameters.
    """
    self._logger.debug(("No parameters were taken from data, therefore none was "
        "also empty."))

  @abstractmethod
  def __str__(self):
    """
      Overload this method to return the string representation
      of the pre-processing.
    """
    pass

  @abstractmethod
  def shortName(self):
    """
      Overload this method to return the short string representation
      of the pre-processing.
    """
    pass

  def isRevertible(self):
    """
      Check whether the PreProc is revertible
    """
    import inspect
    return any([a[0] == '_undo' for a in inspect.getmembers(self) ])

  @abstractmethod
  def _apply(self, data):
    """
      Overload this method to apply the pre-processing
    """
    pass

#  @abstractmethod
#  def train(self, data):
#    """
#      Overload this method to apply the pre-processing
#    """
#    pass

class NoPreProc(PrepObj):
  """
    Do not apply any pre-processing to data.
  """

  def __init__( self, **kw ):
    PrepObj.__init__( self, kw )
    checkForUnusedVars(kw, self._logger.warning )
    del kw

  def __str__(self):
    """
      String representation of the object.
    """
    return "NoPreProc"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "NoPP"

  def _apply(self, data):
    pass

  def _undo(self, data):
    pass

class Norm1(PrepObj):
  """
    Applies norm-1 to data
  """

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._logger.warning )
    del d

  def __retrieveNorm(self, data):
    """
      Calculate pre-processing parameters.
    """
    if isinstance(data, (tuple, list,)):
      norms = []
      for cdata in data:
        cnorm = cdata.sum(axis=1)
        cnorm[cnorm==0] = 1
        norms.append( cnorm )
    else:
      norms = data.sum(axis=1)
      norms[norms==0] = 1
    return norms

  def __str__(self):
    """
      String representation of the object.
    """
    return "Norm1"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "N1"

  def _apply(self, data):
    norms = self.__retrieveNorm(data)
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append( cdata / norms[i][ :, np.newaxis ] )
    else:
      ret = data / norms[ :, np.newaxis ]
    return ret

  def takeParams(self, trnData):
    """
      Take pre-processing parameters for all objects in chain. 
    """
    return self._apply(trnData)

class RingerRp( Norm1 ):
  """
    Apply ringer-rp reprocessing to data.
  """
  _rVec = None
  _alpha = None
  _beta = None

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    self._alpha = d.pop('alpha', 1.)
    self._beta  = d.pop('beta' , 1.)
    Norm1.__init__( self, d )
    checkForUnusedVars(d, self._logger.warning )
    del d
    #Layers resolution
    PS      = 0.025 * np.arange(8)
    EM1     = 0.003125 * np.arange(64)
    EM2     = 0.025 * np.arange(8)
    EM3     = 0.05 * np.arange(8)
    HAD1    = 0.1 * np.arange(4)
    HAD2    = 0.1 * np.arange(4)
    HAD3    = 0.2 * np.arange(4)
    rings   = np.concatenate((PS,EM1,EM2,EM3,HAD1,HAD2,HAD3))
    self._rVec = np.power( rings, self._beta )

  def __str__(self):
    """
      String representation of the object.
    """
    return ("RingerRp_a%g_b%g" % (self._alpha, self._beta)).replace('.','dot')

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "Rp"

  def __retrieveNorm(self, data):
    """
      Calculate pre-processing parameters.
    """
    if isinstance(data, (tuple, list,)):
      norms = []
      for cdata in data:
        norms.append( np.power( data, self._alpha ) )
    else:
      norms = np.power(data, self._alpha)
    return Norm1.__retrieveNorm(self, norms)

  def rVec(self):
    """
      Retrieves the ring pseudo-distance vector
    """
    return self._rVec

  def _apply(self, data):
    self._logger.info('(alpha, beta) = (%f,%f)', self._alpha, self._beta)
    norms = self.__retrieveNorm(data)
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append(np.power( cdata, self._alpha ) * self._rVec) \
            / norms[i][:, np.newaxis ]
    else:
      ret = (np.power( data, self._alpha ) * self._rVec) \
          / norms[:, np.newaxis ]
    return ret

class MapStd( PrepObj ):
  """
    Remove data mean and set unitary standard deviation.
  """

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._logger.warning )
    del d
    self._mean = np.array( [], dtype='float32' )
    self._invRMS  = np.array( [], dtype='float32' )

  def mean(self):
    return self._mean
  
  def rms(self):
    return 1 / self._invRMS

  def params(self):
    return self.mean(), self.rms()

  def takeParams(self, trnData):
    """
      Calculate mean and rms for transformation.
    """
    # Put all classes information into only one representation
    # TODO Make transformation invariant to each class mass.
    if isinstance(trnData, (tuple, list,)):
      trnData = np.concatenate( trnData )
    self._mean = np.mean( trnData, axis=0 )
    trnData = trnData - self._mean
    self._invRMS = 1 / np.sqrt( np.mean( np.square( trnData ), axis=0 ) )
    self._invRMS[self._invRMS==0] = 1
    trnData *= self._invRMS
    return trnData

  def __str__(self):
    """
      String representation of the object.
    """
    return "MapStd"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "std"

  def _apply(self, data):
    if not self._mean.size or not self._invRMS.size:
      raise RuntimeError("Attempted to apply MapStd before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        ret.append( ( cdata - self._mean ) * self._invRMS )
    else:
      ret = ( data - self._mean ) * self._invRMS
    return ret

  def _undo(self, data):
    if not self._mean.size or not self._invRMS.size:
      raise RuntimeError("Attempted to undo MapStd before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append( ( cdata / self._invRMS ) + self._mean )
    else:
      ret = ( data / self._invRMS ) + self._mean
    return ret

class MapStd_MassInvariant( MapStd ):
  """
    Remove data mean and set unitary standard deviation but "invariant" to each
    class mass.
  """

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    MapStd.__init__( self, d )
    del d

  def takeParams(self, trnData):
    """
      Calculate mean and rms for transformation.
    """
    # Put all classes information into only one representation
    if isinstance(trnData, (tuple, list,)):
      means = []
      means = np.zeros(shape=( len(trnData), trnData[0].shape[1]), dtype='float32' )
      for idx, cTrnData in enumerate(trnData):
        means[idx,] = np.mean( cTrnData, axis=0, dtype='float32' )
      print 'data means: ', means
      self._mean = np.mean( means, axis=0 )
      print 'self.-mean:', self._mean
      trnData = np.concatenate( trnData )
    else:
      self._mean = np.mean( trnData, axis=0 )
    trnData = trnData - self._mean
    self._invRMS = 1 / np.sqrt( np.mean( np.square( trnData ), axis=0 ) )
    self._invRMS[self._invRMS==0] = 1
    trnData *= self._invRMS
    return trnData

  def __str__(self):
    """
      String representation of the object.
    """
    return "MapStd_MassInv"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "stdI"


class PCA( PrepObj ):
  """
    PCA preprocessing 
  """
  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    self.energy = d.pop('energy' , None)

    checkForUnusedVars(d, self._logger.warning )
    from sklearn import decomposition
    self._pca = decomposition.PCA(n_components = self.energy)

    #fix energy value
    if self.energy:  self.energy=int(100*self.energy)
    else:  self.energy=100 #total energy
    del d

  def params(self):
    return self._pca

  def variance(self):
    return self._pca.explained_variance_ratio_

  def cov(self):
    return self._pca.get_covariance()

  def ncomponents(self):
    return self.variance().shape[0]

  def takeParams(self, trnData):
    if isinstance(trnData, (tuple, list,)):
      trnData = np.concatenate( trnData )
    self._pca.fit(trnData)
    self._logger.info('PCA are aplied (%d of energy). Using only %d components of %d',
                      self.energy, self.ncomponents(), trnData.shape[1])
    return trnData

  def __str__(self):
    """
      String representation of the object.
    """
    return "PrincipalComponentAnalysis_"+str(self.energy)

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "PCA_"+str(self.energy)

  def _apply(self, data):
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        ret.append( self._pca.transform(cdata) )
    else:
      ret = self._pca.transform(data)
    return ret

  #def _undo(self, data):
  #  if isinstance(data, (tuple, list,)):
  #    ret = []
  #    for i, cdata in enumerate(data):
  #      ret.append( self._pca.inverse_transform(cdata) )
  #  else:
  #    ret = self._pca.inverse_transform(cdata)
  #  return ret


class KernelPCA( PrepObj ):
  """
    Kernel PCA preprocessing 
  """
  _explained_variance_ratio = None
  _cov = None

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )

    self._kernel                    = d.pop('kernel'                , 'rbf' )
    self._gamma                     = d.pop('gamma'                 , None  )
    self._n_components              = d.pop('n_components'          , None  )
    self._energy                    = d.pop('energy'                , None  )
    self._max_samples               = d.pop('max_samples'           , 5000  )
    self._fit_inverse_transform     = d.pop('fit_inverse_transform' , False )
    self._remove_zero_eig           = d.pop('remove_zero_eig'       , False )


    checkForUnusedVars(d, self._logger.warning )

    if (self._energy) and (self._energy > 1):
      raise RuntimeError('Energy value must be in: [0,1]')

    from sklearn import decomposition
    self._kpca  = decomposition.KernelPCA(kernel = self._kernel, 
                                          n_components = self._n_components,
                                          eigen_solver = 'auto', 
                                          gamma=self._gamma,
                                          fit_inverse_transform= self._fit_inverse_transform, 
                                          remove_zero_eig=self._remove_zero_eig)
    del d

  def params(self):
    return self._kpca

  def takeParams(self, trnData):

    #FIXME: try to reduze the number of samples for large 
    #datasets. There is some problem into sklearn related
    #to datasets with more than 20k samples. (lock to 16K samples)
    data = trnData
    if isinstance(data, (tuple, list,)):
      pattern=0
      for cdata in data:
        print cdata.shape
        if cdata.shape[0] > self._max_samples*0.5:
          self._logger.warning('pattern with more than %d samples. reduze!',self._max_samples*0.5)
          data[pattern] = cdata[np.random.permutation(cdata.shape[0])[0:self._max_samples],:]
        pattern+1
      data = np.concatenate( data )
      trnData = np.concatenate( trnData )
    else:
      if data.shape[0] > self._max_samples:
        data = data[np.random.permutation(data.shape[0])[0:self._max_samples],:]

    self._logger.info('fitting dataset...')
    #fitting kernel pca
    self._kpca.fit(data)
    #apply transformation into data
    data_transf = self._kpca.transform(data)
    #get load curve from variance accumulation for each component
    explained_variance = np.var(data_transf,axis=0)
    self._cov = np.cov(data_transf.T)
    self._explained_variance_ratio = explained_variance / np.sum(explained_variance)
    max_components_found = data_transf.shape[1]
    #release space
    data = [] 
    data_transf = []

    #fix n components by load curve
    if self._energy:
      cumsum = np.cumsum(self._explained_variance_ratio)
      self._n_components = np.where(cumsum > self._energy)[0][0]
      self._energy=int(self._energy*100) #fix representation
      self._logger.info('Variance cut. Using components = %d of %d',self._n_components,max_components_found)
    #free, the n components will be max
    else:
      self._n_components = max_components_found

    return trnData[:,0:self._n_components]

  def kernel(self):
    return self._kernel

  def variance(self):
    return self._explained_variance_ratio

  def cov(self):
    return self._cov

  def ncomponents(self):
    return self._n_components

  def __str__(self):
    """
      String representation of the object.
    """
    if self._energy:
      return "KernelPCA_energy_"+str(self._energy)
    else:
      return "KernelPCA_ncomp_"+str(self._n_components)
      

  def shortName(self):
    """
      Short string representation of the object.
    """
    if self._energy:
      return "kPCAe"+str(self._energy)
    else:
      return "kPCAc"+str(self._n_components)


  def _apply(self, data):
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        ret.append( self._kpca.transform(cdata)[:,0:self._n_components] )
    else:
      ret = self._kpca.transform(data)[:,0:self._n_components]
    return ret

  #def _undo(self, data):
  #  if isinstance(data, (tuple, list,)):
  #    ret = []
  #    for i, cdata in enumerate(data):
  #      ret.append( self._kpca.inverse_transform(cdata) )
  #  else:
  #    ret = self._kpca.inverse_transform(cdata)
  #  return ret




from RingerCore.LimitedTypeList import LimitedTypeList

class PreProcChain ( Logger ):
  """
    The PreProcChain is the object to hold the pre-processing chain to be
    applied to data. They will be sequentially applied to the input data.
  """

  # Use class factory
  __metaclass__ = LimitedTypeList

  # These are the list (LimitedTypeList) accepted objects:
  _acceptedTypes = (PrepObj,)

  def __init__(self, *args, **kw):
    Logger.__init__(self, kw)
    from RingerCore.LimitedTypeList import _LimitedTypeList____init__
    _LimitedTypeList____init__(self, *args)

  def __call__(self, data, revert = False):
    """
      Apply/revert pre-processing chain.
    """
    if not self:
      self._logger.warning("No pre-processing available in this chain.")
      return
    for pp in self:
      data = pp(data, revert)
    return data

  def __str__(self):
    """
      String representation of the object.
    """
    string = 'NoPreProc'
    if self:
      string = ''
      for pp in self:
        string += (str(pp) + '->')
      string = string[:-2]
    return string

  def shortName(self):
    string = 'NoPreProc'
    if self:
      string = ''
      for pp in self:
        string += (pp.shortName() + '-')
      string = string[:-1]
    return string

  def isRevertible(self):
    """
      Check whether the PreProc is revertible
    """
    for pp in self:
      if not pp.isRevertible():
        return False
    return True

  def takeParams(self, trnData):
    """
      Take pre-processing parameters for all objects in chain. 
    """
    if not self:
      self._logger.warning("No pre-processing available in this chain.")
      return
    for pp in self:
      trnData = pp.takeParams(trnData)

class PreProcCollection():
  """
    The PreProcCollection will hold a series of PreProcChain objects to be
    tested. The TuneJob will apply them one by one, looping over the testing
    configurations for each case.
  """

  # Use class factory
  __metaclass__ = LimitedTypeList

  # These are the list (LimitedTypeList) accepted objects:
  _acceptedTypes = (PreProcChain,)

# The PreProcCollection can hold a collection of itself:
PreProcCollection._acceptedTypes = PreProcCollection._acceptedTypes + (PreProcCollection,)

