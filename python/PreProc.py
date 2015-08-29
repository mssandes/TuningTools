from FastNetTool.Logger import Logger
import numpy as np
from FastNetTool.util import checkForUnusedVars

from abc import ABCMeta, abstractmethod

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
  def __init__(self, d = None, **kw):
    d.update( kw )
    Logger.__init__(self, d)

  def __call__(self, data, revert = False):
    """
      The revert should be used to undo the pre-processing.
    """
    if revert:
      try:
        self._logger.info('Reverting %s...', self.__class__.__name__)
        data = self._undo(data)
      except AttributeError:
        raise RuntimeError("It is impossible to revert PreProc %s" % \
            self.__name__)
    else:
      self._logger.info('Applying %s...', self.__class__.__name__)
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
      of the normalization.
    """
    pass

  def isRevertible(self):
    """
      Check whether the PreProc is revertible
    """
    return self.__dict__.has_key['_undo']

  @abstractmethod
  def _apply(self, data):
    """
      Overload this method to apply the pre-processing
    """
    pass

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

  def _apply(self, data):
    pass

  def _undo(self, data):
    pass

class Norm1(PrepObj):
  """
    Applies norm-1 to data
  """

  _norms = None

  def __init__( self, **kw ):
    PrepObj.__init__( self, kw )
    checkForUnusedVars(kw, self._logger.warning )
    del kw

  def takeParams(self, data):
    """
      Calculate pre-processing parameters.
    """
    if isinstance(data, (tuple, list,)):
      self._norms = []
      for cdata in data:
        norms = cdata.sum(axis=1)
        norms[norms==0] = 1
        self._norms.append( norms )
    else:
      self._norms = data.sum(axis=1)
      self._norms[self._norms==0] = 1

  def norms(self):
    """
      Get normalizing factors
    """
    return self._norms

  def __str__(self):
    """
      String representation of the object.
    """
    return "Norm1"

  def release(self):
    """
      Release calculated pre-proessing parameters.
    """
    self._norms = None

  def _apply(self, data):
    if not self._norms:
      self.takeParams(data)
    try:
      if isinstance(data, (tuple, list,)):
        ret = []
        for i, cdata in enumerate(data):
          ret.append( cdata / self._norms[i][ :, np.newaxis ] )
      else:
        ret = data / self._norms[ :, np.newaxis ]
    except ValueError:
      raise ValueError(("Cannot apply norm on data, norm params do not match "
          "dimension. Use Norm1.release() to apply norm1 to new data (this will "
          "make it impossible to revert previous norm)."))
    return ret

  def _undo(self, data):
    if self._norms:
      try:
        if isinstance(data, (tuple, list,)):
          ret = []
          for i, cdata in enumerate(data):
            ret.append( cdata * self._norms[i][ :, np.newaxis ] )
        else:
          ret = data * self._norms[ :, np.newaxis ]
      except ValueError:
        raise ValueError(("Cannot apply norm on data, norm params do not match "
            "dimension. Use Norm1.release() to apply norm1 to new data (this will "
            "make it impossible to revert previous norm)."))
      # try
    else:
      raise RuntimeError(("Cannot revert norm1: its parameters wasn't retrieven "
          "from any data."))
    return ret


class RingerRp( Norm1 ):
  """
    Apply ringer-rp reprocessing to data.
  """
  _rVec = None
  _alpha = None
  _beta = None

  def __init__(self, **kw):
    PrepObj.__init__(self, kw)
    self._alpha = kw.pop('alpha', 1.)
    self._beta  = kw.pop('beta' , 1.)
    checkForUnusedVars(kw, self._logger.warning )
    del kw
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

  def takeParams(self, data):
    """
      Calculate pre-processing parameters.
    """
    if isinstance(data, (tuple, list,)):
      self._norms = []
      for cdata in data:
        self._norms.append( np.power( data, self._alpha ) )
    else:
      self._norms = np.power(data, self._alpha)
    Norm1.takeParams(self, self._norms)

  def rVec(self):
    """
      Retrieves the ring pseudo-distance vector
    """
    return self._rVec

  def _apply(self, data):
    self._logger.info('(alpha, beta) = (%f,%f)', self._alpha, self._beta)
    if not self._norms:
      self.takeParams(data)
    try:
      if isinstance(data, (tuple, list,)):
        ret = []
        for i, cdata in enumerate(data):
          ret.append(np.power( cdata, self._alpha ) * self._rVec) \
              / self._norms[i][:, np.newaxis ]
      else:
        ret = (np.power( data, self._alpha ) * self._rVec) \
            / self._norms[:, np.newaxis ]
    except ValueError,e:
      raise ValueError(("Cannot apply RingerRp on data, norm params do not match "
          "dimension. Use RingerRp.release() to apply rp-normalization to new data "
          "(this will make it impossible to revert previous norm)."))
    return ret

from FastNetTool.LimitedTypeList import LimitedTypeList

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
    from FastNetTool.LimitedTypeList import _LimitedTypeList____init__
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
      string = 'pp_'
      for pp in self:
        string += (str(pp) + '_')
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

  def takeParams(self, data):
    """
      Take pre-processing parameters for all objects in chain. 
    """
    if not self:
      self._logger.warning("No pre-processing available in this chain.")
      return
    for pp in self:
      pp.takeParams(data)

  def release(self):
    """
      Release pre-processing chain pp parameters
    """
    if not self:
      self._logger.warning("No pre-processing available in this chain.")
      return
    for pp in self:
      pp.release()

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

