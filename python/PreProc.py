__all__ = ['PreProcArchieve', 'PrepObj', 'Projection',  'RemoveMean', 'RingerRp',
           'UndoPreProcError', 'UnitaryRMS', 'FirstNthPatterns', 'KernelPCA',
           'MapStd', 'MapStd_MassInvariant', 'NoPreProc', 'Norm1', 'PCA',
           'PreProcChain', 'PreProcCollection', 'RingerEtaMu', 'RingerFilterMu']

from RingerCore import ( Logger, LoggerStreamable, checkForUnusedVars
                       , save, load, LimitedTypeList, LoggingLevel, LoggerRawDictStreamer
                       , LimitedTypeStreamableList, RawDictStreamer, RawDictCnv )
#from RingerCore import LimitedTypeListRDC, LoggerLimitedTypeListRDS, \
#                       LimitedTypeListRDS
from TuningTools.coreDef import npCurrent
import numpy as np

class PreProcArchieve( Logger ):
  """
  Context manager for Pre-Processing archives

  Version 3: - saves raw version of pp collection
  Version 2: - saved a pp collection for each eta/et bin and each sort.
  Version 1: - saved a pre-processing chain.
  """

  _type = 'PreProcFile'
  _version = 3

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with PreProcArchieve("/path/to/file") as data:
      BLOCK

    PreProcArchieve( "file/path", ppCol = Norm1() )
    """
    Logger.__init__(self, kw)
    self._filePath = filePath
    self._ppCol = kw.pop( 'ppCol', None )
    checkForUnusedVars( kw, self._warning )

  @property
  def filePath( self ):
    return self._filePath

  def filePath( self, val ):
    self._filePath = val

  @property
  def ppCol( self ):
    return self._ppCol

  def getData( self ):
    if not self._ppCol:
       self._fatal("Attempted to retrieve empty data from PreProcArchieve.")
    return {'type' : self._type,
            'version' : self._version,
            'ppCol' : self._ppCol.toRawObj() }

  def save(self, compress = True):
    return save( self.getData(), self._filePath, compress = compress )

  def __enter__(self):
    from cPickle import PickleError
    try:
      ppColInfo = load( self._filePath )
    except PickleError:
      # It failed without renaming the module, retry renaming old module
      # structure to new one.
      import sys
      sys.modules['FastNetTool.PreProc'] = sys.modules[__name__]
      ppColInfo = load( self._filePath )
    try: 
      if ppColInfo['type'] != self._type:
        self._fatal(("Input crossValid file is not from PreProcFile " 
            "type."))
      if ppColInfo['version'] == 3:
        ppCol = PreProcCollection.fromRawObj( ppColInfo['ppCol'] )
      elif ppColInfo['version'] == 2:
        ppCol = ppColInfo['ppCol']
      elif ppColInfo['version'] == 1:
        ppCol = PreProcCollection( ppColInfo['ppCol'] )
      else:
        self._fatal("Unknown job configuration version.")
    except RuntimeError, e:
      self._fatal(("Couldn't read PreProcArchieve('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))
    return ppCol
    
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

class PrepObj( LoggerStreamable ):
  """
    This is the base class of all pre-processing objects.
  """

  def __init__(self, d = {}, **kw):
    d.update( kw )
    LoggerStreamable.__init__(self, d)

  def __call__(self, data, revert = False):
    """
      The revert should be used to undo the pre-processing.
    """
    if revert:
      try:
        self._debug('Reverting %s...', self.__class__.__name__)
        data = self._undo(data)
      except AttributeError:
        self._fatal("It is impossible to revert PreProc ")#%s" % \
        #    self.__class__.___name__)
    else:
      self._debug('Applying %s...', self.__class__.__name__)
      data = self._apply(data)
    return data

  def takeParams(self, data):
    """
      Calculate pre-processing parameters.
    """
    self._debug("No need to retrieve any parameters from data.")
    return self._apply(data)

  def release(self):
    """
      Release calculated pre-proessing parameters.
    """
    self._debug(("No parameters were taken from data, therefore none was "
        "also empty."))

  # TODO: Do something smart here, this is needed at the moment.
  def concatenate(self, data, extra):
    """
      Concatenate extra patterns if needed
    """
    return data

  def __str__(self):
    """
      Overload this method to return the string representation
      of the pre-processing.
    """
    pass

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

  def _apply(self, data):
    """
      Overload this method to apply the pre-processing
    """
    return data



class NoPreProc(PrepObj):
  """
    Do not apply any pre-processing to data.
  """

  def __init__( self, **kw ):
    PrepObj.__init__( self, kw )
    checkForUnusedVars(kw, self._warning )
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

  def _undo(self, data):
    return data


class Projection(PrepObj):
  """
    Do not apply any pre-processing to data.
  """

  # FIXME: This will probably give problematic results if data is saved 
  # with one numpy type and executed with other type

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_mat'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_mat'})

  def __init__( self, **kw ):
    PrepObj.__init__( self, kw )
    self._mat = kw.pop( 'matrix', npCurrent.fp_array([[]]) )
    checkForUnusedVars(kw, self._warning )
    del kw

  def __str__(self):
    """
      String representation of the object.
    """
    return "Proj"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "P"

  def _apply(self, data):
    if isinstance(data, (tuple, list,)):
      ret = []
      for cData in data:
        if npCurrent.useFortran:
          ret.append( np.dot( self._mat, cData ) )
        else:
          ret.append( np.dot( cData, self._mat ) )
    else:
      if npCurrent.useFortran:
        ret = np.dot( self._mat , data )
      else:
        ret = np.dot( data , self._mat )
    return ret

  def takeParams(self, trnData, **kw):
    return self._apply(trnData)

class RemoveMean( PrepObj ):
  """
    Remove data mean
  """
  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_mean'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_mean'})

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    del d
    self._mean = np.array( [], dtype=npCurrent.dtype )

  def mean(self):
    return self._mean
  
  def params(self):
    return self.mean()

  def takeParams(self, trnData):
    """
      Calculate mean for transformation.
    """
    import copy
    data = copy.deepcopy(trnData)
    if isinstance(trnData, (tuple, list,)):
      data = np.concatenate( trnData, axis=npCurrent.odim )
    self._mean = np.mean( data, axis=npCurrent.odim, dtype=data.dtype ).reshape( 
            npCurrent.access( pidx=data.shape[npCurrent.pdim],
                              oidx=1 ) )
    return self._apply(trnData)

  def __str__(self):
    """
      String representation of the object.
    """
    return "rm_mean"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "no_mu"

  def _apply(self, data):
    if not self._mean.size:
      self._fatal("Attempted to apply RemoveMean before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        ret.append( cdata - self._mean )
    else:
      ret = data - self._mean
    return ret

  def _undo(self, data):
    if not self._mean.size:
      self._fatal("Attempted to undo RemoveMean before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append( cdata + self._mean )
    else:
      ret = data + self._mean
    return ret

class UnitaryRMS( PrepObj ):
  """
    Set unitary RMS.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_invRMS'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_invRMS'})

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    del d
    self._invRMS  = np.array( [], dtype=npCurrent.dtype )

  def rms(self):
    return 1 / self._invRMS

  def params(self):
    return self.rms()

  def takeParams(self, trnData):
    """
      Calculate rms for transformation.
    """
    # Put all classes information into only one representation
    # TODO Make transformation invariant to each class mass.
    import copy
    data = copy.deepcopy(trnData)
    if isinstance(data, (tuple, list,)):
      data = np.concatenate( data, axis=npCurrent.odim )
    tmpArray = np.sqrt( np.mean( np.square( data ), axis=npCurrent.odim ) ).reshape( 
                npCurrent.access( pidx=data.shape[npCurrent.pdim],
                                  oidx=1 ) )
    tmpArray[tmpArray==0] = 1
    self._invRMS = 1 / tmpArray
    return self._apply(trnData)

  def __str__(self):
    """
      String representation of the object.
    """
    return "UnitRMS"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "1rms"

  def _apply(self, data):
    if not self._invRMS.size:
      self._fatal("Attempted to apply UnitaryRMS before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        ret.append( cdata * self._invRMS )
    else:
      ret = ( data * self._invRMS )
    return ret

  def _undo(self, data):
    if not self._invRMS.size:
      self._fatal("Attempted to undo UnitaryRMS before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append( cdata / self._invRMS )
    else:
      ret = ( data / self._invRMS )
    return ret

# class Sigma2( PrepObj ):
#   """
#     Divide data by the standard deviation.
#   """
#   _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_deviation'})
#   _cnvObj = RawDictCnv(toProtectedAttrs = {'_deviation'})
# 
#   def __init__(self, d = {}, **kw):
#     d.update( kw ); del kw
#     PrepObj.__init__(self, d)
#     checkForUnusedVars(d, self._warning)
#     del d
#     self._deviation = {}
#   
#   def __str__(self):
#     """
#     	String representation of the object.
#     """
#     return "Sigma2"
#   
#   def shortName(self):
#     """
#     	Short string representation of the object.
#     """
#     return "S2"
#   
#   def takeParams(self, trnData):
#     """
#     		Calculate standard deviation for each variable.
#     """
#     # NOTE: Check if the data is given in numpy.ndarray, only from one bin.
#     for i in range(len(trnData[0])):
#     		mean = sum(trnData[:,i])/len(trnData[:,i])
#     		self._deviation[i] = math.sqrt(sum((trnData[:,i]-mean)**2)/len(trnData[:,i]))
#     return self._deviation
#   
#   def _apply(self, data):
#     if len(self._deviation)==0:
#     		self._fatal("Attempted to apply Sigma2 before taking its parameters.")
#     import numpy as np
#     ret = np.concatenate((data[0],data[1]),axis=0)
#     for i in range(len(ret[0])):
#     		ret[:,i] = ret[:,I]/(2*self._deviation[i])
#     return ret


class TrackSimpleNorm( PrepObj ):
  """
    Specific normalization for track parameters in mc14, 13TeV.
    Six variables, normalized through simple multiplications.
  """
  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_factors'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_factors'})
  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__(self, d)
    checkForUnusedVars(d, self._warning)
    self._factors = [0.05,  # deltaeta1
                     1.0,   # deltaPoverP
                     0.05,  # deltaPhiReescaled
                     6.0,   # d0significance
                     0.2,   # d0pvunbiased
                     1.0  ] # TRT_PID (from eProbabilityHT)
    del d

  def __str__(self):
    """
      String representation of the object.
    """
    return "Tracking data simple normalization."

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "TrackSimple"

  def _apply(self, data):
    # NOTE: is it a problem that I don't deepcopy the data?
    if isinstance(data, (list,tuple)):
      if len(data) == 0: return data
      import numpy as np
      import copy
      ret = []
      for conj in data:
        tmp = copy.deepcopy(conj)
        for i in range(len(self._factors)):
          tmp[:,i] = tmp[:,i]/self._factors[i]
        ret.append(tmp)
      return ret
    else:
      self._fatal("Data is not in the right format, must be list or tuple")

  def takeParams(self, trnData):
    return self._apply(trnData)

# FIXME
#   def _undo(self, data):
#     import numpy as np
#     ret = np.concatenate((data[0],data[1]),axis=0)
#     for i in range(len(self._factors)):
#       ret[:,i] = ret[:,i]*self._factors[i]
#     return ret


class Norm1(PrepObj):
  """
    Applies norm-1 to data
  """

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    del d

  def __retrieveNorm(self, data):
    """
      Calculate pre-processing parameters.
    """
    if isinstance(data, (tuple, list,)):
      norms = []
      for cdata in data:
        cnorm = cdata.sum(axis=npCurrent.pdim).reshape( 
            npCurrent.access( pidx=1,
                              oidx=cdata.shape[npCurrent.odim] ) )
        cnorm[cnorm==0] = 1
        norms.append( cnorm )
    else:
      norms = data.sum(axis=npCurrent.pdim).reshape( 
            npCurrent.access( pidx=1,
                              oidx=data.shape[npCurrent.odim] ) )
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
        ret.append( cdata / norms[i] )
    else:
      ret = data / norms
    return ret

class ExpertNetworksSimpleNorm(PrepObj):
  """
    Specific normalization for calorimeter and tracking parameters
    to be used in the Expert Neural Networks training.
    Usage of Norm1 normalization for calorimeter data and
    TrackSimpleNorm for tracking data.
  """
  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_factors'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_factors'})
  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__(self, d)
    checkForUnusedVars(d, self._warning)
    self._factors = [0.05,  # deltaeta1
                     1.0,   # deltaPoverP
                     0.05,  # deltaPhiReescaled
                     6.0,   # d0significance
                     0.2,   # d0pvunbiased
                     1.0  ] # eProbabilityHT
    del d

  def __str__(self):
    """
      String representation of the object.
    """
    return "Expert Neural Networks simple normalization."

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "Exp1"

  def __retrieveNorm(self, data):
    """
      Calculate pre-processing parameters of Norm1 normalization.
    """
    if isinstance(data, (tuple, list,)):
      norms = []
      for cdata in data:
        cnorm = cdata.sum(axis=npCurrent.pdim).reshape( 
            npCurrent.access( pidx=1,
                              oidx=cdata.shape[npCurrent.odim] ) )
        cnorm[cnorm==0] = 1
        norms.append( cnorm )
    else:
      norms = data.sum(axis=npCurrent.pdim).reshape( 
            npCurrent.access( pidx=1,
                              oidx=data.shape[npCurrent.odim] ) )
      norms[norms==0] = 1
    return norms

  def _apply(self, data):
    data_calo = data[0]
    norms = self.__retrieveNorm(data_calo)
    if isinstance(data_calo, (tuple, list,)):
      ret_calo = []
      for i, cdata in enumerate(data_calo):
        ret_calo.append( cdata / norms[i] )
    else:
      ret_calo = data_calo / norms

    data_track = data[1]
    if not isinstance(data_track, (list,tuple)):
      self._fatal("Data is not in the right format, must be list or tuple")
    else:
      if len(data_track) == 0:
        ret_track = data_track
      else:
        import numpy as np
        import copy
        ret_track = []
        for conj in data_track:
          tmp = copy.deepcopy(conj)
          for i in range(len(self._factors)):
            tmp[:,i] = tmp[:,i]/self._factors[i]
          ret_track.append(tmp)
    return [ret_calo,ret_track]


class FirstNthPatterns(PrepObj):
  """
    Get first nth patterns from data
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_n'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_n'})

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    self._n = d.pop('nth',0)
    checkForUnusedVars(d, self._warning )
    del d

  def __str__(self):
    """
      String representation of the object.
    """
    return "First_%dPat" % self._n

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "F%dP" % self._n

  def _apply(self, data):
    try: 
      if isinstance(data, (tuple, list,)):
        ret = []
        for cdata in data:
          ret.append( cdata[npCurrent.access( pidx=slice(0,self._n), oidx=':'  ) ] )
      else:
        ret = data[ npCurrent.access( pidx=slice(0,self._n), oidx=':'  ) ]  
    except IndexError, e:
      self._fatal("Data has not enought patterns!\n%s", str(e), IndexError)
    return ret

class RingerRp( Norm1 ):
  """
    Apply ringer-rp reprocessing to data.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_alpha', '_beta','_rVec'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_alpha','_beta','_rVec'})

  def __init__(self, alpha = 1., beta = 1., d = {}, **kw):
    d.update( kw ); del kw
    Norm1.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    del d
    self._alpha = alpha
    self._beta = beta
    #Layers resolution
    #PS      = 0.025 * np.arange(1,8)
    #EM1     = 0.003125 * np.arange(1,64)
    #EM2     = 0.025 * np.arange(1,8)
    #EM3     = 0.05 * np.arange(1,8)
    #HAD1    = 0.1 * np.arange(1,4)
    #HAD2    = 0.1 * np.arange(1,4)
    #HAD3    = 0.2 * np.arange(1,4)
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
    return Norm1._Norm1__retrieveNorm(self, data)

  def rVec(self):
    """
      Retrieves the ring pseudo-distance vector
    """
    return self._rVec

#array([  0.00000000e+00,   8.15707375e-04,  -1.10010558e-03,
#         4.47313348e-03,  -3.97166889e-03,   6.39257906e-03,
#         6.54621050e-03,   4.77211224e-03,   0.00000000e+00,
#         2.11066343e-02,   1.80540066e-02,   6.83453772e-03,
#         4.41724900e-03,   1.64411089e-03,   1.15571858e-03,
#         1.44910149e-03,   1.63055456e-03,   5.98416897e-04,
#         6.79160818e-04,   5.41001151e-04,   1.09213439e-03,
#         5.80448133e-04,  -5.42422466e-04,  -1.03258807e-03,
#        -1.13338057e-03,   2.36474923e-04,   4.88943188e-04,
#        -9.18365316e-04,   8.67576513e-04,  -4.77118243e-04,
#        -1.93166343e-04,  -4.25095845e-04,  -3.32333409e-04,
#         8.03846226e-04,   9.86088635e-05,  -2.34487306e-04,
#         9.99490381e-04,   2.20256252e-03,   2.03745579e-03,
#         1.77585275e-03,   1.58940803e-03,   1.73320470e-03,
#         4.69416118e-04,   7.04462989e-04,  -2.59370463e-05,
#         5.32700913e-04,  -1.34548778e-03,  -7.31210283e-04,
#        -7.96360662e-04,  -1.30150875e-03,   8.97139136e-04,
#         8.90310446e-04,  -5.41164773e-04,   5.86333394e-04,
#         1.66354538e-03,  -4.91637446e-04,  -1.42299489e-03,
#         7.11409026e-04,   6.05857407e-04,  -1.24913862e-03,
#        -3.88187502e-04,  -7.03075377e-04,  -1.28103315e-03,
#        -4.57881652e-05,   1.85933823e-04,  -1.82768883e-04,
#        -1.53428817e-04,   9.02294065e-04,   8.74958467e-04,
#        -1.02110242e-03,   9.73318354e-04,   6.48773566e-04,
#         0.00000000e+00,   7.45570883e-02,   1.83086284e-02,
#         1.42294550e-02,   1.23070702e-02,   1.14151677e-02,
#         5.50619932e-03,   3.57280398e-04,   0.00000000e+00,
#         2.06541806e-03,   1.05943729e-03,   3.96153657e-03,
#         6.60419464e-03,   1.94086856e-03,  -2.14856467e-03,
#        -1.65478513e-03,  -0.00000000e+00,  -6.19104016e-04,
#         1.81093253e-03,   0.00000000e+00,   0.00000000e+00,
#         5.59639127e-04,  -1.18495105e-03,   0.00000000e+00,
#        -0.00000000e+00,   1.52364664e-03,   9.87489126e-04,
#         0.00000000e+00], dtype=float32)

  def _apply(self, data):
    # FIXME This is here just as a temporary workaround, Layers resolution
    #PS      = 0.025 * np.arange(1,8)
    #EM1     = 0.003125 * np.arange(1,64)
    #EM2     = 0.025 * np.arange(1,8)
    #EM3     = 0.05 * np.arange(1,8)
    #HAD1    = 0.1 * np.arange(1,4)
    #HAD2    = 0.1 * np.arange(1,4)
    #HAD3    = 0.2 * np.arange(1,4)
    #rings   = np.concatenate((PS,EM1,EM2,EM3,HAD1,HAD2,HAD3))
    #self._rVec = np.power( rings, self._beta )
    self._info('(alpha, beta) = (%f,%f)', self._alpha, self._beta)
    mask = np.ones(100, dtype=bool)
    mask[np.cumsum([0,8,64,8,8,4,4])] = False
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        rpEnergy = np.sign(cdata)*np.power( abs(cdata), self._alpha )
        #rpEnergy = rpEnergy[ npCurrent.access( pidx=mask, oidx=':') ]
        norms = self.__retrieveNorm(rpEnergy)
        rpRings = ( ( rpEnergy * self._rVec ) / norms[i][ npCurrent.access( pidx=':', oidx=np.newaxis) ] ).astype( npCurrent.fp_dtype )
        ret.append(rpRings)
    else:
      rpEnergy = np.sign(data)*np.power( abs(cdata), self._alpha )
      #rpEnergy = rpEnergy[ npCurrent.access( pidx=mask, oidx=':') ]
      norms = self.__retrieveNorm(rpEnergy)
      rpRings = ( ( rpEnergy * self._rVec ) / norms[i][ npCurrent.access( pidx=':', oidx=np.newaxis) ] ).astype( npCurrent.fp_dtype )
      ret = rpRings
    return ret

class MapStd( PrepObj ):
  """
    Remove data mean and set unitary standard deviation.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_mean', '_invRMS'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_mean','_invRMS'})

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    del d
    self._mean = np.array( [], dtype=npCurrent.dtype )
    self._invRMS  = np.array( [], dtype=npCurrent.dtype )

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
    import copy
    data = copy.deepcopy(trnData)
    if isinstance(data, (tuple, list,)):
      data = np.concatenate( data, axis=npCurrent.odim )
    self._mean = np.mean( data, axis=npCurrent.odim, dtype=data.dtype ).reshape( 
            npCurrent.access( pidx=data.shape[npCurrent.pdim],
                              oidx=1 ) )
    data = data - self._mean
    tmpArray = np.sqrt( np.mean( np.square( data ), axis=npCurrent.odim ) ).reshape( 
                npCurrent.access( pidx=data.shape[npCurrent.pdim],
                                  oidx=1 ) )
    tmpArray[tmpArray==0] = 1
    self._invRMS = 1 / tmpArray
    return self._apply(trnData)

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
      self._fatal("Attempted to apply MapStd before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        ret.append( ( cdata - self._mean ) * self._invRMS )
    else:
      ret = ( data - self._mean ) * self._invRMS
    return ret

  def _undo(self, data):
    if not self._mean.size or not self._invRMS.size:
      self._fatal("Attempted to undo MapStd before taking its parameters.")
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
    self._fatal('MapStd_MassInvariant still needs to be validated.')
    #if isinstance(trnData, (tuple, list,)):
    #  means = []
    #  means = np.zeros(shape=( trnData[0].shape[npCurrent.odim], len(trnData) ), dtype=trnData.dtype )
    #  for idx, cTrnData in enumerate(trnData):
    #    means[:,idx] = np.mean( cTrnData, axis=npCurrent.pdim, dtype=npCurrent.fp_dtype )
    #  self._mean = np.mean( means, axis=npCurrent.pdim )
    #  trnData = np.concatenate( trnData )
    #else:
    #  self._mean = np.mean( trnData, axis=0 )
    #trnData = trnData - self._mean
    #self._invRMS = 1 / np.sqrt( np.mean( np.square( trnData ), axis=0 ) )
    #self._invRMS[self._invRMS==0] = 1 # FIXME, not on invRMS, but before dividing it.
    #trnData *= self._invRMS
    #return trnData

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

    checkForUnusedVars(d, self._warning )
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
    return self.variance().shape[npCurrent.pdim]

  def takeParams(self, trnData):
    if isinstance(trnData, (tuple, list,)):
      trnData = np.concatenate( trnData )
    self._pca.fit(trnData)
    self._info('PCA are aplied (%d of energy). Using only %d components of %d',
                      self.energy, self.ncomponents(), trnData.shape[np.pdim])
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
        # FIXME Test this!
        if npCurrent.isfortran:
          ret.append( self._pca.transform(cdata.T).T )
        else:
          ret.append( self._pca.transform(cdata) )
    else:
      ret = self._pca.transform(data)
      if npCurrent.isfortran:
        ret = self._pca.transform(cdata.T).T
      else:
        ret = self._pca.transform(cdata)
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


    checkForUnusedVars(d, self._warning )

    if (self._energy) and (self._energy > 1):
      self._fatal('Energy value must be in: [0,1]')

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
      # FIXME Test kpca dimensions
      pattern=0
      for cdata in data:
        if cdata.shape[npCurrent.odim] > self._max_samples*0.5:
          self._warning('Pattern with more than %d samples. Reduce!',self._max_samples*0.5)
          data[pattern] = cdata[
            npCurrent.access( pdim=':',
                              odim=(0,np.random.permutation(cdata.shape[npCurrent.odim])[0:self._max_samples]))
                               ] 
        pattern+=1
      data = np.concatenate( data, axis=npCurrent.odim )
      trnData = np.concatenate( trnData, axis=npCurrent.odim )
    else:
      if data.shape[0] > self._max_samples:
        data = data[
          npCurrent.access( pdim=':',
                            odim=(0,np.random.permutation(data.shape[npCurrent.odim])[0:self._max_samples]))
                   ]

    self._info('fitting dataset...')
    # fitting kernel pca
    if npCurrent.isfotran:
      self._kpca.fit(data.T)
      # apply transformation into data
      data_transf = self._kpca.transform(data.T).T
      self._cov = np.cov(data_transf.T)
    else:
      self._kpca.fit(data)
      # apply transformation into data
      data_transf = self._kpca.transform(data)
      self._cov = np.cov(data_transf)
    #get load curve from variance accumulation for each component
    explained_variance = np.var(data_transf,axis=npCurrent.pdim)
    self._explained_variance_ratio = explained_variance / np.sum(explained_variance)
    max_components_found = data_transf.shape[0]
    # release space
    data = [] 
    data_transf = []

    #fix n components by load curve
    if self._energy:
      cumsum = np.cumsum(self._explained_variance_ratio)
      self._n_components = np.where(cumsum > self._energy)[0][0]
      self._energy=int(self._energy*100) #fix representation
      self._info('Variance cut. Using components = %d of %d',self._n_components,max_components_found)
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
        if npCurrent.isfortran:
          ret.append( self._kpca.transform(cdata.T).T[0:self._n_components,:] )
        else:
          ret.append( self._kpca.transform(cdata)[:,0:self._n_components] )
    else:
      ret = self._kpca.transform(data)[0:self._n_components]
      if npCurrent.isfortran:
        ret = self._kpca.transform(data.T).T[0:self._n_components,:]
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

class RingerEtaMu(Norm1):
  """
    Applies norm-1+MapMinMax to data
  """

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    self._etamin           = d.pop('etamin'           , 0  )
    self._etamax           = d.pop('etamax'           , 2.5)
    self._pileupThreshold  = d.pop('pileupThreshold'  , 60 )
    checkForUnusedVars(d, self._warning )
    del d

  def __str__(self):
    """
      String representation of the object.
    """
    return "ExpertNormalizationRingerEtaMu"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "ExNREM"


  def __retrieveNorm(self, data):
    """
      Calculate pre-processing parameters.
    """
    if isinstance(data, (tuple, list,)):
      norms = []
      for cdata in data:
        cnorm = cdata.sum(axis=npCurrent.pdim).reshape( 
            npCurrent.access( pidx=1,
                              oidx=cdata.shape[npCurrent.odim] ) )
        cnorm[cnorm==0] = 1
        norms.append( cnorm )
    else:
      norms = data.sum(axis=npCurrent.pdim).reshape( 
            npCurrent.access( pidx=1,
                              oidx=data.shape[npCurrent.odim] ) )
      norms[norms==0] = 1
    return norms

  def concatenate(self, data, extra):

    self._logger.info('Concatenate extra patterns...')
    from TuningTools.dataframe import BaseInfo
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        cdata = np.concatenate((cdata, extra[i][BaseInfo.Eta], extra[i][BaseInfo.PileUp]),axis=1)
        ret.append(cdata)
    else:
      ret = np.concatenate((data, extra[i][BaseInfo.Eta], extra[i][BaseInfo.PileUp]),axis=1)
    return ret

  def _apply(self, data):

    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        norms = self.__retrieveNorm(cdata[ npCurrent.access( pidx=(0, 100) ) ])
        rings = cdata[ npCurrent.access( pidx=(0, 100) ) ] / norms[i]
        eta   = cdata[ npCurrent.access( pidx=(100,101), oidx=':' )] 
        eta   = ((np.abs(eta) - np.abs(self._etamin))*np.sign(eta))/np.max(self._etamax)
        mu    = cdata[ npCurrent.access( pidx=(101,102) ,oidx=':') ]
        mu[mu > self._pileupThreshold] = self._pileupThreshold
        mu = mu/self._pileupThreshold
        cdata  = np.concatenate((rings,eta,mu),axis=1)
        ret.append(cdata)
    else:
      norms = self.__retrieveNorm(data)
      norms = self.__retrieveNorm(data[ npCurrent.access( pidx=(0, 100) ) ])
      rings = data[ npCurrent.access( pidx=(0, 100) ) ] / norms[i]
      eta   = data[ npCurrent.access( pidx=(100,101), oidx=':' )] 
      eta   = ((np.abs(eta) - np.abs(self._etamin))*np.sign(eta))/np.max(self._etamax)
      mu[mu > self._pileupThreshold] = self._pileupThreshold
      mu = mu/self._pileupThreshold
      ret  = np.concatenate((rings,eta,mu),axis=1)

    return ret


class RingerFilterMu(PrepObj):
    '''
    Applies a filter at the mu values for training.
    '''
    def __init__(self, mu_min=0, mu_max=70):
        # Calling Norm1 Constructor
        super(RingerFilterMu, self).__init__()
        self._mu_min = mu_min
        self._mu_max = mu_max

    def __str__(self):
        return 'RingerFilterMu'

    def shortName(self):
        return 'RinFMU'

    def concatenate(self, data, extra):
        self._logger.info('Concatenate extra patterns...')
        from TuningTools.dataframe import BaseInfo
        if isinstance(data, (tuple, list,)):
            self._logger.info('Data is in list format...')
            ret = []
            for i, cdata in enumerate(data):
                self._logger.info('Filtering mu values %d...' % i)
                cdata = np.concatenate((cdata, extra[i][BaseInfo.PileUp]),axis=1)
                cdata = cdata[(cdata[:,-1] >= self._mu_min) & (cdata[:,-1] <= self._mu_max)]
                ret.append(cdata)
        else:
            self._logger.info('Filtering mu values...')
            ret = np.concatenate((data, extra[i][BaseInfo.PileUp]),axis=1)
            ret = ret[(ret[:,-1] >= self._mu_min) & (ret[:,-1] <= self._mu_max)]
        return ret


class PreProcChain ( Logger ):
  """
    The PreProcChain is the object to hold the pre-processing chain to be
    applied to data. They will be sequentially applied to the input data.
  """

  # Use class factory
  __metaclass__ = LimitedTypeStreamableList
  #_streamerObj  = LoggerLimitedTypeListRDS( level = LoggingLevel.VERBOSE )
  #_cnvObj       = LimitedTypeListRDC( level = LoggingLevel.VERBOSE )

  # These are the list (LimitedTypeList) accepted objects:
  _acceptedTypes = (PrepObj,)

  def __init__(self, *args, **kw):
    from RingerCore.LimitedTypeList import _LimitedTypeList____init__
    _LimitedTypeList____init__(self, *args)
    Logger.__init__(self, kw)

  def __call__(self, data, revert = False):
    """
      Apply/revert pre-processing chain.
    """
    if not self:
      self._warning("No pre-processing available in this chain.")
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
      self._warning("No pre-processing available in this chain.")
      return
    for pp in self:
      trnData = pp.takeParams(trnData)
    return trnData

  def concatenate(self, trnData, extraData):
    """
      Concatenate extra patterns into the data input
    """
    if not self:
      self._warning("No pre-processing available in this chain.")
      return
    for pp in self:
      trnData = pp.concatenate(trnData, extraData)

    return trnData

  def setLevel(self, value):
    """
      Override Logger setLevel method so that we can be sure that every
      pre-processing will have same logging level than that was set for the
      PreProcChain instance.
    """
    for pp in self:
      pp.level = LoggingLevel.retrieve( value )
    self._level = LoggingLevel.retrieve( value )
    self._logger.setLevel(self._level)

  level = property( Logger.getLevel, setLevel )


class PreProcCollection( object ):
  """
    The PreProcCollection will hold a series of PreProcChain objects to be
    tested. The TuneJob will apply them one by one, looping over the testing
    configurations for each case.
  """

  # Use class factory
  __metaclass__ = LimitedTypeStreamableList
  #_streamerObj  = LimitedTypeListRDS( level = LoggingLevel.VERBOSE )
  #_cnvObj       = LimitedTypeListRDC( level = LoggingLevel.VERBOSE )
  _acceptedTypes = type(None),
# The PreProcCollection can hold a collection of itself besides PreProcChains:
PreProcCollection._acceptedTypes = PreProcChain, PreProcCollection

