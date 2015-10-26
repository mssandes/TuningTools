from RingerCore.Logger import Logger
from RingerCore.util   import checkForUnusedVars, reshape
from RingerCore.FileIO import save, load
import os
import numpy as np

# FIXME This should be integrated into a class so that save could check if it
# is one instance of this base class and use its save method
class TuningDataArchive( Logger ):
  """
  Context manager for Tuning Data archives
  """

  _type = np.array('TuningData', dtype='|S10')
  _version = np.array(1)
  _signal_rings = np.array([])
  _background_rings = np.array([])
  _filePath = None

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with TuningDataArchive("/path/to/file") as data:
      BLOCK

    TuningDataArchive( signal_rings = np.array(...),
                       background_rings = np.array(...)
    """
    Logger.__init__(self, kw)
    self._filePath = filePath
    self._signal_rings = kw.pop( 'signal_rings', np.array([]) )
    self._background_rings = kw.pop( 'background_rings', np.array([]) )
    checkForUnusedVars( kw, self._logger.warning )

  @property
  def filePath( self ):
    return self._filePath

  @filePath.setter
  def filePath( self, val ):
    self._filePath = val

  @property
  def signal_rings( self ):
    return self._signal_rings

  @signal_rings.setter
  def signal_rings( self, val ):
    if val:
      if isinstance(val, np.ndarray):
        self._signal_rings = val
      else:
        raise TypeError("Rings must be an numpy array.")
    else:
      self._signal_rings = np.array([])

  @property
  def background_rings( self ):
    return self._background_rings

  @background_rings.setter
  def background_rings( self, val ):
    if val:
      if isinstance(val, np.ndarray):
        self._background_rings = val
      else:
        raise TypeError("Rings must be an numpy array.")
    else:
      self._background_rings = np.array([])

  def getData( self ):
    return {'type' : self._type,
            'version' : self._version,
            'signal_rings' : self._signal_rings,
            'background_rings' : self._background_rings }

  def save(self):
    return save(self.getData(), self._filePath, protocol = 'savez_compressed')

  def __enter__(self):
    from cPickle import PickleError
    npData = load( self._filePath )
    try:
      if type(npData) is np.ndarray:
        # Legacy type:
        data = reshape( npData[0] ) 
        target = reshape( npData[1] ) 
        self._signal_rings, self._background_rings = \
            TuningDataArchive.__separateClasses( data, target )
        data = (self._signal_rings, self._background_rings)
      elif type(npData) is np.lib.npyio.NpzFile:
        if npData['type'] != self._type:
          raise RuntimeError("Input file is not of TuningData type!")
        if npData['version'] == self._version:
          data = (npData['signal_rings'], npData['background_rings'])
        else:
          raise RuntimeError("Unknown file version!")
      elif isinstance(npData, dict) and 'type' in npData:
        raise RuntimeError("Attempted to read archive of type: %s_v%d" % (npData['type'],
                                                                          npData['version']))
      else:
        raise RuntimeError("Object on file is of unkown type.")
    except RuntimeError, e:
      raise RuntimeError(("Couldn't read TuningDataArchive('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))
    return data
    
  def __exit__(self, exc_type, exc_value, traceback):
    # Remove bound to data array
    self.signal_rings = None 
    self.background_rings = None

  @classmethod
  def __separateClasses( cls, data, target ):
    """
    Function for dealing with legacy data.
    """
    sgn = data[np.where(target==1)]
    bkg = data[np.where(target==-1)]
    return sgn, bkg


class CreateData(Logger):

  def __init__( self, logger = None ):
    Logger.__init__( self, logger = logger )
    from TuningTools.FilterEvents import filterEvents
    self._filter = filterEvents

  def __call__(self, sgnFileList, bkgFileList, ringerOperation, **kw):
    """
      Creates a numpy file ntuple with rings and its targets
      Arguments:
        - sgnFileList: A python list or a comma separated list of the root files
            containing the TuningTool TTree for the signal dataset
        - bkgFileList: A python list or a comma separated list of the root files
            containing the TuningTool TTree for the background dataset
        - ringerOperation: Set Operation type to be used by the filter
      Optional arguments:
        - output ['tuningData']: Name for the output file
        - referenceSgn [Reference.Truth]: Filter reference for signal dataset
        - referenceBkg [Reference.Truth]: Filter reference for background dataset
        - treePath: Sets tree path on file to be used as the TChain. The default
            value depends on the operation. If set to None, it will be set to 
            the default value.
        - nClusters [None]: Number of clusters to export. If set to None, export
            full PhysVal information.
        - getRatesOnly [False]: Do not create data, but retrieve the efficiency
            for benchmark on the chosen operation.
        - etBins [None]: E_T bins where the data should be segmented
        - etaBins [None]: eta bins where the data should be segmented
        - ringConfig [100]: A list containing the number of rings available in the data
          for each eta bin.
    """
    from TuningTools.FilterEvents import FilterType, Reference

    output       = kw.pop('output',         'tuningData'   )
    referenceSgn = kw.pop('referenceSgn',  Reference.Truth )
    referenceBkg = kw.pop('referenceBkg',  Reference.Truth )
    treePath     = kw.pop('treePath',           None       )
    l1EmClusCut  = kw.pop('l1EmClusCut',        None       )
    l2EtCut      = kw.pop('l2EtCut',            None       )
    offEtCut     = kw.pop('offEtCut',           None       )
    nClusters    = kw.pop('nClusters',          None       )
    getRatesOnly = kw.pop('getRatesOnly',       False      )
    etBins       = kw.pop('etBins',             None       )
    etaBins      = kw.pop('etaBins',            None       )
    ringConfig   = kw.pop('ringConfig',        None       )
    if ringConfig is None:
      ringConfig = [100]*(len(etaBins)-1) if etaBins else [100]
    if 'level' in kw: 
      self.level = kw.pop('level') # log output level
      self._filter.level = self.level
    checkForUnusedVars( kw, self._logger.warning )

    nEtBins  = len(etBins)-1 if not etBins is None else 1
    nEtaBins = len(etaBins)-1 if not etaBins is None else 1
    useBins = True if nEtBins > 1 or nEtaBins > 1 else False

    self._logger.info('Extracting signal dataset information...')

    # List of operation arguments to be propagated
    kwargs = { 'treePath':     treePath,
               'l1EmClusCut':  l1EmClusCut,
               'l2EtCut':      l2EtCut,
               'offEtCut':     offEtCut,
               'nClusters':    nClusters,
               'getRatesOnly': getRatesOnly,
               'etBins':       etBins,
               'etaBins':      etaBins,
               'ringConfig':   ringConfig, }

    npSgn, sgnEffList  = self._filter(sgnFileList,
                                      ringerOperation,
                                      filterType = FilterType.Signal,
                                      reference = referenceSgn,
                                      **kwargs)
    if npSgn.size: self.__printShapes(npSgn,'Signal')

    self._logger.info('Extracting background dataset information...')
    npBkg, bkgEffList = self._filter(bkgFileList, 
                                     ringerOperation,
                                     filterType = FilterType.Background,
                                     reference = referenceBkg,
                                     **kwargs)
    if npBkg.size: self.__printShapes(npBkg,'Background')

    if not getRatesOnly:
      savedPath = TuningDataArchive( output,
                                     signal_rings = npSgn,
                                     background_rings = npBkg ).save()
      self._logger.info('Saved data file at path: %s', savedPath )

    for etBin in range(nEtBins):
      for etaBin in range(nEtaBins):
        for sgnEff, bkgEff in zip(sgnEffList, bkgEffList) if not useBins else \
                              zip(sgnEffList[etBin][etaBin], bkgEffList[etBin][etaBin]):
          self._logger.info('Efficiency for %s: Det(%%): %s | FA(%%): %s',
                            sgnEff.name,
                            sgnEff.eff_str(), 
                            bkgEff.eff_str())
        # for eff
      # for eta
    # for et
  # end __call__

  def __printShapes(self, npArray, name):
    "Print numpy shapes"
    if not npArray.dtype.type is np.object_:
      self._logger.info('Extracted %s rings with size: %r',name, (npArray.shape))
    else:
      shape = npArray.shape
      for etBin in range(shape[0]):
        for etaBin in range(shape[1]):
          self._logger.info('Extracted %s rings (et=%d,eta=%d) with size: %r', 
                            name, 
                            etBin,
                            etaBin,
                            (npArray[etBin][etaBin].shape))
        # etaBin
      # etBin

createData = CreateData()

