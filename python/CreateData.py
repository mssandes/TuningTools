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
    """
    from TuningTools.FilterEvents import FilterType, Reference

    output            = kw.pop('output',        'tuningData'   )
    referenceSgn      = kw.pop('referenceSgn', Reference.Truth )
    referenceBkg      = kw.pop('referenceBkg', Reference.Truth )
    treePath          = kw.pop('treePath',          None       )
    l1EmClusCut       = kw.pop('l1EmClusCut',       None       )
    l2EtCut           = kw.pop('l2EtCut',           None       )
    nClusters         = kw.pop('nClusters',         None       )
    getRatesOnly      = kw.pop('getRatesOnly',      False      )
    if 'level' in kw: 
      self.level = kw.pop('level') # log output level
      self._filter.level = self.level
    
    self._logger.info('Extracting signal dataset information...')

    # List of operation arguments to be propagated
    kwargs = { 'treePath':     treePath,
              'l1EmClusCut':  l1EmClusCut,
              'l2EtCut':      l2EtCut,
              'nClusters':    nClusters,
              'getRatesOnly': getRatesOnly}

    if not getRatesOnly:
      npSgn  = self._filter(sgnFileList,
                            ringerOperation,
                            filterType = FilterType.Signal,
                            reference = referenceSgn,
                            **kwargs)
      self._logger.info('Extracted signal rings with size: %r',(npSgn.shape))
    else:
      sgn_rate, sgn_passed_counts, sgn_total_count  = self._filter(sgnFileList,
                                                                   ringerOperation,
                                                                   filterType = FilterType.Signal,
                                                                   reference = referenceSgn,
                                                                   **kwargs)

    self._logger.info('Extracting background dataset information...')
    if not getRatesOnly:
      npBkg = self._filter(bkgFileList, 
                           ringerOperation,
                           filterType = FilterType.Background,
                           reference = referenceBkg,
                           **kwargs)
      self._logger.info('Extracted background rings with size: %r',(npBkg.shape))
    else:
      bkg_rate, bkg_passed_counts, bkg_total_count  = self._filter(bkgFileList,
                                                                   ringerOperation,
                                                                   filterType = FilterType.Background,
                                                                   reference = referenceBkg,
                                                                   **kwargs)

    if not getRatesOnly:
      savedPath = TuningDataArchive( output,
                                     signal_rings = npSgn,
                                     background_rings = npBkg ).save()
      self._logger.info('Saved data file at path: %s', savedPath )
    else:
      for sgn_eff, sgn_passed, \
          bkg_eff, bkg_passed in zip(sgn_rate, sgn_passed_counts, \
                                     bkg_rate, bkg_passed_counts):
        self._logger.info('Det(%%): %.6f (%d/%d) | FA(%%): %.6f (%d/%d)',
                          sgn_eff, sgn_passed, sgn_total_count,
                          bkg_eff, bkg_passed, bkg_total_count)

createData = CreateData()

