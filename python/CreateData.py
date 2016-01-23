from RingerCore.Logger import Logger
from RingerCore.util   import checkForUnusedVars, reshape
from RingerCore.FileIO import save, load
from TuningTools.npdef import npCurrent
import numpy as np

# FIXME This should be integrated into a class so that save could check if it
# is one instance of this base class and use its save method
class TuningDataArchive( Logger ):
  """
  Context manager for Tuning Data archives

  Version 3: - added eta/et bins compatibility
             - added benchmark efficiency information
             - improved fortran/C integration
  Version 2: - started fotran/C order integration
  Version 1: - save compressed npz file
             - removed target information: classes are flaged as
               signal_rings/background_rings
  Version 0: - save pickle file with numpy data
  """

  _type = np.array('TuningData', dtype='|S10')
  _version = np.array(3)

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with TuningDataArchive("/path/to/file", 
                           [eta_bin = 0],
                           [et_bin = 0]) as data, eff:
      BLOCK

    When setting eta_bin or et_bin to None, the function will return data and
    efficiency for all bins instead of the just one selected.

    TuningDataArchive( signal_rings = np.array(...),
                       background_rings = np.array(...),
                       eta_bins = np.array(...),
                       et_bins = np.array(...),
                       benchmark_effs = np.array(...), )
    """
    Logger.__init__(self, kw)
    self._filePath         = filePath
    self._signal_rings     = kw.pop( 'signal_rings', npCurrent.array([]))
    self._background_rings = kw.pop( 'background_rings', npCurrent.array([]))
    self._eta_bins         = kw.pop( 'eta_bins', npCurrent.array([]))
    self._et_bins          = kw.pop( 'et_bins',  npCurrent.array([]))
    self._eta_bin          = kw.pop( 'eta_bin', None )
    self._et_bin           = kw.pop( 'et_bin', None )
    #self._benchmark_effs   = kw.pop( 'benchmark_effs', npCurrent.array([],dtype=np.object))
    checkForUnusedVars( kw, self._logger.warning )

  @property
  def filePath( self ):
    return self._filePath

  @property
  def signal_rings( self ):
    return self._signal_rings

  @property
  def background_rings( self ):
    return self._background_rings

  def getData( self ):
    return {'type' : self._type,
            'version' : self._version,
            'signal_rings' : self._signal_rings,
            'background_rings' : self._background_rings,
            'eta_bins' : self._eta_bins,
            'et_bins' : self._et_bins, }
            #'efficiency' : }

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
        data = [self._signal_rings, self._background_rings]
      elif type(npData) is np.lib.npyio.NpzFile:
        if npData['type'] != self._type:
          raise RuntimeError("Input file is not of TuningData type!")
        if npData['version'] == np.array(3): # self._version:
          data = [npData['signal_rings'], npData['background_rings']]
          eta_bins = npData['eta_bins']; et_bins = npData['et_bins']
        elif npData['version'] <= np.array(2): # self._version:
          data = [npData['signal_rings'], npData['background_rings']]
          eta_bins = npCurrent.array([]); et_bins = npCurrent.array([]);
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
      ## Treat case where eta bin and et bin were set to 0 and there is no binning
    #if not eta_bins.size and not et_bins.size \
    #    and self._eta_bin == 0 and self._et_bin == 0:
    #  self._eta_bin = None
    #  self._et_bin = None
    # Check if eta/et bin requested can be retrieved.
    errmsg = ""
    if (self._eta_bin and not eta_bins.size > 1) or (self._eta_bin + 1 >= eta_bins.size):
      errmsg += "Cannot retrieve eta_bin(%d) as eta_bins (%r) max bin is (%d). " % (self._eta_bin, eta_bins, eta_bins.size - 2 if eta_bins.size - 2 > 0 else 0)
    if (self._et_bin and not et_bins.size > 1) or (self._et_bin + 1 >= et_bins.size):
      errmsg += "Cannot retrieve et_bin(%d) as et_bins (%r) max bin is (%d).  " % (self._et_bin, et_bins, et_bins.size - 2 if et_bins.size - 2 > 0 else 0)
    if errmsg:
      raise ValueError(errmsg)
    # Ok, all good from now on. Only need to test if user forgot to change index:
    if self._eta_bin is not None or self._et_bin is not None:
      # Handle cases where user didn't specify eta/et bins b/c it isn't binned
      if self._eta_bin is None and ( self._et_bin is not None and et_bins.size <= 1 ):
        self._et_bin = 0
      if self._et_bin is None and ( self._eta_bin is not None and eta_bins.size <= 1 ):
        self._eta_bin = 0
      # Handle no dependency but bins specificied as 0
      if not eta_bins.size and not et_bins.size \
          and not self._eta_bin and not self._et_bin:
        # data will still be data
        pass
      # Here only binned cases survived
      self._logger.info( 'Choosing et_bin%d%s and eta_bin%d%s)', 
          self._et_bin,
          (' (%g->%g)' % (et_bins[self._et_bin], et_bins[self._et_bin+1])) if et_bins.size else '',
          self._eta_bin,
          (' (%g->%g)' % (eta_bins[self._eta_bin], eta_bins[self._eta_bin+1])) if eta_bins.size else '',
          )
      data = [cData[self._et_bin][self._eta_bin] for cData in data]
    # Now that data is defined, check if numpy information fits with the
    # information representation we need:
    from RingerCore.util import traverse
    for cData, idx, parent, _, _ in traverse(data, (list,tuple,np.ndarray), 1):
      #print cData, idx, parent
      if cData.dtype != npCurrent.fp_dtype:
        self._logger.info( 'Changing data type from %s to %s', cData.dtype, npCurrent.fp_dtype)
        cData = cData.astype( npCurrent.fp_dtype )
        parent[idx] = cData
      if cData.flags['F_CONTIGUOUS'] != npCurrent.isfortran:
        # Transpose data to either C or Fortran representation...
        self._logger.info( 'Changing data fortran order from %s to %s', 
                            cData.flags['F_CONTIGUOUS'], 
                            npCurrent.isfortran)
        cData = cData.T
        data[idx] = cData
    # for data
    data = tuple(data)
    return data
    
  def __exit__(self, exc_type, exc_value, traceback):
    pass

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
            When it is different for signal and background, you can inform a list
            which will be passed to them, respectively.
        - nClusters [None]: Number of clusters to export. If set to None, export
            full PhysVal information.
        - getRatesOnly [False]: Do not create data, but retrieve the efficiency
            for benchmark on the chosen operation.
        - etBins [None]: E_T bins where the data should be segmented
        - etaBins [None]: eta bins where the data should be segmented
        - ringConfig [100]: A list containing the number of rings available in the data
          for each eta bin.
        - crossVal [None]: Whether to measure benchmark efficiency separing it
          by the crossVal-validation datasets
    """
    from TuningTools.FilterEvents import FilterType, Reference, Dataset, BranchCrossEffCollector
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
    ringConfig   = kw.pop('ringConfig',         None       )
    crossVal     = kw.pop('crossVal',           None       )
    if ringConfig is None:
      ringConfig = [100]*(len(etaBins)-1) if etaBins else [100]
    if 'level' in kw: 
      self.level = kw.pop('level') # log output level
      self._filter.level = self.level
    if type(treePath) is not list:
      treePath = [treePath]
    if len(treePath) == 1:
      treePath.append( treePath[0] )
    checkForUnusedVars( kw, self._logger.warning )

    if etaBins is None: etaBins = npCurrent.fp_array([])
    if etBins is None: etBins = npCurrent.fp_array([])

    nEtBins  = len(etBins)-1 if not etBins is None else 1
    nEtaBins = len(etaBins)-1 if not etaBins is None else 1
    useBins = True if nEtBins > 1 or nEtaBins > 1 else False

    self._logger.info('Extracting signal dataset information...')

    # List of operation arguments to be propagated
    kwargs = { 'l1EmClusCut':  l1EmClusCut,
               'l2EtCut':      l2EtCut,
               'offEtCut':     offEtCut,
               'nClusters':    nClusters,
               'getRatesOnly': getRatesOnly,
               'etBins':       etBins,
               'etaBins':      etaBins,
               'ringConfig':   ringConfig,
               'crossVal':     crossVal, }

    npSgn, sgnEffList, sgnCrossEffList  = self._filter(sgnFileList,
                                                       ringerOperation,
                                                       filterType = FilterType.Signal,
                                                       reference = referenceSgn,
                                                       treePath = treePath[0],
                                                       **kwargs)
    if npSgn.size: self.__printShapes(npSgn,'Signal')

    self._logger.info('Extracting background dataset information...')
    npBkg, bkgEffList, bkgCrossEffList = self._filter(bkgFileList, 
                                                      ringerOperation,
                                                      filterType = FilterType.Background,
                                                      reference = referenceBkg,
                                                      treePath = treePath[1],
                                                      **kwargs)
    if npBkg.size: self.__printShapes(npBkg,'Background')

    if not getRatesOnly:
      savedPath = TuningDataArchive( output,
                                     signal_rings = npSgn,
                                     background_rings = npBkg,
                                     eta_bins = etaBins,
                                     et_bins = etBins ).save()
      self._logger.info('Saved data file at path: %s', savedPath )

    for idx in range(len(sgnEffList)) if not useBins else \
               range(len(sgnEffList[0][0])):
      for etBin in range(nEtBins):
        for etaBin in range(nEtaBins):
          sgnEff = sgnEffList[etBin][etaBin][idx]
          bkgEff = bkgEffList[etBin][etaBin][idx]
          self._logger.info('Efficiency for %s: Det(%%): %s | FA(%%): %s', 
                            sgnEff.name,
                            sgnEff.eff_str(),
                            bkgEff.eff_str() )
          if crossVal is not None:
            for ds in BranchCrossEffCollector.dsList:
              try:
                sgnEffCross = sgnCrossEffList[etBin][etaBin][idx]
                bkgEffCross = bkgCrossEffList[etBin][etaBin][idx]
                self._logger.info( '%s_%s: Det(%%): %s | FA(%%): %s',
                                  Dataset.tostring(ds),
                                  sgnEffCross.name,
                                  sgnEffCross.eff_str(ds),
                                  bkgEffCross.eff_str(ds))
              except KeyError, e:
                pass
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

