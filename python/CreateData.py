from RingerCore.Logger import Logger
import numpy as np

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
            containing the FastNet TTree for the signal dataset
        - bkgFileList: A python list or a comma separated list of the root files
            containing the FastNet TTree for the background dataset
        - ringerOperation: Set Operation type to be used by the filter
      Optional arguments:
        - output ['tuningData']: Name for the output file
        - referenceSgn [Reference.Truth]: Filter reference for signal dataset
        - referenceBkg [Reference.Truth]: Filter reference for background dataset
        - treePath: Sets tree path on file to be used as the TChain. The default
            value depends on the operation. If set to None, it will be set to 
            the default value.
    """
    from TuningTools.FilterEvents import FilterType, Reference
    from RingerCore.FileIO import save

    output            = kw.pop('output',        'tuningData'   )
    referenceSgn      = kw.pop('referenceSgn', Reference.Truth )
    referenceBkg      = kw.pop('referenceBkg', Reference.Truth )
    treePath          = kw.pop('treePath',          None       )
    l1EmClusCut       = kw.pop('l1EmClusCut',       None       )
    nClusters         = kw.pop('nClusters',         None       )
    if 'level' in kw: 
      self.level = kw.pop('level') # log output level
      self._filter.level = self.level
    
    self._logger.info('Extracting signal dataset rings...')
    npSgn  = self._filter(sgnFileList,
                          ringerOperation,
                          filterType = FilterType.Signal,
                          reference = referenceSgn, 
                          treePath = treePath,
                          l1EmClusCut = l1EmClusCut,
                          nClusters = nClusters)
  
    self._logger.info('Extracted signal rings with size: %r',(npSgn.shape))

    self._logger.info('Extracting background dataset rings...')
    npBkg = self._filter(bkgFileList, 
                         ringerOperation,
                         filterType = FilterType.Background, 
                         reference = referenceBkg,
                         treePath = treePath,
                         l1EmClusCut = l1EmClusCut,
                         nClusters = nClusters)

    self._logger.info('Extracted background rings with size: %r',(npBkg.shape))
    objSave = {'type' : np.array('TuningData'),
               'version' : np.array(1),
               'signal_rings' : npSgn,
               'background_rings' : npBkg }

    savedPlace = save(objSave, output, protocol = 'savez_compressed')

createData = CreateData()
    
