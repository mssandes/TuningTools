__all__ = ['TuningDataArchieve', 'CreateData', 'createData']

from RingerCore import Logger, checkForUnusedVars, reshape, save, load, traverse, \
                       retrieve_kw, NotSet
from TuningTools.coreDef import retrieve_npConstants

npCurrent, _ = retrieve_npConstants()
import numpy as np

# FIXME This should be integrated into a class so that save could check if it
# is one instance of this base class and use its save method
class TuningDataArchieve( Logger ):
  """
  Context manager for Tuning Data archives

  Version 5: - Changes _rings for _patterns
  Version 4: - keeps the operation requested by user when creating data
  Version 3: - added eta/et bins compatibility
             - added benchmark efficiency information
             - improved fortran/C integration
             - loads only the indicated bins to memory
  Version 2: - started fotran/C order integration
  Version 1: - save compressed npz file
             - removed target information: classes are flaged as
               signal_rings/background_rings
  Version 0: - save pickle file with numpy data
  """

  _type = np.array('TuningData', dtype='|S10')
  _version = np.array(5)

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with TuningDataArchieve("/path/to/file", 
                           [eta_bin = None],
                           [et_bin = None]) as data:
      data['signal_patterns'] # access patterns from signal dataset 
      data['background_patterns'] # access patterns from background dataset
      data['benchmark_effs'] # access benchmark efficiencies

    When setting eta_bin or et_bin to None, the function will return data and
    efficiency for all bins instead of the just one selected.

    TuningDataArchieve( signal_patterns = np.array(...),
                       background_patterns = np.array(...),
                       eta_bins = np.array(...),
                       et_bins = np.array(...),
                       benchmark_effs = np.array(...), )
    """
    # Both
    Logger.__init__(self, kw)
    self._filePath                      = filePath
    # Saving
    self._signal_patterns               = kw.pop( 'signal_patterns',               npCurrent.fp_array([]) )
    self._background_patterns           = kw.pop( 'background_patterns',           npCurrent.fp_array([]) )
    self._eta_bins                      = kw.pop( 'eta_bins',                      npCurrent.fp_array([]) )
    self._et_bins                       = kw.pop( 'et_bins',                       npCurrent.fp_array([]) )
    self._signal_efficiencies           = kw.pop( 'signal_efficiencies',           None                   )
    self._background_efficiencies       = kw.pop( 'background_efficiencies',       None                   )
    self._signal_cross_efficiencies     = kw.pop( 'signal_cross_efficiencies',     None                   )
    self._background_cross_efficiencies = kw.pop( 'background_cross_efficiencies', None                   )
    self._toMatlab                      = kw.pop( 'toMatlab',                      False                  )
    # Loading
    self._eta_bin                       = kw.pop( 'eta_bin',                       None                   )
    self._et_bin                        = kw.pop( 'et_bin',                        None                   )
    self._operation                     = kw.pop( 'operation',                     None                   )
    checkForUnusedVars( kw, self._logger.warning )
    # Make some checks:
    if type(self._signal_patterns) != type(self._background_patterns):
      raise TypeError("Signal and background types do not match.")
    if type(self._signal_patterns) == list:
      if len(self._signal_patterns) != len(self._background_patterns) \
          or len(self._signal_patterns[0]) != len(self._background_patterns[0]):
        raise ValueError("Signal and background patterns lenghts do not match.")
    if type(self._eta_bins) is list: self._eta_bins=npCurrent.fp_array(self._eta_bins)
    if type(self._et_bins) is list: self._et_bins=npCurrent.fp_array(self._eta_bins)
    if self._eta_bins.size == 1 or self._eta_bins.size == 1:
      raise ValueError("Eta or et bins size are 1.")

  @property
  def filePath( self ):
    return self._filePath

  @property
  def signal_patterns( self ):
    return self._signal_patterns

  @property
  def background_patterns( self ):
    return self._background_patterns

  def getData( self ):
    from TuningTools.ReadData import RingerOperation
    kw_dict =  {
                'type': self._type,
             'version': self._version,
            'eta_bins': self._eta_bins,
             'et_bins': self._et_bins,
           'operation': RingerOperation.retrieve( self._operation ),
               }
    max_eta = self.__retrieve_max_bin(self._eta_bins)
    max_et = self.__retrieve_max_bin(self._et_bins)
    # Handle patterns:
    if max_eta is None and max_et is None:
      kw_dict['signal_patterns'] = self._signal_patterns
      kw_dict['background_patterns'] = self._background_patterns
    else:
      if max_eta is None: max_eta = 0
      if max_et is None: max_et = 0
      for et_bin in range( max_et + 1 ):
        for eta_bin in range( max_eta + 1 ):
          bin_str = self.__get_bin_str(et_bin, eta_bin) 
          sgn_key = 'signal_patterns_' + bin_str
          kw_dict[sgn_key] = self._signal_patterns[et_bin][eta_bin]
          bkg_key = 'background_patterns_' + bin_str
          kw_dict[bkg_key] = self._background_patterns[et_bin][eta_bin]
        # eta loop
      # et loop
    # Handle efficiencies
    from copy import deepcopy
    kw_dict['signal_efficiencies']           = deepcopy(self._signal_efficiencies)
    kw_dict['background_efficiencies']       = deepcopy(self._background_efficiencies)
    kw_dict['signal_cross_efficiencies']     = deepcopy(self._signal_cross_efficiencies)
    kw_dict['background_cross_efficiencies'] = deepcopy(self._background_cross_efficiencies)
    def efficiency_to_raw(d):
      for key, val in d.iteritems():
        for cData, idx, parent, _, _ in traverse(val):
          if parent is None:
            d[key] = cData.toRawObj()
          else:
            parent[idx] = cData.toRawObj()
    if self._signal_efficiencies and self._background_efficiencies:
      efficiency_to_raw(kw_dict['signal_efficiencies'])
      efficiency_to_raw(kw_dict['background_efficiencies'])
    if self._signal_cross_efficiencies and self._background_cross_efficiencies:
      efficiency_to_raw(kw_dict['signal_cross_efficiencies'])
      efficiency_to_raw(kw_dict['background_cross_efficiencies'])
    return kw_dict
  # end of getData

  def _toMatlabDump(self, data):
    import scipy.io as sio
    import pprint
    crossval = None
    kw_dict_aux = dict()

    # Retrieve efficiecies
    for key_eff in ['signal_','background_']:# sgn and bkg efficiencies
      key_eff+='efficiencies'
      kw_dict_aux[key_eff] = dict()
      for key_trigger in data[key_eff].keys():# Trigger levels
        kw_dict_aux[key_eff][key_trigger] = list()
        etbin = 0; etabin = 0
        for obj  in data[key_eff][key_trigger]: #Et
          kw_dict_aux[key_eff][key_trigger].append(list())
          for obj_  in obj: # Eta
            obj_dict = dict()
            obj_dict['count']  = obj_['_count'] if obj_.has_key('_count') else 0
            obj_dict['passed'] = obj_['_passed'] if obj_.has_key('_passed') else 0
            if obj_dict['count'] > 0:
              obj_dict['efficiency'] = obj_dict['passed']/float(obj_dict['count']) * 100
            else:
              obj_dict['efficiency'] = 0
            obj_dict['branch'] = obj_['_branch']
            kw_dict_aux[key_eff][key_trigger][etbin].append(obj_dict)
          etbin+=1 

    # Retrieve patterns 
    for key in data.keys():
      if 'rings' in key or \
         'patterns' in key:
        kw_dict_aux[key] = data[key]

    # Retrieve crossval
    crossVal = data['signal_cross_efficiencies']['L2CaloAccept'][0][0]['_crossVal']
    kw_dict_aux['crossVal'] = {
                                'nBoxes'          : crossVal['_nBoxes'],
                                'nSorts'          : crossVal['_nSorts'],
                                'nTrain'          : crossVal['_nTrain'],
                                'nTest'           : crossVal['_nTest'],
                                'nValid'          : crossVal['_nValid'],
                                'sort_boxes_list' : crossVal['_sort_boxes_list'],
                                }

    self._logger.info( 'Saving data to matlab...')
    sio.savemat(self._filePath+'.mat', kw_dict_aux)
  #end of matlabDump


  def save(self):
    self._logger.info( 'Saving data using following numpy flags: %r', npCurrent)
    data = self.getData()
    if self._toMatlab:  self._toMatlabDump(data)
    return save(data, self._filePath, protocol = 'savez_compressed')



  def __enter__(self):
    data = {'et_bins' : npCurrent.fp_array([]),
            'eta_bins' : npCurrent.fp_array([]),
            'operation' : None,
            'signal_patterns' : npCurrent.fp_array([]),
            'background_patterns' : npCurrent.fp_array([]),
            'signal_efficiencies' : {},
            'background_efficiencies' : {},
            'signal_efficiencies' : {},
            'background_efficiencies' : {},
            }
    npData = load( self._filePath )
    try:
      if type(npData) is np.ndarray:
        # Legacy type:
        data = reshape( npData[0] ) 
        target = reshape( npData[1] ) 
        self._signal_patterns, self._background_patterns = TuningDataArchieve.__separateClasses( data, target )
        data = {'signal_patterns' : self._signal_patterns, 
                'background_patterns' : self._background_patterns}
      elif type(npData) is np.lib.npyio.NpzFile:
        if npData['type'] != self._type:
          raise RuntimeError("Input file is not of TuningData type!")
        # Retrieve operation, if any
        if npData['version'] <= np.array(4):
          sgn_base_key = 'signal_rings'
          bkg_base_key = 'background_rings'
        else:
          sgn_base_key = 'signal_patterns'
          bkg_base_key = 'background_patterns'
        if npData['version'] == np.array(4):
          data['operation'] = npData['operation']
        else:
          from TuningTools.ReadData import RingerOperation
          data['operation'] = RingerOperation.EFCalo
        # Retrieve bins information, if any
        if npData['version'] <= np.array(4) and npData['version'] >= np.array(3): # self._version:
          eta_bins = npData['eta_bins'] if 'eta_bins' in npData else \
                     npCurrent.array([])
          et_bins  = npData['et_bins'] if 'et_bins' in npData else \
                     npCurrent.array([])
          self.__check_bins(eta_bins, et_bins)
          max_eta = self.__retrieve_max_bin(eta_bins)
          max_et = self.__retrieve_max_bin(et_bins)
          if self._eta_bin == self._et_bin == None:
            data['eta_bins'] = npCurrent.fp_array(eta_bins) if max_eta else npCurrent.fp_array([])
            data['et_bins'] = npCurrent.fp_array(et_bins) if max_et else npCurrent.fp_array([])
          else:
            data['eta_bins'] = npCurrent.fp_array([eta_bins[self._eta_bin],eta_bins[self._eta_bin+1]]) if max_eta else npCurrent.fp_array([])
            data['et_bins'] = npCurrent.fp_array([et_bins[self._et_bin],et_bins[self._et_bin+1]]) if max_et else npCurrent.fp_array([])
        # Retrieve data (and efficiencies):
        from TuningTools.ReadData import BranchEffCollector, BranchCrossEffCollector
        def retrieve_raw_efficiency(d, et_bins = None, eta_bins = None, cl = BranchEffCollector):
          if d is not None:
            if type(d) is np.ndarray:
              d = d.item()
            for key, val in d.iteritems():
              if et_bins is None or eta_bins is None:
                for cData, idx, parent, _, _ in traverse(val):
                  if parent is None:
                    d[key] = cl.fromRawObj(cData)
                  else:
                    parent[idx] = cl.fromRawObj(cData)
              else:
                if type(et_bins) == type(eta_bins) == list:
                  d[key] = []
                  for cEtBin, et_bin in enumerate(et_bins):
                    d[key].append([])
                    for eta_bin in eta_bins:
                      d[key][cEtBin].append(cl.fromRawObj(val[et_bin][eta_bin]))
                else:
                  d[key] = cl.fromRawObj(val[et_bins][eta_bins])
          return d
        if npData['version'] <= np.array(4) and npData['version'] >= np.array(3): # self._version:
          if self._eta_bin is None and max_eta is not None:
            self._eta_bin = range( max_eta + 1 )
          if self._et_bin is None and max_et is not None:
            self._et_bin = range( max_et + 1)
          if self._et_bin is None and self._eta_bin is None:
            data['signal_patterns'] = npData[sgn_base_key]
            data['background_patterns'] = npData[bkg_base_key]
            try:
              data['signal_efficiencies']           = retrieve_raw_efficiency(npData['signal_efficiencies'])
              data['background_efficiencies']       = retrieve_raw_efficiency(npData['background_efficiencies'])
            except KeyError:
              pass
            try:
              data['signal_cross_efficiencies']     = retrieve_raw_efficiency(npData['signal_cross_efficiencies'], BranchCrossEffCollector)
              data['background_cross_efficiencies'] = retrieve_raw_efficiency(npData['background_cross_efficiencies'], BranchCrossEffCollector)
            except KeyError:
              pass
          else:
            if self._eta_bin is None: self._eta_bin = 0
            if self._et_bin is None: self._et_bin = 0
            if type(self._eta_bin) == type(self._eta_bin) != list:
              bin_str = self.__get_bin_str(self._et_bin, self._eta_bin) 
              sgn_key = sgn_base_key + '_' + bin_str
              bkg_key = bkg_base_key + '_' + bin_str
              data['signal_patterns']                  = npData[sgn_key]
              data['background_patterns']              = npData[bkg_key]
              try:
                data['signal_efficiencies']           = retrieve_raw_efficiency(npData['signal_efficiencies'], 
                                                                                self._et_bin, self._eta_bin)
                data['background_efficiencies']       = retrieve_raw_efficiency(npData['background_efficiencies'],
                                                                                self._et_bin, self._eta_bin)
              except KeyError:
                pass
              try:
                data['signal_cross_efficiencies']     = retrieve_raw_efficiency(npData['signal_cross_efficiencies'],
                                                                                self._et_bin, self._eta_bin, BranchCrossEffCollector)
                data['background_cross_efficiencies'] = retrieve_raw_efficiency(npData['background_cross_efficiencies'],
                                                                                self._et_bin, self._eta_bin, BranchCrossEffCollector)
              except KeyError:
                pass
            else:
              if not type(self._eta_bin) is list:
                self._eta_bin = [self._eta_bin]
              if not type(self._et_bin) is list:
                self._et_bin = [self._et_bin]
              sgn_list = []
              bkg_list = []
              for et_bin in self._et_bin:
                sgn_local_list = []
                bkg_local_list = []
                for eta_bin in self._eta_bin:
                  bin_str = self.__get_bin_str(et_bin, eta_bin) 
                  sgn_key = sgn_base_key + '_' + bin_str
                  sgn_local_list.append(npData[sgn_key])
                  bkg_key = bkg_base_key + '_' + bin_str
                  bkg_local_list.append(npData[bkg_key])
                # Finished looping on eta
                sgn_list.append(sgn_local_list)
                bkg_list.append(bkg_local_list)
              # Finished retrieving data
              data['signal_patterns'] = sgn_list
              data['background_patterns'] = bkg_list
              indexes = self._eta_bin[:]; indexes.append(self._eta_bin[-1]+1)
              data['eta_bins'] = eta_bins[indexes]
              indexes = self._et_bin[:]; indexes.append(self._et_bin[-1]+1)
              data['et_bins'] = et_bins[indexes]
              try:
                data['signal_efficiencies']           = retrieve_raw_efficiency(npData['signal_efficiencies'], 
                                                                                self._et_bin, self._eta_bin)
                data['background_efficiencies']       = retrieve_raw_efficiency(npData['background_efficiencies'], 
                                                                                self._et_bin, self._eta_bin)
              except KeyError:
                pass
              try:
                data['signal_cross_efficiencies']     = retrieve_raw_efficiency(npData['signal_cross_efficiencies'], 
                                                                                self._et_bin, self._eta_bin, 
                                                                                BranchCrossEffCollector)
                data['background_cross_efficiencies'] = retrieve_raw_efficiency(npData['background_cross_efficiencies'], 
                                                                                self._et_bin, self._eta_bin, 
                                                                                BranchCrossEffCollector)
              except KeyError:
                pass
        elif npData['version'] <= np.array(2): # self._version:
          data['signal_patterns']     = npData['signal_patterns']
          data['background_patterns'] = npData['background_patterns']
        else:
          raise RuntimeError("Unknown file version!")
      elif isinstance(npData, dict) and 'type' in npData:
        raise RuntimeError("Attempted to read archive of type: %s_v%d" % (npData['type'],
                                                                          npData['version']))
      else:
        raise RuntimeError("Object on file is of unkown type.")
    except RuntimeError, e:
      raise RuntimeError(("Couldn't read TuningDataArchieve('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))
    data['eta_bins'] = npCurrent.fix_fp_array(data['eta_bins'])
    data['et_bins'] = npCurrent.fix_fp_array(data['et_bins'])
    # Now that data is defined, check if numpy information fits with the
    # information representation we need:
    if type(data['signal_patterns']) is list:
      for cData, idx, parent, _, _ in traverse(data['signal_patterns'], (list,tuple,np.ndarray), 2):
        cData = npCurrent.fix_fp_array(cData)
        parent[idx] = cData
      for cData, idx, parent, _, _ in traverse(data['background_patterns'], (list,tuple,np.ndarray), 2):
        cData = npCurrent.fix_fp_array(cData)
        parent[idx] = cData
    else:
      data['signal_patterns'] = npCurrent.fix_fp_array(data['signal_patterns'])
      data['background_patterns'] = npCurrent.fix_fp_array(data['background_patterns'])
    return data
    
  def __exit__(self, exc_type, exc_value, traceback):
    pass

  def nEtBins(self):
    """
      Return maximum eta bin index. If variable is not dependent on bin, return none.
    """
    et_max = self.__max_bin('et_bins') 
    return et_max + 1 if et_max is not None else et_max

  def nEtaBins(self):
    """
      Return maximum eta bin index. If variable is not dependent on bin, return none.
    """
    eta_max = self.__max_bin('eta_bins')
    return eta_max + 1 if eta_max is not None else eta_max

  def __max_bin(self, var):
    """
      Return maximum dependent bin index. If variable is not dependent on bin, return none.
    """
    npData = load( self._filePath )
    try:
      if type(npData) is np.ndarray:
        return None
      elif type(npData) is np.lib.npyio.NpzFile:
        if npData['type'] != self._type:
          raise RuntimeError("Input file is not of TuningData type!")
        arr  = npData[var] if var in npData else npCurrent.array([])
        return self.__retrieve_max_bin(arr)
    except RuntimeError, e:
      raise RuntimeError(("Couldn't read TuningDataArchieve('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))

  def __retrieve_max_bin(self, arr):
    """
    Return  maximum dependent bin index. If variable is not dependent, return None.
    """
    max_size = arr.size - 2
    return max_size if max_size >= 0 else None

  def __check_bins(self, eta_bins, et_bins):
    """
    Check if self._eta_bin and self._et_bin are ok, through otherwise.
    """
    max_eta = self.__retrieve_max_bin(eta_bins)
    max_et = self.__retrieve_max_bin(et_bins)
    # Check if eta/et bin requested can be retrieved.
    errmsg = ""
    if self._eta_bin > max_eta:
      errmsg += "Cannot retrieve eta_bin(%d) from eta_bins (%r). %s" % (self._eta_bin, eta_bins, 
          ('Max bin is: ' + str(max_eta) + '. ') if max_eta is not None else ' Cannot use eta bins.')
    if self._et_bin > max_et:
      errmsg += "Cannot retrieve et_bin(%d) from et_bins (%r). %s" % (self._et_bin, et_bins,
          ('Max bin is: ' + str(max_et) + '. ') if max_et is not None else ' Cannot use E_T bins. ')
    if errmsg:
      raise ValueError(errmsg)

  def __get_bin_str(self, et_bin, eta_bin):
    return 'etBin_' + str(et_bin) + '_etaBin_' + str(eta_bin)

  @classmethod
  def __separateClasses( cls, data, target ):
    """
    Function for dealing with legacy data.
    """
    sgn = data[np.where(target==1)]
    bkg = data[np.where(target==-1)]
    return sgn, bkg2


class CreateData(Logger):

  def __init__( self, logger = None ):
    Logger.__init__( self, logger = logger )
    from TuningTools.ReadData import readData
    self._reader = readData

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
        - pattern_oFile ['tuningData']: Path for saving the output file
        - efficiency_oFile ['tuningData']: Path for saving the output file efficiency
        - referenceSgn [Reference.Truth]: Filter reference for signal dataset
        - referenceBkg [Reference.Truth]: Filter reference for background dataset
        - treePath [<Same as ReadData default>]: set tree name on file, this may be set to
          use different sources then the default.
        - efficiencyTreePath [None]: Sets tree path for retrieving efficiency
              benchmarks.
            When not set, uses treePath as tree.
        - nClusters [<Same as ReadData default>]: Number of clusters to export. If set to None, export
            full PhysVal information.
        - getRatesOnly [<Same as ReadData default>]: Do not create data, but retrieve the efficiency
            for benchmark on the chosen operation.
        - etBins [<Same as ReadData default>]: E_T bins  (GeV) where the data should be segmented
        - etaBins [<Same as ReadData default>]: eta bins where the data should be segmented
        - ringConfig [<Same as ReadData default>]: A list containing the number of rings available in the data
          for each eta bin.
        - crossVal [<Same as ReadData default>]: Whether to measure benchmark efficiency splitting it
          by the crossVal-validation datasets
        - extractDet [<Same as ReadData default>]: Which detector to export (use Detector enumeration).
        - standardCaloVariables [<Same as ReadData default>]: Whether to extract standard track variables.
        - useTRT [<Same as ReadData default>]: Whether to export TRT information when dumping track
          variables.
        - toMatlab [False]: Whether to also export data to matlab format.
    """
    """
    # TODO Add a way to create new reference files setting operation points as
    # desired. A way to implement this is:
    #"""
    #    - tuneOperationTargets [['Loose', 'Pd' , #looseBenchmarkRef],
    #                            ['Medium', 'SP'],
    #                            ['Tight', 'Pf' , #tightBenchmarkRef]]
    #      The tune operation targets which should be used for this tuning
    #      job. The strings inputs must be part of the ReferenceBenchmark
    #      enumeration.
    #      Instead of an enumeration string (or the enumeration itself),
    #      you can set it directly to a value, e.g.: 
    #        [['Loose97', 'Pd', .97,],['Tight005','Pf',.005]]
    #      This can also be set using a string, e.g.:
    #        [['Loose97','Pd' : '.97'],['Tight005','Pf','.005']]
    #      , which may contain a percentage symbol:
    #        [['Loose97','Pd' : '97%'],['Tight005','Pf','0.5%']]
    #      When set to None, the Pd and Pf will be set to the value of the
    #      benchmark correspondent to the operation level set.
    #"""
    from TuningTools.ReadData import FilterType, Reference, Dataset, BranchCrossEffCollector
    pattern_oFile         = retrieve_kw(kw, 'pattern_oFile',         'tuningData'      )
    efficiency_oFile      = retrieve_kw(kw, 'efficiency_oFile',      NotSet            )
    referenceSgn          = retrieve_kw(kw, 'referenceSgn',          Reference.Truth   )
    referenceBkg          = retrieve_kw(kw, 'referenceBkg',          Reference.Truth   )
    treePath              = retrieve_kw(kw, 'treePath',              NotSet            )
    efficiencyTreePath    = retrieve_kw(kw, 'efficiencyTreePath',    NotSet            )
    l1EmClusCut           = retrieve_kw(kw, 'l1EmClusCut',           NotSet            )
    l2EtCut               = retrieve_kw(kw, 'l2EtCut',               NotSet            )
    efEtCut               = retrieve_kw(kw, 'efEtCut',               NotSet            )
    offEtCut              = retrieve_kw(kw, 'offEtCut',              NotSet            )
    nClusters             = retrieve_kw(kw, 'nClusters',             NotSet            )
    getRatesOnly          = retrieve_kw(kw, 'getRatesOnly',          NotSet            )
    etBins                = retrieve_kw(kw, 'etBins',                NotSet            )
    etaBins               = retrieve_kw(kw, 'etaBins',               NotSet            )
    ringConfig            = retrieve_kw(kw, 'ringConfig',            NotSet            )
    crossVal              = retrieve_kw(kw, 'crossVal',              NotSet            )
    extractDet            = retrieve_kw(kw, 'extractDet',            NotSet            )
    standardCaloVariables = retrieve_kw(kw, 'standardCaloVariables', NotSet            )
    useTRT                = retrieve_kw(kw, 'useTRT',                NotSet            )
    toMatlab              = retrieve_kw(kw, 'toMatlab',              False             )
    if 'level' in kw: 
      self.level = kw.pop('level') # log output level
      self._reader.level = self.level
    checkForUnusedVars( kw, self._logger.warning )
    # Make some checks:
    if type(treePath) is not list:
      treePath = [treePath]
    if len(treePath) == 1:
      treePath.append( treePath[0] )
    if efficiencyTreePath in (NotSet, None):
      efficiencyTreePath = treePath
    if type(efficiencyTreePath) is not list:
      efficiencyTreePath = [efficiencyTreePath]
    if len(efficiencyTreePath) == 1:
      efficiencyTreePath.append( efficiencyTreePath[0] )
    if etaBins is None: etaBins = npCurrent.fp_array([])
    if etBins is None: etBins = npCurrent.fp_array([])
    if type(etaBins) is list: etaBins=npCurrent.fp_array(etaBins)
    if type(etBins) is list: etBins=npCurrent.fp_array(etBins)
    if efficiency_oFile in (NotSet, None):
      efficiency_oFile = pattern_oFile
      listOfEndings = ['.npz','.npy','.npz.tgz','.npy.tgz','.npz.gz','.npy.gz']
      addupStr = '-eff'
      for ending in listOfEndings:
        if efficiency_oFile.endswith(ending):
          efficiency_oFile = efficiency_oFile.strip(ending) + addupStr + ending
          break
      else:
        efficiency_oFile += addupStr

    nEtBins  = len(etBins)-1 if not etBins is None else 1
    nEtaBins = len(etaBins)-1 if not etaBins is None else 1
    #useBins = True if nEtBins > 1 or nEtaBins > 1 else False

    #FIXME: problems to only one bin. print eff doest work as well
    useBins=True

    # List of operation arguments to be propagated
    kwargs = { 'l1EmClusCut':           l1EmClusCut,
               'l2EtCut':               l2EtCut,
               'efEtCut':               efEtCut,
               'offEtCut':              offEtCut,
               'nClusters':             nClusters,
               'getRatesOnly':          getRatesOnly,
               'etBins':                etBins,
               'etaBins':               etaBins,
               'ringConfig':            ringConfig,
               'crossVal':              crossVal,
               'extractDet':            extractDet,
               'standardCaloVariables': standardCaloVariables,
               'useTRT':                useTRT,
               }

    if efficiencyTreePath[0] == treePath[0]:
      self._logger.info('Extracting signal dataset information...')
      npSgn, sgnEff, sgnCrossEff  = self._reader(sgnFileList,
                                                 ringerOperation,
                                                 filterType = FilterType.Signal,
                                                 reference = referenceSgn,
                                                 treePath = treePath[0],
                                                 **kwargs)
      if npSgn.size: self.__printShapes(npSgn, 'Signal')
    else:
      if not getRatesOnly:
        self._logger.info("Extracting signal data...")
        npSgn, _, _  = self._reader(sgnFileList,
                                    ringerOperation,
                                    filterType = FilterType.Signal,
                                    reference = referenceSgn,
                                    treePath = treePath[0],
                                    getRates = False,
                                    **kwargs)
        self.__printShapes(npSgn, 'Signal')
      else:
        self._logger.warning("Informed treePath was ignored and used only efficiencyTreePath.")

      self._logger.info("Extracting signal efficiencies...")
      _, sgnEff, sgnCrossEff  = self._reader(sgnFileList,
                                             ringerOperation,
                                             filterType = FilterType.Signal,
                                             reference = referenceSgn,
                                             treePath = efficiencyTreePath[0],
                                             **kwargs)

    if efficiencyTreePath[1] == treePath[1]:
      self._logger.info('Extracting background dataset information...')
      npBkg, bkgEff, bkgCrossEff  = self._reader(bkgFileList,
                                                 ringerOperation,
                                                 filterType = FilterType.Background,
                                                 reference = referenceBkg,
                                                 treePath = treePath[1],
                                                 **kwargs)
    else:
      if not getRatesOnly:
        self._logger.info("Extracting background data...")
        npBkg, _, _  = self._reader(bkgFileList,
                                    ringerOperation,
                                    filterType = FilterType.Background,
                                    reference = referenceBkg,
                                    treePath = treePath[1],
                                    getRates = False,
                                    **kwargs)
      else:
        self._logger.warning("Informed treePath was ignored and used only efficiencyTreePath.")

      self._logger.info("Extracting background efficiencies...")
      _, bkgEff, bkgCrossEff  = self._reader(bkgFileList,
                                             ringerOperation,
                                             filterType = FilterType.Background,
                                             reference = referenceBkg,
                                             treePath = efficiencyTreePath[1],
                                             **kwargs)
    if npBkg.size: self.__printShapes(npBkg, 'Background')

    if not getRatesOnly:
      savedPath = TuningDataArchieve(pattern_oFile,
                                     signal_patterns = npSgn,
                                     background_patterns = npBkg,
                                     eta_bins = etaBins,
                                     et_bins = etBins,
                                     signal_efficiencies = sgnEff,
                                     background_efficiencies = bkgEff,
                                     signal_cross_efficiencies = sgnCrossEff,
                                     background_cross_efficiencies = bkgCrossEff,
                                     operation = ringerOperation,
                                     toMatlab = toMatlab,
                                     ).save()
      self._logger.info('Saved data file at path: %s', savedPath )

    for etBin in range(nEtBins):
      for etaBin in range(nEtaBins):
        self.plotMeanRings(npSgn[etBin][etaBin],npBkg[etBin][etaBin],etBin,etaBin)
        for key in sgnEff.iterkeys():
          sgnEffBranch = sgnEff[key][etBin][etaBin] if useBins else sgnEff[key]
          bkgEffBranch = bkgEff[key][etBin][etaBin] if useBins else bkgEff[key]
          self._logger.info('Efficiency for %s: Det(%%): %s | FA(%%): %s', 
                            sgnEffBranch.printName,
                            sgnEffBranch.eff_str(),
                            bkgEffBranch.eff_str() )
          if crossVal not in (None, NotSet):
            for ds in BranchCrossEffCollector.dsList:
              try:
                sgnEffBranchCross = sgnCrossEff[key][etBin][etaBin] if useBins else sgnEff[key]
                bkgEffBranchCross = bkgCrossEff[key][etBin][etaBin] if useBins else bkgEff[key]
                self._logger.info( '%s_%s: Det(%%): %s | FA(%%): %s',
                                  Dataset.tostring(ds),
                                  sgnEffBranchCross.printName,
                                  sgnEffBranchCross.eff_str(ds),
                                  bkgEffBranchCross.eff_str(ds))
              except KeyError, e:
                pass
        # for eff
      # for eta
    # for et
  # end __call__

  def Signal(self,signal,et,eta):

    nSamplesSignal = signal.shape[0]

    y = (np.zeros(100))
    x=(np.arange(100)+1.0)

    for i in range(100):
      aux = 0
      for j in range(nSamplesSignal):
        aux = aux + signal[j][i]
      y[i]= aux/(j+1.0)
    n = "Signal Ring Et = %d Eta = %d " %(et,eta)
    xn = "Ring"
    yn = "Energy"

    return x,y,n,xn,yn

  def Back (self,background,et,eta):

    y = (np.zeros(100))
    x=(np.arange(100)+1.0)
    nSamplesBackground = background.shape[0]

    for k in range(100):
      aux = 0
      for l in range(nSamplesBackground):
        aux = aux + background[l][k]
      y[k] = aux/(l + 1.0)

    return x,y
    
  def plotMeanRings(self,signal,background,et,eta):
    from ROOT import TGraph,TCanvas
    if (signal is not None) and (background is None):

      lista = self.Signal(signal,et,eta)
      c1 = TCanvas("plot_patternsMean_et%d_eta%d" % (0,0), "a",0,0,600,400)
      c1.SetGrid()
      c1.cd()
      sg= TGraph (100,lista[0],lista[1])
      sg.SetTitle(lista[2])
      sg.GetXaxis().SetTitle(lista[3])
      sg.GetYaxis().SetTitle(lista[4])

      sg.GetYaxis().SetTitleOffset(1.3)
      sg.SetFillColor(34)

      sg.Draw("AB")

    if (signal is None) and (background is not None):

      lista = self.Back (background,et,eta)
      c1 = TCanvas("plot_patternsMean_et%d_eta%d" % (0,0), "a",0,0,600,400)
      c1.cd()

      bck = TGraph (100,lista[0],lista[1])
      bck.SetTitle("Background Ring Et = %d Eta = %d " %(et,eta))
      bck.GetXaxis().SetTitle("Ring")
      bck.GetYaxis().SetTitle("Energy")
      bck.SetFillColor(35)
      bck.Draw("AB")

    if (signal is not None) and (background is not None):
      lista = self.Signal(signal,et,eta)
      list1 = self.Back(background,et,eta)

      c1 = TCanvas("plot_patternsMean_et%d_eta%d" % (0,0), "a",0,0,800,400)
      c1.Divide(2,1)
      c1.cd(1).SetGrid()
      sg= TGraph (100,lista[0],lista[1])
      sg.SetTitle(lista[2])
      sg.GetXaxis().SetTitle(lista[3])
      sg.GetYaxis().SetTitle(lista[4])
      sg.GetYaxis().SetTitleOffset(1.9)
      sg.SetFillColor(34)
      sg.Draw("AB")

      c1.cd(2).SetGrid()
      bck = TGraph (100,list1[0],list1[1])
      bck.SetTitle("Background Ring Et = %d Eta = %d " %(et,eta))
      bck.GetXaxis().SetTitle("Ring")
      bck.GetYaxis().SetTitle("Energy")
      bck.GetYaxis().SetTitleOffset(2)
      bck.SetFillColor(2)
      bck.Draw("AB")

    c1.SaveAs('plot_patternsMean_et%d_eta%d.png' % (et, eta))
    c1.Close()


  def __printShapes(self, npArray, name):
    "Print numpy shapes"
    if not npArray.dtype.type is np.object_:
      self._logger.info('Extracted %s patterns with size: %r',name, (npArray.shape))
    else:
      shape = npArray.shape
      for etBin in range(shape[0]):
        for etaBin in range(shape[1]):
          self._logger.info('Extracted %s patterns (et=%d,eta=%d) with size: %r', 
                            name, 
                            etBin,
                            etaBin,
                            (npArray[etBin][etaBin].shape if npArray[etBin][etaBin] is not None else ("None")))
        # etaBin
      # etBin

createData = CreateData()

