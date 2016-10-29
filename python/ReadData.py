__all__ = ['BranchCrossEffCollector','BranchEffCollector', 'ReadData',
    'FilterType',  'Reference', 'RingerOperation', 'Target', 'readData',
    'BaseInfo']

from RingerCore import EnumStringification, Logger, LoggingLevel, traverse, \
                       stdvector_to_list, checkForUnusedVars, expandFolders, \
                       RawDictStreamer, RawDictStreamable, RawDictCnv, retrieve_kw, \
                       csvStr2List, NotSet, progressbar
from TuningTools.coreDef import retrieve_npConstants
npCurrent, _ = retrieve_npConstants()
from collections import OrderedDict
import numpy as np
from copy import deepcopy

class RingerOperation(EnumStringification):
  """
    Select which framework ringer will operate
    - Positive values for Online operation; and 
    - Negative values for Offline operation.
  """
  _ignoreCase = True

  Offline_All = -9
  Offline_CutBased_Tight = -8
  Offline_CutBased_Medium = -7
  Offline_CutBased_Loose = -6
  Offline_CutBased = -5
  Offline_LH_Tight = -4
  Offline_LH_Medium = -3
  Offline_LH_Loose = -2
  Offline_LH = -1
  Offline = -1
  L2  = 1
  EF = 2
  L2Calo  = 3
  EFCalo  = 4
  HLT  = 5

  @classmethod
  def branchName(cls, val):
    # FIXME This should be a dict
    val = cls.retrieve( val )
    if val == cls.L2Calo:
      return 'L2CaloAccept'
    elif val == cls.L2:
      return 'L2ElAccept'
    elif val == cls.EFCalo:
      return 'EFCaloAccept'
    elif val == cls.HLT:
      return 'HLTAccept'
    elif val == cls.Offline_LH_Loose:
      return 'LHLoose'
    elif val == cls.Offline_LH_Medium:
      return 'LHMedium'
    elif val == cls.Offline_LH_Tight:
      return 'LHTight'
    elif val == cls.Offline_LH:
      return ['LHLoose', 'LHMedium', 'LHTight']
    elif val == cls.Offline_CutBased_Loose:
      return 'CutIDLoose'
    elif val == cls.Offline_CutBased_Medium:
      return 'CutIDMedium'
    elif val == cls.Offline_CutBased_Tight:
      return 'CutIDTight'
    elif val == cls.Offline_CutBased:
      return ['CutIDLoose', 'CutIDMedium', 'CutIDTight']
    elif val == cls.Offline_All:
      return [ 'LHLoose',    'LHMedium',    'LHTight',   \
               'CutIDLoose', 'CutIDMedium', 'CutIDTight' ]

class Reference(EnumStringification):
  """
    Reference for training algorithm
  """
  _ignoreCase = True

  Truth = -1
  AcceptAll = 0
  Off_CutID = 1
  Off_Likelihood = 2
  
class FilterType(EnumStringification):
  """
    Enumeration if selection event type w.r.t reference
  """
  _ignoreCase = True

  DoNotFilter = 0
  Background = 1
  Signal = 2

class Target(EnumStringification):
  """ 
    Holds the target value for the discrimination method
  """
  _ignoreCase = True

  Signal = 1
  Background = -1
  Unknown = -999

class Dataset(EnumStringification):
  """
  The possible datasets to use
  """
  _ignoreCase = True

  Unspecified = 0
  Train = 1
  Validation = 2
  Test = 3
  Operation = 4

class Detector(EnumStringification):
  """
  The ATLAS Detector systems.
  """
  Tracking = 1
  Calorimetry = 2
  MuonSpectometer = 3
  CaloAndTrack = 4
  All = 5

class BaseInfo( EnumStringification ):
  Et = 0
  Eta = 1
  Nvtx = 2
  nInfo = 3 # This must always be the last base info

  def __init__(self, baseInfoBranches, dtypes):
    self._baseInfoBranches = baseInfoBranches
    self._dtypes = dtypes

  def retrieveBranch(self, idx):
    idx = self.retrieve(idx)
    return self._baseInfoBranches[idx]

  def dtype(self, idx):
    idx = self.retrieve(idx)
    return self._dtypes[idx]

  def __iter__(self):
    return self.loop()

  def loop(self):
    for baseEnum in range(BaseInfo.nInfo):
      yield baseEnum


class BranchEffCollectorRDS( RawDictStreamer ):
  def treatDict(self, obj, raw):
    """
    Add efficiency value to be readable in matlab
    """
    raw['efficiency'] = obj.efficiency
    #raw['__version '] = obj._version
    return RawDictStreamer.treatDict( self, obj, raw )




class BranchEffCollector(object):
  """
    Simple class for counting passed events using input branch
  """

  __metaclass__ = RawDictStreamable
  _streamerObj  = BranchEffCollectorRDS( toPublicAttrs = {'_etaBin', '_etBin'} )
  _cnvObj       = RawDictCnv( ignoreAttrs = {'efficiency'}, toProtectedAttrs = {'_etaBin', '_etBin'}, )
  _version      = 1

  def __init__(self, name = '', branch = '', etBin = -1, etaBin = -1, crossIdx = -1, ds = Dataset.Unspecified):
    self._ds = ds if ds is None else Dataset.retrieve(ds)
    self.name = name
    self._branch = branch
    self._etBin = etBin
    self._etaBin = etaBin
    self._crossIdx = crossIdx
    self._passed = 0
    self._count = 0

  @property
  def etBin(self):
    return self._etBin

  @property
  def etaBin(self):
    return self._etaBin

  @property
  def crossIdx(self):
    return self._crossIdx

  @property
  def ds(self):
    return self._ds

  @property
  def printName(self):
    return (Dataset.tostring(self.ds) + '_' if self.ds not in (None,Dataset.Unspecified) else '') + \
        self.name + \
        (('_etBin%d') % self.etBin if self.etBin not in (None,-1) else '') + \
        (('_etaBin%d') % self.etaBin if self.etaBin not in (None,-1) else '') + \
        (('_x%d') % self.crossIdx if self.crossIdx not in (None,-1) else '')

  def update(self, event, total = None):
    " Update the counting. "
    if total is not None: 
      self._passed += event
      self._count += total
      return
    elif getattr(event,self._branch): 
      self._passed += 1
    self._count += 1

  @property
  def efficiency(self):
    " Returns efficiency in percentage"
    if self._count:
      return self._passed / float(self._count) * 100.
    else:
      return 0.

  def setEfficiency(self, percentage):
    "Set efficiency in percentage"
    self._passed = (percentage/100.); 
    self._count = 1 

  @property
  def passed(self):
    "Total number of passed occurrences"
    return self._passed

  @property
  def count(self):
    "Total number of counted occurrences"
    return self._count

  def eff_str(self):
    "Retrieve the efficiency string"
    return '%.6f (%d/%d)' % ( self.efficiency,
                              self._passed,
                              self._count )
  def __str__(self):
    return (self.printName + " : " + self.eff_str() )

class BranchCrossEffCollectorRDS(RawDictStreamer):

  def __init__(self, **kw):
    RawDictStreamer.__init__( self, transientAttrs = {'_output'}, toPublicAttrs = {'_etaBin', '_etBin'}, **kw )
    self.noChildren = False

  def treatDict(self, obj, raw):
    """
    Method dedicated to modifications on raw dictionary
    """
    raw['__version'] = obj._version

    # Treat special members:
    if self.noChildren:
      raw.pop('_crossVal')
    self.deepCopyKey( raw, '_branchCollectorsDict')
    if raw['_branchCollectorsDict']:
      from copy import deepcopy
      raw['_branchCollectorsDict'] = deepcopy( raw['_branchCollectorsDict'] )
      for cData, idx, parent, _, _ in traverse(raw['_branchCollectorsDict'].values()):
        if self.noChildren:
          parent[idx] = cData.efficiency
        else:
          parent[idx] = cData.toRawObj()
    else: 
      raw['_branchCollectorsDict'] = ''
    # And now add the efficiency member
    raw['efficiency'] = { Dataset.tostring(key) : val for key, val in obj.allDSEfficiency.iteritems() }
    if not raw['efficiency']: 
      raw['efficiency'] = ''
    # Use default treatment
    RawDictStreamer.treatDict(self, obj, raw)
    return raw

class BranchCrossEffCollectorRDC( RawDictCnv ):

  def __init__(self, **kw):
    RawDictCnv.__init__( self, ignoreAttrs = {'efficiency'}, toProtectedAttrs = {'_etaBin', '_etBin'}, **kw )

  def treatObj( self, obj, d ):

    if not 'version' in d:
      obj._readVersion = 0
    
    if '_crossVal' in d: 
      if type('_crossVal' is dict): # Treat old files
        from TuningTools.CrossValid import CrossValid
        obj._crossVal = CrossValid.fromRawObj( d['_crossVal'] )
    
    if type( obj._branchCollectorsDict ) is dict:
      for cData, idx, parent, _, _ in traverse(obj._branchCollectorsDict.values()):
        if not 'version' in d:
          # Old version
          parent[idx] = BranchEffCollector.fromRawObj( cData )
        else:
          from RingerCore import retrieveRawDict
          parent[idx] = retrieveRawDict( cData )
        if parent[idx] is cData:
          break
    else:
      obj._branchCollectorsDict = {}
    if obj._readVersion < 2:
      obj._valAsTst = True
    return obj

class BranchCrossEffCollector(object):
  """
  Object for calculating the cross-validation datasets efficiencies
  """

  __metaclass__ = RawDictStreamable
  _streamerObj  = BranchCrossEffCollectorRDS()
  _cnvObj       = BranchCrossEffCollectorRDC()
  _version      = 2

  dsList = [ Dataset.Train,
             Dataset.Validation,
             Dataset.Test, ]

  def __init__(self, nevents=-1, crossVal=None, name='', branch='', etBin=-1, etaBin=-1):
    self.name = name
    self._count = 0
    self._branch = branch
    self._output = npCurrent.flag_ones(nevents) * -1 if nevents > 0 else npCurrent.flag_ones([])
    self._etBin = etBin
    self._etaBin = etaBin
    self._valAsTst = crossVal.nTest() if crossVal is not None else False
    from TuningTools.CrossValid import CrossValid
    if crossVal is not None and not isinstance(crossVal, CrossValid): 
      self._logger.fatal('Wrong cross-validation object.')
    self._crossVal = crossVal
    self._branchCollectorsDict = {}
    if self._crossVal is not None:
      for ds in BranchCrossEffCollector.dsList:
        if ds == Dataset.Test and self._valAsTst: continue
        self._branchCollectorsDict[ds] = \
            [BranchEffCollector(name, branch, etBin, etaBin, sort, ds) \
               for sort in range(self._crossVal.nSorts())]

  @property
  def etBin(self):
    return self._etBin

  @property
  def etaBin(self):
    return self._etaBin

  @property
  def allDSEfficiency(self):
    return self.efficiency()

  @property
  def allDSCount(self):
    return self.count()

  @property
  def allDSPassed(self):
    return self.passed()

  @property
  def allDSEfficiencyList(self):
    return self.efficiencyList()

  @property
  def allDSListCount(self):
    return self.countList()

  @property
  def allDSPassedList(self):
    return self.passedList()

  @property
  def printName(self):
    return self.name + \
        (('_etBin%d') % self.etBin if self.etBin is not None else '') + \
        (('_etaBin%d') % self.etaBin if self.etaBin is not None else '')

  def update(self, event):
    " Update the looping data. "
    if getattr(event,self._branch):
      self._output[self._count] = 1
    else:
      self._output[self._count] = 0
    self._count += 1

  def finished(self):
    " Call after looping is finished"
    # Strip uneeded values
    self._output = self._output[self._output != -1]
    #print 'Stripped output is (len=%d: %r)' % (len(self._output), self._output)
    maxEvts = len(self._output)
    for sort in range(self._crossVal.nSorts()):
      for ds, val in self._branchCollectorsDict.iteritems():
        for box in self._crossVal.getBoxIdxs(ds, sort):
          startPos, endPos = self._crossVal.getBoxPosition(sort, box, maxEvts=maxEvts )
          #print 'sort %d: startPos, endPos (%d,%d)' % (sort, startPos, endPos)
          boxPassed = np.sum( self._output[startPos:endPos] == 1 )
          boxTotal = endPos - startPos
          val[sort].update( boxPassed, boxTotal )
          #print '%s_%s=%d/%d' % ( self.name, Dataset.tostring(ds), boxPassed, boxTotal) 
    # Release data, not needed anymore
    self._output = None

  def __retrieveInfo(self, attr, ds = Dataset.Unspecified, sort = None):
    "Helper function to retrieve cross-validation reference mean/std information"
    if ds is Dataset.Unspecified:
      retDict = {}
      for ds, branchEffCol in self._branchCollectorsDict.iteritems():
        if sort is not None:
          retDict[ds] = getattr(branchEffCol[sort], attr)
        else:
          info = self.__retrieveInfoList(attr, ds)
          retDict[ds] = (np.mean(info), np.std(info))
      return retDict
    else:
      if ds is Dataset.Test and not self._valAsTst:
        ds = Dataset.Validation
      if sort is not None:
        return getattr(self._branchCollectorsDict[ds][sort], attr)
      else:
        info = self.__retrieveInfoList(attr, ds)
        return (np.mean(info), np.std(info))

  def __retrieveInfoList(self, attr, ds = Dataset.Unspecified, ):
    " Helper function to retrieve cross-validation information list"
    if ds is Dataset.Unspecified:
      retDict = {}
      for ds, branchEffCol in self._branchCollectorsDict.iteritems():
        retDict[ds] = [ getattr( branchEff, attr) for branchEff in branchEffCol ]
      return retDict
    else:
      if ds is Dataset.Test and not self._valAsTst:
        ds = Dataset.Validation
      return [ getattr( branchEff, attr ) for branchEff in self._branchCollectorsDict[ds] ]

  def efficiency(self, ds = Dataset.Unspecified, sort = None):
    " Returns efficiency in percentage"
    return self.__retrieveInfo( 'efficiency', ds, sort )

  def passed(self, ds = Dataset.Unspecified, sort = None):
    " Returns passed counts"
    return self.__retrieveInfo( 'passed', ds, sort )

  def count(self, ds = Dataset.Unspecified, sort = None):
    " Returns total counts"
    return self.__retrieveInfo( 'count', ds, sort )

  def efficiencyList(self, ds = Dataset.Unspecified ):
    " Returns efficiency in percentage"
    return self.__retrieveInfoList( 'efficiency', ds)

  def passedList(self, ds = Dataset.Unspecified):
    "Total number of passed occurrences"
    return self.__retrieveInfoList( 'passed', ds )

  def countList(self, ds = Dataset.Unspecified):
    "Total number of counted occurrences"
    return self.__retrieveInfoList( 'count', ds )

  def eff_str(self, ds = Dataset.Unspecified, format_ = 'long'):
    "Retrieve the efficiency string"
    if ds is Dataset.Unspecified:
      retDict = {}
      for ds, val in self._branchCollectorsDict.iteritems():
        eff = self.efficiency(ds)
        passed = self.passed(ds)
        count = self.count(ds)
        retDict[ds] = '%.6f +- %.6f (count: %.4f/%.4f +- %.4f/%.4f)' % ( eff[0], eff[1],
                                  passed[0], count[0],
                                  passed[1], count[1],)
      return retDict
    else:
      if ds is Dataset.Test and \
          not self._crossVal.nTest():
        ds = Dataset.Validation
      eff = self.efficiency(ds)
      passed = self.passed(ds)
      count = self.count(ds)
      return '%.6f +- %.6f (count: %.4f/%.4f +- %.4f/%.4f)' % ( eff[0], eff[1],
                                passed[0], count[0],
                                passed[1], count[1],)

  def __str__(self):
    "String representation of the object."
    # FIXME check itertools for a better way of dealing with all of this
    trnEff = self.efficiency(Dataset.Train)
    valEff = self.efficiency(Dataset.Validation)
    return self.printName + ( " : Train (%.6f +- %.6f) | Val (%6.f +- %.6f)" % \
         (trnEff[0], trnEff[1], valEff[0], valEff[1]) ) \
         + ( " Test (%.6f +- %.6f)" % self.efficiency(Dataset.Test) if self._valAsTst else '')

  def dump(self, fcn, **kw):
    "Dump efficiencies using log function."
    printSort = kw.pop('printSort', False)
    sortFcn = kw.pop('sortFcn', None)
    if printSort and sortFcn is None:
      self._logger.fatal(('When the printSort flag is True, it is also needed to '  
          'specify the sortFcn.'), TypeError)
    for ds, str_ in self.eff_str().iteritems():
      fcn(self.printName +  " : " + str_)
      if printSort:
        for branchCollector in self._branchCollectorsDict[ds]:
          sortFcn('%s', branchCollector)

class ReadData(Logger):
  """
    Retrieve from TTree the training information. Use readData object.
  """

  def __setBranchAddress( self, tree, varname, holder ):
    " Set tree branch varname to holder "
    if not tree.GetBranchStatus(varname):
      tree.SetBranchStatus( varname, True )
      from ROOT import AddressOf
      tree.SetBranchAddress( varname, AddressOf(holder, varname) )
      self._logger.debug("Set %s branch address on %s", varname, tree )
    else:
      self._logger.debug("Already set %s branch address on %s", varname, tree)


  def __retrieveBinIdx( self, bins, value ):
    return npCurrent.scounter_dtype.type(np.digitize(npCurrent.fp_array([value]), bins)[0]-1)

  def __init__( self, logger = None ):
    """
      Load TuningTools C++ library and set logger
    """
    # Retrieve python logger
    Logger.__init__( self, logger = logger)



  def __call__( self, fList, ringerOperation, **kw):
    """
      Read ntuple and return patterns and efficiencies.
      Arguments:
        - fList: The file path or file list path. It can be an argument list of 
        two types:
          o List: each element is a string path to the file;
          o Comma separated string: each path is separated via a comma
          o Folders: Expand folders recursively adding also files within them to analysis
        - ringerOperation: Set Operation type. It can be both a string or the
          RingerOperation
      Optional arguments:
        - filterType [None]: whether to filter. Use FilterType enumeration
        - reference [Truth]: set reference for targets. Use Reference enumeration
        - treePath [Set using operation]: set tree name on file, this may be set to
          use different sources then the default.
            Default for:
              o Offline: Offline/Egamma/Ntuple/electron
              o L2: Trigger/HLT/Egamma/TPNtuple/e24_medium_L1EM18VH
        - l1EmClusCut [None]: Set L1 cluster energy cut if operating on the trigger
        - l2EtCut [None]: Set L2 cluster energy cut value if operating on the trigger
        - offEtCut [None]: Set Offline cluster energy cut value
        - nClusters [None]: Read up to nClusters. Use None to run for all clusters.
        - getRatesOnly [False]: Read up to nClusters. Use None to run for all clusters.
        - etBins [None]: E_T bins (GeV) where the data should be segmented
        - etaBins [None]: eta bins where the data should be segmented
        - ringConfig [100]: A list containing the number of rings available in the data
          for each eta bin.
        - crossVal [None]: Whether to measure benchmark efficiency splitting it
          by the crossVal-validation datasets
        - extractDet [None]: Which detector to export (use Detector enumeration).
          Defaults are:
            o L2Calo: Calorimetry
            o L2: Tracking
            o Offline: Calorimetry
            o Others: CaloAndTrack
        - standardCaloVariables [False]: Whether to extract standard track variables.
        - useTRT [False]: Whether to export TRT information when dumping track
          variables.
        - supportTriggers [True]: Whether reading data comes from support triggers
    """
    # Offline information branches:
    __offlineBranches = [#'el_et',
                         #'el_eta',
                         #'el_loose',
                         #'el_medium',
                         #'el_tight',
                         #'el_lhLoose',
                         #'el_lhMedium',
                         'el_lhTight',
                         'mc_hasMC',
                         'mc_isElectron',
                         'mc_hasZMother',
                         'el_nPileupPrimaryVtx',
                         ]
    # Online information branches
    __onlineBranches = ['trig_L1_accept']
    __l2stdCaloBranches = ['trig_L2_calo_et',
                           'trig_L2_calo_eta',
                           'trig_L2_calo_e237', # rEta
                           'trig_L2_calo_e277', # rEta
                           'trig_L2_calo_fracs1', # F1: fraction sample 1
                           'trig_L2_calo_weta2', # weta2
                           'trig_L2_calo_ehad1', # energy on hadronic sample 1
                           'trig_L2_calo_emaxs1', # eratio
                           'trig_L2_calo_e2tsts1', # eratio
                           'trig_L2_calo_wstot',] # wstot
    __l2trackBranches = [ # Do not add non patter variables on this branch list
                         #'trig_L2_el_pt',
                         #'trig_L2_el_eta',
                         #'trig_L2_el_phi',
                         #'trig_L2_el_caloEta',
                         #'trig_L2_el_charge',
                         #'trig_L2_el_nTRTHits',
                         #'trig_L2_el_nTRTHiThresholdHits',
                         'trig_L2_el_etOverPt',
                         'trig_L2_el_trkClusDeta',
                         'trig_L2_el_trkClusDphi',]
    # Retrieve information from keyword arguments
    filterType            = retrieve_kw(kw, 'filterType',            FilterType.DoNotFilter )
    reference             = retrieve_kw(kw, 'reference',             Reference.Truth        )
    l1EmClusCut           = retrieve_kw(kw, 'l1EmClusCut',           None                   )
    l2EtCut               = retrieve_kw(kw, 'l2EtCut',               None                   )
    efEtCut               = retrieve_kw(kw, 'efEtCut',               None                   )
    offEtCut              = retrieve_kw(kw, 'offEtCut',              None                   )
    treePath              = retrieve_kw(kw, 'treePath',              None                   )
    nClusters             = retrieve_kw(kw, 'nClusters',             None                   )
    getRates              = retrieve_kw(kw, 'getRates',              True                   )
    getRatesOnly          = retrieve_kw(kw, 'getRatesOnly',          False                  )
    etBins                = retrieve_kw(kw, 'etBins',                None                   )
    etaBins               = retrieve_kw(kw, 'etaBins',               None                   )
    crossVal              = retrieve_kw(kw, 'crossVal',              None                   )
    ringConfig            = retrieve_kw(kw, 'ringConfig',            100                    )
    extractDet            = retrieve_kw(kw, 'extractDet',            None                   )
    standardCaloVariables = retrieve_kw(kw, 'standardCaloVariables', False                  )
    useTRT                = retrieve_kw(kw, 'useTRT',                False                  )
    supportTriggers       = retrieve_kw(kw, 'supportTriggers',       True                   )
    import ROOT
    #gROOT.ProcessLine (".x $ROOTCOREDIR/scripts/load_packages.C");
    #ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
    if ROOT.gSystem.Load('libTuningTools') < 0:
      self._logger.fatal("Could not load TuningTools library", ImportError)

    if 'level' in kw: self.level = kw.pop('level')
    # and delete it to avoid mistakes:
    checkForUnusedVars( kw, self._logger.warning )
    del kw
    ### Parse arguments
    # Mutual exclusive arguments:
    if not getRates and getRatesOnly:
      self._logger.error("Cannot run with getRates set to False and getRatesOnly set to True. Setting getRates to True.")
      getRates = True
    # Also parse operation, check if its type is string and if we can
    # transform it to the known operation enum:
    fList = csvStr2List ( fList )
    fList = expandFolders( fList )
    ringerOperation = RingerOperation.retrieve(ringerOperation)
    reference = Reference.retrieve(reference)
    if isinstance(l1EmClusCut, str):
      l1EmClusCut = float(l1EmClusCut)
    if l1EmClusCut:
      l1EmClusCut = 1000.*l1EmClusCut # Put energy in MeV
      __onlineBranches.append( 'trig_L1_emClus'  )
    if l2EtCut:
      l2EtCut = 1000.*l2EtCut # Put energy in MeV
      __onlineBranches.append( 'trig_L2_calo_et' )
    if efEtCut:
      efEtCut = 1000.*efEtCut # Put energy in MeV
      __onlineBranches.append( 'trig_EF_calo_et' )
    if offEtCut:
      offEtCut = 1000.*offEtCut # Put energy in MeV
      __offlineBranches.append( 'el_et' )
    # Check if treePath is None and try to set it automatically
    if treePath is None:
      treePath = 'Offline/Egamma/Ntuple/electron' if ringerOperation < 0 else \
                 'Trigger/HLT/Egamma/TPNtuple/e24_medium_L1EM18VH'
    # Check whether using bins
    useBins=False; useEtBins=False; useEtaBins=False
    nEtaBins = 1; nEtBins = 1
    # Set the detector which we should extract the information:
    if extractDet is None:
      if ringerOperation < 0:
        extractDet = Detector.Calorimetry
      elif ringerOperation is RingerOperation.L2Calo:
        extractDet = Detector.Calorimetry
      elif ringerOperation is RingerOperation.L2:
        extractDet = Detector.Tracking
      else:
        extractDet = Detector.CaloAndTrack
    else:
      extractDet = Detector.retrieve( extractDet )

    if etaBins is None: etaBins = npCurrent.fp_array([])
    if type(etaBins) is list: etaBins=npCurrent.fp_array(etaBins)
    if etBins is None: etBins = npCurrent.fp_array([])
    if type(etBins) is list: etBins=npCurrent.fp_array(etBins)

    if etBins.size:
      etBins = etBins * 1000. # Put energy in MeV
      nEtBins  = len(etBins)-1
      if nEtBins >= np.iinfo(npCurrent.scounter_dtype).max:
        self._logger.fatal(('Number of et bins (%d) is larger or equal than maximum '
            'integer precision can hold (%d). Increase '
            'TuningTools.coreDef.npCurrent scounter_dtype number of bytes.'), nEtBins,
            np.iinfo(npCurrent.scounter_dtype).max)
      # Flag that we are separating data through bins
      useBins=True
      useEtBins=True
      self._logger.debug('E_T bins enabled.')    

    if not type(ringConfig) is list and not type(ringConfig) is np.ndarray:
      ringConfig = [ringConfig] * (len(etaBins) - 1) if etaBins.size else 1
    if type(ringConfig) is list: ringConfig=npCurrent.int_array(ringConfig)
    if not len(ringConfig):
      self._logger.fatal('Rings size must be specified.');

    if etaBins.size:
      nEtaBins = len(etaBins)-1
      if nEtaBins >= np.iinfo(npCurrent.scounter_dtype).max:
        self._logger.fatal(('Number of eta bins (%d) is larger or equal than maximum '
            'integer precision can hold (%d). Increase '
            'TuningTools.coreDef.npCurrent scounter_dtype number of bytes.'), nEtaBins,
            np.iinfo(npCurrent.scounter_dtype).max)
      if len(ringConfig) != nEtaBins:
        self._logger.fatal(('The number of rings configurations (%r) must be equal than ' 
                            'eta bins (%r) region config'),ringConfig, etaBins)
      useBins=True
      useEtaBins=True
      self._logger.debug('eta bins enabled.')    
    else:
      self._logger.debug('eta/et bins disabled.')

    ### Prepare to loop:
    # Open root file
    t = ROOT.TChain(treePath)
    for inputFile in progressbar(fList, len(fList),
                                 logger = self._logger,
                                 prefix = "Creating collection tree "):

      # Check if file exists
      f  = ROOT.TFile.Open(inputFile, 'read')
      if not f or f.IsZombie():
        self._logger.warning('Couldn''t open file: %s', inputFile)
        continue
      # Inform user whether TTree exists, and which options are available:
      self._logger.debug("Adding file: %s", inputFile)
      obj = f.Get(treePath)
      if not obj:
        self._logger.warning("Couldn't retrieve TTree (%s)!", treePath)
        self._logger.info("File available info:")
        f.ReadAll()
        f.ReadKeys()
        f.ls()
        continue
      elif not isinstance(obj, ROOT.TTree):
        self._logger.fatal("%s is not an instance of TTree!", treePath, ValueError)
      t.Add( inputFile )

    # Turn all branches off.
    t.SetBranchStatus("*", False)

    # RingerPhysVal hold the address of required branches
    event = ROOT.RingerPhysVal()

    # Add offline branches, these are always needed
    cPos = 0
    for var in __offlineBranches:
      self.__setBranchAddress(t,var,event)

    # Add online branches if using Trigger
    if ringerOperation > 0:
      for var in __onlineBranches:
        self.__setBranchAddress(t,var,event)

    ## Allocating memory for the number of entries
    entries = t.GetEntries()
    nobs = entries if (nClusters is None or nClusters > entries or nClusters < 1) \
                                                                else nClusters

    ## Retrieve the dependent operation variables:
    if useEtBins:
      etBranch = 'el_et' if ringerOperation < 0 else 'trig_L2_calo_et'
      self.__setBranchAddress(t,etBranch,event)
      self._logger.debug("Added branch: %s", etBranch)
      if not getRatesOnly:
        npEt    = npCurrent.scounter_zeros(shape=npCurrent.shape(npat = 1, nobs = nobs))
        self._logger.debug("Allocated npEt    with size %r", npEt.shape)
    
    if useEtaBins:
      etaBranch    = "el_eta" if ringerOperation < 0 else "trig_L2_calo_eta"
      self.__setBranchAddress(t,etaBranch,event)
      self._logger.debug("Added branch: %s", etaBranch)
      if not getRatesOnly:
        npEta    = npCurrent.scounter_zeros(shape=npCurrent.shape(npat = 1, nobs = nobs))
        self._logger.debug("Allocated npEta   with size %r", npEta.shape)

    # The base information holder, such as et, eta and nvtx
    baseInfoBranch = BaseInfo((etBranch, etaBranch, 'el_nPileupPrimaryVtx',),
                              (npCurrent.fp_dtype, npCurrent.fp_dtype, np.uint16,) )
    baseInfo = [None, ] * baseInfoBranch.nInfo

    # Allocate numpy to hold as many entries as possible:
    if not getRatesOnly:
      # Retrieve the rings information depending on ringer operation
      ringerBranch = "el_ringsE" if ringerOperation < 0 else \
                     "trig_L2_calo_rings"
      self.__setBranchAddress(t,ringerBranch,event)
      if ringerOperation > 0:
        if ringerOperation is RingerOperation.L2:
          for var in __l2trackBranches:
            self.__setBranchAddress(t,var,event)
      if standardCaloVariables:
        if ringerOperation in (RingerOperation.L2, RingerOperation.L2Calo,): 
          for var in __l2stdCaloBranches:
            self.__setBranchAddress(t, var, event)
        else:
          self._logger.warning("Unknown standard calorimeters for Operation:%s. Setting operation back to use rings variables.", 
                               RingerOperation.tostring(ringerOperation))
      t.GetEntry(0)
      npat = 0
      if extractDet in (Detector.Calorimetry, 
                        Detector.CaloAndTrack, 
                        Detector.All):
        if standardCaloVariables:
          npat+= 6
        else:
          npat += ringConfig.max()
      if extractDet in (Detector.Tracking, 
                       Detector.CaloAndTrack, 
                       Detector.All):
        if ringerOperation is RingerOperation.L2:
          if useTRT:
            self._logger.info("Using TRT information!")
            npat += 2
            __l2trackBranches.append('trig_L2_el_nTRTHits')
            __l2trackBranches.append('trig_L2_el_nTRTHiThresholdHits')
          npat += 3
          for var in __l2trackBranches:
            self.__setBranchAddress(t,var,event)
          self.__setBranchAddress(t,"trig_L2_el_pt",event)
        elif ringerOperation < 0: # Offline
          self._logger.warning("Still need to implement tracking for the ringer offline.")
      npPatterns = npCurrent.fp_zeros( shape=npCurrent.shape(npat=npat, #getattr(event, ringerBranch).size()
                                                   nobs=nobs)
                                     )
      self._logger.debug("Allocated npPatterns with size %r", npPatterns.shape)

      # Add E_T, eta and luminosity information
      npBaseInfo = [npCurrent.zeros( shape=npCurrent.shape(npat=1, nobs=nobs ), dtype=baseInfoBranch.dtype(idx) )
                                    for idx in baseInfoBranch]
    else:
      npPatterns = npCurrent.fp_array([])
      npBaseInfo = [deepcopy(npCurrent.fp_array([])) for _ in baseInfoBranch]

    ## Allocate the branch efficiency collectors:
    if getRates:
      if ringerOperation < 0:
        benchmarkDict = OrderedDict(
          [( RingerOperation.branchName( RingerOperation.Offline_CutBased_Loose  ), 'el_loose'            ),
           ( RingerOperation.branchName( RingerOperation.Offline_CutBased_Medium ), 'el_medium'           ),
           ( RingerOperation.branchName( RingerOperation.Offline_CutBased_Tight  ), 'el_tight'            ),
           ( RingerOperation.branchName( RingerOperation.Offline_LH_Loose        ), 'el_lhLoose'          ),
           ( RingerOperation.branchName( RingerOperation.Offline_LH_Medium       ), 'el_lhMedium'         ),
           ( RingerOperation.branchName( RingerOperation.Offline_LH_Tight        ), 'el_lhTight'          ),
          ])
      else:
        benchmarkDict = OrderedDict(
          [( RingerOperation.branchName( RingerOperation.L2Calo                  ), 'trig_L2_calo_accept' ),
           ( RingerOperation.branchName( RingerOperation.L2                      ), 'trig_L2_el_accept'   ),
           ( RingerOperation.branchName( RingerOperation.EFCalo                  ), 'trig_EF_calo_accept' ),
           ( RingerOperation.branchName( RingerOperation.HLT                     ), 'trig_EF_el_accept'   ),
          ])
      branchEffCollectors = OrderedDict()
      branchCrossEffCollectors = OrderedDict()
      for key, val in benchmarkDict.iteritems():
        branchEffCollectors[key] = list()
        branchCrossEffCollectors[key] = list()
        # Add efficincy branch:
        self.__setBranchAddress(t,val,event)
        for etBin in range(nEtBins):
          if useBins:
            branchEffCollectors[key].append(list())
            branchCrossEffCollectors[key].append(list())
          for etaBin in range(nEtaBins):
            etBinArg = etBin if useBins else -1
            etaBinArg = etaBin if useBins else -1
            argList = [ key, val, etBinArg, etaBinArg ]
            branchEffCollectors[key][etBin].append(BranchEffCollector( *argList ) )
            if crossVal:
              branchCrossEffCollectors[key][etBin].append(BranchCrossEffCollector( entries, crossVal, *argList ) )
          # etBin
        # etaBin
      # benchmark dict
      if self._logger.isEnabledFor( LoggingLevel.DEBUG ):
        self._logger.debug( 'Retrieved following branch efficiency collectors: %r', 
            [collector[0].printName for collector in traverse(branchEffCollectors.values())])
    # end of (getRates)

    etaBin = 0; etBin = 0
    step = int(entries/100) if int(entries/100) > 0 else 1
    ## Start loop!
    self._logger.info("There is available a total of %d entries.", entries)
    for entry in progressbar(range(entries), entries, 
                             step = step, logger = self._logger,
                             prefix = "Looping over entries "):
     
      #self._logger.verbose('Processing eventNumber: %d/%d', entry, entries)
      t.GetEntry(entry)

      # Check if it is needed to remove energy regions (this means that if not
      # within this range, it will be ignored as well for efficiency measuremnet)
      if event.el_et < offEtCut: 
        self._logger.verbose("Ignoring entry due to offline E_T cut.")
        continue
      if ringerOperation > 0:
        # Remove events which didn't pass L1_calo
        if not supportTriggers and not event.trig_L1_accept: 
          #self._logger.verbose("Ignoring entry due to L1Calo cut (trig_L1_accept = %r).", event.trig_L1_accept)
          continue
        if event.trig_L1_emClus  < l1EmClusCut: 
          #self._logger.verbose("Ignoring entry due to L1Calo E_T cut (%d < %r).", event.trig_L1_emClus, l1EmClusCut)
          continue
        if event.trig_L2_calo_et < l2EtCut: 
          #self._logger.verbose("Ignoring entry due to L2Calo E_T cut.")
          continue
        if event.trig_L2_calo_accept and efEtCut is not None:
          # EF calo is a container, search for electrons objects with et > cut
          trig_EF_calo_et_list = stdvector_to_list(event.trig_EF_calo_et)
          found=False
          for v in trig_EF_calo_et_list:
            if v < efEtCut:  found=True
          if found: 
            #self._logger.verbose("Ignoring entry due to EFCalo E_T cut.")
            continue

      # Set discriminator target:
      target = Target.Unknown
      if reference is Reference.Truth:
        if event.mc_isElectron and event.mc_hasZMother: 
          target = Target.Signal 
        elif not (event.mc_isElectron and (event.mc_hasZMother or event.mc_hasWMother) ): 
          target = Target.Background
      elif reference is Reference.Off_Likelihood:
        if event.el_lhTight: target = Target.Signal
        elif not event.el_lhLoose: target = Target.Background
      elif reference is Reference.AcceptAll:
        target = Target.Signal if filterType is FilterType.Signal else Target.Background
      else:
        if event.el_tight: target = Target.Signal 
        elif not event.el_loose: target = Target.Background 

      # Run filter if it is defined
      if filterType and \
         ( (filterType is FilterType.Signal and target != Target.Signal) or \
           (filterType is FilterType.Background and target != Target.Background) or \
           (target == Target.Unknown) ):
        #self._logger.verbose("Ignoring entry due to filter cut.")
        continue

      # Retrieve base information:
      for idx in baseInfoBranch:
        lInfo = getattr(event, baseInfoBranch.retrieveBranch(idx))
        baseInfo[idx] = lInfo
        if not getRatesOnly: npBaseInfo[idx][cPos] = lInfo
      # Retrieve dependent operation region
      if useEtBins:
        etBin  = self.__retrieveBinIdx( etBins, baseInfo[0] )
      if useEtaBins:
        etaBin = self.__retrieveBinIdx( etaBins, np.fabs( baseInfo[1]) )

      # Check if bin is within range (when not using bins, this will always be true):
      if (etBin < nEtBins and etaBin < nEtaBins):
        # Retrieve patterns:
        if not getRatesOnly:
          if useEtBins:  npEt[cPos] = etBin
          if useEtaBins: npEta[cPos] = etaBin
          ## Retrieve calorimeter information:
          cPat = 0
          caloAvailable = True
          if extractDet in (Detector.Calorimetry, 
                           Detector.CaloAndTrack, 
                           Detector.All):
            if standardCaloVariables:
              patterns = []
              if ringerOperation is RingerOperation.L2Calo:
                from math import cosh
                cosh_eta = cosh( event.trig_L2_calo_eta )
                # second layer ratio between 3x7 7x7
                rEta = event.trig_L2_calo_e237 / event.trig_L2_calo_e277
                base = event.trig_L2_calo_emaxs1 + event.trig_L2_calo_e2tsts1
                # Ratio between first and second highest energy cells
                eRatio = ( event.trig_L2_calo_emaxs1 - event.trig_L2_calo_e2tsts1 ) / base if base > 0 else 0
                # ratio of energy in the first layer (hadronic particles should leave low energy)
                F1 = event.trig_L2_calo_fracs1 / ( event.trig_L2_calo_et * cosh_eta )
                # weta2 is calculated over the middle layer using 3 x 5
                weta2 = event.trig_L2_calo_weta2
                # wstot is calculated over the first layer using (typically) 20 strips
                wstot = event.trig_L2_calo_wstot
                # ratio between EM cluster and first hadronic layers:
                Rhad1 = ( event.trig_L2_calo_ehad1 / cosh_eta ) / event.trig_L2_calo_et
                # allocate patterns:
                patterns = [rEta, eRatio, F1, weta2, wstot, Rhad1]
                for pat in patterns:
                  npPatterns[npCurrent.access( pidx=cPat, oidx=cPos) ] = pat
                  cPat += 1
              # end of ringerOperation
            else:
              # Remove events without rings
              if getattr(event,ringerBranch).empty(): 
                caloAvailable = False
              # Retrieve rings:
              if caloAvailable:
                try:
                  patterns = stdvector_to_list( getattr(event,ringerBranch) )
                  lPat = len(patterns) 
                  if lPat == ringConfig[etaBin]:
                    npPatterns[npCurrent.access(pidx=slice(cPat,ringConfig[etaBin]),oidx=cPos)] = patterns
                  else:
                    oldEtaBin = etaBin
                    if etaBin > 0 and ringConfig[etaBin - 1] == lPat:
                      etaBin -= 1
                    elif etaBin + 1 < len(ringConfig) and ringConfig[etaBin + 1] == lPat:
                      etaBin += 1
                    npPatterns[npCurrent.access(pidx=slice(cPat, ringConfig[etaBin]),oidx=cPos)] = patterns
                    self._logger.warning(("Recovered event which should be within eta bin (%d: %r) " 
                                          "but was found to be within eta bin (%d: %r). "
                                          "Its read eta value was of %f."),
                                          oldEtaBin, etaBins[oldEtaBin:oldEtaBin+2],
                                          etaBin, etaBins[etaBin:etaBin+2], 
                                          np.fabs( getattr(event,etaBranch)))
                except ValueError:
                  self._logger.error(("Patterns size (%d) do not match expected "
                                    "value (%d). This event eta value is: %f, and ringConfig is %r."),
                                    lPat, ringConfig[etaBin], np.fabs( getattr(event,etaBranch)), ringConfig 
                                    )
                  continue
              else:
                if extractDet is Detector.Calorimetry:
                  # Also display warning when extracting only calorimetry!
                  self._logger.warning("Rings not available")
                  continue
                self._logger.warning("Rings not available")
                continue
              cPat += ringConfig.max()
            # which calo variables
          # end of (extractDet needed calorimeter)
          # And track information:
          if extractDet in (Detector.Tracking, 
                           Detector.CaloAndTrack, 
                           Detector.All):
            if caloAvailable or extractDet is Detector.Tracking:
              if ringerOperation is RingerOperation.L2:
                # Retrieve nearest deta/dphi only, so we need to find each one is the nearest:
                if event.trig_L2_el_trkClusDeta.size():
                  clusDeta = npCurrent.fp_array( stdvector_to_list( event.trig_L2_el_trkClusDeta ) )
                  clusDphi = npCurrent.fp_array( stdvector_to_list( event.trig_L2_el_trkClusDphi ) )
                  bestTrackPos = int( np.argmin( clusDeta**2 + clusDphi**2 ) )
                  for var in __l2trackBranches:
                    npPatterns[npCurrent.access( pidx=cPat,oidx=cPos) ] = getattr(event, var)[bestTrackPos] 
                    cPat += 1
                else:
                  #self._logger.verbose("Ignoring entry due to track information not available.")
                  continue
                  #for var in __l2trackBranches:
                  #  npPatterns[npCurrent.access( pidx=cPat,oidx=cPos) ] = np.nan
                  #  cPat += 1
              elif ringerOperation < 0: # Offline
                pass
            # caloAvailable or only tracking
          # end of (extractDet needs tracking)
        # end of (getRatesOnly)

        ## Retrieve rates information:
        if getRates:
          for branch in branchEffCollectors.itervalues():
            if not useBins:
              branch.update(event)
            else:
              branch[etBin][etaBin].update(event)
          if crossVal:
            for branchCross in branchCrossEffCollectors.itervalues():
              if not useBins:
                branchCross.update(event)
              else:
                branchCross[etBin][etaBin].update(event)
        # end of (getRates)

        # We only increment if this cluster will be computed
        cPos += 1
      # end of (et/eta bins)

      # Limit the number of entries to nClusters if desired and possible:
      if not nClusters is None and cPos >= nClusters:
        break
    # for end

    ## Treat the rings information
    if not getRatesOnly:

      ## Remove not filled reserved memory space:
      if npPatterns.shape[npCurrent.odim] > cPos:
        npPatterns = np.delete( npPatterns, slice(cPos,None), axis = npCurrent.odim)

      ## Segment data over bins regions:
      # Also remove not filled reserved memory space:
      if useEtBins:
        npEt  = npCurrent.delete( npEt, slice(cPos,None))
      if useEtaBins:
        npEta = npCurrent.delete( npEta, slice(cPos,None))
      # Treat 
      npObject = self.treatNpInfo(cPos, npEt, npEta, useEtBins, useEtaBins, 
                                  nEtBins, nEtaBins, standardCaloVariables, ringConfig, 
                                  npPatterns, )
      data = [self.treatNpInfo(cPos, npEt, npEta, useEtBins, useEtaBins, 
                                                      nEtBins, nEtaBins, standardCaloVariables, ringConfig,
                                                      npData) for npData in npBaseInfo]
      npBaseInfo = npCurrent.array( data, dtype=np.object )
    else:
      npObject = npCurrent.array([], dtype=npCurrent.dtype)
    # not getRatesOnly

    if getRates:
      if crossVal:
        for etBin in range(nEtBins):
          for etaBin in range(nEtaBins):
            for branchCross in branchCrossEffCollectors.itervalues():
              if not useBins:
                branchCross.finished()
              else:
                branchCross[etBin][etaBin].finished()

      # Print efficiency for each one for the efficiency branches analysed:
      for etBin in range(nEtBins) if useBins else range(1):
        for etaBin in range(nEtaBins) if useBins else range(1):
          for branch in branchEffCollectors.itervalues():
            lBranch = branch if not useBins else branch[etBin][etaBin]
            self._logger.info('%s',lBranch)
          if crossVal:
            for branchCross in branchCrossEffCollectors.itervalues():
              lBranchCross = branchCross if not useBins else branchCross[etBin][etaBin]
              lBranchCross.dump(self._logger.debug, printSort = True,
                                 sortFcn = self._logger.verbose)
          # for branch
        # for eta
      # for et
    else:
      branchEffCollectors = None
      branchCrossEffCollectors = None
    # end of (getRates)

    outputs = []
    if not getRatesOnly:
      outputs.extend((npObject, npBaseInfo))
    if getRates:
      outputs.extend((branchEffCollectors, branchCrossEffCollectors))
    #outputs = tuple(outputs)
    return outputs
  # end __call__

  def treatNpInfo(self, cPos, npEt, npEta, useEtBins, 
                  useEtaBins, nEtBins, nEtaBins, standardCaloVariables, 
                  ringConfig, npInput, ):
    ## Remove not filled reserved memory space:
    if npInput.shape[npCurrent.odim] > cPos:
      npInput = np.delete( npInput, slice(cPos,None), axis = npCurrent.odim)

    # Separation for each bin found
    if useEtBins or useEtaBins:
      npObject = np.empty((nEtBins,nEtaBins),dtype=object)
      npObject.fill((npCurrent.array([[]]),))
      for etBin in range(nEtBins):
        for etaBin in range(nEtaBins):
          if useEtBins and useEtaBins:
            # Retrieve all in current eta et bin
            idx = np.all([npEt==etBin,npEta==etaBin],axis=0).nonzero()[0]
            if len(idx): 
              npObject[etBin][etaBin]=npInput[npCurrent.access(oidx=idx)]
              # Remove extra features in this eta bin
              if not standardCaloVariables:
                npObject[etBin][etaBin]=npCurrent.delete(npObject[etBin][etaBin],slice(ringConfig[etaBin],ringConfig.max()),
                                                         axis=npCurrent.pdim)
          elif useEtBins:
            # Retrieve all in current et bin
            idx = (npEt==etBin).nonzero()[0]
            if len(idx):
              npObject[etBin][etaBin]=npInput[npCurrent.access(oidx=idx)]
          else:# useEtaBins
            # Retrieve all in current eta bin
            idx = (npEta==etaBin).nonzero()[0]
            if len(idx): 
              npObject[etBin][etaBin]=npInput[npCurrent.access(oidx=idx)]
              # Remove extra rings:
              if not standardCaloVariables:
                npObject[etBin][etaBin]=npCurrent.delete(npObject[etBin][etaBin],slice(ringConfig[etaBin],ringConfig.max()),
                                                         axis=npCurrent.pdim)
        # for etaBin
      # for etBin
    else:
      npObject = npInput
    # useBins
    return npObject
  # end of (ReadData.treatNpInfo)

# Instantiate object
readData = ReadData()
