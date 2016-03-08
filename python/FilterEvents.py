from RingerCore.util import EnumStringification
from TuningTools.coreDef import retrieve_npConstants
npCurrent, _ = retrieve_npConstants()
from collections import OrderedDict
import numpy as np

class RingerOperation(EnumStringification):
  """
    Select which framework ringer will operate

    - Positive values for Online operation; and 
    - Negative values for Offline operation.
  """
  Offline = -1
  L2  = 1
  EF = 2
  L2Calo  = 3
  EFCalo  = 4

class Reference(EnumStringification):
  """
    Reference for training algorithm
  """
  Truth = -1
  Off_CutID = 1
  Off_Likelihood = 2
  

class FilterType(EnumStringification):
  """
    Enumeration if selection event type w.r.t reference
  """
  DoNotFilter = 0
  Background = 1
  Signal = 2

class Target(EnumStringification):
  """ 
    Holds the target value for the discrimination method
  """
  Signal = 1
  Background = -1
  Unknown = -999

class Dataset(EnumStringification):
  """
  The possible datasets to use
  """
  Unspecified = 0
  Train = 1
  Validation = 2
  Test = 3
  Operation = 4

from RingerCore.Logger import Logger, LoggingLevel

class BranchEffCollector(object):
  """
    Simple class for counting passed events using input branch
  """

  _passed = 0
  _count = 0
  _version = 1

  def __init__(self, name = None, branch = None, etBin = None, etaBin = None, crossIdx = None, ds = None):
    self._ds = ds if ds is None else Dataset.retrieve(ds)
    self.name = name
    self._branch = branch
    self._etBin = etBin
    self._etaBin = etaBin
    self._crossIdx = crossIdx

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
    return (Dataset.tostring(self.ds) + '_' if self.ds is not None else '') + \
        self.name + \
        (('_etBin%d') % self.etBin if self.etBin is not None else '') + \
        (('_etaBin%d') % self.etaBin if self.etaBin is not None else '') + \
        (('_x%d') % self.crossIdx if self.crossIdx is not None else '')

  def update(self, event, total = None):
    " Update the counting. "
    if total is not None: 
      self._passed += event
      self._count += total
      return
    elif getattr(event,self._branch): 
      self._passed += 1
    self._count += 1

  def efficiency(self):
    " Returns efficiency in percentage"
    if self._count:
      return self._passed / float(self._count) * 100.
    else:
      return 0.

  def passed(self):
    "Total number of passed occurrences"
    return self._passed

  def count(self):
    "Total number of counted occurrences"
    return self._count

  def eff_str(self):
    "Retrieve the efficiency string"
    return '%.6f (%d/%d)' % ( self.efficiency(),
                              self._passed,
                              self._count )
  def __str__(self):
    return (self.printName + " : " + self.eff_str() )

  def toRawObj(self):
    "Return a raw dict object from itself"
    from copy import copy
    raw = copy(self.__dict__)
    raw['version'] = BranchEffCollector._version
    return raw

  def buildFromDict(self, d):
    if d.pop('version') == self.__class__._version:
      for k, val in d.iteritems():
        self.__dict__[k] = d[k]
    return self

  @classmethod
  def fromRawObj(cls, obj):
    from copy import copy
    obj = copy(obj)
    self = cls().buildFromDict(obj)
    return self

class BranchCrossEffCollector(object):
  """
  Object for calculating the cross-validation datasets efficiencies
  """

  dsList = [ Dataset.Train,
             Dataset.Validation,
             Dataset.Test, ]

  _count = 0

  _version = 1

  def __init__(self, nevents=None, crossVal=None, name=None, branch=None, etBin=None, etaBin=None):
    self.name = name
    self._branch = branch
    self._output = npCurrent.flag_ones(nevents) * -1
    self._etBin = etBin
    self._etaBin = etaBin
    from TuningTools.CrossValid import CrossValid
    if crossVal is not None and not isinstance(crossVal, CrossValid): 
      raise ValueError('Wrong crossVal-validation object.')
    self._crossVal = crossVal
    self._branchCollectorsDict = {}
    if self._crossVal:
      for ds in BranchCrossEffCollector.dsList:
        fill = True if ds != Dataset.Test or self._crossVal.nTest() \
               else False
        if fill:
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

  def efficiency(self, ds = Dataset.Unspecified):
    " Returns efficiency in percentage"
    if ds is Dataset.Unspecified:
      retDict = {}
      for ds, val in self._branchCollectorsDict.iteritems():
        effs = [ branchEffCol.efficiency() for branchEffCol in val ]
        retDict[ds] = (np.mean(effs), np.std(effs))
      return retDict
    else:
      effs = [ branchEffCol.efficiency() for branchEffCol in self._branchCollectorsDict[ds] ]
      return (np.mean(effs), np.std(effs))

  def passed(self, ds = Dataset.Unspecified):
    "Total number of passed occurrences"
    if ds is Dataset.Unspecified:
      retDict = {}
      for ds, val in self._branchCollectorsDict.iteritems():
        passeds = [ branchEffCol.passed() for branchEffCol in val ]
        retDict[ds] = (np.mean(passeds), np.std(passeds))
      return retDict
    else:
      passeds = [ branchEffCol.passed() for branchEffCol in self._branchCollectorsDict[ds] ]
      return (np.mean(passeds), np.std(passeds))

  def count(self, ds = Dataset.Unspecified):
    "Total number of counted occurrences"
    if ds is Dataset.Unspecified:
      retDict = {}
      for ds, val in self._branchCollectorsDict.iteritems():
        counts = [ branchEffCol.count() for branchEffCol in val ]
        retDict[ds] = (np.mean(counts), np.std(counts))
      return retDict
    else:
      counts = [ branchEffCol.count() for branchEffCol in self._branchCollectorsDict[ds] ]
      return (np.mean(counts), np.std(counts))

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
         + ( " Test (%.6f +- %.6f)" % self.efficiency(Dataset.Test) if self._crossVal.nTest() else '')

  def dump(self, fcn, **kw):
    "Dump efficiencies using log function."
    printSort = kw.pop('printSort', False)
    sortFcn = kw.pop('sortFcn', None)
    if printSort and sortFcn is None:
      raise TypeError(('When the printSort flag is True, it is also needed to '  
          'specify the sortFcn.'))
    for ds, str_ in self.eff_str().iteritems():
      fcn(self.printName +  " : " + str_)
      if printSort:
        for branchCollector in self._branchCollectorsDict[ds]:
          sortFcn('%s', branchCollector)

  def toRawObj(self):
    "Return a raw dict object from itself"
    from copy import copy
    raw = copy(self.__dict__)
    raw['_crossVal'] = self._crossVal.toRawObj()
    from RingerCore.util import traverse
    for cData, idx, parent, _, _ in traverse(self._branchCollectorsDict.values()):
      parent[idx] = cData.toRawObj()
    raw['version'] = self.__class__._version
    return raw

  def buildFromDict(self, d):
    if d.pop('version') == self.__class__._version:
      for k, val in d.iteritems():
        self.__dict__[k] = d[k]
      from TuningTools.CrossValid import CrossValid
      self._crossVal = CrossValid.fromRawObj( self._crossVal )
      from RingerCore.util import traverse
      for cData, idx, parent, _, _ in traverse(self._branchCollectorsDict.values()):
        parent[idx] = BranchEffCollector.fromRawObj( cData )
    return self

  @classmethod
  def fromRawObj(cls, obj):
    from copy import copy
    obj = copy(obj)
    self = cls().buildFromDict(obj)
    return self

class FilterEvents(Logger):
  """
    Retrieve from TTree the training information. Use filterEvents object.
  """

  # Offline information branches:
  __offlineBranches = ['el_et',
                       'el_eta',
                       'el_phi',
                       'el_loose',
                       'el_medium',
                       'el_tight',
                       'el_lhLoose',
                       'el_lhMedium',
                       'el_lhTight',
                       'mc_hasMC',
                       'mc_isElectron',
                       'mc_hasZMother',]

  # Online information branches
  __onlineBranches = ['trig_L1_emClus',
                      'trig_L1_accept',
                      'trig_L2_calo_et',
                      'trig_L2_calo_eta',
                      'trig_L2_calo_accept',
                      'trig_L2_el_accept',
                      'trig_EF_calo_accept',
                      'trig_EF_el_accept']

  def __setBranchAddress( self, tree, varname, holder ):
    " Set tree branch varname to holder "
    import ROOT
    tree.SetBranchAddress(varname, ROOT.AddressOf(holder,varname) )  


  def __retrieveBinIdx( self, bins, value ):
    return npCurrent.scounter_dtype.type(np.digitize(npCurrent.fp_array([value]), bins)[0]-1)

  def __init__( self, logger = None ):
    """
      Load TuningTools C++ library and set logger
    """
    # Retrieve python logger
    Logger.__init__( self, logger = logger)

    import ROOT
    #gROOT.ProcessLine (".x $ROOTCOREDIR/scripts/load_packages.C");
    #ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
    if ROOT.gSystem.Load('libTuningTools') < 0:
      raise ImportError("Could not load TuningTools library")


  def __call__( self, fList, ringerOperation, **kw):
    """
      Returns ntuple with rings and its targets
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
        - efficiencyTreePath [None]: Sets tree path for retrieving efficiency
              benchmarks.
            When not set, uses treePath as tree.
        - l1EmClusCut [None]: Set L1 cluster energy cut if operating on the trigger
        - l2EtCut [None]: Set L2 cluster energy cut value if operating on the trigger
        - offEtCut [None]: Set Offline cluster energy cut value
        - nClusters [None]: Read up to nClusters. Use None to run for all clusters.
        - getRatesOnly [False]: Read up to nClusters. Use None to run for all clusters.
        - etBins [None]: E_T bins where the data should be segmented
        - etaBins [None]: eta bins where the data should be segmented
        - ringConfig [100]: A list containing the number of rings available in the data
          for each eta bin.
        - crossVal [None]: Whether to measure benchmark efficiency splitting it
          by the crossVal-validation datasets
    """
    # Retrieve information from keyword arguments
    filterType         = kw.pop('filterType',         FilterType.DoNotFilter )
    reference          = kw.pop('reference',          Reference.Truth        )
    l1EmClusCut        = kw.pop('l1EmClusCut',        None                   )
    l2EtCut            = kw.pop('l2EtCut',            None                   )
    efEtCut            = kw.pop('efEtCut',            None                   )
    offEtCut           = kw.pop('offEtCut',           None                   )
    treePath           = kw.pop('treePath',           None                   )
    efficiencyTreePath = kw.pop('efficiencyTreePath', None                   )
    nClusters          = kw.pop('nClusters',          None                   )
    getRatesOnly       = kw.pop('getRatesOnly',       False                  )
    etBins             = kw.pop('etBins',             None                   )
    etaBins            = kw.pop('etaBins',            None                   )
    ringConfig         = kw.pop('ringConfig',         None                   )
    crossVal           = kw.pop('crossVal',           None                   )
    if ringConfig is None:
      ringConfig = [100]*(len(etaBins)-1) if etaBins else [100]

    if 'level' in kw: self.level = kw.pop('level')
    # and delete it to avoid mistakes:
    from RingerCore.util import checkForUnusedVars, stdvector_to_list
    checkForUnusedVars( kw, self._logger.warning )
    del kw
    ### Parse arguments
    # Also parse operation, check if its type is string and if we can
    # transform it to the known operation enum:
    if isinstance(fList, str): # transform comma separated list to a list
      fList = fList.split(',')
    if len(fList) == 1 and ',' in fList[0]:
      fList = fList[0].split(',')
    from RingerCore.FileIO import expandFolders
    fList = expandFolders( fList )
    if isinstance(ringerOperation, str):
      ringerOperation = RingerOperation.fromstring(ringerOperation)
    if isinstance(reference, str):
      reference = Reference.fromstring(reference)
    if isinstance(l1EmClusCut, str):
      l1EmClusCut = float(l1EmClusCut)
    if l1EmClusCut:
      l1EmClusCut = 1000.*l1EmClusCut # Put energy in MeV
    if l2EtCut:
      l2EtCut = 1000.*l2EtCut # Put energy in MeV
    if efEtCut:
      efEtCut = 1000.*efEtCut # Put energy in MeV
    if offEtCut:
      offEtCut = 1000.*offEtCut # Put energy in MeV
    # Check if treePath is None and try to set it automatically
    if treePath is None:
      treePath = 'Offline/Egamma/Ntuple/electron' if ringerOperation < 0 else \
                 'Trigger/HLT/Egamma/TPNtuple/e24_medium_L1EM18VH'
    if efficiencyTreePath is None:
      efficiencyTreePath = treePath
    # Check whether using bins
    useBins=False; useEtBins=False; useEtaBins=False
    nEtaBins = 1; nEtBins = 1

    if etaBins is None: etaBins = npCurrent.fp_array([])
    if type(etaBins) is list: etaBins=npCurrent.fp_array(etaBins)
    if etBins is None: etBins = npCurrent.fp_array([])
    if type(etBins) is list: etBins=npCurrent.fp_array(etBins)

    if etBins.size:
      etBins = etBins * 1000. # Put energy in MeV
      nEtBins  = len(etBins)-1
      if nEtBins >= np.iinfo(npCurrent.scounter_dtype).max:
        raise RuntimeError(('Number of et bins (%d) is larger or equal than maximum '
            'integer precision can hold (%d). Increase '
            'TuningTools.coreDef.npCurrent scounter_dtype number of bytes.'), nEtBins,
            np.iinfo(npCurrent.scounter_dtype).max)
      # Flag that we are separating data through bins
      useBins=True
      useEtBins=True
      self._logger.debug('E_T bins enabled.')    

    if not type(ringConfig) is list and not type(ringConfig) is np.ndarray:
      ringConfig = [ringConfig]
    if type(ringConfig) is list: ringConfig=npCurrent.int_array(ringConfig)
    if not len(ringConfig):
      raise RuntimeError('Rings size must be specified.');

    if etaBins.size:
      nEtaBins = len(etaBins)-1
      if nEtaBins >= np.iinfo(npCurrent.scounter_dtype).max:
        raise RuntimeError(('Number of eta bins (%d) is larger or equal than maximum '
            'integer precision can hold (%d). Increase '
            'TuningTools.coreDef.npCurrent scounter_dtype number of bytes.'), nEtaBins,
            np.iinfo(npCurrent.scounter_dtype).max)
      if len(ringConfig) != nEtaBins:
        raise RuntimeError(('The number of rings configurations (%r) must be equal than ' 
                            'eta bins (%r) region config') % (ringConfig, etaBins))
      useBins=True
      useEtaBins=True
      self._logger.debug('eta bins enabled.')    
    else:
      self._logger.debug('eta/et bins disabled.')

    ### Prepare to loop:
    # Open root file
    import ROOT
    t = ROOT.TChain(treePath)
    if treePath != efficiencyTreePath:
      tEff = ROOT.TChain(efficiencyTreePath) 
    else: 
      tEff = t
    for inputFile in fList:
      # Check if file exists
      f  = ROOT.TFile.Open(inputFile, 'read')
      if f.IsZombie():
        raise RuntimeError('Couldn''t open file: %s', f)
      # Inform user whether TTree exists, and which options are available:
      self._logger.debug("Adding file: %s", inputFile)
      if not f.Get(treePath) or ( ( treePath != efficiencyTreePath ) and not f.Get(efficiencyTreePath)):
        self._logger.warning("Couldn't retrieve TTree (%s)!", treePath)
        self._logger.info("File available info:")
        f.ReadKeys()
        f.ls()
      t.Add( inputFile )
      if tEff is not t: tEff.Add( inputFile )

    # RingerPhysVal hold the address of required branches
    event = ROOT.RingerPhysVal()

    # Add offline branches, these are always needed
    cPos = 0
    for var in self.__offlineBranches:
      self.__setBranchAddress(t,var,event)
      self._logger.debug("Added branch: %s", var)

    # Add online branches if using Trigger
    if ringerOperation > 0:
      for var in self.__onlineBranches:
        self.__setBranchAddress(t,var,event)
        self._logger.debug("Added branch: %s", var)

    if not getRatesOnly:
      # Retrieve the rings information depending on ringer operation
      ringerBranch = "el_ringsE" if ringerOperation < 0 else \
                     "trig_L2_calo_rings"
      self.__setBranchAddress(t,ringerBranch,event)
      self._logger.debug("Added branch: %s", ringerBranch)

    ## Allocating memory for the number of entries
    entries = t.GetEntries() if not getRatesOnly else tEff.GetEntries()
    if tEff is not t and not getRatesOnly:
      effEntries = tEff.GetEntries()
      if effEntries != entries:
        raise RuntimeError("Size of efficiency tree (%d) does not match with base tree (%d)." % \
            (effEntries, entries))

    # Allocate numpy to hold as many entries as possible:
    if not getRatesOnly:
      t.GetEntry(0)
      npRings = np.zeros( shape=npCurrent.shape(npat=ringConfig.max(), #getattr(event, ringerBranch).size()
                                                nobs=(entries if (nClusters is None or nClusters > entries or nClusters < 1) \
                                                      else nClusters)
                                               ), 
                         dtype=npCurrent.fp_dtype,order=npCurrent.order)
      self._logger.debug("Allocated npRings with size %r", npRings.shape)
      
    else:
      npRings = npCurrent.fp_array([])

    ## Retrieve the dependent operation variables:
    if useEtBins:
      etBranch = 'el_et' if ringerOperation < 0 else 'trig_L2_calo_et'
      if offEtCut: self.__setBranchAddress(t, 'el_et', event)
      if l2EtCut:  self.__setBranchAddress(t,'trig_L2_calo_et',event)
      if efEtCut:  self.__setBranchAddress(t, 'trig_EF_calo_et',event)
      if not getRatesOnly:
        npEt    = np.zeros(shape=npRings.shape[npCurrent.odim],dtype=npCurrent.scounter_dtype)
        self._logger.debug("Allocated npEt    with size %r", npEt.shape)
    
    if useEtaBins:
      etaBranch    = "el_eta" if ringerOperation < 0 else "trig_L2_calo_eta"
      self.__setBranchAddress(t,etaBranch,event)
      self._logger.debug("Added branch: %s", etaBranch)
      if not getRatesOnly:
        npEta   = np.zeros(shape=npRings.shape[npCurrent.odim],dtype=npCurrent.scounter_dtype)
        self._logger.debug("Allocated npEta   with size %r", npEta.shape)

    ## Allocate the branch efficiency collectors:
    if ringerOperation < 0:
      benchmarkDict = OrderedDict(
        [('CutIDLoose',  'el_loose'),   
         ('CutIDMedium', 'el_medium'),  
         ('CutIDTight',  'el_tight'),   
         ('LHLoose',     'el_lhLoose'), 
         ('LHMedium',    'el_lhMedium'),
         ('LHTight',     'el_lhTight'), 
        ])
    else:
      benchmarkDict = OrderedDict(
        [('L2CaloAccept', 'trig_L2_calo_accept'), 
        ('L2ElAccept',    'trig_L2_el_accept'),   
        ('EFCaloAccept',  'trig_EF_calo_accept'), 
        ('EFElAccept',    'trig_EF_el_accept'),   
        ])
    branchEffCollectors = OrderedDict()
    branchCrossEffCollectors = OrderedDict()
    for key, val in benchmarkDict.iteritems():
      branchEffCollectors[key] = list()
      branchCrossEffCollectors[key] = list()
      # Add efficincy branch:
      self.__setBranchAddress(tEff,val,event)
      for etBin in range(nEtBins):
        if useBins:
          branchEffCollectors[key].append(list())
          branchCrossEffCollectors[key].append(list())
        for etaBin in range(nEtaBins):
          etBinArg = etBin if useBins else None
          etaBinArg = etaBin if useBins else None
          argList = [ key, val, etBinArg, etaBinArg ]
          branchEffCollectors[key][etBin].append(BranchEffCollector( *argList ) )
          if crossVal:
            branchCrossEffCollectors[key][etBin].append(BranchCrossEffCollector( entries, crossVal, *argList ) )
        # etBin
      # etaBin
    # benchmark dict
    if self._logger.isEnabledFor( LoggingLevel.DEBUG ):
      from RingerCore.util import traverse
      self._logger.debug( 'Retrieved following branch efficiency collectors: %r', 
          [collector[0].printName for collector in traverse(branchEffCollectors.values())])

    etaBin = 0; etBin = 0
    ## Start loop!
    self._logger.info("There is available a total of %d entries.", entries)
    for entry in range(entries):
     
      #self._logger.verbose('Processing eventNumber: %d/%d', entry, entries)
      t.GetEntry(entry)
      if not t is tEff: tEff.GetEntry(entry)

      # Check if it is needed to remove energy regions (this means that if not
      # within this range, it will be ignore for efficiency measuremnet)
      if (event.el_et < offEtCut): continue
      if ringerOperation > 0:
        
        # Remove events which didn't pass L1_calo
        if not event.trig_L1_accept: continue
        if (l1EmClusCut) and (event.trig_L1_emClus  < l1EmClusCut): continue
        if (l2EtCut)     and (event.trig_L2_calo_et < l2EtCut    ): continue
        passed=False
        if (event.trig_L2_calo_accept) and (efEtCut):
          # EF calo is a container, search for electrons objects with et > cut
          trig_EF_calo_et_list = stdvector_to_list(event.trig_EF_calo_et)
          for et in trig_EF_calo_et_list: 
            if et >= efEtCut:  
              passed=True
              break
        else:
          passed=True

        #Remove events which didn't pass EF calo energy
        if efEtCut and not passed: continue
        

      # Remove events without rings
      if not getRatesOnly:
        if getattr(event,ringerBranch).empty(): 
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
      else:
        if event.el_tight: target = Target.Signal 
        elif not event.el_loose: target = Target.Background 

      # Run filter if it is defined
      if filterType and \
         ( (filterType is FilterType.Signal and target != Target.Signal) or \
           (filterType is FilterType.Background and target != Target.Background) or \
           (target == Target.Unknown) ):
        continue

      # Retrieve dependent operation region
      if useEtBins:
        etBin  = self.__retrieveBinIdx( etBins, getattr(event, etBranch) )
      if useEtaBins:
        etaBin = self.__retrieveBinIdx( etaBins, np.fabs( getattr(event,etaBranch) ) )

      # Check if bin is within range (when not using bins, this will always be true):
      if (etBin < nEtBins and etaBin < nEtaBins):
        # Retrieve rates information:
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

        # Retrieve rings:
        if not getRatesOnly:
          npRings[npCurrent.access(oidx=cPos)] = stdvector_to_list( getattr(event,ringerBranch))
          if useEtBins:  npEt[cPos] = etBin
          if useEtaBins: npEta[cPos] = etaBin

        # We only increment if this cluster will be computed
        cPos += 1
     
      # Limit the number of entries to nClusters if desired and possible:
      if not nClusters is None and cPos >= nClusters:
        break
    # for end

    ## Treat the rings information
    if not getRatesOnly:

      ## Remove not filled reserved memory space:
      if npRings.shape[npCurrent.odim] > cPos:
        npRings = np.delete( npRings, slice(cPos,None), axis = npCurrent.odim)

      ## Segment data over bins regions:
      # Also remove not filled reserved memory space:
      if useEtBins:
        npEt  = npCurrent.delete( npEt, slice(cPos,None))
      if useEtaBins:
        npEta = npCurrent.delete( npEta, slice(cPos,None))
        
      # Separation for each bin found
      if useBins:
        npObject = np.empty((nEtBins,nEtaBins),dtype=object)
        for etBin in range(nEtBins):
          for etaBin in range(nEtaBins):
            if useEtBins and useEtaBins:
              # Retrieve all in current eta et bin
              idx = np.all([npEt==etBin,npEta==etaBin],axis=0).nonzero()[0]
              if len(idx): 
                npObject[etBin][etaBin]=npRings[npCurrent.access(oidx=idx)]
                # Remove extra features in this eta bin
                npObject[etBin][etaBin]=npCurrent.delete(npObject[etBin][etaBin],slice(ringConfig[etaBin],None),
                                                         axis=npCurrent.pdim)
            elif useEtBins:
              # Retrieve all in current et bin
              idx = (npEt==etBin).nonzero()[0]
              if len(idx):
                npObject[etBin][etaBin]=npRings[npCurrent.access(oidx=idx)]
            else:# useEtaBins
              # Retrieve all in current eta bin
              idx = (npEta==etaBin).nonzero()[0]
              if len(idx): 
                npObject[etBin][etaBin]=npRings[npCurrent.access(oidx=idx)]
                # Remove extra rings:
                npObject[etBin][etaBin]=npCurrent.delete(npObject[etBin][etaBin],slice(ringConfig[etaBin],None),
                                                         axis=npCurrent.pdim)
          # for etaBin
        # for etBin
      else:
        npObject = npRings
      # useBins
    else:
      npObject = npCurrent.array([], dtype=npCurrent.dtype)
    # not getRatesOnly

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
    return npObject, branchEffCollectors, branchCrossEffCollectors
  # end __call__

# Instantiate object
filterEvents = FilterEvents()

