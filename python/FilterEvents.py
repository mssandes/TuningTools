from RingerCore.util import EnumStringification
import numpy as np

class RingerOperation(EnumStringification):
  """
    Select which framework ringer will operate
  """
  Offline = -1
  L2  = 1
  EF = 2

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


from RingerCore.Logger import Logger, LoggingLevel
import ROOT

class BranchEffCollector(object):
  """
    Simple class for counting passed events using input branch
  """

  _passed = 0
  _count = 0

  def __init__(self, name, branchBuffer):
    self.name = name
    self._branchBuffer = branchBuffer

  def update(self):
    " Update the counting. "
    if self._branchBuffer[0]: self._passed += 1
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
    return (self.name + " efficiency is: " + self.eff_str() )

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
                       'mc_hasZMother']

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
    tree.SetBranchAddress(varname, ROOT.AddressOf(holder,varname) )  


  def __retrieveBinIdx( self, bins, value ):
    return int(np.digitize(np.array([value]), bins)[0]-1)

  def __init__( self, logger = None ):
    """
      Load TuningTools C++ library and set logger
    """
    # Retrieve python logger
    Logger.__init__( self, logger = logger)

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
        - ll2EtCut [None]: Set L2 cluster energy cut value if operating on the trigger
        - nClusters [None]: Read up to nClusters. Use None to run for all clusters.
        - getRatesOnly [False]: Read up to nClusters. Use None to run for all clusters.
        - etBins [None]: E_T bins where the data should be segmented
        - etaBins [None]: eta bins where the data should be segmented
        - ringConfig [100]: A list containing the number of rings available in the data
          for each eta bin.
    """
    # Retrieve information from keyword arguments
    filterType    = kw.pop('filterType', FilterType.DoNotFilter )
    reference     = kw.pop('reference',     Reference.Truth     )
    l1EmClusCut   = kw.pop('l1EmClusCut',        None           )
    l2EtCut       = kw.pop('l2EtCut',            None           )
    treePath      = kw.pop('treePath',           None           )
    nClusters     = kw.pop('nClusters',          None           )
    getRatesOnly  = kw.pop('getRatesOnly',      False           )
    etBins        = kw.pop('etBins',             None           )
    etaBins       = kw.pop('etaBins',            None           )
    ringConfig    = kw.pop('ringConfig',         None           )
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
    if isinstance(ringerOperation, str):
      ringerOperation = RingerOperation.fromstring(ringerOperation)
    if isinstance(reference, str):
      reference = Reference.fromstring(reference)
    if isinstance(l1EmClusCut, str):
      l1EmClusCut = float(l1EmClusCut)
    if l1EmClusCut:
      l1EmClusCut = 1000*l1EmClusCut # Put energy in MeV
    if l2EtCut:
      l2EtCut = 1000*l2EtCut # Put energy in MeV
    # Check if treePath is None and try to set it automatically
    if treePath is None:
      treePath = 'Offline/Egamma/Ntuple/electron' if ringerOperation is RingerOperation.Offline else \
                 'Trigger/HLT/Egamma/TPNtuple/e24_medium_L1EM18VH'
    # Check whether using bins
    useBins=False; useEtBins=False; useEtaBins=False
    nEtaBins = 1; nEtBins = 1

    if etBins:
      if type(etBins)  is list: etBins=np.array(etBins)
      nEtBins  = etBins.shape[0]-1
      # Flag that we are separating data through bins
      useBins=True
      useEtBins=True
      self._logger.debug('E_T bins enabled.')    

    if not type(ringConfig) is list and not type(ringConfig) is np.ndarray:
      ringConfig = [ringConfig]
    if type(ringConfig) is list: ringConfig=np.array(ringConfig)
    if not len(ringConfig):
      raise RuntimeError('Rings size must be specified.');

    if etaBins:
      if type(etaBins) is list: etaBins=np.array(etaBins)
      nEtaBins = etaBins.shape[0]-1
      if ringConfig.shape[0] != nEtaBins:
        raise RuntimeError(('The number of rings configurations (%r) must be equal than ' 
                            'eta bins (%r) region config') % (ringConfig, etaBins))
      useBins=True
      useEtaBins=True
      self._logger.debug('eta bins enabled.')    
    else:
      self._logger.debug('eta/et bins disabled.')

    ### Prepare to loop:
    # Open root file
    t = ROOT.TChain(treePath)
    for inputFile in fList:
      # Check if file exists
      f  = ROOT.TFile.Open(inputFile, 'read')
      if f.IsZombie():
        raise RuntimeError('Couldn''t open file: %s', f)
      self._logger.debug("Adding file: %s", inputFile)
      t.Add( inputFile )

    # IEVentModel hold the address of required branches
    event = ROOT.IEventModel()

    # Add offline branches, these are always needed
    cPos = 0
    for var in self.__offlineBranches:
      self.__setBranchAddress(t,var,event)
      self._logger.debug("Added branch: %s", var)

    # Add online branches if using Trigger
    if not getRatesOnly:
      if ringerOperation is RingerOperation.L2 or ringerOperation is RingerOperation.EF:
        for var in self.__onlineBranches:
          self.__setBranchAddress(t,var,event)
          self._logger.debug("Added branch: %s", var)

      # Retrieve the rings information depending on ringer operation
      ringerBranch = "el_ringsE" if ringerOperation is RingerOperation.Offline else \
                     "trig_L2_calo_rings"
      self.__setBranchAddress(t,ringerBranch,event)
      self._logger.debug("Added branch: %s", ringerBranch)

    ## Allocating memory for the number of entries
    entries = t.GetEntries()

    # Allocate numpy to hold as many entries as possible:
    if not getRatesOnly:
      t.GetEntry(0)
      npRings = np.zeros(shape=(entries if (nClusters is None or nClusters > entries or nClusters < 1) \
                                      else nClusters,
                                #getattr(event, ringerBranch).size()          
                                ringConfig.max()
                               ), 
                         dtype='float32' )
      self._logger.debug("Allocated npRings with size %r", (npRings.shape,))
      
    else:
      npRings = np.array([], dtype='float32')

    ## Retrieve the dependent operation variables:
    if useEtBins:
      etBranch     = "el_et" if ringerOperation is RingerOperation.Offline else \
                     "trig_L2_calo_et"
      self.__setBranchAddress(t,etBranch,event)
      self._logger.debug("Added branch: %s", etBranch)
      if not getRatesOnly:
        npEt    = np.zeros(shape=npRings.shape[0])
        self._logger.debug("Allocated npEt    with size %r", (npEt.shape,))
    if useEtaBins:
      etaBranch    = "el_eta" if ringerOperation is RingerOperation.Offline else \
                     "trig_L2_calo_eta"
      self.__setBranchAddress(t,etaBranch,event)
      self._logger.debug("Added branch: %s", etaBranch)
      if not getRatesOnly:
        npEta   = np.zeros(shape=npRings.shape[0])
        self._logger.debug("Allocated npEta   with size %r", (npEta.shape,))

    ## Allocate the branch efficiency collectors:
    branchEffCollectors = []
    for etBin in range(nEtBins):
      if useBins:
        branchEffCollectors.append(list())
      for etaBin in range(nEtaBins):
        if useBins:
          branchEffCollectors[etBin].append(list())
        currentCollectorList = branchEffCollectors[etBin][etaBin] if useBins else \
                               branchEffCollectors
        suffix = ('_etBin_%d_etaBin_%d') % (etBin, etaBin) if useBins else ''
        if ringerOperation is RingerOperation.Offline:
          currentCollectorList.append(BranchEffCollector( 'CutIDLoose%s' % suffix,   
                                                      ROOT.AddressOf(event,'el_loose')            ) )
          currentCollectorList.append(BranchEffCollector( 'CutIDMedium%s' % suffix,  
                                                      ROOT.AddressOf(event,'el_medium')           ) )
          currentCollectorList.append(BranchEffCollector( 'CutIDTight%s' % suffix,   
                                                      ROOT.AddressOf(event,'el_tight')            ) )
          currentCollectorList.append(BranchEffCollector( 'LHLoose%s' % suffix,  
                                                      ROOT.AddressOf(event,'el_lhLoose')          ) )
          currentCollectorList.append(BranchEffCollector( 'LHMedium%s' % suffix,     
                                                      ROOT.AddressOf(event,'el_lhMedium')         ) )
          currentCollectorList.append(BranchEffCollector( 'LHTight%s' % suffix,      
                                                      ROOT.AddressOf(event,'el_lhTight')          ) )
        else:
          currentCollectorList.append(BranchEffCollector( 'L2CaloAccept%s' % suffix, 
                                                      ROOT.AddressOf(event,'trig_L2_calo_accept') ) )
          currentCollectorList.append(BranchEffCollector( 'L2ElAccept%s' % suffix, 
                                                      ROOT.AddressOf(event,'trig_L2_el_accept')   ) )
          currentCollectorList.append(BranchEffCollector( 'EFCaloAccept%s' % suffix, 
                                                      ROOT.AddressOf(event,'trig_EF_calo_accept') ) )
          currentCollectorList.append(BranchEffCollector( 'EFElAccept%s' % suffix, 
                                                      ROOT.AddressOf(event,'trig_EF_el_accept')   ) )
        # ringerOperation
      # etBin
    # etaBin
    if self._logger.isEnabledFor( LoggingLevel.DEBUG ):
      from RingerCore.util import traverse
      self._logger.debug( 'Retrieved following branch efficiency collectors: %r', 
          [collector.name for collector in traverse(branchEffCollectors)])

    etaBin = 0; etBin = 0
    ## Start loop!
    self._logger.info("There is available a total of %d entries.", entries)
    for entry in range(entries):
     
      #self._logger.verbose('Processing eventNumber: %d/%d', entry, entries)
      t.GetEntry(entry)

      # Check if it is needed to remove using L1 energy cut
      if ringerOperation is RingerOperation.L2:
        if (event.trig_L1_emClus < l1EmClusCut): continue
        if (event.trig_L2_calo_et < l2EtCut):  continue

      # Remove events without rings
      if not getRatesOnly:
        if getattr(event,ringerBranch).empty(): continue

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
        etBin  = self.__retrieveBinIdx( etBins, getattr(event, etBranch)/1000. )
      if useEtaBins:
        etaBin = self.__retrieveBinIdx( etaBins, np.fabs( getattr(event,etaBranch) ) )

      # Retrieve rates information:
      if (etBin < nEtBins and etaBin < nEtaBins):
        for branch in branchEffCollectors if not useBins else \
                      branchEffCollectors[etBin][etaBin]:
          branch.update()
        # We only increment if this cluster will be computed
        if not getRatesOnly:
          npRings[cPos,] = stdvector_to_list( getattr(event,ringerBranch))
          if useEtBins:  npEt[cPos] = etBin
          if useEtaBins: npEta[cPos] = etaBin
        cPos += 1
     
      # Limit the number of entries to nClusters if desired and possible:
      if not nClusters is None and cPos >= nClusters:
        break
    # for end

    ## Treat the rings information
    if not getRatesOnly:
      ## Remove not filled reserved memory space:
      npRings = np.delete( npRings, slice(cPos,None), axis = 0)

      ## Segment data over bins regions:
      # Also remove not filled reserved memory space:
      if useEtBins:
        npEt  = np.delete( npEt, slice(cPos,None), axis = 0)
      if useEtaBins:
        npEta = np.delete( npEta, slice(cPos,None), axis = 0)
        
      # Separation for each bin found
      if useBins:
        npObject = np.empty((nEtBins,nEtaBins),dtype=object)
        for etBin in range(nEtBins):
          for etaBin in range(nEtaBins):
            if useEtBins and useEtaBins:
              npObject[etBin][etaBin]=npRings[np.all([npEt==etBin,npEta==etaBin],axis=0).nonzero()[0]][:]
            elif useEtBins:
              npObject[etBin][etaBin]=npRings[(npEt==etBin).nonzero()[0]][:]
            else:
              npObject[etBin][etaBin]=npRings[(npEta==etaBin).nonzero()[0]][:]
              # Remove extra rings:
              npObject[etBin][etaBin]=np.delete(npObject[etBin][etaBin],slice(ringConfig[etaBin],None),axis=1)
          # for etaBin
        # for etBin
      else:
        npObject = npRings
      # useBins
    # not getRatesOnly

    # Print efficiency for each one for the efficiency branches analysed:
    for etBin in range(nEtBins) if useBins else range(1):
      for etaBin in range(nEtaBins) if useBins else range(1):
        for branch in branchEffCollectors if not useBins else \
                      branchEffCollectors[etBin][etaBin]:
          self._logger.info('%s',branch)
        # for branch
      # for eta
    # for et

    return npObject, branchEffCollectors
  # end __call__

# Instantiate object
filterEvents = FilterEvents()

