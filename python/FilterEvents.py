from RingerCore.util import EnumStringification

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


from RingerCore.Logger import Logger
import ROOT

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
                      'trig_L2_calo_accept',
                      'trig_L2_el_accept',
                      'trig_EF_calo_accept',
                      'trig_EF_el_accept']

  def __setBranchAddress( self, tree, varname, holder ):
    " Set tree branch varname to holder "
    tree.SetBranchAddress(varname, ROOT.AddressOf(holder,varname) )  

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
    """
    # Retrieve information from keyword arguments
    filterType    = kw.pop('filterType', FilterType.DoNotFilter )
    reference     = kw.pop('reference',     Reference.Truth     )
    l1EmClusCut   = kw.pop('l1EmClusCut',        None           )
    l2EtCut       = kw.pop('l2EtCut',            None           )
    treePath      = kw.pop('treePath',           None           )
    nClusters     = kw.pop('nClusters',          None           )
    getRatesOnly  = kw.pop('getRatesOnly',      False           )
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

    ### Loop and retrieve information:
    entries = t.GetEntries()

    # Allocate numpy to hold as many entries as possible:
    if entries > 0:
      if not getRatesOnly:
        import numpy as np
        t.GetEntry(0)
        npRings = np.zeros(shape=(entries if (nClusters is None or nClusters > entries or nClusters < 1) \
                                        else nClusters,
                                  getattr(event, ringerBranch).size()                               
                                 ), 
                           dtype='float32' )
        self._logger.debug("Allocated npRings with size %r", (npRings.shape,))
    else:
      npRings = np.array([], dtype='float32')

    if ringerOperation is RingerOperation.L2:
      count_l2calo_tot = count_l2calo_passed = 0
    elif ringerOperation is RingerOperation.Offline:
      count_off_tot = count_off_loose = count_off_medium \
                    = count_off_tight = 0

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

      # Update efficiency counters:
      if ringerOperation is RingerOperation.L2:
        count_l2calo_tot += 1
        if event.trig_L2_calo_accept: count_l2calo_passed+=1
      elif ringerOperation is RingerOperation.Offline:
        count_off_tot += 1
        if event.el_lhLoose: count_off_loose += 1
        if event.el_lhMedium: count_off_medium += 1
        if event.el_lhTight: count_off_tight += 1

      # Append information to data
      if not getRatesOnly:
        npRings[cPos,] = stdvector_to_list( getattr(event, ringerBranch) )
      cPos += 1

      # Limit the number of entries to nClusters if desired and possible:
      if not nClusters is None and cPos >= nClusters:
        break
    # for end

    # Retrieve and print effieciencies:
    if ringerOperation is RingerOperation.L2:
      eff = count_l2calo_passed/float(count_l2calo_tot)*100.
      self._logger.info('Efficiency on L2Calo trigger is: %.6f (%d/%d)',
                        eff,count_l2calo_passed, count_l2calo_tot)
      if getRatesOnly:
        return (eff), (count_l2calo_passed), count_l2calo_tot
    elif ringerOperation is RingerOperation.Offline:
      eff_off_loose = count_off_loose/float(count_off_tot)*100.
      eff_off_medium = count_off_medium/float(count_off_tot)*100.
      eff_off_tight = count_off_tight/float(count_off_tot)*100.
      self._logger.info('Efficiency for LH Loose is: %.6f (%d/%d)',
                        eff_off_loose,count_off_loose, count_off_tot)
      self._logger.info('Efficiency for LH Medium is: %.6f (%d/%d)',
                        eff_off_medium,count_off_medium, count_off_tot)
      self._logger.info('Efficiency for LH Tight is: %.6f (%d/%d)',
                        eff_off_tight,count_off_tight, count_off_tot)
      if getRatesOnly:
        return (eff_off_loose, eff_off_medium, eff_off_tight), \
               (count_off_loose, count_off_medium, count_off_tight), \
               count_off_tot

    if not getRatesOnly:
      # Remove not filled reserved memory space:
      npRings = np.delete( npRings, slice(cPos,None), axis = 0)

      # We've finished looping, return ringer np.array
      return npRings

# Instantiate object
filterEvents = FilterEvents()

