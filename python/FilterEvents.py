#!/usr/bin/env python
import ROOT
from FastNetTool.util import sourceEnvFile, stdvector_to_list
#sourceEnvFile()
import numpy as np

class EnumStringification:
  "Adds 'enum' static methods for conversion to/from string"
  @classmethod
  def tostring(cls, val):
    "Transforms val into string."
    for k,v in vars(cls).iteritems():
      if v==val:
        return k

  @classmethod
  def fromstring(cls, str):
    "Transforms string into enumeration."
    return getattr(cls, str, None)

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


from FastNetTool.Logger import Logger

class _FilterEvents(Logger):
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
                      'trig_L2_calo_accept',
                      'trig_L2_el_accept',
                      'trig_EF_calo_accept',
                      'trig_EF_el_accept']

  def __setBranchAddress( self, tree, varname, holder ):
    " Set tree branch varname to holder "
    tree.SetBranchAddress(varname, ROOT.AddressOf(holder,varname) )  

  def __init__( self, logger = None ):
    """
      Load FastNetTool C++ library and sets logger
    """
    # Retrieve python logger
    Logger.__init__( self, logger = logger)

    #gROOT.ProcessLine (".x $ROOTCOREDIR/scripts/load_packages.C");
    #ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
    if ROOT.gSystem.Load('libFastNetTool') < 0:
      self._logger.error("Could not load FastNetTool library")


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
        - treePath: set tree name on file
        - l1EmClusCut [None]: Set L1 cluster energy cut if operating for the trigger
    """
    # Retrieve information from keyword arguments
    filterType = kw.pop('filterType', FilterType.DoNotFilter )
    reference = kw.pop('reference', Reference.Truth )
    l1EmClusCut = kw.pop('l1EmClusCut', None )
    treePath = kw.pop('treePath', None )
    # and delete it to avoid mistakes:
    from FastNetTool.util import checkForUnusedVars
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

    # Python list which will be used to retrieve objects
    ringsList  = []
    targetList = []

    # IEVentModel hold the address of required branches
    event = ROOT.IEventModel()

    # Add offline branches, these are always needed
    for var in self.__offlineBranches:
      self.__setBranchAddress(t,var,event)
      self._logger.debug("Added branch: %s", var)

    # Add online branches if using Trigger
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
    self._logger.info("There is available a total of %d entries.", entries)
    for entry in range(entries):
     
      #self._logger.verbose('Processing eventNumber: %d/%d', entry, entries)
      t.GetEntry(entry)
      
      # Check if it is needed to remove using L1 energy cut
      if ringerOperation is  RingerOperation.L2:
        if (event.trig_L1_emClus < l1EmClusCut): continue
      
      # Remove events without rings
      if getattr(event,ringerBranch).empty(): continue

      # Set discriminator target:
      target = Target.Unknown
      if reference is Reference.Truth:
        if event.mc_isElectron and event.mc_hasZMother: 
          target = Target.Signal 
        if not event.mc_isElectron: target = Target.Background
      elif reference is Reference.Off_Likelihood:
        if event.el_lhTight: target = Target.Signal
        if not event.el_lhLoose: target = Target.Background
      else:
        if event.el_tight: target = Target.Signal 
        if not event.el_loose: target = Target.Background 

      # Run filter if it is defined
      if filterType:
        if (filterType is FilterType.Signal) and target != Target.Signal:
          continue
        if (filterType is FilterType.Background) and target != Target.Background:  
          continue
        if target is Target.Unknown: 
          continue

      # Append information to data
      ringsList.append( stdvector_to_list( getattr(event, ringerBranch) ) )
      targetList.append( target )
    # for end

    # We've finished looping, return numpy arrays from the lists
    return (np.array(ringsList, dtype='float32'), np.array(targetList, dtype='int32'))

# Instantiate object
filterEvents = _FilterEvents()

del _FilterEvents
