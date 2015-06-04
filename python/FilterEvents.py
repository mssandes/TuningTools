#!/usr/bin/env python
import ROOT
import numpy as np
from FastNetTool.util import stdvector_to_list

class RingerOperation:
  """
    Select which framework ringer will operate
  """
  Offline = 0
  Trigger = 1

class Reference:
  """
    Reference for training algorithm
  """
  Truth = 0
  Off_CutID = 1
  Off_Likelihood = 2
  

class FilterType:
  """
    Enumeration if selection event type w.r.t reference
  """
  DoNotFilter = 0
  Background = 1
  Signal = 2

class Target:
  """ 
    Holds the target value for the discrimination method
  """
  Signal = 1
  Background = -1
  Unknown = -999

class _FilterEvents:
  """
    Retrieve from TTree the traning information. Use filterEvents object.
  """

  # Offline information branches:
  __offlineBranches = ['el_pt',
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

  def __call__( self, fname, ringerOperation, **kw):
    """
      Returns ntuple with rings and its targets
      Arguments:
        - fname: The file path
        - ringerOperation: set Operation type
      Optional arguments:
        - filterType [None]: whether to filter. Use FilterType enumeration
        - reference [Truth]: set reference for targets. Use Reference enumeration
        - treeName ['CollectionTree']: set tree name on file
        - l1EmClusCut [None]: Set L1 cluster energy cut if operating for the trigger
    """

    # Retrieve information from keyword arguments
    filterType = kw.pop('filterType', FilterType.DoNotFilter )
    l1EmClusCut = kw.pop('l1EmClusCut', None )
    reference = kw.pop('reference', Reference.Truth )
    treeName = kw.pop('treeName', 'CollectionTree' )
    # and delete it to avoid mistakes:
    del kw

    # Retrieve python logger
    import logging
    logging.basicConfig()
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # Open root file
    f  = ROOT.TFile.Open(fname, 'read')
    if f.IsZombie():
      raise RuntimeError('Couldn''t open file: %s', f)

    t = f.Get(treeName)

    # Python list which will be used to retrieve objects
    ringsList  = []
    targetList = []

    # IEVentModel hold the address of required branches
    #gROOT.ProcessLine (".x $ROOTCOREDIR/scripts/load_packages.C");
    #ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
    if ROOT.gSystem.Load('libFastNetTool') < 0:
      log.error("Could not load FastNetTool library")

    event = ROOT.IEventModel()

    # Add offline branches, these are always needed
    for var in self.__offlineBranches:
      self.__setBranchAddress(t,var,event)

    # Added online branches if using Trigger
    if ringerOperation is RingerOperation.Trigger:
      for var in self.__onlineBranches:
        self.__setBranchAddress(t,var,event)

    # Retrieve the rings information depending on ringer operation
    ringerBranch = "el_ringsE" if ringerOperation is RingerOperation.Offline else "trig_L2_calo_rings"
    self.__setBranchAddress(t,ringerBranch,event)

    # Loop and retrieve information:
    entries = t.GetEntries()
    for entry in range(entries):
     
      #log.info('Processing eventNumber: %d/%d', entry, entries)
      t.GetEntry(entry)

      # Check if it is needed to remove using L1 energy cut
      if ringerOperation is RingerOperation.Trigger and l1EmClusCut:
        if event.trig_L1_emClus*0.001 < l1EmClusCut: continue

      # Remove events without rings
      if getattr(event,ringerBranch).empty(): continue

      # Set discriminator target:
      target = Target.Unknown
      if reference is Reference.Truth:
        if event.mc_isElectron and event.mc_hasZMother: target = Target.Signal 
        if not event.mc_hasMC or not event.mc_isElectron: target = Target.Background
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
    return [np.array(ringsList), np.array(targetList)]

# Instantiate object
filterEvents = _FilterEvents()

del _FilterEvents
