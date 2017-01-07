__all__ = [ 'FilterType',  'Reference', 'RingerOperation', 'Target',
    'BaseInfo','PileupReference', 'Dataset', 'Detector', 'Dataframe']

from RingerCore import EnumStringification

class Dataframe(EnumStringification):
  """
    Select the input data frame type.
    - PhysVal: from Ryan's trigger egamma tool
    - TPNtuple: from Likelihood tag and probe package
  """
  PhysVal = 0
  Egamma  = 1

class RingerOperation(EnumStringification):
  """
    Select which framework ringer will operate
    - Positive values for Online operation; and 
    - Negative values for Offline operation.
  """
  _ignoreCase = True
  Offline_LH_VeryLoose = -10
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
    from TuningTools.coreDef import dataframeConf
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
  PileUp = 2
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

class PileupReference(EnumStringification):
  """
    Reference branch type for luminosity
  """
  _ignoreCase = True

  AverageLuminosity = 0
  avgmu = 0
  NumberOfVertices = 1
  nvtx = 1


