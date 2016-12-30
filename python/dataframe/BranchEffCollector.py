__all__ = ['BranchCrossEffCollector','BranchEffCollector']


from RingerCore import ( Logger, checkForUnusedVars, reshape, save, load, traverse
                       , retrieve_kw, NotSet, appendToFileName, LoggerRawDictStreamer
                       , RawDictCnv, RawDictStreamable, RawDictStreamer, LoggerStreamable
                       ,ensureExtension, secureExtractNpItem, progressbar 
                       )

from TuningTools.coreDef import retrieve_npConstants
from collections         import OrderedDict
from copy                import deepcopy

from TuningTools.dataframe.EnumCollection    import *
npCurrent, _ = retrieve_npConstants()
import numpy as np


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
      if type(d['_crossVal']) is dict: # Treat old files
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



