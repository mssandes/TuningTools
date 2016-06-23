__all__ = []

#from RingerCore import EnumStringification, get_attributes, checkForUnusedVars, \
#    calcSP, save, load, Logger, LoggingLevel, expandFolders, traverse, retrieve_kw, NotSet
#
#from TuningTools.TuningJob import TunedDiscrArchieve, ReferenceBenchmark, ReferenceBenchmarkCollection
#from TuningTools import PreProc
#from TuningTools.ReadData import Dataset
#from pprint import pprint
#from cPickle import UnpicklingError
#import numpy as np
#import os
#from RingerCore import StoreGate
#
#def percentile( data, score ):
#  """
#  val = percentile( data, score )
#  Retrieve the data percentile at score
#  """
#  size = len(data)
#  if size:
#    pos = score*size
#    if pos % 10 or pos == size:
#      return data[pos]
#    else:
#      return data[pos] + data[pos+1]
#  else: return None
#
#def fixReferenceBenchmarkCollection( refCol, nBins, nTuned ):
#  """
#    Make sure that input data is a ReferenceBenchmarkCollection( ReferenceBenchmarkCollection([...]) ) 
#    with dimensions [nBins][nTuned] or transform it to that format if it is possible.
#  """
#  try: 
#    for _, _, _, _, level in traverse(refCol, tree_types = (ReferenceBenchmarkCollection, list, tuple )): pass
#  except TypeError:
#    level = 0
#
#  if level == 0:
#    refCol = ReferenceBenchmarkCollection( [refCol] * nTuned )
#    refCol = ReferenceBenchmarkCollection( [refCol] * nBins )
#  elif level == 1:
#    lRefCol = len(refCol)
#    if lRefCol == 1:
#      refCol = ReferenceBenchmarkCollection( refCol * nTuned )
#      refCol = ReferenceBenchmarkCollection( [refCol] * nBins )
#    elif lRefCol == nBins:
#      refCol = ReferenceBenchmarkCollection( refCol )
#      refCol = ReferenceBenchmarkCollection( [refCol] * nTuned )
#    elif lRefCol == nTuned:
#      refCol = ReferenceBenchmarkCollection( refCol )
#      refCol = ReferenceBenchmarkCollection( [refCol] * nBins )
#    else:
#      raise ValueError(("The ReferenceBenchmark collection size does not " \
#          "match either the number of tuned operating points or the number of bins."))
#  elif level == 2:
#    for obj, idx, parent, depth_dist, level in traverse(refCol, 
#                                                        tree_types = (ReferenceBenchmarkCollection, list, tuple ), 
#                                                        max_depth = 1,
#                                                       ):
#      obj = ReferenceBenchmarkCollection(obj)
#      if len(obj) == 1:
#        obj = obj * nTuned
#      elif len(obj) != nTuned and nTuned != 1:
#        raise ValueError("Internal collection size does not match nTuned size.")
#      parent[idx] = obj
#    refCol = ReferenceBenchmarkCollection( refCol )
#    if len( refCol ) < nBins:
#      raise ValueError(("Cannot use less configurations (%d) than the "
#          "total number of bins available (%d).") % (len(refCol), nBins))
#  else:
#    raise ValueError("Collection dimension is greater than 2!")
#
#  return refCol
#
#class JobFilter( object ):
#  def __call__(self, paths):
#    return []
#
#class GridJobFilter( JobFilter ):
#
#  import re
#  pat = re.compile(r'.*user.[a-zA-Z0-9]+.(?P<jobID>[0-9]+)\..*$')
#  #pat = re.compile(r'user.(?P<user>[A-z0-9]*).(?P<jobID>[0-9]+).*\.tgz')
#
#  def __call__(self, paths):
#    jobIDs = sorted(list(set(['*' + self.pat.match(f).group('jobID') + '*' for f in paths if self.pat.match(f) is not None])))
#    return jobIDs
#
#class CrossValidStatAnalysis( Logger ):
#
#  def __init__(self, paths, **kw):
#    """
#    Usage: 
#
#    # Create object
#    cvStatAna = CrossValidStatAnalysis( paths 
#                                        [,binFilters=None]
#                                        [,logger[,level=INFO]]
#                                      )
#    # Fill the information and save output file with cross-validation summary
#    cvStatAna( refBenchMark, **args...)
#    # Call other methods if necessary.
#    """
#    Logger.__init__(self, kw)    
#    self._binFilters = retrieve_kw(kw, 'binFilters',            None           )
#    self._binFilterJobIdxs  = retrieve_kw(kw, 'binFilterIdxs',  None           )
#    mFName           = retrieve_kw(kw, 'monitoringFileName', 'monitoring'      )
#
#    checkForUnusedVars(kw, self._logger.warning)
#    if not mFName.endswith( '.root' ): mFName += '.root'
#    # Recursively expand all folders in the given paths so that we have all
#    # files lists:
#    if hasattr(self._binFilters,'__call__'):
#      self._paths = expandFolders( paths )
#      #import types
#      #if not type(self._binFilters) is types.FunctionType:
#      if type(self._binFilters) is type:
#        self._binFilters = self._binFilters()
#      self._binFilters = self._binFilters( self._paths )
#
#      #Retrieve only the bin IDx selected by arg
#      if self._binFilterJobIdxs is not None:
#        try:
#          self._binFilters = [self._binFilters[idx] for idx in self._binFilterJobIdxs]
#        except IndexError:
#          raise IndexError('This bin idx doest exist.')
#        self._logger.warning('Taken only the bin with index %r in binFilter list', self._binFilterJobIdxs)
#      
#      self._logger.info('Found following filters: %r', self._binFilters)
#      # FIXME This should be expandFolder, but rather a method for filtering
#      # the data
#      self._paths = expandFolders( self._paths, self._binFilters ) 
#    else:
#      self._paths = expandFolders( paths, self._binFilters )
#    if not(self._binFilters is None):
#      self._nBins = len(self._binFilters)
#    else:
#      self._nBins = 1
#    if self._nBins is 1:
#      self._paths = [self._paths]
#    if self._level <= LoggingLevel.VERBOSE:
#      for binFilt in self._binFilters if self._binFilters is not None else [None]:
#        self._logger.verbose("The stored files are (binFilter=%s):", binFilt)
#        for path in self._paths:
#          self._logger.verbose("%s", path)
#    self._nFiles = [len(l) for l in self._paths]
#    self._logger.info("A total of %r files were found.", self._nFiles )
#    self._sg = None
#
#  def __addPerformance( self, tunedDiscrInfo, path, ref, 
#                              neuron, sort, init, 
#                              etBinIdx, etaBinIdx, 
#                              tunedDiscr, trainEvolution ):
#
#    refName = ref.name
#    # We need to make sure that the key will be available on the dict if it
#    # wasn't yet there
#    if not refName in tunedDiscrInfo:
#      tunedDiscrInfo[refName] = { 'benchmark' : ref,
#                                  'tuningBenchmark' : tunedDiscr['benchmark'] }
#    if not neuron in tunedDiscrInfo[refName]:
#      tunedDiscrInfo[refName][neuron] = dict()
#    if not sort in tunedDiscrInfo[refName][neuron]:
#      tunedDiscrInfo[refName][neuron][sort] = { 'headerInfo' : [], 
#                                                'initPerfTstInfo' : [], 
#                                                'initPerfOpInfo' : [] }
#    # The performance holder, which also contains the discriminator
#    perfHolder = PerfHolder( tunedDiscr, trainEvolution )
#    # Retrieve operating points:
#    (spTst, detTst, faTst, cutTst, idxTst) = perfHolder.getOperatingBenchmarks(ref)
#    (spOp, detOp, faOp, cutOp, idxOp)      = perfHolder.getOperatingBenchmarks(ref, ds = Dataset.Operation)
#    headerInfo = { 'discriminator' : tunedDiscr['discriminator'],
#                   'neuron' : neuron, 'sort' : sort, 'init' : init,
#                 }
#    # Create performance holders:
#    iInfoTst = { 'sp' : spTst, 'det' : detTst, 'fa' : faTst, 'cut' : cutTst, 'idx' : idxTst, }
#    iInfoOp  = { 'sp' : spOp,  'det' : detOp,  'fa' : faOp,  'cut' : cutOp,  'idx' : idxOp,  }
#    if self._level <= LoggingLevel.VERBOSE:
#      self._logger.verbose("Retrieved file '%s' configuration for benchmark '%s' as follows:", 
#                         os.path.basename(path),
#                         ref )
#      pprint({'headerInfo' : headerInfo, 'initPerfTstInfo' : iInfoTst, 'initPerfOpInfo' : iInfoOp })
#    # Append information to our dictionary:
#    tunedDiscrInfo[refName][neuron][sort]['headerInfo'].append( headerInfo )
#    tunedDiscrInfo[refName][neuron][sort]['initPerfTstInfo'].append( iInfoTst )
#    tunedDiscrInfo[refName][neuron][sort]['initPerfOpInfo'].append( iInfoOp )
#
#    #Adding graphs into monitoring file
#    init = len(tunedDiscrInfo[refName][neuron][sort]['initPerfOpInfo'])-1
#    dirname = ('trainEvolution/%s/config_%d/sort_%d/init_%d') % (ref.name,neuron,sort,init)
#    self._sg.mkdir(dirname)
#    graphNames = [ 'mse_trn', 'mse_val', 'mse_tst',
#         'sp_val', 'sp_tst',
#         'det_val', 'det_tst',
#         'fa_val', 'fa_tst',
#         'det_fitted', 'fa_fitted',
#         'roc_tst', 'roc_op',]
#         #'roc_tst_cut', 'roc_op_cut' ]
#
#    #Attach graphs
#    for gname in graphNames:
#      g = perfHolder.getGraph(gname); g.SetName(gname)
#      self._sg.attach(g)
#
#    #Attach stops
#    from RingerCore.util import createRootParameter
#    self._sg.attach( createRootParameter("double","mse_stop", perfHolder.epoch_stop_mse) )
#    self._sg.attach( createRootParameter("double","sp_stop" , perfHolder.epoch_stop_sp ) )
#    self._sg.attach( createRootParameter("double","det_stop", perfHolder.epoch_stop_det) )
#    self._sg.attach( createRootParameter("double","fa_stop" , perfHolder.epoch_stop_fa ) )
#  
#  def __call__(self, **kw ):
#    """
#    Hook for loop method.
#    """
#    self.loop( **kw )
#
#
#  def loop(self, **kw ):
#    """
#    Optional args:
#      * refBenchmarkList: a list of reference benchmark objects which will be used
#        as the operation points.
#      * toMatlab [True]: also create a matlab file from the obtained tuned discriminators
#      * outputName ['crossValStat']: the output file name.
#      * debug [False]: Run only for a small number of files
#    """
#    import gc
#    refBenchmarkList = retrieve_kw( kw, 'refBenchmarkList', None           )
#    toMatlab         = retrieve_kw( kw, 'toMatlab',         True           )
#    outputName       = retrieve_kw( kw, 'outputName',       'crossValStat' )
#    debug            = retrieve_kw( kw, 'debug',            False          )
#    checkForUnusedVars( kw, self._logger.warning )
#    tuningBenchmarks = ReferenceBenchmarkCollection([])
#
#    pbinIdxList=[]
#    for binIdx, binPath in enumerate(self._paths):
#<<<<<<< Updated upstream
#      with TunedDiscrArchieve(binPath[0]) as TDArchieve:
#        tunedArchieveDict = TDArchieve.getTunedInfo( TDArchieve.neuronBounds[0],
#                                                     TDArchieve.sortBounds[0],
#                                                     TDArchieve.initBounds[0] )
#        tunedDiscrList    = tunedArchieveDict['tunedDiscr']
#        try:
#          if nTuned  - len(tunedDiscrList):
#            raise RuntimeError("For now, all bins must have the same tuned benchmarks.")
#        except NameError:
#          pass
#        nTuned            = len(tunedDiscrList)
#        binTuningBench    = ReferenceBenchmarkCollection( 
#                               [tunedDiscrDict['benchmark'] for tunedDiscrDict in tunedDiscrList]
#                            )
#        tuningBenchmarks.append( binTuningBench )
#        etBinIdx          = TDArchieve.etBinIdx
#        etaBinIdx         = TDArchieve.etaBinIdx
#        pbinIdxList.append( (etBinIdx, etaBinIdx) )
#
#      self._logger.debug("Found a total of %d tuned operation points on bin (et:%d,eta:%d). They are: ", 
#          nTuned,
#          etBinIdx,
#          etaBinIdx
#          )
#      for bench in binTuningBench:
#        self._logger.debug("%s", bench)
#=======
#      tdArchieve =  TunedDiscrArchieve.load(binPath[0])
#      tunedArchieveDict = tdArchieve.getTunedInfo( tdArchieve.neuronBounds[0],
#                                                   tdArchieve.sortBounds[0],
#                                                   tdArchieve.initBounds[0] )
#      tunedDiscrList    = tunedArchieveDict['tunedDiscr']
#      try:
#        if nTuned  - len(tunedDiscrList):
#          raise RuntimeError("For now, all bins must have the same tuned benchmarks.")
#      except NameError:
#        pass
#      nTuned            = len(tunedDiscrList)
#      binTuningBench    = ReferenceBenchmarkCollection( 
#                             [tunedDiscrDict['benchmark'] for tunedDiscrDict in tunedDiscrList]
#                          )
#      tuningBenchmarks.append( binTuningBench )
#      etBinIdx          = tdArchieve.etBinIdx
#      etaBinIdx         = tdArchieve.etaBinIdx
#
#    self._logger.debug("Found a total of %d tuned operation points on bin (et:%d,eta:%d). They are: ", 
#        nTuned,
#        etBinIdx,
#        etaBinIdx
#        )
#    for bench in binTuningBench:
#      self._logger.debug("%s", bench)
#>>>>>>> Stashed changes
#
#
#    # Make sure everything is ok with the reference benchmark list:
#    refBenchmarkList = fixReferenceBenchmarkCollection(refBenchmarkList, len(self._paths), nTuned)
#   
#    # Match between benchmarks from pref and files in path
#    if len(refBenchmarkList) != 1 and refBenchmarkList[0][0] is not None:
#      trefBenchmarkList=[]
#      for etBinIdx, etaBinIdx in pbinIdxList:
#        for idx, refBenchmark in enumerate(refBenchmarkList):
#          if refBenchmark[0].checkEtBinIdx(etBinIdx) and refBenchmark[0].checkEtaBinIdx(etaBinIdx):
#            self._logger.info('BenchmarkCollection found in perf file with operation on bin (et:%d,eta:%d). They are:', etBinIdx,etaBinIdx)
#            for cref in refBenchmark:  self._logger.debug('%s',cref)
#            trefBenchmarkList.append(refBenchmarkList.pop(idx))
#      refBenchmarkList=trefBenchmarkList
#
#
#
#    self._logger.info("Started analysing cross-validation statistics...")
#    self._summaryInfo = [ dict() for i in range(self._nBins) ]
#    self._summaryPPInfo = [ dict() for i in range(self._nBins) ]
#
#    # Loop over the files
#    from itertools import product
#    for binIdx, binPath in enumerate(self._paths):
#      if self._binFilters is not None:
#        self._logger.info("Running bin filter '%s'...",self._binFilters[binIdx])
#      tunedDiscrInfo = dict()
#      cSummaryInfo = self._summaryInfo[binIdx]
#      cSummaryPPInfo = self._summaryPPInfo[binIdx]
#      cRefBenchmarkList = refBenchmarkList[binIdx]
#
#      # If user inserted None and 
#      if cRefBenchmarkList[0] is None and len(cRefBenchmarkList) == 1 and  \
#          len(tuningBenchmarks[binIdx]) == 1 and \
#          tuningBenchmarks[binIdx][0].reference in (ReferenceBenchmark.SP, ReferenceBenchmark.MSE):
#        self._logger.info("Found a unique tuned MSE or SP reference. Expanding it to SP/Pd/Pf operation points.")
#        from copy import copy
#        copyRefList = ReferenceBenchmarkCollection( [copy(ref) for ref in cRefBenchmarkList] )
#        if refBenchmark.signal_efficiency is not None:
#          opRefs = [ReferenceBenchmark.SP, ReferenceBenchmark.Pd, ReferenceBenchmark.Pf]
#          for ref, copyRef in zip(opRefs, copyRefList):
#            copyRef.reference = ref
#            copyRef.name = copyRef.name.replace("Tuning_", "OperationPoint_") \
#                                       .replace("_" + ReferenceBenchmark.tostring(cRefBenchmarkList[0].reference),
#                                                "_" + ReferenceBenchmark.tostring(ref))
#        else:
#          if copyRefList.reference is ReferenceBenchmark.MSE:
#            copyRefList[0].name = "OperationPoint_" + copyRefList[0].split("_")[1] + "_SP"
#        cRefBenchmarkList = copyRefList
#      for idx, refBenchmark in enumerate(cRefBenchmarkList):
#        if refBenchmark is None:
#          cRefBenchmarkList[idx] = tuningBenchmarks[binIdx][idx]
#          cRefBenchmarkList[idx].name = cRefBenchmarkList[idx].name.replace('Tuning_', 'OperationPoint_')
#
#      self._logger.info('Using references: %r.', [(ReferenceBenchmark.tostring(ref.reference),ref.refVal) for ref in cRefBenchmarkList])
#
#      #For monitoring file name
#      if self._binFilters is not None:
#        cOutputName = outputName + self._binFilters[binIdx].replace('*','_')
#        if cOutputName.endswith('_'): 
#          cOutputName = cOutputName[:-1]
#      else:
#        cOutputName = outputName
#      
#      self._sg = StoreGate( cOutputName + '_monitoring.root' )
#      self._sg.setProperty( holdObj = False )
#
#      for cFile, path in enumerate(binPath):
#        self._logger.info("Reading file %d/%d (%s)", cFile, self._nFiles[binIdx], path )
#        # And open them as Tuned Discriminators:
#        try:
#<<<<<<< Updated upstream
#          with TunedDiscrArchieve(path) as TDArchieve:
#            if TDArchieve.etaBinIdx != -1 and cFile == 0:
#              self._logger.info("File eta bin index (%d) limits are: %r", 
#                                 TDArchieve.etaBinIdx, 
#                                 TDArchieve.etaBin, 
#                                )
#            if TDArchieve.etBinIdx != -1 and cFile == 0:
#              self._logger.info("File Et bin index (%d) limits are: %r", 
#                                 TDArchieve.etBinIdx, 
#                                 TDArchieve.etBin, 
#                               )
#            # Now we loop over each configuration:
#            for neuron, sort, init in product( TDArchieve.neuronBounds(), 
#                                               TDArchieve.sortBounds(), 
#                                               TDArchieve.initBounds() ):
#
#              tunedDict      = TDArchieve.getTunedInfo( neuron, sort, init )
#              tunedDiscr     = tunedDict['tunedDiscr']
#              tunedPPChain   = tunedDict['tunedPP']
#              trainEvolution = tunedDict['tuningInfo']
#
#              if not len(tunedDiscr) == nTuned:
#                raise ValueError("File %s contains diffenret number of tunings in the collection.")
#
#              # We loop on each reference benchmark we have.
#              for idx, refBenchmark in enumerate(cRefBenchmarkList):
#
#                # Check if binning information matches:
#                if neuron == TDArchieve.neuronBounds.lowerBound() and \
#                   sort == TDArchieve.sortBounds.lowerBound() and \
#                   init == TDArchieve.initBounds.lowerBound() and \
#                   idx == 0:
#                  if not refBenchmark.checkEtaBinIdx(TDArchieve.etaBinIdx):
#                    if refBenchmark.etaBinIdx is None:
#                      self._logger.warning("TunedDiscrArchieve does not contain eta binning information!")
#                    else:
#                      self._logger.error("File (%d) eta binning information does not match with benchmark (%r)!", 
#                          TDArchieve.etaBinIdx,
#                          refBenchmark.etaBinIdx)
#                  if not refBenchmark.checkEtBinIdx(TDArchieve.etBinIdx):
#                    if refBenchmark.etaBinIdx is None:
#                      self._logger.warning("TunedDiscrArchieve does not contain Et binning information!")
#                    else:
#                      self._logger.error("File (%d) Et binning information does not match with benchmark (%r)!", 
#                          TDArchieve.etBinIdx,
#                          refBenchmark.etBinIdx)
#                if type(tunedDiscr) in (list, tuple,):
#                  # fastnet core version
#                  if len(tunedDiscr) == 1:
#                    discr = tunedDiscr[0]
#                  else:
#                    discr = tunedDiscr[idx]
#                else:
#                  # exmachina core version
#                  discr = tunedDiscr
#      
#                #self._logger.debug('Add preproc chain information into the file')
#                self.__addPPChain( cSummaryPPInfo,
#                                   tunedPPChain, 
#                                   sort )                  
#
#                #self._logger.debug('Add performance information into the file')
#                self.__addPerformance( tunedDiscrInfo = tunedDiscrInfo,
#                                       path = path, ref = refBenchmark, 
#                                       neuron = neuron, sort = sort, init = init,
#                                       etBinIdx = TDArchieve.etBinIdx, etaBinIdx = TDArchieve.etaBinIdx,
#                                       tunedDiscr = discr, trainEvolution = trainEvolution ) 
#                # Add bin information to reference benchmark
#              # end of references
#            # end of configurations
#
#          # with file
#=======
#          try:
#            # Try to retrieve as a collection:
#            tdArchieveCol = TunedDiscrArchieveCol(path)
#            for tdArchieve in tdArchieveCol:
#              self.__retrieveFileInfo( tdArchieve, 
#                                       cRefBenchmarkList,
#                                       tunedDiscrInfo,
#                                       cSummaryPPInfo )
#          except (UnpicklingError, ValueError, EOFError), e:
#            # Couldn't read as a collection, add it to log
#            self._logger.debug("File '%s' couldn't be read as TunedDiscrArchieveCol. Reason:\n%s", path, str(e))
#            pass
#          # And try to read it as a common archieve
#          tdArchieve = TunedDiscrArchieve.load(path)
#          self.__retrieveFileInfo( tdArchieve, 
#                                   cRefBenchmarkList,
#                                   tunedDiscrInfo,
#                                   cSummaryPPInfo )
#>>>>>>> Stashed changes
#        except (UnpicklingError, ValueError, EOFError), e:
#          # Couldn't read it as both a common file or a collection:
#          self._logger.warning("Ignoring file '%s'. Reason:\n%s", path, str(e))
#        if debug and cFile == 10:
#          break
#
#        self._sg.collect()
#        gc.collect()
#      # Finished all files in this bin
#      
#
#      # Print information retrieved:
#      if self._level <= LoggingLevel.VERBOSE:
#        for refBenchmark in cRefBenchmarkList:
#          refName = refBenchmark.name
#          self._logger.verbose("Retrieved %d discriminator configurations for benchmark '%s':", 
#              len(tunedDiscrInfo[refName]) - 1, 
#              refBenchmark)
#          for nKey, nDict in tunedDiscrInfo[refName].iteritems():
#            if nKey == 'benchmark': continue
#            self._logger.verbose("Retrieved %d sorts for configuration '%r'", len(nDict), nKey)
#            for sKey, sDict in nDict.iteritems():
#              self._logger.verbose("Retrieved %d inits for sort '%d'", len(sDict['headerInfo']), sKey)
#            # got number of inits
#          # got number of sorts
#        # got number of configurations
#      # finished all references
#
#      # Create summary info object
#      for refKey, refValue in tunedDiscrInfo.iteritems(): # Loop over operations
#        refBenchmark = refValue['benchmark']
#        # Create a new dictionary and append bind it to summary info
#        refDict = { 'rawBenchmark' : refBenchmark.rawInfo(),
#                    'rawTuningBenchmark' : refValue['tuningBenchmark'].rawInfo() }
#        cSummaryInfo[refKey] = refDict
#        for nKey, nValue in refValue.iteritems(): # Loop over neurons
#          if nKey in ('benchmark', 'tuningBenchmark',):
#            continue
#          nDict = dict()
#          refDict['config_' + str(nKey)] = nDict
#          for sKey, sValue in nValue.iteritems(): # Loop over sorts
#            sDict = dict()
#            nDict['sort_' + str(sKey)] = sDict
#            # Retrieve information from outermost initializations:
#            ( sDict['summaryInfoTst'], \
#              sDict['infoTstBest'], sDict['infoTstWorst']) = self.__outermostPerf( sValue['headerInfo'],
#                                                                                   sValue['initPerfTstInfo'], 
#                                                                                   refBenchmark, 
#                                                                                   'sort', 
#                                                                                   sKey )
#            ( sDict['summaryInfoOp'], \
#              sDict['infoOpBest'], sDict['infoOpWorst']) = self.__outermostPerf( sValue['headerInfo'],
#                                                                                 sValue['initPerfOpInfo'], 
#                                                                                 refBenchmark, 
#                                                                                 'sort', 
#                                                                                 sKey )
#          # Retrieve information from outermost sorts:
#          allBestTstSortInfo   = [ sDict['infoTstBest' ] for key, sDict in nDict.iteritems() ]
#          allBestOpSortInfo    = [ sDict['infoOpBest'  ] for key, sDict in nDict.iteritems() ]
#          ( nDict['summaryInfoTst'], \
#            nDict['infoTstBest'], nDict['infoTstWorst']) = self.__outermostPerf( allBestTstSortInfo,
#                                                                                 allBestTstSortInfo, 
#                                                                                 refBenchmark, 
#                                                                                 'config', 
#                                                                                 nKey )
#          ( nDict['summaryInfoOp'], \
#            nDict['infoOpBest'], nDict['infoOpWorst'])   = self.__outermostPerf( allBestOpSortInfo,
#                                                                                 allBestOpSortInfo, 
#                                                                                 refBenchmark, 
#                                                                                 'config', 
#                                                                                 nKey )
#        # Retrieve information from outermost discriminator configurations:
#        allBestTstConfInfo   = [ nDict['infoTstBest' ] for key, nDict in refDict.iteritems() if key not in ('rawBenchmark', 'rawTuningBenchmark',) ]
#        allBestOpConfInfo    = [ nDict['infoOpBest'  ] for key, nDict in refDict.iteritems() if key not in ('rawBenchmark', 'rawTuningBenchmark',) ]
#        ( refDict['summaryInfoTst'], \
#          refDict['infoTstBest'], refDict['infoTstWorst']) = self.__outermostPerf( allBestTstConfInfo,
#                                                                                   allBestTstConfInfo, 
#                                                                                   refBenchmark, 
#                                                                                   'benchmark', 
#                                                                                   refKey )
#        ( refDict['summaryInfoOp'], \
#          refDict['infoOpBest'], refDict['infoOpWorst'])   = self.__outermostPerf( allBestOpConfInfo,  
#                                                                                   allBestOpConfInfo, 
#                                                                                   refBenchmark, 
#                                                                                   'benchmark', 
#                                                                                   refKey )
#      # Finished summary information
#      if self._level <= LoggingLevel.DEBUG:
#        for refKey, refValue in cSummaryInfo.iteritems(): # Loop over operations
#          self._logger.debug("This is the summary information for benchmark %s", refKey )
#          pprint({key : { innerkey : innerval for innerkey, innerval in val.iteritems() if not(innerkey.startswith('sort_'))} 
#                                            for key, val in refValue.iteritems() if type(key) is str} 
#                , depth=3
#                )
#
#      # Append pp collections
#      cSummaryInfo['infoPPChain'] = cSummaryPPInfo
#
#      # Save files
#      if self._binFilters is not None:
#        cOutputName = outputName + self._binFilters[binIdx].replace('*','_')
#        if cOutputName.endswith('_'): 
#          cOutputName = cOutputName[:-1]
#      else:
#        cOutputName = outputName
#      outputPath = save( cSummaryInfo, cOutputName )
#      self._logger.info("Saved file '%s'",outputPath)
#      # Save matlab file
#      if toMatlab:
#        try:
#          import scipy.io
#          scipy.io.savemat( cOutputName + '.mat', cSummaryInfo)
#        except ImportError:
#          raise RuntimeError(("Cannot save matlab file, it seems that scipy is not "
#              "available."))
#      # Finished bin
#    # finished all files
#  # end of loop
#
#  def __retrieveFileInfo(self, tdArchieve, 
#                               cRefBenchmarkList,
#                               tunedDiscrInfo,
#                               cSummaryPPInfo):
#    """
#    Retrieve tdArchieve information
#    """
#    if tdArchieve.etaBinIdx != -1 and cFile == 0:
#      self._logger.info("File eta bin index (%d) limits are: %r", 
#                         tdArchieve.etaBinIdx, 
#                         tdArchieve.etaBin, 
#                        )
#    if tdArchieve.etBinIdx != -1 and cFile == 0:
#      self._logger.info("File Et bin index (%d) limits are: %r", 
#                         tdArchieve.etBinIdx, 
#                         tdArchieve.etBin, 
#                       )
#    # Now we loop over each configuration:
#    for neuron, sort, init in product( tdArchieve.neuronBounds(), 
#                                       tdArchieve.sortBounds(), 
#                                       tdArchieve.initBounds() ):
#
#      tunedDict      = tdArchieve.getTunedInfo( neuron, sort, init )
#      tunedDiscr     = tunedDict['tunedDiscr']
#      tunedPPChain   = tunedDict['tunedPP']
#      trainEvolution = tunedDict['tuningInfo']
#
#      # FIXME The number of refBenchmark should be the same number of tuned reference points
#      # discriminators
#      for refBenchmark in cRefBenchmarkList:
#        # Check if binning information matches:
#        if tdArchieve.etaBinIdx != -1 and refBenchmark.signal_efficiency.etaBin != -1 \
#            and tdArchieve.etaBinIdx != refBenchmark.signal_efficiency.etaBin:
#          self._logger.warning("File (%d) eta binning information does not match with benchmark (%d)!", 
#              tdArchieve.etaBinIdx,
#              refBenchmark.signal_efficiency.etaBin)
#        if tdArchieve.etBinIdx != -1 and refBenchmark.signal_efficiency.etBin != -1 \
#            and tdArchieve.etBinIdx != refBenchmark.signal_efficiency.etBin:
#          self._logger.warning("File (%d) Et binning information does not match with benchmark (%d)!", 
#              tdArchieve.etBinIdx,
#              refBenchmark.signal_efficiency.etBin)
#
#
#        # FIXME, this shouldn't be like that, instead the reference
#        # benchmark should be passed to the TuningJob so that it could
#        # set the best operation point itself.
#        # When this is done, we can then remove the working points list
#        # as it is done here:
#        if type(tunedDiscr) is list:
#          # fastnet core version
#          discr = tunedDiscr[refBenchmark.reference]
#        else:
#          # exmachina core version
#          discr = tunedDiscr
#
#        self.__addPPChain( cSummaryPPInfo,
#                           tunedPPChain, 
#                           sort )                    
#        
#        self.__addPerformance( tunedDiscrInfo,
#                               path,
#                               refBenchmark, 
#                               neuron,
#                               sort,
#                               init,
#                               discr,
#                               trainEvolution ) 
#        # Add bin information to reference benchmark
#      # end of references
#    # end of configurations
#  # end of __retrieveFileInfo
#
#  def __addPerformance( self, tunedDiscrInfo, path, ref, neuron, sort, init, tunedDiscr, trainEvolution ):
#    refName = ref.name
#    # We need to make sure that the key will be available on the dict if it
#    # wasn't yet there
#    if not refName in tunedDiscrInfo:
#      tunedDiscrInfo[refName] = { 'benchmark' : ref }
#    if not neuron in tunedDiscrInfo[refName]:
#      tunedDiscrInfo[refName][neuron] = dict()
#    if not sort in tunedDiscrInfo[refName][neuron]:
#      tunedDiscrInfo[refName][neuron][sort] = { 'headerInfo' : [], 
#                                                'initPerfTstInfo' : [], 
#                                                'initPerfOpInfo' : [] }
#    # The performance holder, which also contains the discriminator
#    perfHolder = PerfHolder( tunedDiscr, trainEvolution )
#    # Retrieve operating points:
#    (spTst, detTst, faTst, cutTst, idxTst) = perfHolder.getOperatingBenchmarks(ref)
#    (spOp, detOp, faOp, cutOp, idxOp) = perfHolder.getOperatingBenchmarks(ref, ds = Dataset.Operation)
#    headerInfo = { 'filepath' : path,
#                   'neuron' : neuron, 'sort' : sort, 'init' : init,
#                   #'perfHolder' : perfHolder, 
#                 }
#    # Create performance holders:
#    iInfoTst = { 'sp' : spTst, 'det' : detTst, 'fa' : faTst, 'cut' : cutTst, 'idx' : idxTst, }
#    iInfoOp  = { 'sp' : spOp,  'det' : detOp,  'fa' : faOp,  'cut' : cutOp,  'idx' : idxOp,  }
#    if self._level <= LoggingLevel.VERBOSE:
#      self._logger.verbose("Retrieved file '%s' configuration for benchmark '%s' as follows:", 
#                         os.path.basename(path),
#                         ref )
#      pprint({'headerInfo' : headerInfo, 'initPerfTstInfo' : iInfoTst, 'initPerfOpInfo' : iInfoOp })
#    # Append information to our dictionary:
#    # FIXME headerInfo shouldn't be connected to refName.
#    tunedDiscrInfo[refName][neuron][sort]['headerInfo'].append( headerInfo )
#    tunedDiscrInfo[refName][neuron][sort]['initPerfTstInfo'].append( iInfoTst )
#    tunedDiscrInfo[refName][neuron][sort]['initPerfOpInfo'].append( iInfoOp )
#
#    #Adding graphs into monitoring file
#    init = len(tunedDiscrInfo[refName][neuron][sort]['initPerfOpInfo'])-1
#    dirname = ('%s/%s/neuron_%d/sort_%d/init_%d') % (self._currentPath,ref.name,neuron,sort,init)
#    self._sg.mkdir(dirname)
#    
#    graphNames = [
#         'mse_trn',
#         'mse_val',
#         'mse_tst',
#         'sp_val',
#         'sp_tst',
#         'det_val',
#         'det_tst',
#         'fa_val',
#         'fa_tst',
#         'det_fitted',
#         'fa_fitted',
#         'roc_tst',
#         'roc_op',
#         'roc_tst_cut',
#         'roc_op_cut'
#         ]
#
#    #Attach graphs
#    for gname in graphNames:
#      g = perfHolder.getGraph(gname); g.SetName(gname)
#      self._sg.attach(g)
#    #Attach stops
#    self._sg.attach(perfHolder.getTree())
#
#  def __addPPChain(self, cSummaryPPInfo, tunedPPChain, sort):
#    if not( 'sort_' + str(sort) in cSummaryPPInfo ) and tunedPPChain:
#      ppData = tunedPPChain.toRawObj()
#      cSummaryPPInfo['sort_' + str( sort ) ] = ppData
#  # end of __addPPChain
#
#
#  def __outermostPerf(self, headerInfoList, perfInfoList, refBenchmark, collectionType, val, **kw):
#
#    self._logger.debug("%s: Retrieving outermost performance for %s %r (done twice, first for test, after for operation).",
#        refBenchmark, collectionType, val )
#
#    summaryDict = {'cut': [], 'sp': [], 'det': [], 'fa': [], 'idx': []}
#    # Fetch all information together in the dictionary:
#    for key in summaryDict.keys():
#      summaryDict[key] = [ perfInfo[key] for perfInfo in perfInfoList ]
#      if not key == 'idx':
#        summaryDict[key + 'Mean'] = np.mean(summaryDict[key],axis=0)
#        summaryDict[key + 'Std']  = np.std(summaryDict[key],axis=0)
#
#    # Put information together on data:
#    benchmarks = [summaryDict['sp'], summaryDict['det'], summaryDict['fa']]
#
#    # The outermost performances:
#    bestIdx  = refBenchmark.getOutermostPerf(benchmarks )
#    worstIdx = refBenchmark.getOutermostPerf(benchmarks, cmpType = -1. )
#    if self._level <= LoggingLevel.DEBUG:
#      self._logger.debug('Retrieved best index as: %d; values: (%f,%f,%f)', bestIdx, 
#          benchmarks[0][bestIdx],
#          benchmarks[1][bestIdx],
#          benchmarks[2][bestIdx])
#      self._logger.debug('Retrieved worst index as: %d; values: (%f,%f,%f)', worstIdx,
#          benchmarks[0][worstIdx],
#          benchmarks[1][worstIdx],
#          benchmarks[2][worstIdx])
#
#    # Retrieve information from outermost performances:
#    def __getInfo( headerInfoList, perfInfoList, idx ):
#      info = dict()
#      wantedKeys = ['discriminator', 'neuron', 'sort', 'init']
#      headerInfo = headerInfoList[idx]
#      for key in wantedKeys:
#        info[key] = headerInfo[key]
#      wantedKeys = ['cut','sp', 'det', 'fa', 'idx']
#      perfInfo = perfInfoList[idx]
#      for key in wantedKeys:
#        info[key] = perfInfo[key]
#      return info
#
#    bestInfoDict  = __getInfo( headerInfoList, perfInfoList, bestIdx )
#    worstInfoDict = __getInfo( headerInfoList, perfInfoList, worstIdx )
#    if self._level <= LoggingLevel.VERBOSE:
#      self._logger.verbose("The best configuration retrieved is: ")
#      pprint(bestInfoDict)
#      self._logger.verbose("The worst configuration retrieved is: ")
#      pprint(worstInfoDict)
#
#    return (summaryDict, bestInfoDict, worstInfoDict)
#  # end of __outermostPerf
#
#  def exportDiscrFiles(self, ringerOperation, **kw ):
#    """
#    Export discriminators operating at reference benchmark list to the
#    ATLAS environment using this CrossValidStat information.
#    """
#    if not self._summaryInfo:
#      raise RuntimeError(("Create the summary information using the loop method."))
#    CrossValidStat.exportDiscrFiles( self._summaryInfo, 
#                                     ringerOperation, 
#                                     level = self._level,
#                                     **kw )
#
#  @classmethod
#  def exportDiscrFiles(cls, summaryInfoList, ringerOperation, **kw):
#    """
#    Export discriminators operating at reference benchmark list to the
#    ATLAS environment using summaryInfo. 
#    
#    If benchmark name on the reference list is not available at summaryInfo, an
#    KeyError exception will be raised.
#    """
#    baseName      = kw.pop( 'baseName'      , 'tunedDiscr'      )
#    refBenchCol   = kw.pop( 'refBenchCol'   , None              )
#    configCol     = kw.pop( 'configCol'     , []                )
#    triggerChains = kw.pop( 'triggerChains' , None              )
#    level         = kw.pop( 'level'         , LoggingLevel.INFO )
#
#    # Initialize local logger
#    logger      = Logger.getModuleLogger("exportDiscrFiles", logDefaultLevel = level )
#    checkForUnusedVars( kw, logger.warning )
#
#    # Treat the summaryInfoList
#    if not isinstance( summaryInfoList, (list,tuple)):
#      summaryInfoList = [ summaryInfoList ]
#    summaryInfoList = list(traverse(summaryInfoList,simple_ret=True))
#    nSummaries = len(summaryInfoList)
#
#    if refBenchCol is None:
#      refBenchCol = summaryInfoList[0].keys()
#
#    # Treat the reference benchmark list
#    if not isinstance( refBenchCol, (list,tuple)):
#      refBenchCol = [ refBenchCol ] * nSummaries
#
#    if len(refBenchCol) == 1:
#      refBenchCol = refBenchCol * nSummaries
#
#    nRefs = len(list(traverse(refBenchCol,simple_ret=True)))
#
#    # Make sure that the lists are the same size as the reference benchmark:
#    nConfigs = len(list(traverse(configCol,simple_ret=True)))
#    if nConfigs == 0:
#      configCol = [None for i in range(nRefs)]
#    elif nConfigs == 1:
#      configCol = configCol * nSummaries
#
#    if nConfigs != nRefs:
#      raise ValueError("Summary size is not equal to the configuration list.")
#    
#    if nRefs == nConfigs == nSummaries:
#      # If user input data without using list on the configuration, put it as a list:
#      for o, idx, parent, _, _ in traverse(configCol):
#        parent[idx] = [o]
#      for o, idx, parent, _, _ in traverse(refBenchCol):
#        parent[idx] = [o]
#
#    configCol   = list(traverse(configCol,max_depth_dist=1,simple_ret=True))
#    refBenchCol = list(traverse(refBenchCol,max_depth_dist=1,simple_ret=True))
#    nConfigs = len(configCol)
#    nSummary = len(refBenchCol)
#
#    if nRefs != nConfigs != nSummary:
#      raise ValueError("Number of references, configurations and summaries do not match!")
#
#    # Retrieve the operation:
#    from TuningTools.ReadData import RingerOperation
#    ringerOperation = RingerOperation.retrieve(ringerOperation)
#    logger.info(('Exporting discrimination info files for the following '
#                'operating point (RingerOperation:%s).'), 
#                RingerOperation.tostring(ringerOperation))
#
#    if ringerOperation is RingerOperation.L2:
#      if triggerChains is None:
#        triggerChains = "custom"
#      if type(triggerChains) not in (list,tuple):
#        triggerChains = [triggerChains]
#      nExports = len(refBenchCol[0])
#      if len(triggerChains) == 1 and nExports != 1:
#        baseChainName = triggerChains[0]
#        triggerChains = ["%s_chain%d" % (baseChainName,i) for i in range(nExports)]
#      if nExports != len(triggerChains):
#        raise ValueError("Number of exporting chains does not match with number of given chain names.")
#
#      output = open('TrigL2CaloRingerConstants.py','w')
#      output.write('def RingerMap():\n')
#      output.write('  signatures=dict()\n')
#      outputDict = dict()
#
#    for summaryInfo, refBenchmarkList, configList in \
#                        zip(summaryInfoList,
#                            refBenchCol,
#                            configCol,
#                           ):
#      if type(summaryInfo) is str:
#        logger.info('Loading file "%s"...', summaryInfo)
#        summaryInfo = load(summaryInfo)
#      elif type(summaryInfo) is dict:
#        pass
#      else:
#        raise ValueError("Cross-valid summary info is not string and not a dictionary.")
#      from itertools import izip, count
#      for idx, refBenchmarkName, config in izip(count(), refBenchmarkList,configList):
#        info = summaryInfo[refBenchmarkName]['infoOpBest'] if config is None else \
#               summaryInfo[refBenchmarkName]['config_' + str(config)]['infoOpBest']
#        logger.info("%s discriminator information is available at file: \n\t%s", 
#                    refBenchmarkName,
#                    info['filepath'])
#        ## Check if user specified parameters for exporting discriminator
#        ## operation information:
#        sort = info['sort']
#        init = info['init']
#        tdArchieve =  TunedDiscrArchieve.load(info['filepath'])
#        tdArchieve.level = level
#        etBinIdx = tdArchieve.etBinIdx
#        etaBinIdx = tdArchieve.etaBinIdx
#        etBin = tdArchieve.etBin
#        etaBin = tdArchieve.etaBin
#        ## Write the discrimination wrapper
#        if ringerOperation is RingerOperation.Offline:
#          # Import athena cpp information
#          try:
#            import cppyy
#          except ImportError:
#            import PyCintex as cppyy
#          try:
#            cppyy.loadDict('RingerSelectorTools_Reflex')
#          except RuntimeError:
#            raise RuntimeError("Couldn't load RingerSelectorTools_Reflex dictionary.")
#          from ROOT import TFile
#          from ROOT import std
#          from ROOT.std import vector
#          # Import Ringer classes:
#          from ROOT import Ringer
#          from ROOT import MsgStream
#          from ROOT import MSG
#          from ROOT.Ringer import IOHelperFcns
#          from ROOT.Ringer import RingerProcedureWrapper
#          from ROOT.Ringer import Discrimination
#          from ROOT.Ringer import IDiscrWrapper
#          from ROOT.Ringer import IDiscrWrapperCollection
#          from ROOT.Ringer import IThresWrapper
#          from ROOT.Ringer.Discrimination import UniqueThresholdVarDep
#          # Extract dictionary:
#          discrData, keep_lifespan_list = tdArchieve.exportDiscr(config, 
#                                                                 sort, 
#                                                                 init, 
#                                                                 ringerOperation, 
#                                                                 summaryInfo[refBenchmarkName]['rawBenchmark'])
#          logger.debug("Retrieved discrimination info!")
#
#          fDiscrName = baseName + '_Discr_' + refBenchmarkName + ".root"
#          # Export the discrimination wrapper to a TFile and save it:
#          discrCol = IDiscrWrapperCollection() 
#          discrCol.push_back(discrData)
#          IDiscrWrapper.writeCol(discrCol, fDiscrName)
#          logger.info("Successfully created file %s.", fDiscrName)
#          ## Export the Threshold Wrapper:
#          RingerThresWrapper = RingerProcedureWrapper("Ringer::Discrimination::UniqueThresholdVarDep",
#                                                      "Ringer::EtaIndependent",
#                                                      "Ringer::EtIndependent",
#                                                      "Ringer::NoSegmentation")
#          BaseVec = vector("Ringer::Discrimination::UniqueThresholdVarDep*")
#          vec = BaseVec() # We are not using eta dependency
#          thres = UniqueThresholdVarDep(info['cut'])
#          if logger.isEnabledFor( LoggingLevel.DEBUG ):
#            thresMsg = MsgStream("ExportedThreshold")
#            thresMsg.setLevel(LoggingLevel.toC(level))
#            thres.setMsgStream(thresMsg)
#            getattr(thres,'print')(MSG.DEBUG)
#          vec.push_back( thres )
#          thresVec = vector(BaseVec)() # We are not using et dependency
#          thresVec.push_back(vec)
#          ## Create pre-processing wrapper:
#          logger.debug('Initiazing Threshold Wrapper:')
#          thresWrapper = RingerThresWrapper(thresVec)
#          fThresName = baseName + '_Thres_' + refBenchmarkName + ".root"
#          IThresWrapper.writeWrapper( thresWrapper, fThresName )
#          logger.info("Successfully created file %s.", fThresName)
#        elif ringerOperation is RingerOperation.L2:
#          triggerChain = triggerChains[idx]
#          if not triggerChain in outputDict:
#            cDict = {}
#            outputDict[triggerChain] = cDict
#          else:
#            cDict = outputDict[triggerChain]
#          config = {}
#          cDict['eta%d_et%d' % (etaBinIdx, etBinIdx) ] = config
#          #config['rawBenchmark'] = summaryInfo[refBenchmarkName]['rawBenchmark']
#          #config['infoOp']       = info
#          # FIXME Index [0] is the discriminator, [1] is the normalization. This should be more organized.
#          discr = tdArchieve.getTunedInfo(info['neuron'],
#                                          info['sort'],
#                                          info['init'])[0]
#          if type(discr) is list:
#            reference = ReferenceBenchmark.retrieve( summaryInfo[refBenchmarkName]['rawBenchmark']['reference'] )
#            discr = discr[reference]
#          else:
#            discr = ['discriminator']
#          discr = { key : (val.tolist() if type(val) == np.ndarray \
#                        else val) for key, val in discr['discriminator'].iteritems()
#                  }
#          config.update( discr )
#          config['threshold'] = info['cut']
#          config['etaBin']     = etaBin.tolist()
#          config['etBin']      = etBin.tolist()
#          logger.info('Exported bin(et=%d,eta=%d) using following configuration:',
#                      etBinIdx,
#                      etaBinIdx)
#          logger.info('neuron = %d, sort = %d, init = %d, thr = %f',
#                      info['neuron'],
#                      info['sort'],
#                      info['init'],
#                      info['cut'])
#        else:
#          raise RuntimeError('You must choose a ringerOperation')
#      # for benchmark
#    # for summay in list
#
#    if ringerOperation is RingerOperation.L2:
#      for key, val in outputDict.iteritems():
#        output.write('  signatures["%s"]=%s\n' % (key, val))
#      output.write('  return signatures\n')
#  # exportDiscrFiles 
#
#  @classmethod
#  def printTables(cls, confBaseNameList,
#                       crossValGrid,
#                       configMap):
#    "Print operation tables for the "
#    # TODO Improve documentation
#
#    # We first loop over the configuration base names:
#    for ds in [Dataset.Test, Dataset.Operation]:
#      for confIdx, confBaseName in enumerate(confBaseNameList):
#        print "{:=^90}".format("  %s ( %s )  " % (confBaseName, Dataset.tostring(ds)) )
#        # And then on et/eta bins:
#        for crossList in crossValGrid:
#          print "{:-^90}".format("  Starting new Et  ")
#          for crossFile in crossList:
#            # Load file and then search the benchmark references with the configuration name:
#            summaryInfo = load(crossFile)
#            etIdx = -1
#            etaIdx = -1
#            for key in summaryInfo.keys():
#              try:
#                rawBenchmark = summaryInfo[key]['rawBenchmark']
#                try:
#                  etIdx = rawBenchmark['signal_efficiency']['etBin']
#                  etaIdx = rawBenchmark['signal_efficiency']['etaBin']
#                except KeyError:
#                  etIdx = rawBenchmark['signal_efficiency']['_etBin']
#                  etaIdx = rawBenchmark['signal_efficiency']['_etaBin']
#                break
#              except (KeyError, TypeError) as e:
#                pass
#            print "{:-^90}".format("  Eta (%d) | Et (%d)  " % (etaIdx, etIdx))
#            #from scipy.io import loadmat
#            #summaryInfo = loadmat(crossFile)
#            confPdKey = confSPKey = confPfKey = None
#            for key in summaryInfo.keys():
#              if key == 'infoPPChain': continue
#              rawBenchmark = summaryInfo[key]['rawBenchmark']
#              reference = rawBenchmark['reference']
#              # Retrieve the configuration keys:
#              if confBaseName in key:
#                if reference == 'Pd':
#                  confPdKey = key 
#                if reference == 'Pf':
#                  confPfKey = key 
#                if reference == 'SP':
#                  confSPKey = key 
#            # Loop over each one of the cases and print ringer performance:
#            print '{:^13}   {:^13}   {:^13} |   {:^13}   |  {}  '.format("Pd (%)","SP (%)","Pf (%)","cut","(ReferenceBenchmark)")
#            print "{:-^90}".format("  Ringer  ")
#            for keyIdx, key in enumerate([confPdKey, confSPKey, confPfKey]):
#              if not key:
#                print '{:-^90}'.format(' Information Unavailable ')
#                continue
#              if ds is Dataset.Test:
#                ringerPerf = summaryInfo[key] \
#                                        ['config_' + str(configMap[confIdx][etIdx][etaIdx][keyIdx])] \
#                                        ['summaryInfoTst']
#                print '%6.3f+-%5.3f   %6.3f+-%5.3f   %6.3f+-%5.3f |   % 5.3f+-%5.3f   |  (%s) ' % ( 
#                    ringerPerf['detMean'] * 100.,   ringerPerf['detStd']  * 100.,
#                    ringerPerf['spMean']  * 100.,   ringerPerf['spStd']   * 100.,
#                    ringerPerf['faMean']  * 100.,   ringerPerf['faStd']   * 100.,
#                    ringerPerf['cutMean']       ,   ringerPerf['cutStd']        ,
#                    key)
#              else:
#                ringerPerf = summaryInfo[key] \
#                                        ['config_' + str(configMap[confIdx][etIdx][etaIdx][keyIdx])] \
#                                        ['infoOpBest']
#                print '{:^13.3f}   {:^13.3f}   {:^13.3f} |   {:^ 13.3f}   |  ({}) '.format(
#                    ringerPerf['det'] * 100.,
#                    ringerPerf['sp']  * 100.,
#                    ringerPerf['fa']  * 100.,
#                    ringerPerf['cut'],
#                    key)
#
#            print "{:-^90}".format("  Baseline  ")
#            reference_sp = calcSP(
#                                  rawBenchmark['signal_efficiency']['efficiency'] / 100.,
#                                  ( 1. - rawBenchmark['background_efficiency']['efficiency'] / 100. )
#                                 )
#            print '{:^13.3f}   {:^13.3f}   {:^13.3f} |{:@<43}'.format(
#                                      rawBenchmark['signal_efficiency']['efficiency']
#                                      ,reference_sp * 100.
#                                      ,rawBenchmark['background_efficiency']['efficiency']
#                                      ,''
#                                     )
#            if ds is Dataset.Test:
#              print "{:.^90}".format("")
#              try:
#                sgnCrossEff    = rawBenchmark['signal_cross_efficiency']['_branchCollectorsDict'][Dataset.Test]
#                bkgCrossEff    = rawBenchmark['background_cross_efficiency']['_branchCollectorsDict'][Dataset.Test]
#                sgnRawCrossVal = rawBenchmark['signal_cross_efficiency']['efficiency']['Test']
#                bkgRawCrossVal = rawBenchmark['background_cross_efficiency']['efficiency']['Test']
#              except KeyError:
#                sgnCrossEff = rawBenchmark['signal_cross_efficiency']['_branchCollectorsDict'][Dataset.Validation]
#                bkgCrossEff = rawBenchmark['background_cross_efficiency']['_branchCollectorsDict'][Dataset.Validation]
#                sgnRawCrossVal = rawBenchmark['signal_cross_efficiency']['efficiency']['Validation']
#                bkgRawCrossVal = rawBenchmark['background_cross_efficiency']['efficiency']['Validation']
#              try:
#                reference_sp = [ calcSP(rawSgn,(100.-rawBkg))
#                                  for rawSgn, rawBkg in zip(sgnCrossEff, bkgCrossEff)
#                               ]
#              except TypeError: # Old format compatibility
#                reference_sp = [ calcSP(rawSgn['efficiency'],(100.-rawBkg['efficiency']))
#                                  for rawSgn, rawBkg in zip(sgnCrossEff, bkgCrossEff)
#                               ]
#              print '{:6.3f}+-{:5.3f}   {:6.3f}+-{:5.3f}   {:6.3f}+-{:5.3f} |{:@<43}'.format( 
#                  sgnRawCrossVal[0]
#                  ,sgnRawCrossVal[1]
#                  ,np.mean(reference_sp)
#                  ,np.std(reference_sp)
#                  ,bkgRawCrossVal[0]
#                  ,bkgRawCrossVal[1]
#                  ,'')
#        print "{:=^90}".format("")
#
#
#class PerfHolder( object ):
#  """
#  Hold the performance values and evolution for a tuned discriminator
#  """
#
#  def __init__(self, tunedDiscrData, tunedEvolutionData ):
#    self.roc_tst        = tunedDiscrData['summaryInfo']['roc_test']
#    self.roc_operation  = tunedDiscrData['summaryInfo']['roc_operation']
#    trainEvo            = tunedEvolutionData
#    self.epoch          = np.array( range(len(trainEvo['mse_trn'])), dtype ='float_')
#    self.nEpoch         = len(self.epoch)
#    self.mse_trn        = np.array( trainEvo['mse_trn'],           dtype = 'float_' )
#    self.mse_val        = np.array( trainEvo['mse_val'],           dtype = 'float_' )
#    self.mse_tst        = np.array( trainEvo['mse_tst'],           dtype = 'float_' )
#    self.sp_val         = np.array( trainEvo['sp_val'],            dtype = 'float_' )
#    self.sp_tst         = np.array( trainEvo['sp_tst'],            dtype = 'float_' )
#    self.det_val        = np.array( trainEvo['det_val'],           dtype = 'float_' )
#    self.det_tst        = np.array( trainEvo['det_tst'],           dtype = 'float_' )
#    self.det_fitted     = np.array( trainEvo['det_fitted'],        dtype = 'float_' ) if 'det_fitted' in trainEvo else np.array([], dtype='float_')
#    self.fa_val         = np.array( trainEvo['fa_val'],            dtype = 'float_' )
#    self.fa_tst         = np.array( trainEvo['fa_tst'],            dtype = 'float_' )
#    self.fa_fitted      = np.array( trainEvo['fa_fitted'],         dtype = 'float_' ) if 'det_fitted' in trainEvo else np.array([], dtype='float_')
#    self.roc_tst_det    = np.array( self.roc_tst.detVec,           dtype = 'float_' )
#    self.roc_tst_fa     = np.array( self.roc_tst.faVec,            dtype = 'float_' )
#    self.roc_tst_cut    = np.array( self.roc_tst.cutVec,           dtype = 'float_' )
#    self.roc_op_det     = np.array( self.roc_operation.detVec,     dtype = 'float_' )
#    self.roc_op_fa      = np.array( self.roc_operation.faVec,      dtype = 'float_' )
#    self.roc_op_cut     = np.array( self.roc_operation.cutVec,     dtype = 'float_' )
#    self.epoch_stop_mse = np.array( trainEvo['epoch_best_mse'],    dtype = 'int_'   ) if 'epoch_best_mse' in trainEvo else np.array(-1, dtype='int_')
#    self.epoch_stop_sp  = np.array( trainEvo['epoch_best_sp'],     dtype = 'int_'   ) if 'epoch_best_sp'  in trainEvo else np.array(-1, dtype='int_')
#    self.epoch_stop_det = np.array( trainEvo['epoch_best_det'],    dtype = 'int_'   ) if 'epoch_best_det' in trainEvo else np.array(-1, dtype='int_')
#    self.epoch_stop_fa  = np.array( trainEvo['epoch_best_fa'],     dtype = 'int_'   ) if 'epoch_best_fa'  in trainEvo else np.array(-1, dtype='int_')
#
#
#  def getOperatingBenchmarks( self, refBenchmark, **kw):
#    """
#    Returns the operating benchmark values for this tunned discriminator
#    """
#    idx = kw.pop('idx', None)
#    ds  = kw.pop('ds', Dataset.Test )
#    if ds is Dataset.Test:
#      detVec = self.roc_tst_det
#      faVec = self.roc_tst_fa
#      cutVec = self.roc_tst_cut
#    elif ds is Dataset.Operation:
#      detVec = self.roc_op_det
#      faVec = self.roc_op_fa
#      cutVec = self.roc_op_cut
#    else:
#      raise ValueError("Cannot retrieve maximum ROC SP for dataset '%s'", ds)
#    spVec = calcSP( detVec, 1 - faVec )
#    if idx is None:
#      if refBenchmark.reference is ReferenceBenchmark.SP:
#        idx = np.argmax( spVec )
#      else:
#        # Get reference for operation:
#        if refBenchmark.reference is ReferenceBenchmark.Pd:
#          ref = detVec
#        elif refBenchmark.reference is ReferenceBenchmark.Pf:
#          ref = faVec
#          idx = np.argmin( np.abs( ref - refBenchmark.refVal ) )
#        idx = np.argmin( np.abs( ref - refBenchmark.refVal ) )
#    sp  = spVec[idx]
#    det = detVec[idx]
#    fa  = faVec[idx]
#    cut = cutVec[idx]
#    return (sp, det, fa, cut, idx)
#
#
#  def getGraph( self, graphType ):
#    """
#      Retrieve a TGraph from the discriminator Tuning data.
#
#      perfHolder.getGraph( option )
#
#      The possible options are:
#        * mse_trn
#        * mse_val
#        * mse_tst
#        * sp_val
#        * sp_tst
#        * det_val
#        * det_tst
#        * fa_val
#        * fa_tst
#        * det_fitted
#        * fa_fitted
#        * roc_val
#        * roc_op
#        * roc_val_cut
#        * roc_op_cut
#    """
#    from ROOT import TGraph
#    if   graphType == 'mse_trn'     : return TGraph(self.nEpoch, self.epoch, self.mse_trn )
#    elif graphType == 'mse_val'     : return TGraph(self.nEpoch, self.epoch, self.mse_val )
#    elif graphType == 'mse_tst'     : return TGraph(self.nEpoch, self.epoch, self.mse_tst )
#    elif graphType == 'sp_val'      : return TGraph(self.nEpoch, self.epoch, self.sp_val  )
#    elif graphType == 'sp_tst'      : return TGraph(self.nEpoch, self.epoch, self.sp_tst  )
#    elif graphType == 'det_val'     : return TGraph(self.nEpoch, self.epoch, self.det_val )
#    elif graphType == 'det_tst'     : return TGraph(self.nEpoch, self.epoch, self.det_tst )
#    elif graphType == 'fa_val'      : return TGraph(self.nEpoch, self.epoch, self.fa_val  )
#    elif graphType == 'fa_tst'      : return TGraph(self.nEpoch, self.epoch, self.fa_tst  )
#    elif graphType == 'det_fitted'  : return TGraph(self.nEpoch, self.epoch, self.det_fitted ) if len(self.det_fitted) else TGraph()
#    elif graphType == 'fa_fitted'   : return TGraph(self.nEpoch, self.epoch, self.fa_fitted  ) if len(self.fa_fitted) else TGraph()
#    elif graphType == 'roc_tst'     : return TGraph(len(self.roc_tst_fa), self.roc_tst_fa, self.roc_tst_det )
#    elif graphType == 'roc_op'      : return TGraph(len(self.roc_op_fa),  self.roc_op_fa,  self.roc_op_det  )
#    elif graphType == 'roc_tst_cut' : return TGraph(len(self.roc_tst_cut),
#                                                    np.array(range(len(self.roc_tst_cut) ), 'float_'), 
#                                                    self.roc_tst_cut )
#    elif graphType == 'roc_op_cut'  : return TGraph(len(self.roc_op_cut), 
#                                                         np.array(range(len(self.roc_op_cut) ),  'float_'), 
#                                                         self.roc_op_cut  )
#    else: raise ValueError( "Unknown graphType '%s'" % graphType )
#
