__all__ = ['percentile','GridJobFilter', 'CrossValidStatAnalysis']

from RingerCore import EnumStringification, get_attributes, checkForUnusedVars, \
    calcSP, save, load, Logger, LoggingLevel, expandFolders, traverse
from TuningTools.TuningJob import TunedDiscrArchieve, ReferenceBenchmark
from TuningTools import PreProc
from TuningTools.FilterEvents import Dataset
from pprint import pprint
from cPickle import UnpicklingError
import numpy as np
import os

def percentile( data, score ):
  """
  val = percentile( data, score )
  Retrieve the data percentile at score
  """
  size = len(data)
  if size:
    pos = score*size
    if pos % 10 or pos == size:
      return data[pos]
    else:
      return data[pos] + data[pos+1]
  else: return None

class GridJobFilter(object):

  import re
  pat = re.compile(r'.*user.[a-zA-Z0-9]+.(?P<jobID>[0-9]+)\..*$')
  #pat = re.compile(r'user.(?P<user>[A-z0-9]*).(?P<jobID>[0-9]+).*\.tgz')

  def __call__(self, paths):
    jobIDs = sorted(list(set(['*' + self.pat.match(f).group('jobID') + '*' for f in paths if self.pat.match(f) is not None])))
    return jobIDs

class CrossValidStatAnalysis( Logger ):

  def __init__(self, paths, **kw):
    """
    Usage: 

    # Create object
    cvStatAna = CrossValidStatAnalysis( paths 
                                        [,logoLabel=TuningTool]
                                        [,binFilters=None]
                                        [,logger[,level=INFO]]
                                      )
    # Fill the information and save output file with cross-validation summary
    cvStatAna( refBenchMark, **args...)
    # Call other methods if necessary.
    """
    Logger.__init__(self, kw)    
    self._logoLabel   = kw.pop('logoLabel',  'TuningTool' )
    self._binFilters  = kw.pop('binFilters',  None        )
    checkForUnusedVars(kw, self._logger.warning)
    # Recursively expand all folders in the given paths so that we have all
    # files lists:
    if hasattr(self._binFilters,'__call__'):
      self._paths = expandFolders( paths )
      import types
      if not type(self._binFilters) is types.FunctionType:
        self._binFilters = self._binFilters()
      self._binFilters = self._binFilters( self._paths )
      self._logger.info('Found following filters: %r', self._binFilters)
      self._paths = expandFolders( paths, self._binFilters ) 
    else:
      self._paths = expandFolders( paths, self._binFilters )
    if not(self._binFilters is None):
      self._nBins = len(self._binFilters)
    else:
      self._nBins = 1
    if self._nBins is 1:
      self._paths = [self._paths]
    if self._level <= LoggingLevel.VERBOSE:
      for binFilt in self._binFilters if self._binFilters is not None else [None]:
        self._logger.verbose("The stored files are (binFilter=%s):", binFilt)
        for path in self._paths:
          self._logger.verbose("%s", path)
    self._nFiles = [len(l) for l in self._paths]
    self._logger.info("A total of %r files were found.", self._nFiles )

  def __call__(self, refBenchmarkList, **kw):
    """
    Hook for loop method.
    """
    self.loop( refBenchmarkList, **kw )

  def __addPerformance( self, tunedDiscrInfo, path, ref, neuron, sort, init, tunedDiscrList ):
    refName = ref.name
    # We need to make sure that the key will be available on the dict if it
    # wasn't yet there
    if not refName in tunedDiscrInfo:
      tunedDiscrInfo[refName] = { 'benchmark' : ref }
    if not neuron in tunedDiscrInfo[refName]:
      tunedDiscrInfo[refName][neuron] = dict()
    if not sort in tunedDiscrInfo[refName][neuron]:
      tunedDiscrInfo[refName][neuron][sort] = { 'headerInfo' : [], 
                                                'initPerfTstInfo' : [], 
                                                'initPerfOpInfo' : [] }
    # The performance holder, which also contains the discriminator
    perfHolder = PerfHolder( tunedDiscrList )
    # Retrieve operating points:
    (spTst, detTst, faTst, cutTst, idxTst) = perfHolder.getOperatingBenchmarks(ref)
    (spOp, detOp, faOp, cutOp, idxOp) = perfHolder.getOperatingBenchmarks(ref, ds = Dataset.Operation)
    headerInfo = { 'filepath' : path,
                   'neuron' : neuron, 'sort' : sort, 'init' : init,
                   #'perfHolder' : perfHolder, 
                 }
    # Create performance holders:
    iInfoTst = { 'sp' : spTst, 'det' : detTst, 'fa' : faTst, 'cut' : cutTst, 'idx' : idxTst, }
    iInfoOp  = { 'sp' : spOp,  'det' : detOp,  'fa' : faOp,  'cut' : cutOp,  'idx' : idxOp,  }
    if self._level <= LoggingLevel.VERBOSE:
      self._logger.verbose("Retrieved file '%s' configuration for benchmark '%s' as follows:", 
                         os.path.basename(path),
                         ref )
      pprint({'headerInfo' : headerInfo, 'initPerfTstInfo' : iInfoTst, 'initPerfOpInfo' : iInfoOp })
    # Append information to our dictionary:
    # FIXME headerInfo shouldn't be connected to refName.
    tunedDiscrInfo[refName][neuron][sort]['headerInfo'].append( headerInfo )
    tunedDiscrInfo[refName][neuron][sort]['initPerfTstInfo'].append( iInfoTst )
    tunedDiscrInfo[refName][neuron][sort]['initPerfOpInfo'].append( iInfoOp )

  def loop(self, refBenchmarkList, **kw ):
    """
    Needed args:
      * refBenchmarkList: a list of reference benchmark objects which will be used
        as the operation points.
    Optional args:
      * toMatlab [True]: also create a matlab file from the obtained tuned discriminators
      * outputName ['crossValStat']: the output file name.
    """
    import gc
    toMatlab        = kw.pop('toMatlab',    True          )
    outputName      = kw.pop('outputName', 'crossValStat' )
    debug           = kw.pop('debug',       False         )
    checkForUnusedVars( kw, self._logger.warning )

    self._logger.info("Started analysing cross-validation statistics...")

    if type(refBenchmarkList[0]) is ReferenceBenchmark:
      refBenchmarkList = [refBenchmarkList for i in range(self._nBins)]

    if len(refBenchmarkList) != self._nBins:
      raise RuntimeError("Number of references (%d) is different from total number of bins (%d)" %
          (len(refBenchmarkList), self._nBins))

    self._summaryInfo = [ dict() for i in range(self._nBins) ]
    self._summaryPPInfo = [ dict() for i in range(self._nBins) ]

    # Loop over the files
    from itertools import product
    for binIdx, binPath in enumerate(self._paths):
      if self._binFilters is not None:
        self._logger.info("Running bin filter '%s'...",self._binFilters[binIdx])
      tunedDiscrInfo = dict()
      cSummaryInfo = self._summaryInfo[binIdx]
      cSummaryPPInfo = self._summaryPPInfo[binIdx]
      cRefBenchmarkList= refBenchmarkList[binIdx]
      self._logger.info('Using references: %r.', [(ReferenceBenchmark.tostring(ref.reference),ref.refVal) for ref in cRefBenchmarkList])
      for cFile, path in enumerate(binPath):
        self._logger.info("Reading file %d/%d (%s)", cFile, self._nFiles[binIdx], path )
        # And open them as Tuned Discriminators:
        try:
          with TunedDiscrArchieve(path) as TDArchieve:
            if TDArchieve.etaBinIdx != -1 and cFile == 0:
              self._logger.info("File eta bin index (%d) limits are: %r", 
                                 TDArchieve.etaBinIdx, 
                                 TDArchieve.etaBin, 
                                )
            if TDArchieve.etBinIdx != -1 and cFile == 0:
              self._logger.info("File Et bin index (%d) limits are: %r", 
                                 TDArchieve.etBinIdx, 
                                 TDArchieve.etBin, 
                               )
            # Now we loop over each configuration:
            for neuron, sort, init in product( TDArchieve.neuronBounds(), 
                                               TDArchieve.sortBounds(), 
                                               TDArchieve.initBounds() ):
              tunedDiscr, tunedPPChain = TDArchieve.getTunedInfo( neuron, sort, init )
              # Check if binning information matches:
              for refBenchmark in cRefBenchmarkList:
                if TDArchieve.etaBinIdx != -1 and refBenchmark.signal_efficiency.etaBin != -1 \
                    and TDArchieve.etaBinIdx != refBenchmark.signal_efficiency.etaBin:
                  self._logger.warning("File (%d) eta binning information does not match with benchmark (%d)!", 
                      TDArchieve.etaBinIdx,
                      refBenchmark.signal_efficiency.etaBin)
                if TDArchieve.etBinIdx != -1 and refBenchmark.signal_efficiency.etBin != -1 \
                    and TDArchieve.etBinIdx != refBenchmark.signal_efficiency.etBin:
                  self._logger.warning("File (%d) Et binning information does not match with benchmark (%d)!", 
                      TDArchieve.etBinIdx,
                      refBenchmark.signal_efficiency.etBin)
                # FIXME, this shouldn't be like that, instead the reference
                # benchmark should be passed to the TuningJob so that it could
                # set the best operation point itself.
                # When this is done, we can then remove the working points list
                # as it is done here:
                if type(tunedDiscr) is list:
                  # fastnet core version
                  discr = tunedDiscr[refBenchmark.reference]
                else:
                  # exmachina core version
                  discr = tunedDiscr

                self.__addPPChain( cSummaryPPInfo,
                                   tunedPPChain, 
                                   sort )                    
                self.__addPerformance( tunedDiscrInfo,
                                       path,
                                       refBenchmark, 
                                       neuron,
                                       sort,
                                       init,
                                       discr ) 
                # Add bin information to reference benchmark
              # end of references
            # end of configurations
          # with file
        except (UnpicklingError, ValueError, EOFError), e:
          self._logger.warning("Ignoring file '%s'. Reason:\n%s", path, str(e))
        if debug and cFile == 10:
          break
        gc.collect()
      # Finished all files in this bin

      # Print information retrieved:
      if self._level <= LoggingLevel.VERBOSE:
        for refBenchmark in cRefBenchmarkList:
          refName = refBenchmark.name
          self._logger.verbose("Retrieved %d discriminator configurations for benchmark '%s':", 
              len(tunedDiscrInfo[refName]) - 1, 
              refBenchmark)
          for nKey, nDict in tunedDiscrInfo[refName].iteritems():
            if nKey == 'benchmark': continue
            self._logger.verbose("Retrieved %d sorts for configuration '%r'", len(nDict), nKey)
            for sKey, sDict in nDict.iteritems():
              self._logger.verbose("Retrieved %d inits for sort '%d'", len(sDict['headerInfo']), sKey)
            # got number of inits
          # got number of sorts
        # got number of configurations
      # finished all references

      # Recreate summary info object
      for refKey, refValue in tunedDiscrInfo.iteritems(): # Loop over operations
        refBenchmark = refValue['benchmark']
        # Create a new dictionary and append bind it to summary info
        refDict = { 'rawBenchmark' : refBenchmark.rawInfo() }
        cSummaryInfo[refKey] = refDict
        for nKey, nValue in refValue.iteritems(): # Loop over neurons
          if nKey == 'benchmark':
            continue
          nDict = dict()
          refDict['config_' + str(nKey)] = nDict
          for sKey, sValue in nValue.iteritems(): # Loop over sorts
            sDict = dict()
            nDict['sort_' + str(sKey)] = sDict
            # Retrieve information from outermost initializations:
            ( sDict['summaryInfoTst'], \
              sDict['infoTstBest'], sDict['infoTstWorst']) = self.__outermostPerf( sValue['headerInfo'],
                                                                                   sValue['initPerfTstInfo'], 
                                                                                   refBenchmark, 
                                                                                   'sort', 
                                                                                   sKey )
            ( sDict['summaryInfoOp'], \
              sDict['infoOpBest'], sDict['infoOpWorst']) = self.__outermostPerf( sValue['headerInfo'],
                                                                                 sValue['initPerfOpInfo'], 
                                                                                 refBenchmark, 
                                                                                 'sort', 
                                                                                 sKey )
          # Retrieve information from outermost sorts:
          allBestTstSortInfo   = [ sDict['infoTstBest' ] for key, sDict in nDict.iteritems() ]
          allBestOpSortInfo    = [ sDict['infoOpBest'  ] for key, sDict in nDict.iteritems() ]
          ( nDict['summaryInfoTst'], \
            nDict['infoTstBest'], nDict['infoTstWorst']) = self.__outermostPerf( allBestTstSortInfo,
                                                                                 allBestTstSortInfo, 
                                                                                 refBenchmark, 
                                                                                 'config', 
                                                                                 nKey )
          ( nDict['summaryInfoOp'], \
            nDict['infoOpBest'], nDict['infoOpWorst'])   = self.__outermostPerf( allBestOpSortInfo,
                                                                                 allBestOpSortInfo, 
                                                                                 refBenchmark, 
                                                                                 'config', 
                                                                                 nKey )
        # Retrieve information from outermost discriminator configurations:
        allBestTstConfInfo   = [ nDict['infoTstBest' ] for key, nDict in refDict.iteritems() if key != 'rawBenchmark' ]
        allBestOpConfInfo    = [ nDict['infoOpBest'  ] for key, nDict in refDict.iteritems() if key != 'rawBenchmark' ]
        ( refDict['summaryInfoTst'], \
          refDict['infoTstBest'], refDict['infoTstWorst']) = self.__outermostPerf( allBestTstConfInfo,
                                                                                   allBestTstConfInfo, 
                                                                                   refBenchmark, 
                                                                                   'benchmark', 
                                                                                   refKey )
        ( refDict['summaryInfoOp'], \
          refDict['infoOpBest'], refDict['infoOpWorst'])   = self.__outermostPerf( allBestOpConfInfo,  
                                                                                   allBestOpConfInfo, 
                                                                                   refBenchmark, 
                                                                                   'benchmark', 
                                                                                   refKey )
      # Finished summary information
      if self._level <= LoggingLevel.DEBUG:
        for refKey, refValue in cSummaryInfo.iteritems(): # Loop over operations
          self._logger.debug("This is the summary information for benchmark %s", refKey )
          pprint({key : { innerkey : innerval for innerkey, innerval in val.iteritems() if not(innerkey.startswith('sort_'))} 
                                            for key, val in refValue.iteritems() if type(key) is str} 
                , depth=3
                )

      # Append pp collections
      cSummaryInfo['infoPPChain'] = self._summaryPPInfo

      # Save files
      if self._binFilters is not None:
        cOutputName = outputName + self._binFilters[binIdx].replace('*','_')
        if cOutputName.endswith('_'): 
          cOutputName = cOutputName[:-1]
      else:
        cOutputName = outputName
      outputPath = save( cSummaryInfo, cOutputName )
      self._logger.info("Saved file '%s'",outputPath)
      # Save matlab file
      if toMatlab:
        try:
          import scipy.io
          scipy.io.savemat( cOutputName + '.mat', cSummaryInfo)
        except ImportError:
          raise RuntimeError(("Cannot save matlab file, it seems that scipy is not "
              "available."))
      # Finished bin
    # finished all files
  # end of loop

  def __outermostPerf(self, headerInfoList, perfInfoList, refBenchmark, collectionType, val, **kw):

    self._logger.debug("%s: Retrieving outermost performance for %s %r (done twice, first for test, after for operation).",
        refBenchmark, collectionType, val )

    summaryDict = {'cut': [], 'sp': [], 'det': [], 'fa': [], 'idx': []}
    # Fetch all information together in the dictionary:
    for key in summaryDict.keys():
      summaryDict[key] = [ perfInfo[key] for perfInfo in perfInfoList ]
      if not key == 'idx':
        summaryDict[key + 'Mean'] = np.mean(summaryDict[key],axis=0)
        summaryDict[key + 'Std']  = np.std(summaryDict[key],axis=0)

    # Put information together on data:
    benchmarks = [summaryDict['sp'], summaryDict['det'], summaryDict['fa']]

    # The outermost performances:
    bestIdx  = refBenchmark.getOutermostPerf(benchmarks )
    worstIdx = refBenchmark.getOutermostPerf(benchmarks, cmpType = -1. )
    if self._level <= LoggingLevel.DEBUG:
      self._logger.debug('Retrieved best index as: %d; values: (%f,%f,%f)', bestIdx, 
          benchmarks[0][bestIdx],
          benchmarks[1][bestIdx],
          benchmarks[2][bestIdx])
      self._logger.debug('Retrieved worst index as: %d; values: (%f,%f,%f)', worstIdx,
          benchmarks[0][worstIdx],
          benchmarks[1][worstIdx],
          benchmarks[2][worstIdx])

    # Retrieve information from outermost performances:
    def __getInfo( headerInfoList, perfInfoList, idx ):
      info = dict()
      wantedKeys = ['filepath', 'neuron', 'sort', 'init']
      headerInfo = headerInfoList[idx]
      for key in wantedKeys:
        info[key] = headerInfo[key]
      wantedKeys = ['cut','sp', 'det', 'fa', 'idx']
      perfInfo = perfInfoList[idx]
      for key in wantedKeys:
        info[key] = perfInfo[key]
      return info

    bestInfoDict  = __getInfo( headerInfoList, perfInfoList, bestIdx )
    worstInfoDict = __getInfo( headerInfoList, perfInfoList, worstIdx )
    if self._level <= LoggingLevel.VERBOSE:
      self._logger.verbose("The best configuration retrieved is: ")
      pprint(bestInfoDict)
      self._logger.verbose("The worst configuration retrieved is: ")
      pprint(worstInfoDict)

    return (summaryDict, bestInfoDict, worstInfoDict)
  # end of __outermostPerf


  def __addPPChain(self, cSummaryPPInfo, tunedPPChain, sort):
    if not( ('sort_'+str(sort) ) in cSummaryPPInfo ):
      ppData=dict();
      for ppID, ppObj in enumerate(tunedPPChain):
        #choose correct type
        if type(ppObj) is PreProc.Norm1:
          ppData[ppObj.shortName()+'_id'+str(ppID)] = 'Norm1'    
        elif type(ppObj) is PreProc.PCA:
          ppData[ppObj.shortName()+'_id'+str(ppID)] = { 'variance'    : ppObj.variance(),
                                                        'n_components': ppObj.ncomponents(),
                                                        'cov'         : ppObj.cov(),
                                                        'components'  : ppObj.params().components_,
                                                        'means'       : ppObj.params().mean_
                                                      }
        elif type(ppObj) is PreProc.KernelPCA:
          ppData[ppObj.shortName()+'_id'+str(ppID)] = { 'variance'    : ppObj.variance(),
                                                        'n_components': ppObj.ncomponents(),
                                                        'cov'         : ppObj.cov(),}
        else:
          self._logger.info('No PreProc type found')
          continue
      #add into sort list    
      cSummaryPPInfo['sort_'+str(sort)] = ppData
  #end of __addPPChain


  #def plot_topo(self, obj, y_limits, outputName):
  #  """
  #    Plot topology efficiency for 
  #  """
  #  def __plot_topo(tpad, obj, var, y_limits, title, xlabel, ylabel):
  #    x_axis = range(*[y_limits[0],y_limits[1]+1])
  #    x_axis_values = np.array(x_axis,dtype='float_')
  #    inds = x_axis_values.astype('int_')
  #    x_axis_error   = np.zeros(x_axis_values.shape,dtype='float_')
  #    y_axis_values  = obj[var+'_mean'].astype('float_')
  #    y_axis_error   = obj[var+'_std'].astype('float_')
  #    graph = ROOT.TGraphErrors(len(x_axis_values),x_axis_values,y_axis_values[inds], x_axis_error, y_axis_error[inds])
  #    graph.Draw('ALP')
  #    graph.SetTitle(title)
  #    graph.SetMarkerColor(4); graph.SetMarkerStyle(21)
  #    graph.GetXaxis().SetTitle('neuron #')
  #    graph.GetYaxis().SetTitle('SP')
  #    tpad.Modified()
  #    tpad.Update()
  #  # end of helper fcn __plot_topo

  #  canvas = ROOT.TCanvas('c1','c1',2000,1300)
  #  canvas.Divide(1,3) 
  #  __plot_topo(canvas.cd(1), obj, 'sp_op', y_limits, 'SP fluctuation', '# neuron', 'SP')
  #  __plot_topo(canvas.cd(2), obj, 'det_op', y_limits, 'Detection fluctuation', '# neuron', 'Detection')
  #  __plot_topo(canvas.cd(3), obj, 'fa_op', y_limits, 'False alarm fluctuation', '# neuron', 'False alarm')
  #  canvas.SaveAs(outputName)

  #def plot_evol(self, bucket, best_id, worse_id, outputName):
  #  """
  #    Plot tuning evolution for the information available on the available
  #    discriminators.
  #  """
  #  def __plot_evol( tpad, curves, y_axis_limits, **kw):
  #    title         = kw.pop('title', '')
  #    xlabel        = kw.pop('xlabel','x axis')
  #    ylabel        = kw.pop('ylabel','y axis')
  #    select_pos1   = kw.pop('select_pop1',-1)
  #    select_pos2   = kw.pop('select_pop2',-1)
  #    color_curves  = kw.pop('color_curves',ROOT.kBlue)
  #    color_select1 = kw.pop('color_select1',ROOT.kBlack)
  #    color_select2 = kw.pop('color_select2',ROOT.kRed)

  #    #create dummy graph
  #    x_max = 0; dummy = None
  #    for i in range(len(curves)):
  #      curves[i].SetLineColor(color_curves)
  #      x = curves[i].GetXaxis().GetXmax()
  #      if x > x_max: x_max = x; dummy = curves[i]
  #    
  #    dummy.SetTitle( title )
  #    dummy.GetXaxis().SetTitle(xlabel)
  #    #dummy.GetYaxis().SetTitleSize( 0.4 ) 
  #    dummy.GetYaxis().SetTitle(ylabel)
  #    #dummy.GetYaxis().SetTitleSize( 0.4 )

  #    #change the axis range for y axis
  #    dummy.GetHistogram().SetAxisRange(y_axis_limits[0],y_axis_limits[1],'Y' )
  #    dummy.Draw('AL')

  #    for c in curves:  c.Draw('same')
  #    if select_pos1 > -1:  curves[select_pos1].SetLineColor(color_select1); curves[select_pos1].Draw('same')
  #    if select_pos2 > -1:  curves[select_pos2].SetLineColor(color_select2); curves[select_pos2].Draw('same')
  #    
  #    tpad.Modified()
  #    tpad.Update()
  #  
  #  red   = ROOT.kRed+2
  #  blue  = ROOT.kAzure+6
  #  black = ROOT.kBlack
  #  canvas = ROOT.TCanvas('c1','c1',2000,1300)
  #  canvas.Divide(1,4) 
  #  mse=list();sp=list();det=list();fa=list()
  #  roc_val=list();roc_op=list()

  #  for graphs in bucket:
  #    mse.append( graphs['train']['mse_val'] )
  #    sp.append( graphs['train']['sp_val'] )
  #    det.append( graphs['train']['det_val'] )
  #    fa.append( graphs['train']['fa_val'] )
  #    roc_val.append( graphs['train']['roc_val'] )
  #    roc_op.append( graphs['train']['roc_op'] )

  #  __plot_evol(canvas.cd(1),mse,[0,.3],title='Mean Square Error Evolution',
  #                                     xlabel='epoch #', ylabel='MSE',
  #                                     select_pos1=best_id,
  #                                     select_pos2=worse_id,
  #                                     color_curves=blue,
  #                                     color_select1=black,
  #                                     color_select2=red)
  #  __plot_evol(canvas.cd(2),sp,[.93,.97],title='SP Evolution',
  #                                     xlabel='epoch #', ylabel='SP',
  #                                     select_pos1=best_id,
  #                                     select_pos2=worse_id,
  #                                     color_curves=blue,
  #                                     color_select1=black,
  #                                     color_select2=red)
  #  __plot_evol(canvas.cd(3),det,[.95,1],title='Detection Evolution',
  #                                     xlabel='epoch #',
  #                                     ylabel='Detection',
  #                                     select_pos1=best_id,
  #                                     select_pos2=worse_id,
  #                                     color_curves=blue,
  #                                     color_select1=black,
  #                                     color_select2=red)
  #  __plot_evol(canvas.cd(4),fa,[0,.3],title='False alarm evolution',
  #                                     xlabel='epoch #', ylabel='False alarm',
  #                                     select_pos1=best_id,
  #                                     select_pos2=worse_id,
  #                                     color_curves=blue,
  #                                     color_select1=black,
  #                                     color_select2=red)
  #   
  #  canvas.cd(1)
  #  logoLabel_obj   = ROOT.TLatex(.65,.65,self._logoLabel);
  #  logoLabel_obj.SetTextSize(.25)
  #  logoLabel_obj.Draw()
  #  canvas.Modified()
  #  canvas.Update()
  #  canvas.SaveAs(outputName)
  #  del canvas 
  #  canvas = ROOT.TCanvas('c2','c2',2000,1300)
  #  canvas.Divide(2,1)
  #  __plot_evol(canvas.cd(1),roc_val,[.80,1],title='ROC (Validation)',
  #              xlabel='false alarm',
  #              ylabel='detection',
  #              select_pos1=best_id,
  #              select_pos2=worse_id,
  #              color_curves=blue,
  #              color_select1=black,
  #              color_select2=red)
  #  __plot_evol(canvas.cd(2),roc_op,[.80,.1],title='ROC (Operation)',
  #              xlabel='false alarm', 
  #              ylabel='detection',
  #              select_pos1=best_id,
  #              select_pos2=worse_id,
  #              color_curves=blue,
  #              color_select1=black,
  #              color_select2=red)
  #  canvas.Modified()
  #  canvas.Update()
  #  canvas.SaveAs('roc_'+outputName)
        

  def exportDiscrFiles(self, ringerOperation, **kw ):
    """
    Export discriminators operating at reference benchmark list to the
    ATLAS environment using this CrossValidStat information.
    """
    if not self._summaryInfo:
      raise RuntimeError(("Create the summary information using the loop method."))
    CrossValidStat.exportDiscrFiles( self._summaryInfo, 
                                     ringerOperation, 
                                     level = self._level,
                                     **kw )

  @classmethod
  def exportDiscrFiles(cls, summaryInfoList, ringerOperation, **kw):
    """
    Export discriminators operating at reference benchmark list to the
    ATLAS environment using summaryInfo. 
    
    If benchmark name on the reference list is not available at summaryInfo, an
    KeyError exception will be raised.
    """
    baseName      = kw.pop( 'baseName'      , 'tunedDiscr'      )
    refBenchCol   = kw.pop( 'refBenchCol'   , None              )
    configCol     = kw.pop( 'configCol'     , []                )
    triggerChains = kw.pop( 'triggerChains' , None              )
    level         = kw.pop( 'level'         , LoggingLevel.INFO )

    # Initialize local logger
    logger      = Logger.getModuleLogger("exportDiscrFiles", logDefaultLevel = level )
    checkForUnusedVars( kw, logger.warning )

    # Treat the summaryInfoList
    if not isinstance( summaryInfoList, (list,tuple)):
      summaryInfoList = [ summaryInfoList ]
    summaryInfoList = list(traverse(summaryInfoList,simple_ret=True))
    nSummaries = len(summaryInfoList)

    if refBenchCol is None:
      refBenchCol = summaryInfoList[0].keys()

    # Treat the reference benchmark list
    if not isinstance( refBenchCol, (list,tuple)):
      refBenchCol = [ refBenchCol ] * nSummaries

    if len(refBenchCol) == 1:
      refBenchCol = refBenchCol * nSummaries

    nRefs = len(list(traverse(refBenchCol,simple_ret=True)))

    # Make sure that the lists are the same size as the reference benchmark:
    nConfigs = len(list(traverse(configCol,simple_ret=True)))
    if nConfigs == 0:
      configCol = [None for i in range(nRefs)]
    elif nConfigs == 1:
      configCol = configCol * nSummaries

    if nConfigs != nRefs:
      raise ValueError("Summary size is not equal to the configuration list.")
    
    if nRefs == nConfigs == nSummaries:
      # If user input data without using list on the configuration, put it as a list:
      for o, idx, parent, _, _ in traverse(configCol):
        parent[idx] = [o]
      for o, idx, parent, _, _ in traverse(refBenchCol):
        parent[idx] = [o]

    configCol   = list(traverse(configCol,max_depth_dist=1,simple_ret=True))
    refBenchCol = list(traverse(refBenchCol,max_depth_dist=1,simple_ret=True))
    nConfigs = len(configCol)
    nSummary = len(refBenchCol)

    if nRefs != nConfigs != nSummary:
      raise ValueError("Number of references, configurations and summaries do not match!")

    # Retrieve the operation:
    from TuningTools.FilterEvents import RingerOperation
    ringerOperation = RingerOperation.retrieve(ringerOperation)
    logger.info(('Exporting discrimination info files for the following '
                'operating point (RingerOperation:%s).'), 
                RingerOperation.tostring(ringerOperation))

    if ringerOperation is RingerOperation.L2:
      if triggerChains is None:
        triggerChains = "custom"
      if type(triggerChains) not in (list,tuple):
        triggerChains = [triggerChains]
      nExports = len(refBenchCol[0])
      if len(triggerChains) == 1 and nExports != 1:
        baseChainName = triggerChains[0]
        triggerChains = ["%s_chain%d" % (baseChainName,i) for i in range(nExports)]
      if nExports != len(triggerChains):
        raise ValueError("Number of exporting chains does not match with number of given chain names.")

      output = open('TrigL2CaloRingerConstants.py','w')
      output.write('def RingerMap():\n')
      output.write('  signatures=dict()\n')
      outputDict = dict()

    for summaryInfo, refBenchmarkList, configList in \
                        zip(summaryInfoList,
                            refBenchCol,
                            configCol,
                           ):
      if type(summaryInfo) is str:
        logger.info('Loading file "%s"...', summaryInfo)
        summaryInfo = load(summaryInfo)
      elif type(summaryInfo) is dict:
        pass
      else:
        raise ValueError("Cross-valid summary info is not string and not a dictionary.")
      from itertools import izip, count
      for idx, refBenchmarkName, config in izip(count(), refBenchmarkList,configList):
        info = summaryInfo[refBenchmarkName]['infoOpBest'] if config is None else \
               summaryInfo[refBenchmarkName]['config_' + str(config)]['infoOpBest']
        logger.info("%s discriminator information is available at file: \n\t%s", 
                    refBenchmarkName,
                    info['filepath'])
        ## Check if user specified parameters for exporting discriminator
        ## operation information:
        sort = info['sort']
        init = info['init']
        with TunedDiscrArchieve(info['filepath'], level = level ) as TDArchieve:
          etBinIdx = TDArchieve.etBinIdx
          etaBinIdx = TDArchieve.etaBinIdx
          etBin = TDArchieve.etBin
          etaBin = TDArchieve.etaBin
          ## Write the discrimination wrapper
          if ringerOperation is RingerOperation.Offline:
            # Import athena cpp information
            try:
              import cppyy
            except ImportError:
              import PyCintex as cppyy
            try:
              cppyy.loadDict('RingerSelectorTools_Reflex')
            except RuntimeError:
              raise RuntimeError("Couldn't load RingerSelectorTools_Reflex dictionary.")
            from ROOT import TFile
            from ROOT import std
            from ROOT.std import vector
            # Import Ringer classes:
            from ROOT import Ringer
            from ROOT import MsgStream
            from ROOT import MSG
            from ROOT.Ringer import IOHelperFcns
            from ROOT.Ringer import RingerProcedureWrapper
            from ROOT.Ringer import Discrimination
            from ROOT.Ringer import IDiscrWrapper
            from ROOT.Ringer import IDiscrWrapperCollection
            from ROOT.Ringer import IThresWrapper
            from ROOT.Ringer.Discrimination import UniqueThresholdVarDep
            # Extract dictionary:
            discrData, keep_lifespan_list = TDArchieve.exportDiscr(config, 
                                                                   sort, 
                                                                   init, 
                                                                   ringerOperation, 
                                                                   summaryInfo[refBenchmarkName]['rawBenchmark'])
            logger.debug("Retrieved discrimination info!")

            fDiscrName = baseName + '_Discr_' + refBenchmarkName + ".root"
            # Export the discrimination wrapper to a TFile and save it:
            discrCol = IDiscrWrapperCollection() 
            discrCol.push_back(discrData)
            IDiscrWrapper.writeCol(discrCol, fDiscrName)
            logger.info("Successfully created file %s.", fDiscrName)
            ## Export the Threshold Wrapper:
            RingerThresWrapper = RingerProcedureWrapper("Ringer::Discrimination::UniqueThresholdVarDep",
                                                        "Ringer::EtaIndependent",
                                                        "Ringer::EtIndependent",
                                                        "Ringer::NoSegmentation")
            BaseVec = vector("Ringer::Discrimination::UniqueThresholdVarDep*")
            vec = BaseVec() # We are not using eta dependency
            thres = UniqueThresholdVarDep(info['cut'])
            if logger.isEnabledFor( LoggingLevel.DEBUG ):
              thresMsg = MsgStream("ExportedThreshold")
              thresMsg.setLevel(LoggingLevel.toC(level))
              thres.setMsgStream(thresMsg)
              getattr(thres,'print')(MSG.DEBUG)
            vec.push_back( thres )
            thresVec = vector(BaseVec)() # We are not using et dependency
            thresVec.push_back(vec)
            ## Create pre-processing wrapper:
            logger.debug('Initiazing Threshold Wrapper:')
            thresWrapper = RingerThresWrapper(thresVec)
            fThresName = baseName + '_Thres_' + refBenchmarkName + ".root"
            IThresWrapper.writeWrapper( thresWrapper, fThresName )
            logger.info("Successfully created file %s.", fThresName)
          elif ringerOperation is RingerOperation.L2:
            triggerChain = triggerChains[idx]
            if not triggerChain in outputDict:
              cDict = {}
              outputDict[triggerChain] = cDict
            else:
              cDict = outputDict[triggerChain]
            config = {}
            cDict['eta%d_et%d' % (etaBinIdx, etBinIdx) ] = config
            #config['rawBenchmark'] = summaryInfo[refBenchmarkName]['rawBenchmark']
            #config['infoOp']       = info
            # FIXME Index [0] is the discriminator, [1] is the normalization. This should be more organized.
            discr = TDArchieve.getTunedInfo(info['neuron'],
                                            info['sort'],
                                            info['init'])[0]
            if type(discr) is list:
              reference = ReferenceBenchmark.retrieve( summaryInfo[refBenchmarkName]['rawBenchmark']['reference'] )
              discr = discr[reference]
            else:
              discr = ['discriminator']
            discr = { key : (val.tolist() if type(val) == np.ndarray \
                          else val) for key, val in discr['discriminator'].iteritems()
                    }
            config.update( discr )
            config['threshold'] = info['cut']
            config['etaBin']     = etaBin.tolist()
            config['etBin']      = etBin.tolist()
            logger.info('Exported bin(et=%d,eta=%d) using following configuration:',
                        etBinIdx,
                        etaBinIdx)
            logger.info('neuron = %d, sort = %d, init = %d, thr = %f',
                        info['neuron'],
                        info['sort'],
                        info['init'],
                        info['cut'])
          else:
            raise RuntimeError('You must choose a ringerOperation')
        # with
      # for benchmark
    # for summay in list

    if ringerOperation is RingerOperation.L2:
      for key, val in outputDict.iteritems():
        output.write('  signatures["%s"]=%s\n' % (key, val))
      output.write('  return signatures\n')
  # exportDiscrFiles 

  @classmethod
  def printTables(cls, confBaseNameList,
                       crossValGrid,
                       configMap):
    "Print operation tables for the "
    # TODO Improve documentation

    # We first loop over the configuration base names:
    for ds in [Dataset.Test, Dataset.Operation]:
      for confIdx, confBaseName in enumerate(confBaseNameList):
        print "{:=^90}".format("  %s ( %s )  " % (confBaseName, Dataset.tostring(ds)) )
        # And then on et/eta bins:
        for crossList in crossValGrid:
          print "{:-^90}".format("  Starting new Et  ")
          for crossFile in crossList:
            # Load file and then search the benchmark references with the configuration name:
            summaryInfo = load(crossFile)
            etIdx = -1
            etaIdx = -1
            for key in summaryInfo.keys():
              try:
                rawBenchmark = summaryInfo[key]['rawBenchmark']
                try:
                  etIdx = rawBenchmark['signal_efficiency']['etBin']
                  etaIdx = rawBenchmark['signal_efficiency']['etaBin']
                except KeyError:
                  etIdx = rawBenchmark['signal_efficiency']['_etBin']
                  etaIdx = rawBenchmark['signal_efficiency']['_etaBin']
                break
              except (KeyError, TypeError) as e:
                pass
            print "{:-^90}".format("  Eta (%d) | Et (%d)  " % (etaIdx, etIdx))
            #from scipy.io import loadmat
            #summaryInfo = loadmat(crossFile)
            confPdKey = confSPKey = confPfKey = None
            for key in summaryInfo.keys():
              if key == 'infoPPChain': continue
              rawBenchmark = summaryInfo[key]['rawBenchmark']
              reference = rawBenchmark['reference']
              # Retrieve the configuration keys:
              if confBaseName in key:
                if reference == 'Pd':
                  confPdKey = key 
                if reference == 'Pf':
                  confPfKey = key 
                if reference == 'SP':
                  confSPKey = key 
            # Loop over each one of the cases and print ringer performance:
            print '{:^13}   {:^13}   {:^13} |   {:^13}   |  {}  '.format("Pd (%)","SP (%)","Pf (%)","cut","(ReferenceBenchmark)")
            print "{:-^90}".format("  Ringer  ")
            for keyIdx, key in enumerate([confPdKey, confSPKey, confPfKey]):
              if not key:
                print '{:-^90}'.format(' Information Unavailable ')
                continue
              if ds is Dataset.Test:
                ringerPerf = summaryInfo[key] \
                                        ['config_' + str(configMap[confIdx][etIdx][etaIdx][keyIdx])] \
                                        ['summaryInfoTst']
                print '%6.3f+-%5.3f   %6.3f+-%5.3f   %6.3f+-%5.3f |   % 5.3f+-%5.3f   |  (%s) ' % ( 
                    ringerPerf['detMean'] * 100.,   ringerPerf['detStd']  * 100.,
                    ringerPerf['spMean']  * 100.,   ringerPerf['spStd']   * 100.,
                    ringerPerf['faMean']  * 100.,   ringerPerf['faStd']   * 100.,
                    ringerPerf['cutMean']       ,   ringerPerf['cutStd']        ,
                    key)
              else:
                ringerPerf = summaryInfo[key] \
                                        ['config_' + str(configMap[confIdx][etIdx][etaIdx][keyIdx])] \
                                        ['infoOpBest']
                print '{:^13.3f}   {:^13.3f}   {:^13.3f} |   {:^ 13.3f}   |  ({}) '.format(
                    ringerPerf['det'] * 100.,
                    ringerPerf['sp']  * 100.,
                    ringerPerf['fa']  * 100.,
                    ringerPerf['cut'],
                    key)

            print "{:-^90}".format("  Baseline  ")
            reference_sp = calcSP(
                                  rawBenchmark['signal_efficiency']['efficiency'] / 100.,
                                  ( 1. - rawBenchmark['background_efficiency']['efficiency'] / 100. )
                                 )
            print '{:^13.3f}   {:^13.3f}   {:^13.3f} |{:@<43}'.format(
                                      rawBenchmark['signal_efficiency']['efficiency']
                                      ,reference_sp * 100.
                                      ,rawBenchmark['background_efficiency']['efficiency']
                                      ,''
                                     )
            if ds is Dataset.Test:
              print "{:.^90}".format("")
              try:
                sgnCrossEff    = rawBenchmark['signal_cross_efficiency']['_branchCollectorsDict'][Dataset.Test]
                bkgCrossEff    = rawBenchmark['background_cross_efficiency']['_branchCollectorsDict'][Dataset.Test]
                sgnRawCrossVal = rawBenchmark['signal_cross_efficiency']['efficiency']['Test']
                bkgRawCrossVal = rawBenchmark['background_cross_efficiency']['efficiency']['Test']
              except KeyError:
                sgnCrossEff = rawBenchmark['signal_cross_efficiency']['_branchCollectorsDict'][Dataset.Validation]
                bkgCrossEff = rawBenchmark['background_cross_efficiency']['_branchCollectorsDict'][Dataset.Validation]
                sgnRawCrossVal = rawBenchmark['signal_cross_efficiency']['efficiency']['Validation']
                bkgRawCrossVal = rawBenchmark['background_cross_efficiency']['efficiency']['Validation']
              try:
                reference_sp = [ calcSP(rawSgn,(100.-rawBkg))
                                  for rawSgn, rawBkg in zip(sgnCrossEff, bkgCrossEff)
                               ]
              except TypeError: # Old format compatibility
                reference_sp = [ calcSP(rawSgn['efficiency'],(100.-rawBkg['efficiency']))
                                  for rawSgn, rawBkg in zip(sgnCrossEff, bkgCrossEff)
                               ]
              print '{:6.3f}+-{:5.3f}   {:6.3f}+-{:5.3f}   {:6.3f}+-{:5.3f} |{:@<43}'.format( 
                  sgnRawCrossVal[0]
                  ,sgnRawCrossVal[1]
                  ,np.mean(reference_sp)
                  ,np.std(reference_sp)
                  ,bkgRawCrossVal[0]
                  ,bkgRawCrossVal[1]
                  ,'')
        print "{:=^90}".format("")


class PerfHolder:
  """
  Hold the performance values and evolution for a tunned discriminator
  """

  def __init__(self, tunedDiscrData ):

    self.roc_tst       = tunedDiscrData['summaryInfo']['roc_test']
    self.roc_operation = tunedDiscrData['summaryInfo']['roc_operation']
    trainEvo           = tunedDiscrData['trainEvolution']
    self.epoch         = np.array( range(len(trainEvo['epoch'])), dtype ='float_')
    self.nEpoch        = len(self.epoch)
    self.mse_trn       = np.array( trainEvo['mse_trn'],           dtype ='float_')
    self.mse_val       = np.array( trainEvo['mse_val'],           dtype ='float_')
    self.mse_tst       = np.array( trainEvo['mse_tst'],           dtype ='float_')
    self.sp_val        = np.array( trainEvo['sp_val'],            dtype ='float_')
    self.sp_tst        = np.array( trainEvo['sp_tst'],            dtype ='float_')
    self.det_val       = np.array( trainEvo['det_val'],           dtype ='float_')
    self.det_tst       = np.array( trainEvo['det_tst'],           dtype ='float_')
    self.fa_val        = np.array( trainEvo['fa_val'],            dtype ='float_')
    self.fa_tst        = np.array( trainEvo['fa_tst'],            dtype ='float_')
    self.roc_tst_det   = np.array( self.roc_tst.detVec,           dtype ='float_')
    self.roc_tst_fa    = np.array( self.roc_tst.faVec,            dtype ='float_')
    self.roc_tst_cut   = np.array( self.roc_tst.cutVec,           dtype ='float_')
    self.roc_op_det    = np.array( self.roc_operation.detVec,     dtype ='float_')
    self.roc_op_fa     = np.array( self.roc_operation.faVec,      dtype ='float_')
    self.roc_op_cut    = np.array( self.roc_operation.cutVec,     dtype ='float_')

  def getOperatingBenchmarks( self, refBenchmark, **kw):
    """
    Returns the operating benchmark values for this tunned discriminator
    """
    idx = kw.pop('idx', None)
    ds  = kw.pop('ds', Dataset.Test )
    if ds is Dataset.Test:
      detVec = self.roc_tst_det
      faVec = self.roc_tst_fa
      cutVec = self.roc_tst_cut
    elif ds is Dataset.Operation:
      detVec = self.roc_op_det
      faVec = self.roc_op_fa
      cutVec = self.roc_op_cut
    else:
      raise ValueError("Cannot retrieve maximum ROC SP for dataset '%s'", ds)
    spVec = calcSP( detVec, 1 - faVec )
    if idx is None:
      if refBenchmark.reference is ReferenceBenchmark.SP:
        idx = np.argmax( spVec )
      else:
        # Get reference for operation:
        if refBenchmark.reference is ReferenceBenchmark.Pd:
          ref = detVec
        elif refBenchmark.reference is ReferenceBenchmark.Pf:
          ref = faVec
          idx = np.argmin( np.abs( ref - refBenchmark.refVal ) )
        idx = np.argmin( np.abs( ref - refBenchmark.refVal ) )
    sp  = spVec[idx]
    det = detVec[idx]
    fa  = faVec[idx]
    cut = cutVec[idx]
    return (sp, det, fa, cut, idx)

  def getGraph( self, graphType ):
    """
      Retrieve a TGraph from the discriminator Tuning data.

      perfHolder.getGraph( option )

      The possible options are:
        * mse_trn
        * mse_val
        * mse_tst
        * sp_val
        * sp_tst
        * det_val
        * det_tst
        * fa_val
        * fa_tst
        * roc_val
        * roc_op
        * roc_val_cut
        * roc_op_cut
    """
    import ROOT
    if   graphType == 'mse_trn'     : return ROOT.TGraph(self.nEpoch, self.epoch, self.mse_val )
    elif graphType == 'mse_val'     : return ROOT.TGraph(self.nEpoch, self.epoch, self.mse_val )
    elif graphType == 'mse_tst'     : return ROOT.TGraph(self.nEpoch, self.epoch, self.mse_tst )
    elif graphType == 'sp_val'      : return ROOT.TGraph(self.nEpoch, self.epoch, self.sp_val  )
    elif graphType == 'sp_tst'      : return ROOT.TGraph(self.nEpoch, self.epoch, self.sp_tst  )
    elif graphType == 'det_val'     : return ROOT.TGraph(self.nEpoch, self.epoch, self.det_val )
    elif graphType == 'det_tst'     : return ROOT.TGraph(self.nEpoch, self.epoch, self.det_tst )
    elif graphType == 'fa_val'      : return ROOT.TGraph(self.nEpoch, self.epoch, self.fa_val  )
    elif graphType == 'fa_tst'      : return ROOT.TGraph(self.nEpoch, self.epoch, self.fa_tst  )
    elif graphType == 'roc_val'     : return ROOT.TGraph(len(self.roc_val_fa), self.roc_val_fa, self.roc_val_det )
    elif graphType == 'roc_op'      : return ROOT.TGraph(len(self.roc_op_fa),  self.roc_op_fa,  self.roc_op_det  )
    elif graphType == 'roc_val_cut' : return ROOT.TGraph(len(self.roc_val_cut),
                                                         np.array(range(len(self.roc_val_cut) ), 'float_'), 
                                                         self.roc_val_cut )
    elif graphType == 'roc_op_cut'  : return ROOT.TGraph(len(self.roc_op_cut), 
                                                         np.array(range(len(self.roc_op_cut) ),  'float_'), 
                                                         self.roc_op_cut  )
    else: raise ValueError( "Unknown graphType '%s'" % graphType )

