from RingerCore.Logger import Logger, LoggingLevel
from RingerCore.util   import EnumStringification, get_attributes
from RingerCore.util   import checkForUnusedVars, calcSP
from RingerCore.FileIO import save, load
from TuningTools.TuningJob import TunedDiscrArchieve
import TuningTools.PreProc as PreProc
from TuningTools.FilterEvents import Dataset
from pprint import pprint
from cPickle import UnpicklingError
import ROOT
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


# FIXME: This should be used by TuningJob to determine the references which
# are to be used by the discriminator tunner
class ReferenceBenchmark(EnumStringification):
  """
  Reference benchmark to set tuned discriminator operation point.
  """
  SP = 0
  Pd = 1
  Pf = 2

  def __init__(self, name, reference, **kw):
    """
    ref = ReferenceBenchmark(name, reference, [, refVal = None] [, removeOLs = False])

      * name: The name for this reference benchmark;
      * reference: The reference benchmark type. It must one of
          ReferenceBenchmark enumerations;
      * refVal [None]: the reference value to operate;
      * removeOLs [False]: Whether to remove outliers from operation.
      * allowLargeDeltas [True]: When set to true and no value is within the operation bounds,
          then it will use operation closer to the reference.
    """
    self.refVal = kw.pop('refVal', None)
    self.removeOLs = kw.pop('removeOLs', False)
    self.allowLargeDeltas = kw.pop('allowLargeDeltas', True)
    if not (type(name) is str):
      raise TypeError("Name must be a string.")
    self.name = name
    self.reference = ReferenceBenchmark.retrieve(reference)
    if reference == ReferenceBenchmark.Pf:
      self.refVal = - self.refVal
  # __init__

  def rawInfo(self):
    """
    Return raw benchmark information
    """
    return { 'reference' : ReferenceBenchmark.tostring(self.reference),
             'refVal'    : (self.refVal if not self.refVal is None else -999),
             'removeOLs' : self.removeOLs }

  def getOutermostPerf(self, data, **kw):
    """
    Get outermost performance for the tuned discriminator performances on data. 
    idx = refBMark.getOutermostPerf( data [, eps = .001 ][, cmpType = 1])

     * data: A list with following struction:
        data[0] : SP
        data[1] : Pd
        data[2] : Pf

     * eps [.001] is used softening. The larger it is, more candidates will be
    possible to be considered, but farther the returned operation may be from
    the reference. The default is 0.1% deviation from the reference value.
     * cmpType [+1.] is used to change the comparison type. Use +1.
    for best performance, and -1 for worst performance.
    """
    # Retrieve optional arguments
    eps = kw.pop('eps', 0.001 )
    cmpType = kw.pop('cmpType', 1.)
    # We will transform data into np array, as it will be easier to work with
    npData = []
    for aData in data:
      npData.append( np.array(aData, dtype='float_') )
    # Retrieve reference and benchmark arrays
    if self.reference is ReferenceBenchmark.Pf:
      refVec = npData[2]
      benchmark = (cmpType) * npData[1]
    elif self.reference is ReferenceBenchmark.Pd:
      refVec = npData[1] 
      benchmark = (-1. * cmpType) * npData[2]
    elif self.reference is ReferenceBenchmark.SP:
      benchmark = (cmpType) * npData[0]
    else:
      raise ValueError("Unknown reference %d" % self.reference)
    # Retrieve the allowed indexes from benchmark which are not outliers
    if self.removeOLs:
      q1=percentile(benchmark,25.0)
      q3=percentile(benchmark,75.0)
      outlier_higher = q3 + 1.5*(q3-q1)
      outlier_lower  = q1 + 1.5*(q1-q3)
      allowedIdxs = np.all([benchmark > q3, benchmark < q1], axis=0).nonzero()[0]
    # Finally, return the index:
    if self.reference is ReferenceBenchmark.SP: 
      if self.removeOLs:
        idx = np.argmax( cmpType * benchmark[allowedIdxs] )
        return allowedIdx[ idx ]
      else:
        return np.argmax( cmpType * benchmark )
    else:
      if self.removeOLs:
        refAllowedIdxs = ( np.abs( refVec[allowedIdxs] - self.refVal) < eps ).nonzero()[0]
        if not refAllowedIdxs.size:
          if not self.allowLargeDeltas:
            # We don't have any candidate, raise:
            raise RuntimeError("eps is too low, no indexes passed constraint! Reference is %r | RefVec is: \n%r" %
                (self.refVal, refVec))
          else:
            # We can search for the closest candidate available:
            return allowedIdxs[ np.argmin( np.abs(refVec[allowedIdxs] - self.refVal) ) ]
        # Otherwise we return best benchmark for the allowed indexes:
        return refAllowedIdxs[ np.argmax( ( benchmark[allowedIdxs] )[ refAllowedIdxs ] ) ]
      else:
        refAllowedIdxs = ( np.abs( refVec - self.refVal ) < eps ).nonzero()[0]
        if not refAllowedIdxs.size:
          if not self.allowLargeDeltas:
            # We don't have any candidate, raise:
            raise RuntimeError("eps is too low, no indexes passed constraint! Reference is %r | RefVec is: \n%r" %
                (self.refVal, refVec))
          else:
            # We can search for the closest candidate available:
            return np.argmin( np.abs(refVec - self.refVal) )
        # Otherwise we return best benchmark for the allowed indexes:
        return refAllowedIdxs[ np.argmax( benchmark[ refAllowedIdxs ] ) ]

  def __str__(self):
    str_ =  self.name + '(' + ReferenceBenchmark.tostring(self.reference) 
    if self.refVal: str_ += ':' + str(self.refVal)
    str_ += ')'
    return str_

class CrossValidStatAnalysis( Logger ):

  _tunedDiscrInfo = dict()
  _summaryInfo    = dict()
  _summaryPPInfo  = dict()
  _nFiles = 0

  def __init__(self, paths, **kw):
    """
    Usage: 

    # Create object
    cvStatAna = CrossValidStatAnalysis( paths [,logoLabel][,logger[,level=INFO]])
    # Fill the information and save output file with cross-validation summary
    cvStatAna( refBenchMark, **args...)
    # Call other methods if necessary.
    """
    Logger.__init__(self, kw)    
    self._logoLabel = kw.pop('logoLabel', 'TuningTool' )
    checkForUnusedVars(kw, self._logger.warning)
    # Recursively expand all folders in the given paths so that we have all
    # files lists:
    from RingerCore.FileIO import expandFolders
    self._paths = expandFolders( paths )
    self._nFiles = len(self._paths)
    if self.level <= LoggingLevel.DEBUG:
      self._logger.debug("The stored files are:")
      for path in self._paths:
        self._logger.debug("%s", path)
    self._logger.info("A total of %d files were found.", self._nFiles )

  def __call__(self, refBenchmarkList, **kw):
    """
    Hook for loop method.
    """
    self.loop( refBenchmarkList, **kw )

  def __addPerformance( self, path, ref, neuron, sort, init, tunedDiscrList ):
    refName = ref.name
    # We need to make sure that the key will be available on the dict if it
    # wasn't yet there
    if not refName in self._tunedDiscrInfo:
      self._tunedDiscrInfo[refName] = { 'benchmark' : ref }
    if not neuron in self._tunedDiscrInfo[refName]:
      self._tunedDiscrInfo[refName][neuron] = dict()
    if not sort in self._tunedDiscrInfo[refName][neuron]:
      self._tunedDiscrInfo[refName][neuron][sort] = { 'headerInfo' : [], 
                                                      'initPerfTstInfo' : [], 
                                                      'initPerfOpInfo' : []}
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
    if self.level <= LoggingLevel.VERBOSE:
      self._logger.verbose("Retrieved file '%s' configuration for benchmark '%s' as follows:", 
                         os.path.basename(path),
                         ref )
      pprint({'headerInfo' : headerInfo, 'initPerfTstInfo' : iInfoTst, 'initPerfOpInfo' : iInfoOp })
    # Append information to our dictionary:
    # FIXME headerInfo shouldn't be connected to refName.
    self._tunedDiscrInfo[refName][neuron][sort]['headerInfo'].append( headerInfo )
    self._tunedDiscrInfo[refName][neuron][sort]['initPerfTstInfo'].append( iInfoTst )
    self._tunedDiscrInfo[refName][neuron][sort]['initPerfOpInfo'].append( iInfoOp )

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

    self._tunedDiscrInfo = dict()

    cFile = 1
    # Loop over the files
    for path in self._paths:
      self._logger.info("Reading file %d/%d", cFile, self._nFiles )
      # And open them as Tuned Discriminators:
      try:
        with TunedDiscrArchieve(path) as TDArchieve:
          # Now we loop over each configuration:
          for neuron in TDArchieve.neuronBounds():
            for sort in TDArchieve.sortBounds():
              for init in TDArchieve.initBounds():
                tunedDiscr, tunedPPChain = TDArchieve.getTunedInfo( neuron, sort, init )
                for refBenchmark in refBenchmarkList:
                  # FIXME, this shouldn't be like that, instead the reference
                  # benchmark should be passed to the TunningJob so that it could
                  # set the best operation point itself.
                  # When this is done, we can then remove the working points list
                  # as it is done here:
                  if type(tunedDiscr) is list:
                    # fastnet core version
                    discr = tunedDiscr[refBenchmark.reference]
                  else:
                    # exmachina core version
                    discr = tunedDiscr

                  self.__addPPChain( tunedPPChain, sort )                    
                  self.__addPerformance( path,
                                         refBenchmark, 
                                         neuron,
                                         sort,
                                         init,
                                         discr ) 
                # end of references
              # end of initializations
            # end of sorts
          # end of neurons
        # with file
      except UnpicklingError, e:
        self._logger.warning("Ignoring file '%s'. Reason:\n%s", str(e))
      if debug and cFile == 10:
        break
      cFile += 1
      gc.collect()
    # finished all files

    # Print information retrieved:
    if self.level <= LoggingLevel.DEBUG:
      for refBenchmark in refBenchmarkList:
        refName = refBenchmark.name
        self._logger.debug("Retrieved %d discriminator configurations for benchmark '%s':", 
            len(self._tunedDiscrInfo[refName]) - 1, 
            refBenchmark)
        for nKey, nDict in self._tunedDiscrInfo[refName].iteritems():
          if nKey == 'benchmark': continue
          self._logger.debug("Retrieved %d sorts for configuration '%r'", len(nDict), nKey)
          for sKey, sDict in nDict.iteritems():
            self._logger.debug("Retrieved %d inits for sort '%d'", len(sDict['headerInfo']), sKey)
          # got number of inits
        # got number of sorts
      # got number of configurations
    # finished all references

    # Recreate summary info object
    self._summaryInfo = dict()
    for refKey, refValue in self._tunedDiscrInfo.iteritems(): # Loop over operations
      refBenchmark = refValue['benchmark']
      # Create a new dictionary and append bind it to summary info
      refDict = { 'rawBenchmark' : refBenchmark.rawInfo() }
      self._summaryInfo[refKey] = refDict
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
        refDict['infoTstBest'], refDict['infoTstWorst']) = self.__outermostPerf(allBestTstConfInfo,
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
    for refKey, refValue in self._summaryInfo.iteritems(): # Loop over operations
      self._logger.info("This is the summary information for benchmark %s", refKey )
      pprint({key : val for key, val in refValue.iteritems() if type(key) is str }, depth=3)

    #append pp collections
    self._summaryInfo['infoPPChain']=self._summaryPPInfo

    # Save files
    save( self._summaryInfo, outputName )
    # Save matlab file
    if toMatlab:
      try:
        import scipy.io
        scipy.io.savemat(outputName + '.mat', self._summaryInfo)
      except ImportError:
        raise RuntimeError(("Cannot save matlab file, it seems that scipy is not "
            "available."))
  # end of loop

  def __outermostPerf(self, headerInfoList, perfInfoList, refBenchmark, collectionType, val, **kw):

    self._logger.debug("Retrieving outermost performance for %s %r", collectionType, val )

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
    if self.level <= LoggingLevel.DEBUG:
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
    if self.level <= LoggingLevel.VERBOSE:
      self._logger.verbose("The best configuration retrieved is: ")
      pprint(bestInfoDict)
      self._logger.verbose("The worst configuration retrieved is: ")
      pprint(worstInfoDict)

    return (summaryDict, bestInfoDict, worstInfoDict)
  # end of __outermostPerf


  def __addPPChain(self, tunedPPChain, sort):
    

    if not self._summaryPPInfo.has_key('sort_'+str(sort)):
      ppData=dict(); ppID=0
      for ppObj in tunedPPChain:
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
        ppID+=1

      #add into sort list    
      self._summaryPPInfo['sort_'+str(sort)]=ppData

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
      self._logger.info(("This CrossValidStat is still empty, it will loop over "
        "file lists to retrieve CrossValidation Statistics."))
      self.loop( refBenchmarkList )
    CrossValidStat.exportDiscrFiles( refBenchmarkList, 
                                     self._summaryInfo, 
                                     ringerOperation, 
                                     **kw )

  @classmethod
  def exportDiscrFiles(cls, summaryInfo, ringerOperation, **kw):
    """
    Export discriminators operating at reference benchmark list to the
    ATLAS environment using summaryInfo. 
    
    If benchmark name on the reference list is not available at summaryInfo, an
    KeyError exception will be raised.
    """
    baseName             = kw.pop( 'baseName',                           'tunedDiscr'         )
    refBenchmarkNameList = kw.pop( 'refBenchmarkNameList',             summaryInfo.keys()     )
    configList           = kw.pop( 'configList',                               []             )
    level                = kw.pop( 'level',                             LoggingLevel.INFO     )

    # Initialize local logger
    logger               = Logger.getModuleLogger("exportDiscrFiles", logDefaultLevel = level )
    checkForUnusedVars( kw, logger.warning )
    import pickle

    # Treat the reference benchmark list
    if not isinstance( refBenchmarkNameList, list):
      refBenchmarkNameList = [ refBenchmarkNameList ]

    nRefs = len(refBenchmarkNameList)

    # Make sure that the lists are the same size as the reference benchmark:
    while not len(configList) == nRefs:
      configList.append( None )

    # Retrieve the operation:
    from TuningTools.FilterEvents import RingerOperation
    if type(ringerOperation) is str:
      ringerOperation = RingerOperation.fromstring(ringerOperation)
    logger.info(('Starting export discrimination info files for the following '
                'operating points (RingerOperation:%s): %r'), 
                RingerOperation.tostring(ringerOperation), 
                refBenchmarkNameList )

    # Import special needed namespaces and modules for each operation:
    if ringerOperation is RingerOperation.Offline:
      try:
        import cppyy
      except ImportError:
        import PyCintex as cppyy
      try:
        cppyy.loadDict('RingerSelectorTools_Reflex')
      except RuntimeError:
        raise RuntimeError("Couldn't load RingerSelectorTools_Reflex dictionary.")
      # Import 
      from ROOT import TFile
      from ROOT import std
      from ROOT.std import vector
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
    # if ringerOperation

    for idx, refBenchmarkName in enumerate(refBenchmarkNameList):
      info = summaryInfo[refBenchmarkName]['infoOpBest'] if configList[idx] is None else \
             summaryInfo[refBenchmarkName]['config_' + str(configList[idx])]['infoOpBest']
      logger.info("%s discriminator information is available at file: \n\t%s", 
                  refBenchmarkName,
                  info['filepath'])
      with TunedDiscrArchieve(info['filepath'], level = level ) as TDArchieve:
        ## Check if user specified parameters for exporting discriminator
        ## operation information:
        config = configList[idx] if not configList[idx] is None else info['neuron']
        sort = info['sort']
        init = info['init']
        ## Write the discrimination wrapper
        if ringerOperation is RingerOperation.Offline:
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
          config=dict()
          config['rawBenchmark']=summaryInfo[refBenchmarkName]['rawBenchmark']
          config['infoOpBest']=info
          discr = TDArchieve.getTunedInfo(info['neuron'],
                                          info['sort'],
                                          info['init'])[0]['network']

          logger.info('neuron = %d, sort = %d, init = %d, thr = %f',
                      info['neuron'],
                      info['sort'],
                      info['init'],
                      info['cut'])

          config['tunedDiscr']=dict()
          config['tunedDiscr']['nodes']     = discr['nodes']
          config['tunedDiscr']['weights']   = discr['weights']
          config['tunedDiscr']['bias']      = discr['bias']
          config['tunedDiscr']['threshold'] = info['cut']
          return config
        else:
          raise RuntimeError('You must choose a ringerOperation')
 

      # with
    # for benchmark
  # exportDiscrFiles 

  @classmethod
  def printTables(cls, confBaseNameList,
                       crossValGrid,
                       configMap):
    "Print operation tables for the "
    # TODO Improve documentation

    # We first loop over the configuration base names:
    for confIdx, confBaseName in enumerate(confBaseNameList):
      print "===================================== ", confBaseName, " ====================================="
      # And then on et/eta bins:
      for etIdx, crossList in enumerate(crossValGrid):
        print "---------------------------    Starting new Et (%d)  -------------------------------" % etIdx 
        for etaIdx, crossFile in enumerate(crossList):
          print "------------- Eta %d | Et %d -----------------" % (etaIdx, etIdx)
          print "----------------------------------------------"
          # Load file and then search the benchmark references with the configuration name:
          summaryInfo = load(crossFile)
          #from scipy.io import loadmat
          #summaryInfo = loadmat(crossFile)
          confPdKey = confSPKey = confPfKey = None
          for key in summaryInfo.keys():
            rawBenchmark = summaryInfo[key]['rawBenchmark']
            reference = rawBenchmark['reference']
            # Retrieve the configuration keys:
            if confBaseName in key:
              if reference == 'Pd':
                confPdKey = key 
                reference_pd = rawBenchmark['refVal']
              if reference == 'Pf':
                confPfKey = key 
                reference_pf = rawBenchmark['refVal']
              if reference == 'SP':
                confSPKey = key 
          reference_sp = calcSP(reference_pd,(1.-reference_pf))
          # Loop over each one of the cases and print ringer performance:
          for keyIdx, key in enumerate([confPdKey, confSPKey, confPfKey]):
            if not key:
              print '--Information Unavailable--'
              continue
            ringerPerf = summaryInfo[key] \
                                    ['config_' + str(configMap[confIdx][etIdx][etaIdx][keyIdx])] \
                                    ['summaryInfoTst']
            print '%.3f+-%.3f  %.3f+-%.3f %.3f+-%.3f' % ( ringerPerf['detMean'] * 100., ringerPerf['detStd']  * 100.,
                                                          ringerPerf['spMean']  * 100.,  ringerPerf['spStd']  * 100.,
                                                          ringerPerf['faMean']  * 100.,  ringerPerf['faStd'] * 100.,
                                                        )

          print "----------------------------------------------"
          print '%.3f  %.3f %.3f' % (reference_pd*100.
                                    ,reference_sp*100.
                                    ,reference_pf*100.)
      print "=============================================================================================="


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

