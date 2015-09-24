from RingerCore.Logger import Logger, LoggingLevel
from RingerCore.util   import EnumStringification, get_attributes
from RingerCore.util   import checkForUnusedVars, calcSP
from RingerCore.FileIO import save, load
from TuningTools.TuningJob import TunedDiscrArchieve
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


class Dataset( EnumStringification ):
  """
  The possible datasets to use
  """
  Train = 1
  Validation = 2
  Test = 3
  Operation = 4

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
    """
    self.refVal = kw.pop('refVal', None)
    self.removeOLs = kw.pop('removeOLs', False)
    if not (type(name) is str):
      raise TypeError("Name must be a string.")
    self.name = name
    if type(reference) is str:
      self.reference = ReferenceBenchmark.fromstring(reference)
    else:
      allowedValues = get_attributes(ReferenceBenchmark)
      if reference in [attr[1] for attr in allowedValues]:
        self.reference = reference
      else:
        raise ValueError(("Attempted to create a reference benchmark "
            "with a enumeration value which is not allowed. Use one of the followings: "
            "%r") % allowedValues)
    if reference == ReferenceBenchmark.Pf:
      self.refVal = - self.refVal
  # __init__

  def rawInfo(self):
    """
    Return raw benchmark information
    """
    return { 'reference' : ReferenceBenchmark.tostring(self.reference),
             'refVal' : self.refVal,
             'removeOLs' : self.removeOLs }

  def getOutermostPerf(self, data, **kw):
    """
    Get outermost performance for the tuned discriminator performances on data. 
    idx = refBMark.getOutermostPerf( data [, eps = .1 ][, cmpType = 1])

     * data: A list with following struction:
        data[0] : SP
        data[1] : Pd
        data[2] : Pf

     * eps [.1] is used softening. The larger it is, more candidates will be
    possible to be considered, and the best returned value will be 
     * cmpType [+1.] is used to change the comparison type. Use +1.
    for best performance, and -1 for worst performance.
    """
    # Retrieve optional arguments
    eps = kw.pop('eps', .1)
    cmpType = kw.pop('cmpType', 1.)
    # Retrieve reference and benchmark arrays
    if self.reference is ReferenceBenchmark.Pf:
      refVec = -cmpType * data[2]
      benchmark = cmpType * data[1]
    elif self.reference is ReferenceBenchmark.Pd:
      refVec = cmpType * data[1] 
      benchmark = - cmpType * data[2]
    elif self.reference is ReferenceBenchmark.SP:
      benchmark = cmpType * data[0]
    else:
      raise ValueError("Unknown reference %d" % self.reference)
    # Retrieve the allowed indexes from benchmark which are not outliers
    if self.removeOLs:
      q1=percentile(benchmark,25.0)
      q3=percentile(benchmark,75.0)
      outlier_higher = q3 + 1.5*(q3-q1)
      outlier_lower  = q1 + 1.5*(q1-q3)
      allowedIdxs = np.all([benchmark > q3, benchmark < q1]).nonzero()[0]
    # Finally, return the index:
    if self.reference is ReferenceBenchmark.SP: 
      if self.removeOLs:
        idx = np.argmax( cmpType * benchmark[allowedIdxs] )
        return allowedIdx[ idx ]
      else:
        return np.argmax( cmpType * benchmark )
    else:
      if self.removeOLs:
        refAllowedIdxs = ( np.power( refVec[allowedIdxs] - ( cmpType * self.refVal[allowedIdxs] ), 2 ) > ( eps ** 2 ) ).nonzero()[0]
        idx = refAllowedIdxs[ np.argmax( ( benchmark[allowedIdxs] )[ refAllowedIdxs ] ) ]
      else:
        refAllowedIdxs = ( np.power( refVec - ( cmpType * self.refVal ), 2 ) > ( eps ** 2 ) ).nonzero()[0]
        idx = refAllowedIdxs[ np.argmax( benchmark[ refAllowedIdxs ] ) ]

  def __str__(self):
    str_ =  self.name + '(' + ReferenceBenchmark.tostring(self.reference) 
    if self.refVal: str_ += ':' + str(self.refVal)
    str_ += ')'
    return str_

class CrossValidStatAnalysis( Logger ):

  _tunedDiscrInfo = dict()
  _summaryInfo = dict()

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
    # We need to make sure that if the key will be available in the dict if it
    # wasn't yet there
    refName = ref.name
    if not refName in self._tunedDiscrInfo:
      self._tunedDiscrInfo[refName] = { 'benchmark' : ref }
    if not neuron in self._tunedDiscrInfo[refName]:
      self._tunedDiscrInfo[refName][neuron] = dict()
    if not sort in self._tunedDiscrInfo[refName][neuron]:
      self._tunedDiscrInfo[refName][neuron][sort] = { 'initPerfInfo' : [] }
    # The performance holder, which also contains the discriminator
    perfHolder = PerfHolder( tunedDiscrList )
    (spTst, detTst, faTst, cutTst, idxTst) = perfHolder.getOperatingBenchmarks(ref)
    (spOp, detOp, faOp, cutOp, idxOp) = perfHolder.getOperatingBenchmarks(ref, ds = Dataset.Operation)
    iInfo = { 'filepath' : path,
              'neuron' : neuron, 'sort' : sort, 'init' : init,
              'perfHolder' : perfHolder, 
              'cutTst' : cutTst, 'spTst' : spTst, 'detTst' : detTst, 'faTst' : faTst, 'idxTst' : idxTst,
              'cutOp' : cutOp, 'spOp' : spOp, 'detOp' : detOp, 'faOp' : faOp, 'idxOp' : idxOp }
    perfHolder = iInfo['perfHolder'] = PerfHolder( tunedDiscrList )
    if self.level <= LoggingLevel.DEBUG:
      import pprint
      self._logger.debug("Retrieved file '%s' configuration for benchmark '%s' as follows:", 
                         os.path.basename(path),
                         ref )
      pprint.pprint(iInfo)
    self._tunedDiscrInfo[refName][neuron][sort]['initPerfInfo'].append( iInfo )

  def loop(self, refBenchmarkList, **kw ):
    """
    Needed args:
      * refBenchmarkList: a list of reference benchmark objects which will be used
        as the operation points.
    Optional args:
      * toMatlab [True]: also create a matlab file from the obtained tuned discriminators
      * outputName ['crossValStat']: the output file name.
    """
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
      with TunedDiscrArchieve(path) as TDArchieve:
        # Now we loop over each configuration:
        for neuron in TDArchieve.neuronBounds():
          for sort in TDArchieve.sortBounds():
            for init in TDArchieve.initBounds():
              tunedDiscr = TDArchieve.getTunedInfo( neuron, sort, init )
              for refBenchmark in refBenchmarkList:
                # FIXME, this shouldn't be like that, instead the reference
                # benchmark should be passed to the TunningJob so that it could
                # set the best operation point itself.
                # On that way, we can then remove a various working points list
                # as it is done here:
                self.__addPerformance( path,
                                       refBenchmark, 
                                       neuron,
                                       sort,
                                       init,
                                       tunedDiscr[refBenchmark.reference] ) 
              # end of references
            # end of initializations
          # end of sorts
        # end of neurons
      # with file
      if debug and cFile == 10:
        break
      cFile += 1
    # finished all files

    # Print total information retrieved:
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
            self._logger.debug("Retrieved %d inits for sort '%d'", len(sDict['initPerfInfo']), sKey)
          # got number of inits
        # got number of sorts
      # got number of configurations
    # finished all references

    # Recreate summary info object
    self.summaryInfo = dict()
    for refKey, refValue in self._tunedDiscrInfo.iteritems(): # Loop over operations
      refBenchmark = refValue['benchmark']
      # Create a new dictionary and append bind it to summary info
      refDict = { 'rawBenchmark' : refBenchmark.rawInfo() }
      self.summaryInfo[refKey] = refDict
      for nKey, nValue in refValue.iteritems(): # Loop over neurons
        nDict = dict()
        refDict[nKey] = nDict
        for sKey, sValue in nValue.iteritems(): # Loop over sorts
          sDict = dict()
          nDict[sKey] = sDict
          # Retrieve information from outermost initializations:
          allInitPerfsInfo = [ initPerfInfo for initPerfInfo in sValue['initPerfInfo'] ]
          ( sDict['summaryInfo'], \
            sDict['infoTstBest'], sDict['infoTstWorst'], \
            sDict['infoOpBest'], sDict['infoOpWorst']) = self.__outermostPerf( allInitPerfsInfo, refBenchmark )
        # Retrieve information from outermost sorts:
        allBestSortInfo = [ sDict['infoTstBest'] for key, sDict in nValue.iteritems() ]
        ( nValue['summaryInfo'], \
          nValue['infoTstBest'], nValue['infoTstWorst'], \
          nValue['infoOpBest'], nValue['infoOpWorst']) = self.__outermostPerf( allBestSortInfo, refBenchmark )
      # Retrieve information from outermost discriminator configurations:
      allBestConfInfo = [ nDict['infoBest'] for key, nDict in refValue.iteritems() ]
      ( refValue['summaryInfo'], \
        refValue['infoTstBest'], refValue['infoTstWorst'], \
        refValue['infoOpBest'], refValue['infoOpWorst']) = self.__outermostPerf( allBestConfInfo, refBenchmark )
    # Finished summary information

    # Save files
    save( outputName, self._summaryInfo )
    # Save matlab file
    if toMatlab:
      try:
        import scipy.io
        scipy.io.savemat(outputName + '.mat', 
                         mdict = {'TuningInfo' : self._summaryInfo})
      except ImportError:
        raise RuntimeError(("Cannot save matlab file, it seems that scipy is not "
            "available."))
  # end of loop

  def __outermostPerf(self, perfInfoList, refBenchmark, **kw):
    summaryDict = {'cut' : [],
                   'spTst': [], 'detTst' : [], 'faTst' : [], 'idxTst' : [],
                   'spOp' : [], 'detOp'  : [], 'faOp' : [] }
    # Fetch all information together in the dictionary:
    for key in summaryDict.keys():
      summaryDict[key] = [ perfInfo[key] for perfInfo in perfInfoList ]
      if not key == 'idxTst':
        summaryDict[key + 'Mean'] = np.mean(summaryDict[key],axis=0)
        summaryDict[key + 'Std']  = np.std(summaryDict[key],axis=0)

    # Put information together on data:
    tstBenchmarks = [summaryDict['spTst'], summaryDict['detTst'], summaryDict['faTst']]
    opBenchmarks  = [summaryDict['spOp'],  summaryDict['detOp'],  summaryDict['faOp'] ]

    # The outermost performances:
    bestTstIdx  = refBenchmark.getOutermostPerf(tstBenchmarks, refBenchmark)
    worstTstIdx = refBenchmark.getOutermostPerf(tstBenchmarks, refBenchmark, cmpType = -1. )
    bestOpIdx   = refBenchmark.getOutermostPerf(opBenchmarks,  refBenchmark)
    worstOpIdx  = refBenchmark.getOutermostPerf(opBenchmarks,  refBenchmark, cmpType = -1. )

    # Retrieve information from outermost performances:
    def __getInfo( perfInfoList, idx ):
      wantedKeys = ['filepath', 'neuron', 'sort', 'init', 
          'cut','spTst', 'detTst', 'faTst', 'idxTst', 
          'spOp', 'detOp', 'faOp']
      info = dict()
      for key in wantedKeys:
        info[key] = perfInfoList[key]

    bestTstInfoDict  = __getInfo( perfInfoList, bestTstIdx )
    worstTstInfoDict = __getInfo( perfInfoList, worstTstIdx )
    bestOpInfoDict   = __getInfo( perfInfoList, bestOpIdx )
    worstOpInfoDict  = __getInfo( perfInfoList, worstOpIdx )

    return (summaryDict, \
            bestTstInfoDict, worstTstInfoDict, \
            bestOpInfoDict,  worstOpInfoDict)
  # end of __outermostPerf

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
        

  def exportBestDiscriminator(self, refBenchmarkList, ringerOperation, **kw ):
    """
    Export best discriminators operating at reference benchmark list to the
    ATLAS environment using this CrossValidStat information.
    """
    if not self._summaryInfo:
      self._logger.info(("This CrossValidStat is still empty, it will loop over "
        "file lists to retrieve CrossValidation Statistics."))
      self.loop( refBenchmarkList )
    CrossValidStat.exportDiscriminator( refBenchmarkList, 
                                        self._summaryInfo, 
                                        ringerOperation, 
                                        **kw )

  @classmethod
  def exportBestDiscriminator(cls, refBenchmarkList, summaryInfo, ringerOperation, **kw):
    """
    Export best discriminators operating at reference benchmark list to the
    ATLAS environment using summaryInfo. 
    
    If benchmark name on the reference list is not available at summaryInfo, an
    KeyError exception will be raised.
    """
    outputName = kw.pop( 'outputName', 'tunedDiscr' )

    if not isinstance( refBenchmarkList, list):
      refBenchmarkList = [ refBenchmarkList ]

    from TuningTools.FilterEvents import RingerOperation
    if type(ringerOperation) is str:
      ringerOperation = RingerOperation.fromstring(ringerOperation)

    for refBenchmark in refBenchmarkList:
      info = summaryInfo[refBenchmark.name]['infoOpBest']
      with TunedDiscrArchieve(info['filepath']) as TDArchieve:
        pass
        #TDArchieve.export()
        #net = dict()
        #net['nodes']      = network.nNodes
        #net['threshold']  = threshold
        #net['bias']       = network.get_b_array()
        #net['weights']    = network.get_w_array()
        #pickle.dump(net,open(outputName,'wb'))

class PerfHolder:
  """
  Hold the performance values and evolution for a tunned discriminator
  """

  def __init__(self, tunedDiscrData ):
    trainEvo           = tunedDiscrData[0].dataTrain
    roc_tst            = tunedDiscrData[1]
    roc_operation      = tunedDiscrData[2]
    self.discriminator = tunedDiscrData[0]
    self.epoch         = np.array( range(len(trainEvo.epoch)), dtype ='float_')
    self.nEpoch        = len(self.epoch)
    self.mse_trn       = np.array( trainEvo.mse_trn,           dtype ='float_')
    self.mse_val       = np.array( trainEvo.mse_val,           dtype ='float_')
    self.sp_val        = np.array( trainEvo.sp_val,            dtype ='float_')
    self.det_val       = np.array( trainEvo.det_val,           dtype ='float_')
    self.fa_val        = np.array( trainEvo.fa_val,            dtype ='float_')
    self.mse_tst       = np.array( trainEvo.mse_tst,           dtype ='float_')
    self.sp_tst        = np.array( trainEvo.sp_tst,            dtype ='float_')
    self.det_tst       = np.array( trainEvo.det_tst,           dtype ='float_')
    self.fa_tst        = np.array( trainEvo.fa_tst,            dtype ='float_')
    self.roc_tst_det   = np.array( roc_tst.detVec,             dtype ='float_')
    self.roc_tst_fa    = np.array( roc_tst.faVec,              dtype ='float_')
    self.roc_tst_cut   = np.array( roc_tst.cutVec,             dtype ='float_')
    self.roc_op_det    = np.array( roc_operation.detVec,       dtype ='float_')
    self.roc_op_fa     = np.array( roc_operation.faVec,        dtype ='float_')
    self.roc_op_cut    = np.array( roc_operation.cutVec,       dtype ='float_')

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
    elif graphType == 'roc_val'     : return ROOT.TGraph(len(roc_val_fa), roc_val_fa, roc_val_det )
    elif graphType == 'roc_op'      : return ROOT.TGraph(len(roc_op_fa),  roc_op_fa,  roc_op_det  )
    elif graphType == 'roc_val_cut' : return ROOT.TGraph(len(roc_val_cut),np.array(range(len(roc_val_cut) ), 'float_'), roc_val_cut )
    elif graphType == 'roc_op_cut'  : return ROOT.TGraph(len(roc_op_cut), np.array(range(len(roc_op_cut) ),  'float_'), roc_op_cut  )
    else: raise ValueError( "Unknown graphType '%s'" % graphType )

