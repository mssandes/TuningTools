
__all__ = ['PlotObjects', 'Summary', 'Performance']

from RingerCore import Logger

class PlotObjects( Logger ):
  """
  Class to hold all objects from monitoring file
  """
  from ROOT import TEnv, TGraph, TCanvas, TParameter, gROOT, kTRUE
  gROOT.SetBatch(kTRUE)

  #Helper names 
  _paramNames = [ 'mse_stop', 
                  'sp_stop', 
                  'det_stop', 
                  'fa_stop' 
                  ]
  _graphNames = [ 'mse_trn', 
                  'mse_val', 
                  'mse_tst',
                  'bestsp_point_sp_val', 
                  'bestsp_point_det_val', 
                  'bestsp_point_fa_val',
                  'bestsp_point_sp_tst', 
                  'bestsp_point_det_tst', 
                  'bestsp_point_fa_tst',
                  'det_point_sp_val', 
                  'det_point_det_val', 
                  'det_point_fa_val', # det_point_det_val is det_fitted
                  'det_point_sp_tst', 
                  'det_point_det_tst', 
                  'det_point_fa_tst', 
                  'fa_point_sp_val', 
                  'fa_point_det_val', 
                  'fa_point_fa_val', # fa_point_fa_val is fa_fitted
                  'fa_point_sp_tst', 
                  'fa_point_det_tst', 
                  'fa_point_fa_tst',  
                  'roc_tst', 
                  'roc_operation'
                  ]

  def __init__(self, name ):

    Logger.__init__( self )  
    self._obj = list()
    self._boundValues = None
    self._bestIdx = 0
    self._worstIdx = 0
    self._name = name
  

  def retrieve(self, rawObj, pathList):
    #Create dictonarys with diff memory locations
    self._obj = [dict() for _ in range(len(pathList))]
    #Loop to retrieve objects from root rawObj
    for idx, path in enumerate(pathList):
      for graphName in self._graphNames:
        #Check if key exist into plot holder dict
        self.__retrieve_graph( rawObj, idx, path, graphName )
      #Loop over graphs
      for paramName in self._paramNames:
        self.__retrieve_param(rawObj, idx, path, paramName )
    #Loop over file list


  def __retrieve_graph(self, rawObj, idx, path, graphName ):
    from ROOT import TGraph, gROOT, kTRUE
    gROOT.SetBatch(kTRUE)
    obj = TGraph()
    rawObj.GetObject( path+'/'+graphName, obj)
    self._obj[idx][graphName] = obj 
    
  #Private method:
  def __retrieve_param(self, rawObj, idx, path, paramName ):
    from ROOT import TParameter
    obj = TParameter("double")()
    rawObj.GetObject( path+'/'+paramName, obj)
    self._obj[idx][paramName] = int(obj.GetVal()) 

  @property
  def best(self):
    return self._bestIdx

  @best.setter
  def best(self, v):
    self._bestIdx = v

  @property
  def worst(self):
    return self._worstIdx

  @worst.setter
  def worst(self, v):
    self._worstIdx = v


  def setBoundValues(self,vec):
    self._boundValues = vec
    if len(vec) != len(self._obj):
      self._logger.warning('The correction vector (Bound Values) and the size of the object its not correct.')

  def getBoundValues(self):
    return self._boundValues

  def getObject(self, idx):
    return self._obj[self._boundValues.index(idx)] if self._boundValues else self._obj[idx]


  def getBestObject(self):
    from copy import deepcopy
    obj = self.getObject(self._bestIdx)
    obj['best'+self._name] = self._bestIdx
    obj['worst'+self._name] = self._worstIdx
    return deepcopy(obj)

  def getWorstObject(self):
    from copy import deepcopy
    obj = self.getObject(self._worstIdx)
    obj['best'+self._name] = self._bestIdx
    obj['worst'+self._name] = self._worstIdx
    return deepcopy(obj)


  def getRawObj(self):
    return self._obj

  def setRawObj(self, obj):
    self._obj = obj

  def append(self, obj):
    self._obj.append(obj)

  def size(self):
    return len(self._obj)

  def clear(self):
    self._obj = []

  def __getitem__(self, idx):
    return self.getObject(idx)

  def keys(self):
    return self._graphNames+self._paramNames

  def tolist(self, name):
    return [obj[name] for obj in self._obj]

  def __del__(self):
    for obj in self._obj:
      for xname in obj.keys():
        del obj[xname]


################################################################################


class Summary(Logger):
  
  def __init__(self, benchmarkName, rawObj):
    Logger.__init__(self)
    self._summary = rawObj
    self._benchmarkName = benchmarkName
    self._initBounds=None

  def neuronBounds(self):
    neuronBounds = [int(neuron.replace('config_','')) for neuron in self._summary.keys() if 'config_' in neuron]
    neuronBounds.sort()
    return neuronBounds

  def sortBounds(self, neuron):
    sortBounds = [int(sort.replace('sort_','')) for sort in self._summary['config_'+str(neuron).zfill(3)].keys() \
                  if 'sort_' in sort]
    sortBounds.sort()
    return sortBounds

  def setInitBounds( self, v ):
    self._initBounds=v

  def initBounds(self, neuron, sort):

    if self._initBounds:
      return self._initBounds
    else:
      # helper function to retrieve all inits bounds
      def GetInits(sDict):
        initBounds = list(set([sDict['infoOpBest']['init'], sDict['infoOpWorst']['init'], \
                       sDict['infoTstBest']['init'], sDict['infoTstWorst']['init']]))
        initBounds.sort()
        return initBounds
      # return the init bounds
      return GetInits(self._summary['config_'+str(neuron).zfill(3)]['sort_'+str(sort).zfill(3)])

  def name(self):
    return self._benchmarkName

  def reference(self):
    return self._summary['rawBenchmark']['reference']

  def rawBenchmark(self):
    return self._summary['rawBenchmark']

  def eps(self):
    try:
      return self._summary['eps']
    except KeyError:
      return self._summary['rawBenchmark']['eps']

  def etBinIdx(self):
    try:
      return self.rawBenchmark()['signalEfficiency']['etBin']
    except:
      return self.rawBenchmark()['signal_efficiency']['etBin']

  def etaBinIdx(self):
    try: 
      return self.rawBenchmark()['signalEfficiency']['etaBin']
    except:
      return self.rawBenchmark()['signal_efficiency']['etaBin']

  def etBin(self):
    try:
      return self._summary['etBin']
    except KeyError:
      return []

  def etaBin(self):
    try:
      return self._summary['etaBin']
    except KeyError:
      return []

  def summary(self):
    return self._summary




class Performance(object):

  def __init__(self, tuning, operation, benchmark):
    
    self._reference = benchmark['reference']
    try:
      self._values = {
          'det_target'   : benchmark['signalEfficiency']['efficiency'],
          'fa_target' : benchmark['backgroundEfficiency']['efficiency'],
          }
    except:
      self._values = {
          'det_target'   : benchmark['signal_efficiency']['efficiency'],
          'fa_target' : benchmark['background_efficiency']['efficiency'],
          }

    wanted_keys = ['detMean','faMean','spMean','detStd', 'faStd', 'spStd']
    for key in wanted_keys:
      self._values[key] = tuning[key] * 100 # put in percentage

    self._values['det'] = operation['det']*100
    self._values['fa'] = operation['fa']*100
    self._values['sp'] = operation['sp']*100


  def getValue( self, key ):
    return self._values[key]

  def reference(self):
    return self._reference





##Helper class to hold the operation information
#class MonitoringOperationInfo( object ):
#  def __init__(self, rawObj):
#    self._rawOp = rawObj
#
#  def rawOp(self):
#    return self._rawOp
#
#  def getDiscr(self):
#    return {'threshold': self._rawOp['cut'],
#            'nodes'    : self._rawOp['discriminator']['nodes'],
#            'bias'     : self._rawOp['discriminator']['bias'],
#            'weights'  : self._rawOp['discriminator']['weights']}
#
#  def print_operation(self):
#    return ('Operation: (det = %.2f, sp = %.2f, fa = %.2f)') %\
#            (self._rawOp['det']*100,self._rawOp['sp']*100,self._rawOp['fa']*100)
#
#



