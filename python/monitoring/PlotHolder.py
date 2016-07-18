from pprint import pprint
from RingerCore import Logger

class PlotHolder( Logger ):
  """
  Class to hold all objects from monitoring file
  """
  from ROOT import TEnv, TGraph, TCanvas, TParameter

  #Helper names 
  _paramNames = [ 'mse_stop', 'sp_stop', 'det_stop', 'fa_stop' ]
  _graphNames = [ 'mse_trn' , 'mse_val', 'mse_tst' , 'sp_val',
                  'sp_tst'  , 'det_val', 'det_tst' , 'fa_val',
                  'fa_tst'  , 'det_fitted', 'fa_fitted', 'roc_tst', 
                  'roc_op'  , ] 

  def __init__(self, **kw):
    #Retrive python logger  
    logger = kw.pop('logger', None)
    self._label = kw.pop('label','')
    Logger.__init__( self, logger = logger)  
    self._obj = []
    self._idxCorr = None
    self.best = 0
    self.worst = 0
     

  def retrieve(self, rawObj, pathList):
    #Create dictonarys with diff memory locations
    self._obj = [dict() for i in range(len(pathList))]

    #Loop to retrieve objects from root rawObj
    for idx, path in enumerate(pathList):
      for graphName in self._graphNames:
        #Check if key exist into plot holder dict
        self.__retrieve_graph( rawObj, idx, path, graphName )
      #Loop over graphs
      for paramName in self._paramNames:
        self.__retrieve_param(rawObj, idx, path, paramName )
    #Loop over file list
    #pprint(self._obj)

  #Private method:
  def __retrieve_graph(self, rawObj, idx, path, graphName ):
    from ROOT import TGraph
    obj = TGraph()
    rawObj.GetObject( path+'/'+graphName, obj)
    self._obj[idx][graphName] = obj 
    
  #Private method:
  def __retrieve_param(self, rawObj, idx, path, paramName ):
    from ROOT import TParameter
    obj = TParameter("double")()
    rawObj.GetObject( path+'/'+paramName, obj)
    self._obj[idx][paramName] = int(obj.GetVal()) 
 
  #Public method:
  
  def set_index_correction(self,vec):
    self._idxCorr = vec
    if len(vec) != len(self._obj):
      self._logger.warning('The correction vector and the size of the object its not correct.')

  def get_index_correction(self):
    return self._idxCorr

  def index_correction(self, idx):
    return self._idxCorr.index(idx)

  def get_best(self):
    obj = self.getObj(self.best)
    obj['best'+self._label] = self.best
    obj['worst'+self._label] = self.worst
    return obj

  def get_worst(self):
    obj = self.getObj(self.worst)
    obj['best'+self._label] = self.best
    obj['worst'+self._label] = self.worst
    return obj

  def set_best_index(self,idx):
    self.best = idx

  def set_worst_index(self,idx):
    self.worst = idx

  #Return the object store into obj dict. Can be a list of objects
  #Or a single object
  def getObj(self, idx):
    if self._idxCorr:
      return self._obj[self._idxCorr.index(idx)]
    else:
      return self._obj[idx]

  def info(self):
    for idx, obj in enumerate(self._obj):
      self._logger.verbose('information from index %d ...')%(idx))
      for name in self._graphNames:
        self._logger.verbose(' The key name [%s] has %d points into you graph class') % (name, obj[name].GetN())

  def rawObj(self):
    return self._obj

  def setRawObj(self, obj):
    self._obj = obj

  def append(self, obj):
    self._obj.append(obj)

  def size(self):
    return len(self._obj)

  def clear(self):
    #for obj in self._obj:
    #  del obj
    self._obj = []

  def __getitem__(self, idx):
    return self.getObj(idx)

  def keys(self):
    return self._graphNames+self._paramNames

  def tolist(self, name):
    return [obj[name] for obj in self._obj]






