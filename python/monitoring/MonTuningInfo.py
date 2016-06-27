#Author:Joao Victor da Fonseca Pinto
#Helper class to hold the information related to
#the configuration loop over monitoring file.

from RingerCore import calcSP, Logger

#Info class to the iterator
class MonIterator(Logger):
  def __init__(self, obj, **kw):
    from pprint import pprint
    Logger.__init__(self, kw)
    self._iterator = []
    #Extract bound index for config and sort names
    for nKey, nValue in obj.iteritems():
      if nKey.startswith('config_'):
        for sKey, sValue in nValue.iteritems():
          if sKey.startswith('sort_'):
            config = int(nKey.replace('config_',''))
            sort = int(sKey.replace('sort_',''))
            self._iterator.append( (config, sort, self.__getInits(sValue)) )

  # Private Helper function to extract four inits from the dictinary
  def __getInits(self,sDict):
    initBounds = list(set([sDict['infoOpBest']['init'], sDict['infoOpWorst']['init'], \
                     sDict['infoTstBest']['init'], sDict['infoTstWorst']['init']]))
    initBounds.sort()
    return initBounds

  #Retrieve iterator
  def iterator(self):
    return self._iterator
  
  #Return config names into a list
  def neuronBounds(self):
    neuronBounds = [int(neuron.replace('config_','')) for neuron in self._summary.keys() if 'config_' in neuron]
    neuronBounds.sort()
    return neuronBounds

  #Return sort names into a list
  def sortBounds(self, neuron):
    sortBounds = [int(sort.replace('sort_','')) for sort in self._summary['config_'+str(neuron)].keys() \
                  if 'sort_' in sort]
    sortBounds.sort()
    return sortBounds

  def initBounds(self, neuron, sort):
    return self.__getInits(self._summary['config_'+str(neuron)]['sort_'+str(sort)])


#Monitoring class
class MonTuningInfo(MonIterator):
  def __init__(self, benchmarkName, rawObj, **kw):
    #Iterator object
    MonIterator.__init__(self,rawObj)
    self._summary = rawObj
    self._benchmarkName = benchmarkName
            
  def name(self):
    return self._benchmarkName

  def reference(self):
    return self._summary['rawBenchmark']['reference']

  def rawBenchmark(self):
    return self._summary['rawBenchmark']

  def etbin(self):
    return self.rawBenchmark()['signal_efficiency']['etBin']

  def etabin(self):
    return self.rawBenchmark()['signal_efficiency']['etaBin']

  def summary(self):
    return self._summary




#Helper class to hold the operation information
class MonOperationInfo( object ):
  def __init__(self, rawObj):
    self._rawOp = rawObj

  def rawOp(self):
    return self._rawOp

  def getDiscr(self):
    return {'threshold': self._rawOp['cut'],
            'nodes'    : self._rawOp['discriminator']['nodes'],
            'bias'     : self._rawOp['discriminator']['bias'],
            'weights'  : self._rawOp['discriminator']['weights']}

  def print_operation(self):
    return ('Operation: (det = %.2f, sp = %.2f, fa = %.2f)') %\
            (self._rawOp['det']*100,self._rawOp['sp']*100,self._rawOp['fa']*100)


#Helper class to hold all performance values
class MonPerfInfo(MonOperationInfo):
  
  _keys = ['detMean','faMean','spMean',
           'detStd', 'faStd', 'spStd']
  #tobj: tuning object
  #bobj: benchmark object
  #infoOp: infoOpBest or infoOpWorst
  def __init__(self, name, ref, tobj, opObj, bobj):
    MonOperationInfo.__init__(self, opObj)
    #name
    self._name   = name
    self._perf = dict()
    self._ref = dict()

    #Retrieve information from benchmark
    self._ref['reference'] = bobj['reference']
    self._ref['det'] = bobj['signal_efficiency']['efficiency']
    self._ref['fa'] = bobj['background_efficiency']['efficiency']
    #Calculate SP from reference
    self._ref['sp'] = calcSP(self._ref['det'], 100-self._ref['fa'])
    #Hold values
    for key in self._keys:  self._perf[key] = tobj[key]*100

  def name(self):
    return self._name

  def getRef(self):
    return self._ref

  def getPerf(self):
    return self._perf

  #Display information
  def __str__(self):
    #Get operation string information
    opString = self.print_operation()

    refString = ('Reference: %s (det = %.2f, sp = %.2f fa = %.2f') % (self._ref['reference'],\
                  self._ref['det'],self._ref['sp'],self._ref['fa'])

    tuningString = ('Tuning  (det = %.2f+/-%.2f, sp = %.2f+/-%.2f, fa = %.2f+/-%.2f') % \
                   (self._perf['detMean'], self._perf['detStd'], self._perf['spMean'], \
                    self._perf['spStd'], self._perf['faMean'], self._perf['faStd'])

    return opString+'\n'+refString+'\n'+tuningString





