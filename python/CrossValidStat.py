#!/usr/bin/env python
from FastNetTool.util import sourceEnvFile, genRoc
import numpy as np
import logging
import ROOT
from FastNetTool.Logger import Logger

class EnumStringification:
  "Adds 'enum' static methods for conversion to/from string"
  @classmethod
  def tostring(cls, val):
    "Transforms val into string."
    for k,v in vars(cls).iteritems():
      if v==val:
        return k

  @classmethod
  def fromstring(cls, str):
    "Transforms string into enumeration."
    return getattr(cls, str, None)

class Criteria(EnumStringification):
  SP  = 0
  DET = 1
  FA  = 2
 
class Performance:

  def __init__(self, vecList):
    self._spVec  = vecList[0]
    self._detVec = vecList[1]
    self._faVec  = vecList[2]
    self._cutVec = vecList[3]
    self.sp  = self._spVec[self.maxsp()[1]]
    self.det = self._detVec[self.maxsp()[1]]
    self.fa  = self._faVec[self.maxsp()[1]]
    self.cutID = self.maxsp()[1]

  def __call__(self, pos = None):
    if pos: return (self._spVec[pos], self._detVec[pos], self._faVec[pos], self._cutVec[pos])
    else: return (self._spVec, self._detVec, self._faVec, self._cutVec)

  def adaptCut(self, newReference, crit ):
    cut_pos = 0
    if crit == Criteria.SP:  cut_pos = np.where( self._spVec  > newReference )[0]
    if crit == Criteria.DET: cut_pos = np.where( self._detVec > newReference )[0]
    if crit == Criteria.FA:  cut_pos = np.where( self._faVec  > newReference )[0]
    self.sp  = self._spVec[ cut_pos]
    self.det = self._detVec[cut_pos]
    self.fa  = self._faVec[ cut_pos]
    self.cutID = cut_pos

  def setCut(self, cut_pos):
    self.sp  = self._spVec[ cut_pos]
    self.det = self._detVec[cut_pos]
    self.fa  = self._faVec[ cut_pos]
    self.cutID = cut_pos


  def maxsp(self):
    pos = np.argmax(self._spVec)
    return (self._spVec[pos], pos)


class CrossValidStat( Logger ):

  def __init__(self, filesList, **kw):

    import pickle
    Logger.__init__(self, kw)
    self._filesList         = filesList
    self._nSorts            = kw.pop('nSorts',50)
    self._neurons           = treatRangeVec( kw.pop('neurons',[5,20]))
    self._cross             = kw.pop('Cross', None)
    self._data              = kw.pop('Data', None)
    self._target            = kw.pop('Target', None)
    self._initChoosenNet    = kw.pop('InitChosenNet',   Criteria.SP)
    self._initChoosenAdapt  = kw.pop('InitChosenAdapt',  Criteria.SP)
    self._initChoosenCrit   = kw.pop('SortChosenCrit', Criteria.SP)
    self._initReference     = kw.pop('InitReference', None)
    
    # and delete it to avoid mistakes:
    checkForUnusedVars( kw, self._logger.warning )
    self._store = self.__alloc_space()
    self._storeBestInits = self.__alloc_space()
    self._logger.info('space allocated with size: %dX%dX%d',\
                       self._neurons[1],self._nSorts,self.nInits)
  
    for file in self._filesList:
      self._logger.debug('reading file: %s', file)
      objLoad     = pickle.load(open(file,'r'))
      neuron      = objLoad[0]
      sort        = objLoad[1]
      initBounds  = objLoad[2]
      train       = objLoad[3]
      del objLoad
      for init in train:
        self._store[neuron][sort].append(init)

    self.__only_best_inits()
    #self.__calculate_statistical_errors()
      
  '''
    Private methods
  '''
  def __alloc_space( self ):
    store = self.neurons[1]*[None]
    for n in range(*self._neurons):
      sorts = self._nSorts*[None]
      for s in range(self._nSorts):
        sorts[s] = []
      store[n] = sorts
    return store

  def __calculateRoc( self, discriminator, data ):
    sgnOutput = discriminator( self._data[0].tolist() )
    bkgOutput = discriminator( self._data[1].tolist() )
    return Performance(genRoc( sgnOutput, bkgOutput ))
         

  def __only_best_inits( self ):
    for neuron in range(len(self._store)):
      if not self._store[neuron]: continue
      for sort in range(len(self._store[neuron])):
        crit = self._initRefCrit
        bestInit  = None
        initCounter=0
        split = self._cross( self._data, self._target, sort)
        for init in self._store[neuron][sort]:

          self._logger.info('calculating performance for init step using:\
                             neuron %d,  sort: %d and init %d',neuron,sort, initCounter)

          initTested = [init[self._netStop], \
            Performance(self.__calculateRoc(init[self._netStop],split[len(split)]) ) ]
          if self._initRefValue: initTested[1].adaptCut( self._initRefValue, crit)

          if not bestInit: bestInit = initTested
          if crit == Criteria.SP  and initTested[1].sp  > bestInit[1].sp:     
            bestInit = initTested
          if crit == Criteria.DET and initTested[1].det > bestInit[1].det: 
            bestInit = initTested
          if crit == Criteria.FA  and initTested[1].fa  < bestInit[1].fa:    
            bestInit = initTested
          initCounter+=1

        self._storeBestInits[neuron][sort] = bestInit

  '''
  def __calculate_statistical_errors( self ):
    
    sp_matrix  = np.zeros([self._nSorts, self._neurons[1]])
    det_matrix = np.zeros([self._nSorts, self._neurons[1]])
    fa_matrix  = np.zeros([self._nSorts, self._neurons[1]])

    for neuron in range( len(self._storeBestInits) ):
      if not self._storeBestInits:  continue
      for sort in range( len(self._storeBestInits[neuron]) ):
        self._logger.info('calculating performance for operation step using:\
                             neuron %d and sort: %d',neuron,sort)

        init = self._storeBestInits[neuron][sort]
        init.append( self.__calculateRoc(init[0], self._sortData) )
        init[2].setCut( init[1].cutID )
        self._storeBestInits[neuron][sort] = init
        sp_matrix[sort][neuron]  = init[2].sp
        det_matrix[sort][neuron] = init[2].det
        fa_matrix[sort][neuron]  = init[2].fa
  '''














