#!/usr/bin/env python
"""
  Class: CrossValid
  Author: Joao Victor da Fonseca Pinto
  Email: jodafons@cern.ch
  
  Class CrossValid is used to sort and randomize the dataset for training step.
  To use this you need the numpy lib installed into your workspace. 
"""
import math 
import numpy as np
import logging
from   random import randint
from FastNetTool.Logger import Logger

"""
  CrossValid main class
"""
class CrossValid (Logger):
  def __init__(self, **kw ):
    Logger.__init__( self, **kw  )
    
    self._nSorts = kw.pop('nSorts', 10)
    self._nBoxes = kw.pop('nBoxes', 10)
    self._nTrain = kw.pop('nTrain', 5 )
    self._nValid = kw.pop('nValid', 5 )
    self._nTest  = kw.pop('nTest',  None )
 

    self._sort_boxes_list = []
    for i in range(self._nSorts):
      random_boxes = np.random.permutation(self._nBoxes)
      self._sort_boxes_list.append( random_boxes )
    self._logger.info('class crossValid was created.')

  def __call__(self, data, target, sort):
    
    classes       = self.__separateClasses(target, range(target.shape[0]))
    sort_boxes    = self._sort_boxes_list[sort]
   
    class_split_list = []
    for class_ in classes:
      evts = class_.shape[0]
      rest = evts % self._nBoxes
      evts_rest = class_[evts-rest:evts]
      class_ = class_[0:evts-rest]
      class_split = np.split(class_, self._nBoxes)
      if rest > 0:
        rand_box = randint(0,self._nBoxes-1)
        class_split[rand_box] = np.concatenate((class_split[rand_box], evts_rest), axis=1)
      class_split_list.append( class_split )

    train = self.__concatenateBoxes( class_split_list, sort_boxes, 0, self._nTrain)
    valid = self.__concatenateBoxes( class_split_list, sort_boxes, self._nValid, self._nTrain)
    if self._nTest:  test = self.__concatenateBoxes( class_split_list,
                                                    sort_boxes,
                                                    self._nTrain+self.nValid, 
                                                    self._nTrain+self._nValid+self.nTest)
    else: 
      test = None
      testData = None
    self._logger.info('train: [%s, %s]', train[0].shape[0], train[1].shape[0])
    self._logger.info('valid: [%s, %s]', valid[0].shape[0], valid[1].shape[0])
    if self._nTest:  
      self._logger.info('test:  [%s, %s]', test[0].shape[0], test[1].shape[0])
      testData = (data[test[0],:],data[test[1],:])
   
    print data.shape
    print data[train[0][0],:]
    print train[0]
    trainData = (data[train[0].astype(int),:],data[train[1].astype(int),:])
    validData = (data[valid[0].astype(int),:],data[valid[1].astype(int),:])
    

    return (trainData, validData, testData)

  
  '''
    Private method
  '''
  def __separateClasses( self, target, indexs):
    sgn = []
    bkg = []
    for i in indexs:
      if target[i] == 1: sgn.append(i)
      else: bkg.append(i)
    return [np.array(sgn, dtype='int32'),np.array(bkg,dtype='int32')]

  def __concatenateBoxes( self, class_split_list, sort_boxes, minBox, maxBox):
    sgn = np.array([])
    bkg = np.array([])
    self._logger.debug('concatenate: box: %s to %s',minBox,maxBox)
    for box in range( minBox, maxBox ):
      sgn = np.concatenate( (sgn, class_split_list[0][sort_boxes[box]]), axis=1)
      bkg = np.concatenate( (bkg, class_split_list[1][sort_boxes[box]]), axis=1)
    return [sgn, bkg]

   



