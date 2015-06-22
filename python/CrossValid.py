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

"""
  CrossValid main class
"""
class CrossValid:
  def __init__(self, target, **kw ):
    
    self._logger = None
    self.nSorts = kw.pop('nSorts', 10)
    self.nBoxes = kw.pop('nBoxes', 10)
    self.nTrain = kw.pop('nTrain', 5 )
    self.nValid = kw.pop('nValid', 5 )
    self.nTest  = kw.pop('nTest',  None )
    self.level  = kw.pop('level',  logging.DEBUG )

    nEvts      = target.shape[0]
    rest       = nEvts % self.nBoxes
    sort       = np.random.permutation(nEvts)
    sort_rest  = sort[nEvts-rest:nEvts]
    sort       = sort[0:nEvts-rest]
    self.sort_split = np.split(sort,self.nBoxes)

    if not rest == 0: 
      rand_box = randint(0,self.nBoxes-1)
      self.sort_split[rand_box] = np.concatenate((self.sort_split[rand_box], sort_rest), axis=1)

    self.sort_box_list = []
    for i in range(self.nSorts):
      self.sort_box_list.append( np.random.permutation(self.nBoxes))


  def __call__(self, data, target, sort):
    
    # Retrieve python logger
    if not self._logger:
      import logging
      logging.basicConfig( level = self.level)
      self._logger = logging.getLogger(__name__)

    sort_box      = self.sort_box_list[sort]
    train_indexs  = np.array([])
    val_indexs    = np.array([])
    tst_indexs    = np.array([])

    for box in range(self.nTrain):
      train_indexs = np.concatenate( (train_indexs, self.sort_split[sort_box[box]]), axis=1 )
    
    for box in range(self.nValid):
      val_indexs = np.concatenate( (val_indexs, self.sort_split[sort_box[self.nTrain + box]]), axis=1 )
   
    if self.nTest:
      for box in range(nTest):
        tst_indexs = np.concatenate( (tst_indexs, self.sort_split[sort_box[self.nTrain + self.nValid +box]]), axis=1 )
      test = self.__separate(target, tst_indexs.astype(int))  

    train = self.__separate(target, train_indexs.astype(int))
    valid = self.__separate(target, val_indexs.astype(int))
    
    train = (data[train[0],:],data[train[1],:])
    valid = (data[valid[0],:],data[valid[1],:])
    if self.nTest:  test = (data[test[0],:],   data[test[1],:])
    else: test = None

    self._logger.debug('train: [%s, %s]', train[0].shape[0], train[1].shape[0])
    self._logger.debug('valid: [%s, %s]', valid[0].shape[0], valid[1].shape[0])
    if self.nTest:  self._logger.debug('test:  [%s, %s]', test[0].shape[0], test[1].shape[0])

    return (train, valid, test)

  
  '''
    Private method
  '''
  def __separate( self, target, indexs):
    a = []
    b = []
    for i in indexs:
      if target[i] == 1: a.append(i)
      else:
        b.append(i)
    return [np.array(a),np.array(b)]


