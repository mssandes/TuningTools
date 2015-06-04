
#Author: Joao Victor da Fonseca Pinto
#Email: jodafons@cern.ch
#
#Class CrossValid is used to sort and randomize the dataset for training step.
#To use this you need the numpy lib installed into your workspace. 
#This class has constructor:
#         CrossValid( target, nSort, nBox, nTrain, nValid, nTest)
#Methods:
#         [train, val, test] = getSort( sort )

import math 
import numpy as np
from random import randint

#Helper function to separate the class 1 and class 2
def separate( target, indexs):
  a = []
  b = []
  for i in indexs:
    if target[i] == 1: a.append(i)
    else:
      b.append(i)
  return [np.array(a),np.array(b)]


class CrossValid:
  def __init__(self, target, nSorts, nBox, nTrain, nValid, nTest = 0):
    
    self.train = []
    self.val = []
    self.tst = []
    if not nTest == 0:  self.useTst = True
    else: self.useTst = False

    nEvts = target.shape[0]
    rest = nEvts % nBox
    sort       = np.random.permutation(nEvts)
    sort_rest  = sort[nEvts-rest:nEvts]
    sort       = sort[0:nEvts-rest]
    sort_split = np.split(sort,nBox)
    if not rest == 0: 
      rand_box = randint(0,nBox-1)
      sort_split[rand_box] = np.concatenate((sort_split[rand_box], sort_rest), axis=1)

    for sort_k in range(nSorts):
      sort_box  = np.random.permutation(nBox)
      train_indexs  = np.array([])
      val_indexs    = np.array([])
      tst_indexs    = np.array([])

      for box in range(nTrain):
        train_indexs = np.concatenate( (train_indexs, sort_split[sort_box[box]]), axis=1 )
      
      for box in range(nValid):
        val_indexs = np.concatenate( (val_indexs, sort_split[sort_box[nTrain + box]]), axis=1 )
     
      if self.useTst:
        for box in range(nTest):
          tst_indexs = np.concatenate( (tst_indexs, sort_split[sort_box[nTrain + nValid +box]]), axis=1 )
        self.tst.append(separate(target, tst_indexs.stype(int)))
      #Append the indexs for this sort
      self.train.append(separate(target, train_indexs.astype(int)))
      self.val.append(separate(target, val_indexs.astype(int)))
      
    
  def getSort(self, data ,sort ):
      
      train = [data[self.train[sort][0],:], data[self.train[sort][1],:]]
      val   = [data[self.val[sort][0],:], data[self.val[sort][1],:]]
      if self.useTst:   
        return [train, val, [data[self.tst[sort][0],:], data[self.tst[sort][1],:]]]
      else:
        return [train, val]

  def showSort(self, sort):
    print 'train: [', self.train[sort][0].shape[0], ', ',self.train[sort][1].shape[0],']'
    print 'val  : [', self.val[sort][0].shape[0], ', '  ,self.val[sort][1].shape[0],']'
    if not len(self.tst) == 0:
      print 'tst  : [', self.tst[sort][0].shape[0], ', ',self.tst[sort][1].shape[0],']'



