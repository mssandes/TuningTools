
import os
import ROOT
import math
from decimal import Decimal
import numpy as np
import pickle

def load(input):
  return pickle.load(open(input, 'r'))

def save(output, object):
  filehandler = open(output,"wb")
  pickle.dump(object,filehandler)
  filehandler.close()

def alloc_list_space(size):
  l = []
  for i in range(size):
    l.append( None )
  return l

def normalizeSumRow(data):
  for row in xrange(data.shape[0]):
        data[row] /= np.sum(data[row])
  return data

def stdvector_to_list(vec):
    size = vec.size()
    l = size*[0]
    for i in range(size):
      l[i] = vec[i]
    return l



