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

#def include(filename):
#  if os.path.exists(filename): 
#    execfile(filename)


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
#end


#NeuralNetowrk structure functions
def makeW(i, j, fill=0.0):
    n = []
    for m in range(i):
        n.append([fill]*j)
    return n

def sigmoid(x):
    return math.tanh(x)

def CALL_TRF_FUNC(input, type):
    return sigmoid(input)



def mapMinMax( x, yMin, yMax ):
 # y = []
  xMax = max(x)
  xMin = min(x)
  for i in range( len(x) ):
    y.append( ( (yMax-yMin)*(x[i]-xMin) )/(xMax-xMin)  + yMin)
  return y

def geomean(nums):
  return (reduce(lambda x, y: x*y, nums))**(1.0/len(nums))

def mean(nums):
  return (sum(nums)/len(nums))


def getEff( outSignal, outNoise, cut ):
  #[detEff, faEff] = getEff(outSignal, outNoise, cut)
  #Returns the detection and false alarm probabilities for a given input
  #vector of detector's perf for signal events(outSignal) and for noise 
  #events (outNoise), using a decision threshold 'cut'. The result in whithin
  #[0,1].
  
  detEff = np.where(outSignal >= cut)[0].shape[0]/ float(outSignal.shape[0]) 
  faEff  = np.where(outNoise >= cut)[0].shape[0]/ float(outNoise.shape[0])
  return [detEff, faEff]

def calcSP( pd, pf ):
  #ret  = calcSP(x,y) - Calculates the normalized [0,1] SP value.
  #effic is a vector containing the detection efficiency [0,1] of each
  #discriminating pattern.  
  return math.sqrt(geomean([pd,pf]) * mean([pd,pf]))

def genRoc( outSignal, outNoise, numPts = 1000 ):
  #[spVec, cutVec, detVec, faVec] = genROC(out_signal, out_noise, numPts, doNorm)
  #Plots the RoC curve for a given dataset.
  #Input Parameters are:
  #   out_signal     -> The perf generated by your detection system when
  #                     electrons were applied to it.
  #   out_noise      -> The perf generated by your detection system when
  #                     jets were applied to it
  #   numPts         -> (opt) The number of points to generate your ROC.
  #
  #If any perf parameters is specified, then the ROC is plot. Otherwise,
  #the sp values, cut values, the detection efficiency and false alarm rate 
  # are returned (in that order).
  cutVec = np.arange( -1,1,2 /float(numPts))
  cutVec = cutVec[0:numPts - 1]
  detVec = np.array(cutVec.shape[0]*[float(0)])
  faVec  = np.array(cutVec.shape[0]*[float(0)])
  spVec  = np.array(cutVec.shape[0]*[float(0)])
  for i in range(cutVec.shape[0]):
    [detVec[i],faVec[i]] = getEff( np.array(outSignal), np.array(outNoise),  cutVec[i] ) 
    spVec[i] = calcSP(detVec[i],1-faVec[i])
  return [spVec.tolist(), cutVec.tolist(), detVec.tolist(), faVec.tolist()]

  













