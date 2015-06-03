#author: Joao Victor da Fonseca Pinto
#email: joao.victor.da.fonseca.pinto@cern.ch
#
#Neural Netowrk single class using to export the weghts and bias from 
#FastNet c++ core to python object. This can be uses to save into a file

import math
import string
from util import *
from defines import *


class Neural:
  def __init__(self, net, train):

    #Extract the information from c++ wrapper code
    self.nNodes = []        
    self.numberOfLayers = net.getNumLayers()
    self.w = []
    self.b = []
    self.trfFunc = []
    self.layerOutput = []

    #Hold the train evolution information
    self.dataTrain = DataTrain(train)

    #Hold performance values
    self.perf_sim = []
    self.perf_tst = []

    #Get nodes information  
    for l in range(self.numberOfLayers):
      self.nNodes.append( net.getNumNodes(l) )

    self.layerOutput.append([])
    for l in range(len(self.nNodes) - 1):
      self.trfFunc.append( net.getTrfFuncName(l) )
      #alloc space for weight
      self.w.append( makeW(self.nNodes[l+1], self.nNodes[l]) )
      #alloc space for bias          
      self.b.append( [0]*self.nNodes[l+1] )
      self.layerOutput.append( [0]*self.nNodes[l+1] )

    #Population matrix from DiscriminatorpyWrapper
    for l in range( len(self.nNodes) - 1 ):
      for n in range( self.nNodes[l+1] ):
        for k in range( self.nNodes[l] ):
          self.w[l][n][k] = net.getWeight(l,n,k)
        self.b[l][n] = net.getBias(l,n)


  def propagateInput(self, input):

    self.layerOutput[0] = input 
    for l in range( len(self.nNodes) - 1 ):
      for n in range( self.nNodes[l+1] ):
        self.layerOutput[l+1][n] = self.b[l][n]
        for k in range( self.nNodes[l] ):
          self.layerOutput[l+1][n] = self.layerOutput[l+1][n] + self.layerOutput[l][k]*self.w[l][n][k]
        self.layerOutput[l+1][n] = CALL_TRF_FUNC(self.layerOutput[l+1][n],  self.trfFunc[l])

    if(self.nNodes[len(self.nNodes)-1]) == 1: 
      return self.layerOutput[len(self.nNodes)-1][0]
    else:  
      return self.layerOutput[len(self.nNodes)-1]

  def propagate(self, input):

    [numEvents, numInputs] = size(input)
    if numEvents == 1:
      return self.propagateInput( input )
    else:
      outputVec = []
      for i in range(numEvents):
        outputVec.append( self.propagateInput( input[i] ))
      return outputVec      







