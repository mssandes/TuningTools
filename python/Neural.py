#author: Joao Victor da Fonseca Pinto
#email: joao.victor.da.fonseca.pinto@cern.ch
#
#Neural Netowrk single class using to export the weghts and bias from 
#FastNet c++ core to python object. This can be uses to save into a file

import math
import string

# Make a matrix
def makeW(i, j, fill=0.0):
    n = []
    for m in range(i):
        n.append([fill]*j)
    return n


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

def CALL_TRF_FUNC(input, type):
    return sigmoid(input)

class Neural:
  def __init__(self, net, train):

    #Extract the information from c++ wrapper code
    self.nNodes = []        
    self.numberOfLayers = net.getNumLayers()
    self.w = []
    self.b = []
    self.trfFunc = []
    self.layerOutput = []

    #Train evolution information
    self.epoch = []
    self.mse_trn = []
    self.mse_val = []
    self.sp_val = []
    self.mse_tst = []
    self.sp_tst = []
    self.is_best_mse = []
    self.is_best_sp = []
    self.num_fails_mse = []
    self.num_fails_sp = []
    self.stop_mse = []
    self.stop_sp = []

    #Get train evolution information from TrainDatapyWrapper
    for i in range(len(train)):
      self.epoch.append(train[i].getEpoch())
      self.mse_trn.append(train[i].getMseTrn())
      self.mse_val.append(train[i].getMseVal())
      self.sp_val.append(train[i].getSPVal())
      self.mse_tst.append(train[i].getMseTst())
      self.sp_tst.append(train[i].getSPTst())
      self.is_best_mse.append(train[i].getIsBestMse())
      self.is_best_sp.append(train[i].getIsBestSP())
      self.stop_mse.append(train[i].getStopMse())
      self.stop_sp.append(train[i].getStopSP())

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
          #print 'w[',l,'][',n,'][',k,'] = ', self.w[l][n][k]
        self.b[l][n] = net.getBias(l,n)
        #print 'b[',l,'][',n,']  = ', self.b[l][n]


  def propagateInput(self, input):

    self.layerOutput[0] = input 
    for l in range( len(self.nNodes) - 1 ):
      for n in range( self.nNodes[l+1] ):
        self.layerOutput[l+1][n] = self.b[l][n]
        for k in range( self.nNodes[l] ):
          self.layerOutput[l+1][n] = self.layerOutput[l+1][n] + self.layerOutput[l][k]*self.w[l][n][k]
        self.layerOutput[l+1][n] = CALL_TRF_FUNC(self.layerOutput[l+1][n],  self.trfFunc[l])
   
    return self.layerOutput[len(self.nNodes)-1]





















