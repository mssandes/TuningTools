#FastNet interface
#Author: Joao Victor da Fonseca Pinto
#Email: jodafons@cern.ch
#
#FastNet: This class is used to connect the python interface and
#the c++ fastnet core. The class FastnetPyWrapper have some methods
#thad can be used to set some train param. Please check the list 
#of methods below:
#
#Class methods:
#           FastNet( msgLevel = INFO = 2 )
#           setData( trnDatam, valData, tstData )
#           setTop( unsigned )
#           initialize()
#           execute()
#           getNeural()
#
#
#Inheritance from FastNetPyWrapper:
#
#           setEpochs( unsigned )
#           setBatchSize(unsigned )
#           setShow( unsigned )
#           new( [input, hidden, output], ['tansig','tansig'],'trainrp')
#           train()
#           py::list output sim( py::list data)
#           showInfo()
#           setFrozenNode(layer, node, status)
#           setTrainData( py::list Data )
#           setValData( py::list ata )
#           setTestData( py::list Data )
#           setShow( unsigned )           
#           setUseSp( bool = true )                            
#           setMaxFail( unsigned = 50)                      
#           setBatchSize( unsigned = 10)                
#           setSPNoiseWeight( unsigned = 1)              
#           setSPSignalWeight( unsigned = 1)           
#           setLearningRate( REAL = 0.05)                   
#           setDecFactor( REAL = 1)                  
#           setDeltaMax( REAL = 50)                     
#           setDeltaMin( REAL = 1E-6 )
#           setIncEta( REAL = 1.10)                     
#           setDecEta( REAL = 0.5)
#           setInitEta( REAL = 0.1 )                    
#           setEpochs( REAL = 1000)                    

import sys
sys.path.append('../../RootCoreBin/lib/x86_64-slc6-gcc48-opt/')
from libFastNetTool import FastnetPyWrapper
from Neural import *
from util import *

class FastNet(FastnetPyWrapper):
  #Default contructor
  def __init__(self, msglevel = 2):
    FastnetPyWrapper.__init__(self, msglevel)
    self.nNodes      = []
    self.top         = 2
    self.inputNumber = 0
    self.trainFcn    = 'trainrp'
    #Datasets
    self.trnData     = []
    self.valData     = []
    self.tstData     = []
    self.simData     = []
    #Neural object
    self.net         = []
    
  #Set number of neurons in the hidden layer
  def setTop(self, n):
    self.top = n

  #Set all datasets
  def setData(self, trnData, valData, tstData, simData):
    #Get the number of inputs
    self.inputNumber = len(trnData[0][0])
    self.trnData = trnData
    self.valData = valData
    self.simData = simData
    self.setTrainData( trnData )
    self.setValData(   valData )
    if len(tstData) > 0:
      self.tstData = tstData
      self.setTestData(  tstData )
    else:
      self.tstData = valData

  #Create the c++ neuralNetowork object into the memory
  def initialize(self):

    self.nNodes = [self.inputNumber, self.top, 1]
    self.newff(self.nNodes, ['tansig','tansig'], self.trainFcn)

  #Run the training
  def execute(self):

    self.train()
    self.net = Neural( self.getNetwork()[0], self.getTrainEvolution() )
    out_sim  = [self.sim(self.simData[0]), self.sim(self.simData[1])]
    out_tst  = [self.sim(self.tstData[0]), self.sim(self.tstData[1])]
    [spVec, cutVec, detVec, faVec] = genRoc( out_sim[0], out_sim[1], 1000 )
    #self.net.setSimPerformance(spVec,detVec,faVec,cutVec)
    [spVec, cutVec, detVec, faVec] = genRoc( out_tst[0], out_tst[1], 1000 )
    #self.net.setTstPerformance(spVec,detVec,faVec,cutVec)

  #Get the Neural object. This object hold all the information about the train
  #and performance values. You can use this as a discriminator.
  def getNeural(self):
    return self.net




