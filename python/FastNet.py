

# Author: Joao Victor da Fonseca Pinto
# Email: jodafons@cern.ch
#
# Description:
#      FastNet: This class is used to connect the python interface and
#      the c++ fastnet core. The class FastnetPyWrapper have some methods
#      thad can be used to set some train param. Please check the list 
#      of methods below:
# 
# Class methods:
#       FastNet( msgLevel = INFO = 2 )
#       setData( trnDatam, valData, tstData )
#       initialize()
#       execute()
#       getNeuralObjectsList()

# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
#Imports
import sys
import os
rootcore = os.environ["ROOTCOREBIN"]
sys.path.append(rootcore + '/lib/x86_64-slc6-gcc48-opt/')
from libFastNetTool import FastnetPyWrapper
from Neural import *
from util import *
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
#Default contructor
class FastNet(FastnetPyWrapper):

  def __init__(self, msglevel = 2):
    FastnetPyWrapper.__init__(self, msglevel)
    
    self.nNodes              = []
    self.top                 = 2
    self.inputNumber         = 0
    self.batchSize           = 100
    self.trainFcn            = 'trainrp'
    self.doPerformance       = True
    self.trnData             = []
    self.valData             = []
    self.tstData             = []
    self.simData             = []
    self.networksList        = []
    self.trainEvolutionData  = []

# ================================================================================
# ================================================================================

  #Set all datasets
  def setData(self, trnData, valData, tstData, simData):
    
    print 'Setting dataset into c++ core...'
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

# ================================================================================
# ================================================================================

  def initialize(self):

    self.nNodes = [self.inputNumber, self.top, 1]
    self.newff(self.nNodes, ['tansig','tansig'], self.trainFcn)

# ================================================================================
# ================================================================================

  #Run the training
  def execute(self):

    [DiscriminatorPyWrapperList , TrainDataPyWrapperList] = self.train()
    self.trainEvolutionData = TrainDataPyWrapperList

    for netPyWrapper in DiscriminatorPyWrapperList:
      
      net  = Neural( netPyWrapper, TrainDataPyWrapperList ) 
      
      if self.doPerformance:
        out_sim  = [self.sim(netPyWrapper, self.simData[0]), self.sim( netPyWrapper, self.simData[1])]
        out_tst  = [self.sim(netPyWrapper, self.tstData[0]), self.sim( netPyWrapper, self.tstData[1])]

        [spVec, cutVec, detVec, faVec] = genRoc( out_sim[0], out_sim[1], 1000 )
        net.setSimPerformance( Performance(spVec,detVec,faVec,cutVec) )
      
        [spVec, cutVec, detVec, faVec] = genRoc( out_tst[0], out_tst[1], 1000 )
        net.setTstPerformance(spVec,detVec,faVec,cutVec)
      self.networksList.append( net )

# ================================================================================
# ================================================================================

  def getNeuralObjectsList(self):
    return self.networksList



