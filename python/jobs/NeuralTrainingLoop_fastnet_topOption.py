
import sys
import os
import pickle
import numpy as np

from FastNet    import *
from defines    import *
from util       import *
from Neural     import *

try: 
  DatasetLocationInput
except:
  #Dataset default configuration for test
  DatasetLocationInput = '/afs/cern.ch/user/j/jodafons/public/valid_ringear_sample.pic'
  objecsFromFile    = load( DatasetLocationInput )
  Data              = normalizeSumRow( objectsFromFile[0] )
  target            = objectsFromFile[1]
  CrossValidObject  = objectsFromFile[2]
 
try:
  OutputName
except:
  OutputName = 'output.save'

try:
  MonitoringLevel
except:
  MonitoringLevel = 2

try:
  NumberOfInitsPerSort
except:  
  NumberOfInitsPerSort = 10

try:
  NumberOfNeuronsInHiddenLayerMin
except:  
  NumberOfNeuronsInHiddenLayerMin = 2

try:
  NumberOfNeuronsInHiddenLayerMax
except:  
  NumberOfNeuronsInHiddenLayerMax = 2

try:
  NumberOfSortsPerConfigurationMin
except:  
  NumberOfSortsPerConfigurationMin = 1

try:
  NumberOfSortsPerConfigurationMax
except:  
  NumberOfSortsPerConfigurationMax = 1

try:
  DoMultiStops
except:
  DoMultiStops = True

try:
  ShowEvolution 
except:
  ShowEvolution = 4

try:
  Epochs
except:
  Epochs = 1000

print 'start loop...'
listOfNetworksPerConfiguration = []
for hiddenLayerConfiguration in range( NumberOfNeuronsInHiddenLayerMin, NumberOfNeuronsInHiddenLayerMax+1 ):
  
  listOfNetworksPerSort = []
  for sort in range( NumberOfSortsPerConfigurationMin-1, NumberOfSortsPerConfigurationMax  ):

    split     = CrossValidObject.getSort( Data, sort )
    trainData = [split[0][0].tolist(), split[0][1].tolist()]
    valData   = [split[1][0].tolist(), split[1][1].tolist()]

    simData   = [np.vstack((split[0][0],split[1][0])).tolist(), np.vstack((split[0][1], split[1][1])).tolist()]

    fastnetObject               = FastNet( MonitoringLevel )
    fastnetObject.setData( trainData, valData, [], simData )
    fastnetObject.epochs        = Epochs
    fastnetObject.show          = ShowEvolution
    fastnetObject.doPerformance = False
    fastnetObject.top           = hiddenLayerConfiguration
    fastnetObject.batchSize     = len(trainData[1]) 

    if DoMultiStops:
      fastnetObject.useAll()
    else:
      fastnetObject.useSP()
 
    listOfNetworksPerInits  = []
    for init in range( NumberOfInitsPerSort ):
      print 'Start: hidden layer = ', hiddenLayerConfiguration, ', sort = ', sort, ', init = ', init
      fastnetObject.initialize()
      fastnetObject.execute()
      listOfNetworksPerInits.append( fastnetObject.getNeuralObjectsList() )
   
    del fastnetObject

    listOfNetworksPerSort.append( listOfNetworksPerInits )
  
  listOfNetworksPerConfiguration.append( listOfNetworksPerSort )


save(OutputName, listOfNetworksPerConfiguration)

