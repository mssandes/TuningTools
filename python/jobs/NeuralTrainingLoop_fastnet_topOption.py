
import sys
import os
import pickle
import numpy as np

from FastNet    import *
from defines    import *
from util       import *
from Neural     import *

NUMBER_MAX_OF_ARQ   = 20
NUMBER_MAX_OF_SORTS = 50
NUMBER_MAX_OF_INITS = 100

def alloc_space_networks_matrix( neurons, sort, inits ):

  list_neurons  = alloc_list_space( neurons )
  for i in range(neurons):  
    list_sorts    = alloc_list_space( sort )
    for j in range(sort):
      list_inits    = alloc_list_space( inits )
      list_sorts[j] = list_inits
    list_neurons[i] = list_sorts
  return list_neurons



try: 
  DatasetLocationInput
except:
  #Dataset default configuration for test
  DatasetLocationInput = '/afs/cern.ch/user/j/jodafons/public/valid_ringear_sample.pic'
  objecsFromFile    = load( DatasetLocationInput )
  Data              = normalizeSumRow( objectsFromFile[0] )
  Target            = objectsFromFile[1]
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


networks = alloc_space_networks_matrix(NUMBER_MAX_OF_ARQ, NUMBER_MAX_OF_SORTS, NUMBER_MAX_OF_INITS)

for hiddenLayerConfiguration in range( NumberOfNeuronsInHiddenLayerMin, NumberOfNeuronsInHiddenLayerMax+1 ):
  
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
 
    for init in range( NumberOfInitsPerSort ):
      print 'Start: hidden layer = ', hiddenLayerConfiguration, ', sort = ', sort, ', init = ', init
      fastnetObject.initialize()
      fastnetObject.execute()
      networks[hiddenLayerConfiguration][sort][init] = fastnetObject.getNeuralObjectsList() 
    del fastnetObject


save(OutputName, networks)

