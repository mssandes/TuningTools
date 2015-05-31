
#======================================================================================
#======================================================================================
#======================================================================================
import pickle
import numpy as np
from FastNet import *
from CrossValid import *
from data    import DataIris
#======================================================================================
#======================================================================================
#======================================================================================
#======================================================================================
#======================================================================================
DatasetLocationInput = '/afs/cern.ch/user/j/jodafons/public/dataset_ringer_e24_medium_L1EM20VH.pic'
OutputName = 'valid_networks_train.pic'
MonitoringLevel = 2
NumberOfInitializesPerSort = 1
NumberOfSortsPerConfigurationMin  = 1
NumberOfSortsPerConfigurationMax  = 1
NumberOfNeuronsInHiddenLayerMin   = 2
NumberOfNeuronsInHiddenLayerMax   = 2
doMultiStops = True
ShowEvolution = 4
#======================================================================================
#======================================================================================
#======================================================================================
#Open file and check list of objects
print 'openning data and normalize...'
filehandler       = open(DatasetLocationInput, 'r')
objectsFromFile   = pickle.load(filehandler)
data              = normalizeSumRow( objectsFromFile[0] )
target            = objectsFromFile[1]
cross             = objectsFromFile[2]
#======================================================================================
#======================================================================================
#======================================================================================
#======================================================================================
#======================================================================================
#======================================================================================
#======================================================================================
#Loop
print 'start loop...'
listOfNetworksPerConfiguration = []
for hiddenLayerConfiguration in range( NumberOfNeuronsInHiddenLayerMin, NumberOfNeuronsInHiddenLayerMax+1 ):
  
  listOfNetworksPerSort = []
  for sort in range( NumberOfSortsPerConfigurationMin, NumberOfSortsPerConfigurationMax + 1 ):

    split     = cross.getSort( data, sort )
    trainData = [split[0][0].tolist(), split[0][1].tolist()]
    valData   = [split[1][0].tolist(), split[1][1].tolist()]
 
    fastnetObject               = FastNet( MonitoringLevel )
    fastnetObject.setData( trainData, valData, [], [] )
    fastnetObject.epochs        = 1000
    fastnetObject.show          = ShowEvolution
    fastnetObject.doPerformance = False
    fastnetObject.top           = hiddenLayerConfiguration
    fastnetObject.batchSize     = len(trainData[1]) 

    if doMultiStops:
      fastnetObject.useAll()
    else:
      fastnetObject.useSP()
 
    listOfNetworksPerInits  = []
    for init in range( NumberOfInitializesPerSort ):
      print 'Start: hidden layer = ', hiddenLayerConfiguration, ', sort = ', sort, ', init = ', init
      fastnetObject.initialize()
      fastnetObject.execute()
      listOfNetworksPerInits.append( fastnetObject.getNeuralObjectsList() )
   
    del fastnetObject

    listOfNetworksPerSort.append( listOfNetworksPerInits )
  
  listOfNetworksPerConfiguration.append( listOfNetworksPerSort )
  

#======================================================================================
#======================================================================================
#======================================================================================
#======================================================================================

print 'done!'  

