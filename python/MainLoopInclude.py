
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

if not DatasetLocationInput in locals() or DatasetLocationInput in globals():
  DatasetLocationInput = ''
if not in locals() or DatasetLocationInput in globals():
  DatasetLocationInput = ''
if not DatasetLocationInput in locals() or DatasetLocationInput in globals():
  DatasetLocationInput = ''
if not DatasetLocationInput in locals() or DatasetLocationInput in globals():
  DatasetLocationInput = ''
if not DatasetLocationInput in locals() or DatasetLocationInput in globals():
  DatasetLocationInput = ''
if not DatasetLocationInput in locals() or DatasetLocationInput in globals():
  DatasetLocationInput = ''
if not DatasetLocationInput in locals() or DatasetLocationInput in globals():
  DatasetLocationInput = ''










#
