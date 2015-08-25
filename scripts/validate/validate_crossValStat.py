

from FastNetTool.util            import include, normalizeSumRow, reshape, load
from FastNetTool.CrossValidStat  import *
import numpy as np


DatasetLocationInput = '/afs/cern.ch/work/j/jodafons/public/mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH.npy'

from FastNetTool.Logger import Logger
mainLogger = Logger.getModuleLogger(__name__)

mainLogger.info('Opening data...')
objDataFromFile                   = np.load( DatasetLocationInput )
Data                              = normalizeSumRow( reshape( objDataFromFile[0] ) )
Target                            = reshape(objDataFromFile[1])
nSorts                            = 50
neurons                           = [5,20]

stat = CrossValidStat( Cross=Cross, Data=Data,
                                    Target=Target,
                                    nSorts=nSorts,
                                    neurons=neurons,



