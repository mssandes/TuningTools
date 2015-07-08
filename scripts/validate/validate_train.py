from FastNetTool.util import sourceEnvFile, normalizeSumRow, reshape
sourceEnvFile()
import numpy as np
from FastNetTool.FastNet import FastNet
from FastNetTool.CrossValid import CrossValid

DatasetLocationInput ='/afs/cern.ch/work/w/wsfreund/public/mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH.npy'
objDataFromFile                   = np.load( DatasetLocationInput )
Data                              = normalizeSumRow( reshape(objDataFromFile[0] ) )
Target                            = reshape(objDataFromFile[1])
Cross                             = CrossValid( nSorts=50, nBoxes=10, nTrain=6, nValid=4)


split = Cross(Data,Target,0)
trnData=split[0]
valData=split[1]

#import scipy.io
#scipy.io.savemat('data.mat',{'trn':trnData,'val':valData})

fastnet = FastNet( trnData, valData,
                   tstData=None,
                   doMultiStop=False,
                   doPerf=False,
                   epochs=1000,
                   showEvo=1,
                   batchSize=len(trnData[1].tolist()))


fastnet.new_ff([100, 5, 1], ['tansig', 'tansig'])
fastnet.train_ff()
