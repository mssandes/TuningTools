#/usr/bin/env python
from timeit import default_timer as timer
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.TuningJob import TuningJob
from TuningTools.PreProc import *
import logging


start = timer()


from TuningTools.TuningJob import fixPPCol
from TuningTools.coreDef      import coreConf, TuningToolCores
from TuningTools.TuningJob    import ReferenceBenchmark,   ReferenceBenchmarkCollection, BatchSizeMethod
from RingerCore.Configure import Development
Development.set( True )
coreConf.set(TuningToolCores.keras)



from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(10,10,1)) ) # 8X8
model.add(Conv2D(32, (3, 3), activation='relu')) # 6X6
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='tanh'))


for et in range(5):
  for eta in range(5):
    #et=2; eta=0
    DatasetLocationInput = '/afs/cern.ch/work/j/jodafons/CERN-DATA/data17_13TeV.AllPeriods.sgn.probes_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et%d_eta%d.npz'%(et,eta)
    tuningJob = TuningJob()
    tuningJob( DatasetLocationInput, 
               #neuronBoundsCol = [5, 5], 
               #sortBoundsCol = [0, 1],
               #initBoundsCol =1, 
               #modelBoundsCol = model,
               epochs = 1000,
               batchSize= 1024*4,
               showEvo = 1,
               #ppCol = ppCol,
               level = 9,
               etBins = et,
               etaBins = eta,
               doMultiStop=False,
               confFileList = 'config/job.hn0001.sl0000.su0009.i0000.pic.gz',
               refFile='/afs/cern.ch/work/j/jodafons/CERN-DATA/data17_13TeV.allPeriods.medium_effs.GRL_v97.npz',
               )

end = timer()

print 'execution time is: ', (end - start)      
