
import pickle
import numpy as np
from FastNet import *
from CrossValid import *
from data    import DataIris

#open file and check list of objects
filehandler = open('/afs/cern.ch/user/j/jodafons/public/dataset_ringer_e24_medium_L1EM20VH.pic', 'r')
#filehandler = open('dataset_ringer_e24_medium_L1EM20VH.pic', 'r')
obj = pickle.load(filehandler)

data    = obj[0]
target  = obj[1]
cross   = obj[2]
data    = normalizeSumRow( data )
samples = cross.getSort( data, 1)

train   = [samples[0][0].tolist(), samples[0][1].tolist()]
val     = [samples[1][0].tolist(), samples[1][1].tolist()]

net = FastNet(1)
net.setData( train , val, [] , val)
net.setEpochs(1000)
net.setBatchSize( len(train[1]) )
net.setMaxFail( 100 )
net.useAll()
#net.useSP()

for i in range(1):
  net.setTop( 8 )
  net.initialize()
  net.execute()
  
  #m_net.perf_sim.showInfo()


