
import pickle
import numpy as np
from FastNet import *
from CrossValid import *
from data    import DataIris

def normalize(data):
  for row in xrange(data.shape[0]):
        data[row] /= np.sum(data[row])
  return data

#open file and check list of objects
filehandler = open('dataset_ringer_e24_medium_L1EM20VH', 'r')
obj = pickle.load(filehandler)

data    = obj[0]
target  = obj[1]
cross   = obj[2]
print data[0][0]
print data[0].sum
data    = normalize( data )
print data[0][0]
samples = cross.getSort( data,0)

train   = [samples[0][0].tolist(), samples[0][1].tolist()]
val     = [samples[1][0].tolist(), samples[1][1].tolist()]

print cross
print len(train[0])
print len(train[0][0])


net = FastNet()
net.setData( train , val, [] , val)
net.setEpochs(1000)
net.setBatchSize( len(train[1]) )
net.setTop( 15 )
for i in range(1):
  net.initialize()
  net.execute()
  #m_net = net.getNeural()
  #m_net.perf_sim.showInfo()


