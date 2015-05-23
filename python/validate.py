import sys
import os

os.environ['OMP_NUM_THREADS']='4'
sys.path.append('../../RootCoreBin/lib/x86_64-slc6-gcc48-opt/')
from libFastNetTool import FastnetPyWrapper
from data_iris import *

iris = DataIris()


net = FastnetPyWrapper(2)

net.setTrainData( iris.getValData() )
net.setValData( iris.getValData() )
net.setTestData( iris.getTrnData() )
sim_signal = iris.getTrnData()[0]
sim_noise = iris.getTrnData()[1]
net.setEpochs(500)

net.newff([4,6,1], ['tansig','tansig'], 'trainrp')
net.train()
out = net.sim(sim_signal)
print out
out = net.sim(sim_noise)
print out

trnEvo = net.getTrainEvolution()
#print trnEvo
print len(trnEvo)
print trnEvo[0].epoch()

print trnEvo[0].mse_trn()
print trnEvo[0].is_best_sp()

