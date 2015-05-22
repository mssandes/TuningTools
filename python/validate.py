import sys

sys.path.append('../../RootCoreBin/lib/x86_64-slc6-gcc48-opt/')
from libFastNetTool import FastnetPyWrapper
from data_iris import *

iris = DataIris()

in_trn = 100*[300*[1]]
out_trn = 300*[1]
in_val = 100*[20*[1]]
out_val = 20*[1]



net = FastnetPyWrapper(2)

net.setTrainData( iris.getTrnData() )
net.setValData( iris.getValData() )
net.setTestData( iris.getValData() )

net.setEpochs(1000)

net.newff([4,18,1], ['tansig','tansig'], 'rprop')
net.train()


