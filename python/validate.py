import sys

sys.path.append('../../RootCoreBin/lib/x86_64-slc6-gcc48-opt/')
from libFastNetTool import FastnetPyWrapper


in_trn = 100*[300000*[1]]
out_trn = 300000*[1]
in_val = 100*[2000*[1]]
out_val = 2000*[1]



net = FastnetPyWrapper()

net.set_in_trn_2D(in_trn, 100, 300000)
net.set_out_trn_1D(out_trn, 300000)
net.set_in_val_2D(in_val, 100, 2000)
net.set_out_val_1D(out_val, 2000)


net.newff([100,18,1], ['tansig','tansig'], 'rprop', 100, True)
net.train()


