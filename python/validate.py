import sys

sys.path.append('../../RootCoreBin/lib/x86_64-slc6-gcc48-opt/')
from libFastNetTool import IFastNetTool

net = IFastNetTool()
net.newff([100,18,1], ['tansig','tansig'], 'rprop', True)



