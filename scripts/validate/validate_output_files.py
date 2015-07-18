import pickle
from FastNetTool.CrossValid import CrossValid
from FastNetTool.Neural     import *
from FastNetTool.util       import  getModuleLogger, Roc
mainLogger = getModuleLogger(__name__)
inputPath  ='../../data/valid.output.n0005.s0000.id0002.iu0002.pic'
loadObjs   = pickle.load(open(inputPath,'r'))

neuron     = loadObjs[0]
sort       = loadObjs[1]
initBounds = loadObjs[2]
train      = loadObjs[3]

mainLogger.info('neuron = %d ,sort = %d and train length = %d' ,\
                neuron,sort,len(train))

objs = train[0]
objs_sp = objs[0]
net_sp = objs_sp[0]
net_sp.showInfo()

print objs
print objs_sp
print net_sp
print objs_sp[1].label
print objs_sp[2].label


