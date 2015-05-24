
from FastNet import *
from data    import DataIris

dataIris = DataIris()


net = FastNet()
net.setData(dataIris.getTrnData(), dataIris.getValData(), dataIris.getValData())
net.setEpochs(1000)
net.setBatchSize( len(dataIris.getTrnData()[1] ))
net.setTop( 2 )
net.initialize()
net.execute()

myNet = net.getNeural()
print 'input is ', dataIris.getValData()[0][0]

print 'output is ', myNet.propagateInput( dataIris.getValData()[0][0] )












