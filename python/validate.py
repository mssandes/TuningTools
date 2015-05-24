
import pickle
from FastNet import *
from data    import DataIris

dataIris = DataIris()
net = FastNet()
net.setData(dataIris.getTrnData(), dataIris.getValData(), [] , dataIris.getValData())
net.setEpochs(1000)
net.setBatchSize( len(dataIris.getTrnData()[1] ))
net.setTop( 3 )



net.initialize()
net.execute()


m_net = net.getNeural()
print m_net.propagate( dataIris.getValData()[0] )


filehandler = open('neuralFromValidate', 'w')  
pickle.dump(m_net, filehandler)

del m_net
del filehandler
del net

#open file and check list of objects
filehandler = open('neuralFromValidate', 'r')
m_net = pickle.load(filehandler)

print 'output is ', m_net.propagate( dataIris.getValData()[0] )
print m_net.dataTrain.mse_trn

m_net.perf_tst.showInfo()
m_net.dataTrain.showInfo()




