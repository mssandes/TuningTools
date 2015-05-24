
import pickle
from FastNet import *
from data    import DataIris

dataIris = DataIris()
net = FastNet()
net.setData(dataIris.getTrnData(), dataIris.getValData(), dataIris.getValData())
net.setEpochs(1000)
net.setBatchSize( len(dataIris.getTrnData()[1] ))
net.setTop( 3 )

#list of neural objects for each initialization
neural_inits = []


for init in range(10):
  net.initialize()
  net.execute()
  neural_inits.append( net.getNeural() )

filehandler = open('neuralFromValidate', 'w')  
pickle.dump(neural_inits, filehandler)

del neural_inits
del filehandler
del net

#open file and check list of objects
filehandler = open('neuralFromValidate', 'r')
neural_inits = pickle.load(filehandler)

print 'input is ', dataIris.getValData()[0][0]
print 'output is ', neural_inits[0].propagateInput( dataIris.getValData()[0][0] )











