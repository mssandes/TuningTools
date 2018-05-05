
def tolist(a):
  if isinstance(a,list): return a
  elif isinstance(a,tuple): return a
  else: return a.tolist()
 

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from pprint import pprint

import json
doRaw=1

if doRaw:
  from RingerCore import load
  raw = load('~/nn.tuned.pp-N1.hn0001.sl0000.su0009.i0000.et0000.eta0000.pic.gz')
  dmodel = raw['tunedDiscr'][0][0]
  from keras.models import model_from_json
  pprint(dmodel['discriminator']['model'])
  model = model_from_json( json.dumps(dmodel['discriminator']['model'], separators=(',', ':')) )
  model.set_weights( dmodel['discriminator']['weights'] )
else:
  model = Sequential()
  model.add(Conv2D(2, kernel_size=(3, 3), activation='linear', input_shape=(10,10,1), bias_initializer='random_uniform') ) # 8X8
  model.add(Conv2D(4, (3, 3), activation='linear', bias_initializer='random_uniform')) # 6X6
  model.add(Flatten())
  #model.add(Dropout(0.25))
  #model.add(Dense(4, activation='linear', bias_initializer='zeros'))
  model.add(Dense(5, activation='linear', bias_initializer='random_uniform'))
  model.add(Dense(1, activation='linear', bias_initializer='random_uniform'))
  model.add(Activation('linear'))
  
  dmodel = {'discriminator':{
              'model':  json.loads(model.to_json()),
              'weights': model.get_weights(),
            } }

### Extract and reshape Keras weights
dense_weights = []; dense_bias = []; dense_nodes = []; dense_tfnames = []
conv_kernel = [];  conv_kernel_i = []; conv_kernel_j = []; conv_bias = []; conv_tfnames = []; conv_nodes = []


discrData = {}
useConvLayer = False
### Loop over layers
for idx, obj in enumerate(model.layers):
  
  dobj = dmodel['discriminator']['model']['config'][idx]['config']
  pprint(obj)  
  if type(obj) is Conv2D:
    
    useConvLayer=True
    conv_nodes.append( dobj['filters'] )
    conv_tfnames.append( str(dobj['activation']) )
    w, b = obj.get_weights()

    for wn in w.T:
      for wu in wn:
        conv_kernel.extend( wu.T.reshape((9,)).tolist() )
    
    #b=b[::-1]
    #conv_bias.extend( b.reshape(-1,order='F').tolist() )
    conv_bias.extend( b.tolist() )
    conv_kernel_i.append( dobj['kernel_size'][0] )
    conv_kernel_j.append( dobj['kernel_size'][1] )

  elif type(obj) is Dense:
    dense_nodes.append( dobj['units'] )
    dense_tfnames.append( str(dobj['activation']) )
    w, b = obj.get_weights()
    for wn in w.T:
      dense_weights.extend( wn.tolist() )


    dense_bias.extend( b.tolist() )

  # TODO: Need to implement something smart to tread this case
  elif type(obj) is Activation:
    dense_tfnames.pop(); dense_tfnames.append( str(dobj['activation']) )

  else:
    continue



discrData['dense_nodes']     = tolist( dense_nodes   )
discrData['dense_bias']      = tolist( dense_bias    )
discrData['dense_weights']   = tolist( dense_weights )
discrData['dense_tfnames']   = tolist( dense_tfnames )

# Convolutional neural network
if useConvLayer:
  discrData['conv_nodes']       = tolist( conv_nodes      )
  discrData['conv_tfnames']     = tolist( conv_tfnames    )
  discrData['conv_kernel_i']    = tolist( conv_kernel_i   )
  discrData['conv_kernel_j']    = tolist( conv_kernel_j   )
  discrData['conv_kernel']      = tolist( conv_kernel     )
  discrData['conv_bias']        = tolist( conv_bias       )
  discrData['conv_input_i']     = dmodel['discriminator']['model']['config'][0]['config']['batch_input_shape'][1]
  discrData['conv_input_j']     = dmodel['discriminator']['model']['config'][0]['config']['batch_input_shape'][2]
  i = discrData['conv_input_i'] - (sum(conv_kernel_i)-len(conv_kernel_i))
  j = discrData['conv_input_j'] - (sum(conv_kernel_j)-len(conv_kernel_j))
  input_layer = i*j*discrData['conv_nodes'][-1]
  discrData['dense_nodes']    = [input_layer]+discrData['dense_nodes']
      

from pprint import pprint

import numpy as np
rings = np.random.rand(100)
#rings=[1]*100
krings = np.array(rings).reshape((1,10,10,1))

import ROOT,cppyy
ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
import cppyy
cppyy.loadDict("rDev")
from ROOT import HLT
from ROOT.HLT import AsgElectronRingerSelector
asg = AsgElectronRingerSelector('CNN')
from RingerCore import list_to_stdvector

from keras import backend as K
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
#outputs=[]
#for o in model.layers:
#  if type(o) is Dense or type(o) is Conv2D or type(o) is Activation:
#    outputs.append(o.output)
functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
layer_outs = functor([krings, 0])
e=len(layer_outs)
print 'CNN keras input flatten sum- >', sum(sum(layer_outs[e-5]))


output = asg.testCNN( list_to_stdvector('float',rings),
             list_to_stdvector( 'unsigned int', discrData['dense_nodes']),
             list_to_stdvector( 'double', discrData['dense_weights']),
             list_to_stdvector( 'double', discrData['dense_bias']),
             list_to_stdvector( 'string', discrData['dense_tfnames']),
             discrData['conv_input_i'],
             discrData['conv_input_j'],
             list_to_stdvector( 'unsigned int', discrData['conv_nodes']),
             list_to_stdvector( 'unsigned int', discrData['conv_kernel_i']),
             list_to_stdvector( 'unsigned int', discrData['conv_kernel_j']),
             list_to_stdvector( 'double', discrData['conv_kernel']),
             list_to_stdvector( 'double', discrData['conv_bias']),
             list_to_stdvector( 'string', discrData['conv_tfnames']))





print '====================> c++ cnn   = ', output
print '====================> CNN keras = ', layer_outs[e-2]


print 'keras MLP = ', sum(sum(layer_outs[e-5]))
print 'keras MLP = ', sum(sum(layer_outs[e-4]))
print 'keras MLP = ', sum(sum(layer_outs[e-3]))
print 'keras MLP = ', layer_outs[e-2]
print 'keras MLP = ', layer_outs[e-1]


#print layer_outs 
