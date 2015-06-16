import ROOT 
import sys
import pickle
from FastNetTool.FilterEvents import *
from FastNetTool.CrossValid import *

data_jf17 = filterEvents('/afs/cern.ch/work/w/wsfreund/private/jf17.e3059_s1982_s2008_r5993_rr0008.fastnet.root',
                         RingerOperation.Offline,
                         filterType = FilterType.Background,
                         reference = Reference.Truth )

print 'jf17 rings size: %r' % [data_jf17[0].shape]

data_zee  = filterEvents('/afs/cern.ch/work/w/wsfreund/private/zee.e3059_s1982_s2008_r5993_rr0008.fastnet.root',
                         RingerOperation.Offline,
                         filterType = FilterType.Signal,
                         reference = Reference.Truth )

print 'zee  rings size: %r' % [data_zee[0].shape]


rings = np.concatenate( (data_zee[0],data_jf17[0]), axis=0)
target = np.concatenate( (data_zee[1],data_jf17[1]), axis=0)


print 'rings size: %r | target size: %r' % (rings.shape, target.shape)

cross = CrossValid( target, 10, 10, 6, 4 )

cross.showSort(0)

objSave = [rings, target, cross]
filehandler = open('dataset_ringer_e24_medium_L1EM20VH.pic', 'w')
pickle.dump(objSave, filehandler)
