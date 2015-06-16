#!/usr/bin/env python
import ROOT 
import sys
import pickle
from FastNetTool.FilterEvents import *
from FastNetTool.CrossValid import *

data_jf17 = filterEvents('/afs/cern.ch/user/j/jodafons/public/ringer_samples/mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.e3084_s2045_s2008_r5989_ntupleEGamma_20.1.4.1_v02.root', 
                         RingerOperation.L2,
                         treeName = 'NavNtuple/e24_medium_L1EM18VH', 
                         l1EmClusCut = 20, 
                         filterType = FilterType.Background, 
                         reference = Reference.Truth )

print 'jf17 rings size: %r' % [data_jf17[0].shape]

data_zee  = filterEvents('/afs/cern.ch/user/j/jodafons/public/ringer_samples/mc14_13TeV.147446.PowhegPythia8_AZNLO_Zee_DiLeptonFilter.recon.e3059_s2045_s2008_r5989_ntupleEGamma_20.1.4.1_v02.root',
                         RingerOperation.L2,
                         treeName = 'NavNtuple/e24_medium_L1EM18VH',
                         l1EmClusCut = 20,
                         filterType = FilterType.Signal,
                         reference = Reference.Off_Likelihood )

print 'zee  rings size: %r' % [data_zee[0].shape]


rings = np.concatenate( (data_zee[0],data_jf17[0]), axis=0)
target = np.concatenate( (data_zee[1],data_jf17[1]), axis=0)


print 'rings size: %r | target size: %r' % (rings.shape, target.shape)

cross = CrossValid( target, 10, 10, 6, 4 )

cross.showSort(0)

objSave = [rings, target, cross]
filehandler = open('dataset_ringer_e24_medium_L1EM20VH.pic', 'w')
pickle.dump(objSave, filehandler)
