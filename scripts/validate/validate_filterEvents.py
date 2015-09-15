#!/usr/bin/env python
import logging
import ROOT 
import sys
import pickle
from TuningTools.FilterEvents import *
from TuningTools.CrossValid import *

output   = 'mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH'
basepath = '/afs/cern.ch/work/j/jodafons/public'
bkgName  = \
    'sample.user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.1.0.4.e3084_s2044_s2008_r5988.rr0003.ph0001_PhysVal.root'
sgnName  = \
    'sample.user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.0.4.e3059_s1982_s2008_r5993_rr0003.ph0001_PhysVal.root'

data_jf17 = filterEvents(basepath+'/'+bkgName, 
                         RingerOperation.L2,
                         treePath= 'Trigger/HLT/Egamma/Ntuple/e24_medium_L1EM18VH', 
                         l1EmClusCut = 20, 
                         filterType = FilterType.Background, 
                         reference = Reference.Truth )
'''
print 'jf17 rings size: %r' % [data_jf17[0].shape]
data_zee  = filterEvents(basepath+'/'+sgnName,
                         RingerOperation.L2,
                         treePath = 'Trigger/HLT/Egamma/Ntuple/e24_medium_L1EM18VH',
                         l1EmClusCut = 20,
                         filterType = FilterType.Signal,
                         reference = Reference.Off_CutID )

print 'zee  rings size: %r' % [data_zee[0].shape]

rings = np.concatenate( (data_zee[0],data_jf17[0]), axis=0)
target = np.concatenate( (data_zee[1],data_jf17[1]), axis=0)
print 'rings size: %r | target size: %r' % (rings.shape, target.shape)

objSave = np.array([rings, target])
del rings, target
np.save(output, objSave)
'''
