#!/usr/bin/env python
import logging
import ROOT 
import sys
import pickle
from TuningTools.FilterEvents import *
#from TuningTools.CrossValid import *


import numpy as np
etaBins = [0, 0.8 , 1.37, 1.54, 2.5]
etBins  = [0,30, 50, 20000]# in GeV


output   = 'mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH'
#basepath = '/afs/cern.ch/work/j/jodafons/public'
basepath = '/afs/cern.ch/work/j/jodafons/news'
bkgName  = \
    'sample.user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.1.0.4.e3084_s2044_s2008_r5988.rr0002.ph0007_PhysVal.root'
sgnName  = \
    'sample.user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.0.4.e3059_s1982_s2008_r5993_rr0002_ph0007_PhysVal.root'


print 'Background:'

npBkg = filterEvents(basepath+'/'+bkgName, 
                         RingerOperation.L2,
                         treePath= 'Trigger/HLT/Egamma/Ntuple/e24_medium_L1EM18VH', 
                         l1EmClusCut = 20, 
                         l2EtCut = 19,
                         filterType = FilterType.Background, 
                         reference = Reference.Truth,
                         etaBins=etaBins,
                         etBins=etBins
                         )


print 'Signal:'

npSgn = filterEvents(basepath+'/'+sgnName,
                         RingerOperation.L2,
                         treePath = 'Trigger/HLT/Egamma/Ntuple/e24_medium_L1EM18VH',
                         l1EmClusCut = 20,
                         l2EtCut = 19,
                         filterType = FilterType.Signal,
                         reference = Reference.Off_Likelihood,
                         etaBins=etaBins,
                         etBins=etBins
                         )


output = 'mc14_13TeV.147406.129160.sgn.offLH.bkg.truth.trig.l1cluscut_20.l2etcut_19.e24_medium_L1EM18VH'
from TuningTools.CreateData import TuningDataArchive

for nEt in range(len(etBins)-1):
  for nEta in range(len(etaBins)-1):
    sufix=('_etBin_%d_etaBin_%d')%(nEt,nEta)
    print ('Saving position: [%d][%d]')%(nEt,nEta)
    print 'sgn shape is ',npSgn[nEt][nEta].shape
    print 'bkg shape is ',npBkg[nEt][nEta].shape
    savedPath = TuningDataArchive( output+sufix,
                                   signal_rings = npSgn[nEt][nEta],
                                   background_rings = npBkg[nEt][nEta] ).save()
    print ('Saved path is %s')%(savedPath)
