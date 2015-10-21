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


output   = 'mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.multiFEX.e24_medium_L1EM20VH'
basepath = '/afs/cern.ch/work/w/wsfreund/public/Online/PhysVal/'
#basepath = '/afs/cern.ch/work/j/jodafons/news'
bkgName  = \
    'user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.1.0.4.e3084_s2044_s2008_r5988.multiFEX.ph0001_PhysVal'
sgnName  = \
    'user.nbullacr.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.0.4.e3059_s1982_s2008_r5993.multiFEX.rr0001_ph001_PhysVal'


print 'Background:'

npBkg = filterEvents(basepath+'/'+bkgName, 
                         RingerOperation.L2,
                         treePath= 'Trigger/HLT/Egamma/Ntuple/e24_medium_L1EM18VH', 
                         l1EmClusCut = 20, 
                         l2EtCut = 19,
                         filterType = FilterType.Background, 
                         reference = Reference.Truth,
                         etaBins=etaBins,
                         etBins=etBins,
                         #nClusters=200,
                         #getRatesOnly=True,
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
                         #etBins=etBins,
                         #getRatesOnly=True,
                         )


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
