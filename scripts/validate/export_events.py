#!/usr/bin/env python
import logging
import ROOT 
import sys
import pickle
from TuningTools.FilterEvents import *
#from TuningTools.CrossValid import *


import numpy as np
etaBins    = [0, 1.57, 2.5]
ringConfig = [100, 60]
#etBins     = [0,30, 50, 20000]
etBins     = [0,20000]

basepath = '/afs/cern.ch/work/j/jodafons/workspace_ringer_physval'
fileName = 'PhysVal.root'


data = filterEvents(basepath+'/'+fileName, 
                         RingerOperation.L2,
                         treePath= 'Trigger/HLT/Egamma/Ntuple/e5_loose', 
                         l1EmClusCut = 0, 
                         l2EtCut = 0,
                         filterType = FilterType.Signal, 
                         reference = Reference.Off_CutID,
                         etaBins=etaBins,
                         etBins=etBins,
                         ringConfig=ringConfig
                         )

from TuningTools.CreateData import TuningDataArchive
import scipy.io 
matlab=dict()
for nEt in range(len(etBins)-1):
  for nEta in range(len(etaBins)-1):
    sufix=('_etBin_%d_etaBin_%d')%(nEt,nEta)
    print ('Saving position: [%d][%d]')%(nEt,nEta)
    print 'data shape is ',data[nEt][nEta].shape
    matlab['rings'+sufix]=data[nEt][nEta]

scipy.io.savemat('rings.mat', matlab)