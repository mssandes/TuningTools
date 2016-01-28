#!/usr/bin/env python
import sys
import pickle
from TuningTools.FilterEvents import *
from TuningTools.CreateData import createData
from RingerCore.FileIO import save, load, expandFolders
from RingerCore.Logger import LoggingLevel
import os
import numpy as np

from TuningTools.CrossValid import CrossValidArchieve
with CrossValidArchieve( "/afs/cern.ch/work/w/wsfreund/private/crossValid.pic.gz" ) as CVArchieve:
  crossVal = CVArchieve
del CVArchieve

RatesOnly=False
etaBins  = [0, 0.8 , 1.37, 1.54, 2.5]
etBins   = [0,30, 50, 20000]# in GeV
output   = 'mc14_13TeV.147406.129160.sgn.offLikelihood.bkg.truth.trig.e24_lhmedium_nod0_L1EM20VH'
#basepath = '/afs/cern.ch/work/j/jodafons/public/Online/PhysVal/'
#bkgName  = 'user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.1.0.4.e3084_s2044_s2008_r5988.rr0104_a0001_PhysVal.root'
#sgnName  = 'user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.0.4.e3059_s1982_s2008_r5993_rr0104_a0001_PhysVal.root'
basepath = '/tmp/jodafons/TriggerTuning2016/samples'
bkgName  = 'user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.7.3.6.e3084_s2044_s2008_r5988.rr0001_p1_PhysVal'
sgnName  = 'user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.7.3.6.e3059_s1982_s2008_r5993_rr0001_p1_PhysVal'

createData( basepath + '/' + sgnName, 
            basepath + '/' + bkgName,
            RingerOperation.L2,
            referenceSgn    = Reference.Off_Likelihood,
            referenceBkg    = Reference.Truth,
            treePath        = ['Trigger/HLT/Egamma/ZeeNtuple/e24_lhmedium_nod0_L1EM20VH', \
                               'Trigger/HLT/Egamma/BackgroundNtuple/e24_lhmedium_nod0_L1EM20VH'],
            l1EmClusCut     = 20,
            l2EtCut         = 19,
            #level           = LoggingLevel.VERBOSE,
            nClusters       = 2000,
            getRatesOnly    = RatesOnly,
            etBins          = etBins,
            etaBins         = etaBins,
            crossVal        = crossVal )


