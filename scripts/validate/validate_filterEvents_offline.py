#!/usr/bin/env python

import logging
import ROOT 
import sys
import pickle
from FastNetTool.FilterEvents import *
from FastNetTool.CrossValid import *
import os

output   = 'mc14_13TeV.147406.129160.sgn.truth.bkg.truth.env.off.test-larger-sample'
basePath = '/afs/cern.ch/work/w/wsfreund/private'

bkgFiles = []                          
bkgFolders = [ \
  'user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.1.4.7.e3084_s2044_s2008_r5988.rr0001.0002_PhysVal.33781066',
  'user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.1.4.7.e3084_s2044_s2008_r5988.rr0001.0002_PhysVal.33782976',
  'user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.1.4.7.e3084_s2044_s2008_r5988.rr0001.0002_PhysVal.33783021',
  'user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.1.4.7.e3084_s2044_s2008_r5988.rr0001.0002_PhysVal.33783402',
  'user.jodafons.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.rel20.1.4.7.e3084_s2044_s2008_r5988.rr0001.0002_PhysVal.33784137',
  ]

for bkgFolder in bkgFolders:
  f = os.popen('ls ' + basePath + '/' + bkgFolder)
  for j in f:
    i = j[0:-1]
    bkgFiles += [ basePath + '/' + bkgFolder + '/' + i]

sgnFiles = []
sgnFolders = [ \
  'user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.4.7.e3059_s1982_s2008_r5993_rr0001.0002_PhysVal.33781182',
  'user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.4.7.e3059_s1982_s2008_r5993_rr0001.0002_PhysVal.33783257',
  'user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.4.7.e3059_s1982_s2008_r5993_rr0001.0002_PhysVal.33783268',
  'user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.4.7.e3059_s1982_s2008_r5993_rr0001.0002_PhysVal.33783274',
  'user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.4.7.e3059_s1982_s2008_r5993_rr0001.0002_PhysVal.33783734',
  'user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.rel20.1.4.7.e3059_s1982_s2008_r5993_rr0001.0002_PhysVal.33783750',
  ]
for sgnFolder in sgnFolders:
  f = os.popen('ls ' + basePath + '/' + sgnFolder)
  for j in f:
    i = j[0:-1]
    sgnFiles += [ basePath + '/' + sgnFolder + '/' + i]

data_jf17 = filterEvents(bkgFiles, 
                         RingerOperation.Offline,
                         filterType = FilterType.Background, 
                         reference = Reference.Truth,
                         nClusters = 10000)

print 'jf17 rings size: %r' % [data_jf17[0].size()]
data_zee  = filterEvents(sgnFiles,
                         RingerOperation.Offline,
                         filterType = FilterType.Signal,
                         reference = Reference.Truth,
                         nClusters = 10000)

print 'zee  rings size: %r' % [data_zee[0].size()]

rings = np.concatenate( (data_zee[0],data_jf17[0]), axis=0)
target = np.concatenate( (data_zee[1],data_jf17[1]), axis=0)
print 'rings size: %r | target size: %r' % (rings.shape, target.shape)

objSave = np.array([rings, target])
del rings, target
np.save(output, objSave)

