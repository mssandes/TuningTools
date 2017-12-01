#!/usr/bin/env python

from RingerCore import LoggingLevel, Logger
mainLogger = Logger.getModuleLogger("FileJuicer")

import argparse
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

parser.add_argument('-i','--inputFile', action='store', 
    dest='inputFile', required = True, help = "File to Juice!")

import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
args = parser.parse_args()


from RingerCore import load,save
from RingerCore import changeExtension, ensureExtension, appendToFileName
import numpy as np
f = load(args.inputFile)
# Copy all metada information
baseDict = { k : f[k] for k in f.keys() if not '_etBin_' in k and not '_etaBin_' in k }
for etIdx in xrange(f['nEtBins'].item()):
  for etaIdx in xrange(f['nEtaBins'].item()):
    binDict= {k:f[k] for k in f.keys()  if 'etBin_%d_etaBin_%d'%(etIdx,etaIdx) in k}
    binDict.update(baseDict)
    mainLogger.info('Saving Et: %d Eta: %d',etIdx,etaIdx)
    outFile = appendToFileName(args.inputFile, 'et%d_eta%d' % (etIdx, etaIdx) )
    save(binDict,outFile, protocol = 'savez_compressed' )
    
