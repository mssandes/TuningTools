#!/usr/bin/env python

import sys

from RingerCore.Logger import Logger, LoggingLevel
mainLogger = Logger.getModuleLogger(__name__)
mainLogger.info("Entering main job.")

crossValidFile = '/afs/cern.ch/work/w/wsfreund/private/crossValid.pic.gz'
dataLocation = '/afs/cern.ch/work/w/wsfreund/public/mc14_13TeV.147406.129160.sgn.truth.bkg.truth.off.npy'

from TuningTools.CrossValid import CrossValidArchieve
with CrossValidArchieve( crossValidFile ) as CVArchieve:
  crossValid = CVArchieve
del CVArchieve
mainLogger.info('CrossValid is: \n%s',crossValid)

self._logger.info('Opening data...')
from TuningTools.CreateData import TuningDataArchive
with TuningDataArchive(dataLocation) as TDArchieve:
  data = TDArchieve
del TDArchieve

import numpy as np

for sort in range( crossValid.nSorts() ):
  trnData, valData, tstData = crossValid( data, sort ) 
  revertedData = crossValid.revert( trnData, valData, tstData, sort = sort )
  try:
    delta = np.abs( data - revertedData )
    cases = ( delta > 0 ).nonzero()[0]
    if cases:
      mainLogger.fatal( 'Found differencies when reverting cross-val...')
      mainLogger.fatal( 'Indexes are: %r', cases )
      sys.exit(1)
  except Exception, e:
    mainLogger.fatal( 'There were an issue when trying to compare reverted crossVal. Reason:\n%r', e )
    sys.exit(1)

sys.exit(0)
