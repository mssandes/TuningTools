#!/usr/bin/env python

from RingerCore import LoggingLevel, expandFolders, Logger, mkdir_p
from TuningTools import CrossValidStatAnalysis, RingerOperation
from pprint import pprint
import os
mainLogger = Logger.getModuleLogger( __name__ )


basepath = 'data_cern/crossval/'
crossval =    [
      [basepath],
             ]

filenameWeights = [
                    'ElectronRingerTightConstants',
                    #'ElectronRingerMediumConstants',
                    #'ElectronRingerLooseConstants',
                    #'ElectronRingerVeryloooseConstants',
                  ]

filenameThres = [
                    'ElectronRingerTightThresholds',
                    #'ElectronRingerMediumThresholds',
                    #'ElectronRingerLooseThresholds',
                    #'ElectronRingerVeryLooseThresholds',
                  ]


ref = 'SP'



####################### Extract Ringer Configuration #########################

from TuningTools import CreateSelectorFiles, TrigMultiVarHypo_v4

export = CreateSelectorFiles(  model = TrigMultiVarHypo_v4() )
export( crossval, filenameWeights, filenameThres, ref )

















