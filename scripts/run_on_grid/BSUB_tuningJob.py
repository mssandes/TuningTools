#!/usr/bin/env python

from RingerCore.Logger import Logger, LoggingLevel
mainLogger = Logger.getModuleLogger(__name__)
mainLogger.info("Entering main job.")

from FastNetTool.TuningJob import TuningJob

tuningJob = TuningJob()

import sys
if len(sys.argv) == 6:
  tuningJob( sys.argv[1], 
             confFileList = sys.argv[2],
             ppFileList = sys.argv[3],
             crossValidFile = sys.argv[4],
             epochs = 10000,
             showEvo = 1000, 
             doMultiStop = True,
             doPerf = True,
             outputFileBase = sys.argv[5],
             level = LoggingLevel.INFO )
else:
  tuningJob( sys.argv[1], 
             confFileList = sys.argv[2],
             crossValidFile = sys.argv[2],
             epochs = 10000,
             showEvo = 1000, 
             doMultiStop = True,
             doPerf = True,
             outputFileBase = sys.argv[3],
             level = LoggingLevel.INFO )


mainLogger.info("Finished.")
