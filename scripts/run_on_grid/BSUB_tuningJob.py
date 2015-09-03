#!/usr/bin/env python

from FastNetTool.Logger import Logger, LoggingLevel
mainLogger = Logger.getModuleLogger(__name__)
mainLogger.info("Entering main job.")

from FastNetTool.TuningJob import TuningJob

tuningJob = TuningJob()

import sys
if sys.nargs == 6:
  tuningJob( sys.argv[1], 
             confFileList = sys.argv[2],
             ppFileList = sys.argv[3],
             crossValidFile = sys.argv[4],
             epochs = 100,
             showEvo = 10, 
             doMultiStop = True,
             doPerf = True,
             outputFileBase = sys.argv[5],
             level = LoggingLevel.INFO )
else:
  tuningJob( sys.argv[1], 
             confFileList = sys.argv[2],
             crossValidFile = sys.argv[2],
             epochs = 100,
             showEvo = 10, 
             doMultiStop = True,
             doPerf = True,
             outputFileBase = sys.argv[3],
             level = LoggingLevel.INFO )


mainLogger.info("Finished.")
