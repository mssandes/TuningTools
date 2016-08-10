#!/usr/bin/env python


def filterPaths(paths, grid=False):
  oDict = dict()
  if grid is True:
    import re
    pat = re.compile(r'.*user.[a-zA-Z0-9]+.(?P<jobID>[0-9]+)\..*$')
    jobIDs = sorted(list(set([pat.match(f).group('jobID')  for f in paths if pat.match(f) is not None]))) 
    for jobID in jobIDs:
      oDict[jobID] = dict()
      for xname in paths:
        if jobID in xname and xname.endswith('.root'): oDict[jobID]['root'] = xname
        if jobID in xname and '.pic' in xname: oDict[jobID]['pic'] = xname
  else:
    oDict['unique'] = {'root':'','pic':''}
    for xname in paths:
      if xname.endswith('.root'): oDict['unique']['root'] = xname
      if '.pic' in xname: oDict['unique']['pic'] = xname

  return oDict


from RingerCore import csvStr2List, str_to_class, NotSet, BooleanStr
from TuningTools.parsers import argparse, loggerParser, crossValStatsMonParser, LoggerNamespace
from TuningTools import GridJobFilter, TuningMonitoringTool

parser = argparse.ArgumentParser(description = 'Retrieve performance information from the Cross-Validation method.',
                                 parents = [crossValStatsMonParser, loggerParser])

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

# Retrieve parser args:
args = parser.parse_args(namespace = LoggerNamespace() )

from RingerCore import Logger, LoggingLevel, printArgs
logger = Logger.getModuleLogger( __name__, args.output_level )

printArgs( args, logger.debug )


#Find files
from RingerCore import expandFolders
logger.info('Expand folders and filter')
paths = expandFolders(args.file)
paths = filterPaths(paths, args.grid)


from pprint import pprint
logger.info('Grid mode is: %s',args.grid)
pprint(paths)



#Loop over job grid, basically loop over user...
for jobID in paths:
  logger.info( ('Start from job tag: %s')%(jobID))
  #If files from grid, we must put the bin tag
  basepath = args.basePath+'_'+jobID if args.grid else args.basePath
  tuningReport = args.tuningReport+'_'+jobID if args.grid is True else args.tuningReport
  #Create the monitoring object
  monitoring = TuningMonitoringTool( paths[jobID]['pic'], 
                                     paths[jobID]['root'], 
                                     refFile = args.refFile,
                                     level = args.output_level)
  #Start!
  monitoring( basePath     = basepath,
              doBeamer     = args.doBeamer,
              shortSlides  = args.doShortSlides,
              debug        = args.debug,
              tuningReport = tuningReport)
  del monitoring
#Loop over jobs








