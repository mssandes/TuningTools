#!/usr/bin/env python

#Helper local function to separate each type file and each job tag
def filters(paths, doTag=False):
  tags = {}
  for name in paths:
    xname = name.replace('_monitoring','') if '_monitoring' in name else name
    tag = xname.split('_').pop().split('.')[0] if doTag else 'files'
    if not tag in tags.keys():  
      tags[tag] = {'root':list(),'mat':list(),'pic':list()}
    if name.endswith('.root'):  
      tags[tag]['root'].append(name)
    elif name.endswith('.mat'): 
      tags[tag]['mat'].append(name)
    else:
      tags[tag]['pic'].append(name)
  return tags
#*************************************************************************


from RingerCore import csvStr2List, str_to_class, NotSet, BooleanStr
from TuningTools.parsers import argparse, loggerParser, crossValStatsMonParser, LoggerNamespace
from TuningTools import GridJobFilter, MonTuningTool

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
paths = filters(paths,args.grid)
from pprint import pprint
pprint(paths)



#Loop over job grid, basically loop over user...
for job in paths.keys():
  
  logger.info( ('Start from job tag: %s')%(job))
  #If files from grid, we must put the bin tag
  basepath = args.basePath+'_'+job if args.grid else args.basePath
  tuningReport = args.tuningReport+'_'+job if args.grid else args.tuningReport
  #Create the monitoring object
  mon = MonTuningTool( paths[job]['pic'][0], 
                       paths[job]['root'][0], 
                       perfFile = args.perfFile,
                       level = args.output_level)
  #Start!
  mon( basePath     = basepath,
       doBeamer     = args.doBeamer,
       shortSlides  = False,
       tuningReport = tuningReport)

#Loop over jobs








