#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from RingerCore import argparse

from RingerCore.Parser import loggerParser, LoggerNamespace
parser = argparse.ArgumentParser( description = """Change data memory representation 
																											 without changing its dimensions.""",
																		  parents = [loggerParser])
parser.add_argument('inputs', action='store', 
    metavar='INPUT',  nargs='+',
    help = "Files to change representation")

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args( namespace = LoggerNamespace() )

from RingerCore.Logger import Logger, LoggingLevel
import numpy as np
from TuningTools.coreDef import retrieve_npConstants
npCurrent, _ = retrieve_npConstants()
npCurrent.level = args.output_level
from RingerCore.FileIO import save, load, expandFolders
logger = Logger.getModuleLogger( __name__, args.output_level )

files = expandFolders( args.inputs )

from zipfile import BadZipfile
from copy import deepcopy
for f in files:
  logger.info("Turning numpy matrix file '%s' into pre-processing file...", f)
  fileparts = f.split('/')
  folder = '/'.join(fileparts[0:-1]) + '/'
  fname = fileparts[-1]
  try:
    data = dict(load(f))
  except BadZipfile, e:
    logger.warning("Couldn't load file '%s'. Reason:\n%s", f, str(e))
    continue
  logger.debug("Finished loading file '%s'...", f)
  for key in data:
    if key == 'W':
      ppCol = deepcopy( data['W'] )
      from TuningTools.PreProc import *
      from RingerCore.util import traverse
      for obj, idx,  parent, _, _ in traverse(ppCol,
                                              tree_types = (np.ndarray,),
                                              max_depth = 3):
        parent[idx] = PreProcChain( RemoveMean(), Projection(matrix = obj) )
      # Turn arrays into mutable objects:
      ppCol = ppCol.tolist()
      from TuningTools.TuningJob import fixPPCol
      ppCol = fixPPCol( ppCol, len(ppCol[0][0]),
                               len(ppCol[0]),
                               len(ppCol))
  if fname.endswith('.npz'):
    fname = fname[:-4]
  newFilePath = folder + fname + '.pic'
  logger.info('Saving to: "%s"...', newFilePath) 
  place = PreProcArchieve( newFilePath, ppCol = ppCol ).save( compress = False )
  logger.info("File saved at path: '%s'", place)
