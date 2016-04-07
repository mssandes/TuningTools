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

files = expandFolders( args.inputs ) # FIXME *.npz

from zipfile import BadZipfile
for f in files:
  logger.info("Changing representation of file '%s'...", f)
  try:
    data = dict(load(f))
  except BadZipfile, e:
    logger.warning("Couldn't load file '%s'. Reason:\n%s", f, str(e))
    continue
  logger.debug("Finished loading file '%s'...", f)
  for key in data:
    if key == 'W':
      from RingerCore.util import traverse
      for obj, idx,  parent, _, _ in traverse(data[key],
                                              tree_types = (np.ndarray,),
                                              max_depth = 3):
        parent[idx] = obj.T
    elif type(data[key]) is np.ndarray:
      logger.debug("Checking key '%s'...", key)
      data[key] = npCurrent.toRepr(data[key])
  path = save(data, f, protocol = 'savez_compressed')
  logger.info("Overwritten file '%s'",f)
