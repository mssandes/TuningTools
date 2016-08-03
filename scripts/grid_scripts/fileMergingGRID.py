#!/usr/bin/env python

from RingerCore import csvStr2List, str_to_class, NotSet, BooleanStr, WriteMethod, \
                       get_attributes, expandFolders, Logger, getFilters, select, \
                       appendToFileName, ensureExtension, progressbar, LoggingLevel, \
                       printArgs, conditionalOption

from TuningTools.parsers import argparse, ioGridParser, loggerParser, \
                                TuningToolGridNamespace

from TuningTools import GridJobFilter

parser = argparse.ArgumentParser(description = 'Merge files into unique file on the GRID.',
                                 parents = [ioGridParser, loggerParser],
                                 conflict_handler = 'resolve')
# Hide outputs and make it use tunedDiscr
parser.add_argument('--outputs', action='store_const',
    required = False, default = '"tunedDiscr*"', const = '"tunedDiscr*"', 
    dest = 'grid_outputs',
    help = argparse.SUPPRESS )
# Hide forceStaged and make it always be true
parser.add_argument('--forceStaged', action='store_const',
    required = False,  dest = 'grid_forceStaged', default = True, 
    const = True, help = argparse.SUPPRESS)
# Hide forceStagedSecondary and make it always be true
parser.add_argument('--forceStagedSecondary', action='store_const',
    required = False, dest = 'grid_forceStagedSecondary', default = True,
    const = True, help = argparse.SUPPRESS)
parser.add_argument('--mergeOutput', action='store_const',
    required = False, default = True, const = True, 
    dest = 'grid_mergeOutput',
    help = argparse.SUPPRESS)

mainLogger = Logger.getModuleLogger(__name__)

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args( namespace = TuningToolGridNamespace('prun') )

mainLogger = Logger.getModuleLogger( __name__, args.output_level )
printArgs( args, mainLogger.debug )

args.grid_allowTaskDuplication = True

# Set primary dataset number of files:
import os.path
user_scope = 'user.%s' % os.path.expandvars('$USER')
try:
  # The input files can be send via a text file to avoid very large command lines?
  mainLogger.info(("Retrieving files on the data container to separate "
                  "the jobs accordingly to each tunned bin region."))
  from rucio.client import DIDClient
  from rucio.common.exception import DataIdentifierNotFound
  didClient = DIDClient()
  parsedDataDS = args.grid_inDS.split(':')
  did = parsedDataDS[-1]
  if len(parsedDataDS) > 1:
    scope = parsedDataDS
  else:
    import re
    pat = re.compile(r'(?P<scope>user.[a-zA-Z]+)\..*')
    m = pat.match(did)
    if m:
      scope = m.group('scope')
    else:
      scope = user_scope
  try:
    files = [d['name'] for d in didClient.list_files(scope, did)]
    from TuningTools import GridJobFilter
    ffilter = GridJobFilter()
    jobFilters = ffilter( files )
    mainLogger.info('Found following filters: %r', jobFilters)
    jobFileCollection = select( files, jobFilters )
    nFilesCollection = [len(l) for l in jobFileCollection]
    mainLogger.info("A total of %r files were found.", nFilesCollection )
  except DataIdentifierNotFound, e:
    raise RuntimeError("Could not retrieve number of files on informed data DID. Rucio error:\n%s" % str(e))
except ImportError, e:
  raise ImportError("rucio environment was not set, please set rucio and try again. Full error:\n%s" % str(e))

args.setMergeExec("""source ./setrootcore.sh --grid;
                     {fileMerging}
                      -i %IN
                      -o %OUT
                      {OUTPUT_LEVEL}
                  """.format( 
                              fileMerging  = r"\\\$ROOTCOREBIN/user_scripts/TuningTools/standalone/fileMerging.py" ,
                              OUTPUT_LEVEL = conditionalOption("--output-level",   args.output_level   ) if args.output_level is not LoggingLevel.INFO else '',
                            )
                 )

startBin = True
for jobFiles, nFiles, jobFilter in zip(jobFileCollection, nFilesCollection, jobFilters):
  #output_file = '{USER_SCOPE}.{MERGING_JOBID}.merge._000001.tunedDiscrXYZ.tgz'.format(
  #                USER_SCOPE = user_scope,
  #                MERGING_JOBID = jobFilter)
  output_file = 'merge.tunedDiscr.tgz'.format(
                  USER_SCOPE = user_scope,
                  MERGING_JOBID = jobFilter)
  if startBin:
    if args.grid_outTarBall is None:
      args.grid_outTarBall = 'workspace.tar'
    startBin = False
  else:
    if args.grid_outTarBall is not None:
      # Swap outtar with intar
      args.grid_inTarBall = args.grid_outTarBall
      args.grid_outTarBall = None
  # Now set information to grid argument
  args.grid_nFiles = nFiles
  if args.gridExpand_debug != '--skipScout' and args.grid_nFiles > 800:
    args.grid_nFiles = 800
  args.grid_nFilesPerJob = args.grid_nFiles
  args.grid_maxNFilesPerJob = args.grid_nFiles
  args.grid_match = '"' + jobFilter + '"'  
  args.setExec("""source ./setrootcore.sh --grid;
                  {fileMerging} 
                    -i %IN
                    -o {OUTPUT_FILE}
                    {OUTPUT_LEVEL}
               """.format( fileMerging = "\$ROOTCOREBIN/user_scripts/TuningTools/standalone/fileMerging.py" ,
                           OUTPUT_FILE = output_file,
                           OUTPUT_LEVEL   = conditionalOption("--output-level",   args.output_level   ) if args.output_level is not LoggingLevel.INFO else '',
                         )
              )
  args.grid_outputs = "{datasetNameSuffix}:{outputFileName}".format(
                        datasetNameSuffix = 'tunedDiscrXYZ.tgz',
                        outputFileName = '"' + output_file + '"',
                       )
  # And run
  args.run_cmd()
  # FIXME We should want something more sofisticated
  if args.gridExpand_debug != '--skipScout':
    break
# Finished submitting all bins
