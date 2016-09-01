#!/usr/bin/env python

from TuningTools.parsers import argparse, ioGridParser, loggerParser, \
                                createDataParser, TuningToolGridNamespace, crossValStatsJobParser

from RingerCore import printArgs, NotSet, conditionalOption, \
                       Logger, LoggingLevel, expandFolders, \
                       select, BooleanStr

## Create our paser
# Add base parser options (this is just a wrapper so that we can have this as
# the first options to show, as they are important options)
parentParser = argparse.ArgumentParser(add_help = False)
parentReqParser = parentParser.add_argument_group("Required arguments", '')
parentReqParser.add_argument('-d','--discrFilesDS', 
    required = True, metavar='DATA', action='store', dest = 'grid_inDS',
    help = "The dataset with the tuned discriminators.")
parentOptParser = parentParser.add_argument_group("Optional arguments", '')
parentOptParser.add_argument('-r','--refFileDS', metavar='REF_FILE', 
    required = False,  default = None, action='store',
    help = """Input dataset to loop upon files to retrieve configuration. There
              will be one job for each file on this container.""")
## The main parser
parser = argparse.ArgumentParser(description = 'Retrieve performance information from the Cross-Validation method on the GRID.',
                                 parents = [crossValStatsJobParser, parentParser, ioGridParser, loggerParser],
                                 conflict_handler = 'resolve')
# Remove tuningJob options:
parser.add_argument('--doMatlab', action='store_const',
    required = False, default = True, const = True, 
    help = argparse.SUPPRESS
       )
parser.add_argument('--binFilters', action='store_const',
    required = False, default = "", const = "",
    help = argparse.SUPPRESS)
parser.add_argument('--discrFiles', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
parser.add_argument('--refFile', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
parser.add_argument('--test', action='store_const', default = False, const = False,
    help = argparse.SUPPRESS)
parser.add_argument('--outputFileBase', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
# Force secondary to be reusable:
parser.add_argument('--reusableSecondary', action='store_const',
    required = False, default = None, const = None, 
    dest = 'grid_reusableSecondary',
    help = argparse.SUPPRESS)
# Make inDS point to inDS-SGN if used
parser.add_argument('--inDS','-i', action='store', nargs='?',
    required = False, default = False,  dest = 'grid_inDS',
    help = argparse.SUPPRESS)
# Make outputs not usable by user
parser.add_argument('--outputs', action='store_const',
    required = False, default = None, const = None, # Will be set accordingly to the job outputs
    dest = 'grid_outputs',
    help = argparse.SUPPRESS )
# Make nFiles not usable by user
parser.add_argument('--nFiles', action='store_const',
    required = False, default = False, const = False, dest = 'grid_nFiles',
    help = argparse.SUPPRESS)
# User cannot set match property
parser.add_argument('--match', action='store_const',
    required = False, default = False, const = False, dest = 'grid_match',
    help = argparse.SUPPRESS)
# User cannot set antiMatch property
parser.add_argument('--antiMatch', action='store_const',
    required = False, default = False, const = False, dest = 'grid_antiMatch',
    help = argparse.SUPPRESS)
# write input to txt is default
parser.add_argument('--writeInputToTxt',  action='store_const',
    dest = 'grid_writeInputToTxt',
    required = False, default = 'IN:input.csv', const = 'IN:input.csv', 
    help = argparse.SUPPRESS)
# write input to txt is default
parser.add_argument('--allowTaskDusplication',  action='store_const',
    dest = 'grid_allowTaskDuplication',
    required = False, default = True, const = True, 
    help = argparse.SUPPRESS)
# Make nFilesPerJob not usable by user
parser.add_argument('--nFilesPerJob', action='store_const',
    required = False, default = 1, const = 1, dest = 'grid_nFilesPerJob',
    help = argparse.SUPPRESS)
# Make nJobs not usable by user
parser.add_argument('-nJobs', action='store_const',
    required = False, default = 1, const = 1, dest = 'grid_nJobs',
    help = argparse.SUPPRESS)
# Make maxNFilesPerJob not usable by user
parser.add_argument('--maxNFilesPerJob', action='store_const',
    required = False, default = 1, const = 1, dest = 'grid_maxNFilesPerJob',
    help = argparse.SUPPRESS)
# Hide forceStaged and make it always be false
parser.add_argument('--forceStaged', action='store_const',
    required = False,  dest = 'grid_forceStaged', default = True, 
    const = True, help = argparse.SUPPRESS)
# Hide forceStagedSecondary and make it always be false
parser.add_argument('--forceStagedSecondary', action='store_const',
    required = False, dest = 'grid_forceStagedSecondary', default = True,
    const = True, help = argparse.SUPPRESS)
parser.add_argument('--doCompress', action='store_const',  dest = '_doCompress',
    default = "False", const = "False", required = False, 
    help = argparse.SUPPRESS)
parser.add_argument('--crossSite',
    default = 1, required = False, 
    help = argparse.SUPPRESS)

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

# Retrieve parser args:
args = parser.parse_args( namespace = TuningToolGridNamespace('prun') )
args.setBExec('source ./buildthis.sh --grid --with-scipy || source ./buildthis.sh --grid --with-scipy')
args.grid_allowTaskDuplication = True
mainLogger = Logger.getModuleLogger( __name__, args.output_level )
printArgs( args, mainLogger.debug )

if args.gridExpand_debug != '--skipScout':
  args.grid_nFiles = 1

# Force jobs to run only in one site.
#sites = args.grid_site.split(',')
#if args.grid_site in (None, NotSet) or \
#    len(sites) > 1 or len(sites) == 0 or sites[0] == "AUTO":
#  raise RuntimeError("--site option must be set to a value different to AUTO!")

# Set primary dataset number of files:
try:
  # The input files can be send via a text file to avoid very large command lines?
  mainLogger.info(("Retrieving files on the data container to separate "
                  "the jobs accordingly to each tunned bin reagion."))
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
      import os
      scope = 'user.%s' % os.path.expandvars('$USER')
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

# Fix secondaryDSs string if using refFile
refPerfArg = ""
if args.refFileDS:
  args.grid_secondaryDS = "REF_FILE:1:%s" % ( args.refFileDS )
  args.grid_reusableSecondary = "REF_FILE"
  refPerfArg = "%REF_FILE"

# Set output:
args.grid_outputs = '"pic:crossValStat.pic","mat:crossValStat.mat"'
# FIXME The default is to create the root files. Change this to a more automatic way.
if args._doMonitoring is NotSet or BooleanStr.retrieve( args._doMonitoring ):
  args.grid_outputs += ',"root:crossValStat_monitoring.root"'

args.grid_nJobs = 1

startBin = True
for jobFiles, nFiles, jobFilter in zip(jobFileCollection, nFilesCollection, jobFilters):
  if startBin:
    if args.grid_outTarBall is None and not args.grid_inTarBall:
      args.grid_outTarBall = 'workspace.tar'
    startBin = False
  else:
    if args.grid_outTarBall is not None:
      # Swap outtar with intar
      args.grid_inTarBall = args.grid_outTarBall
      args.grid_outTarBall = None
  ## Now set information to grid argument
  #args.grid_nFiles = nFiles
  args.grid_nFilesPerJob = nFiles
  #args.grid_maxNFilesPerJob = nFiles
  args.grid_match = '"' + jobFilter + '"'  
  # Set execute:
  args.setExec("""source ./setrootcore.sh --grid;
                  {tuningJob} 
                    -d @input.csv
                    {REF_PERF}
                    {DO_MONITORING}
                    {DO_MATLAB}
                    {DO_COMPRESS}
                    {DEBUG}
                    {OUTPUT_LEVEL}
               """.format( tuningJob = "\$ROOTCOREBIN/user_scripts/TuningTools/standalone/crossValStatAnalysis.py" ,
                           BINFILTERS    = conditionalOption("--binFilters",   args.binFilters    ) ,
                           REF_PERF      = conditionalOption("--refFile",      refPerfArg         ) ,
                           OPERATION     = conditionalOption("--operation",    args.operation     ) ,
                           DO_MONITORING = conditionalOption("--doMonitoring", args._doMonitoring ) if args._doMonitoring is not NotSet else '',
                           DO_MATLAB     = conditionalOption("--doMatlab",     args.doMatlab      ) if args.doMatlab is not NotSet else '',
                           DO_COMPRESS   = conditionalOption("--doCompress",   args._doCompress   ) ,
                           OUTPUT_LEVEL  = conditionalOption("--output-level", args.output_level  ) if args.output_level is not LoggingLevel.INFO else '',
                           DEBUG         = "--test" if ( args.gridExpand_debug != "--skipScout" ) or args.test else '',
                         )
              )
  # And run
  args.run_cmd()
  # FIXME We should want something more sofisticated
  if args.gridExpand_debug != '--skipScout':
    break
# Finished running all bins

