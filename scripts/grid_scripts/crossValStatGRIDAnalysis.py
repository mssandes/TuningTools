#!/usr/bin/env python

from TuningTools.parsers import ( ArgumentParser, ioGridParser, loggerParser
                                , createDataParser, TuningToolGridNamespace, crossValStatsJobParser
                                )

from RingerCore import ( printArgs, NotSet, conditionalOption
                       , Logger, LoggingLevel, expandFolders
                       , select, BooleanStr
                       , GridOutput, GridOutputCollection
                       , SecondaryDataset, SecondaryDatasetCollection
                       , emptyArgumentsPrintHelp
                       )

crossValStatsJobParser.suppress_arguments( doMatlab = True
                                         , test = False 
                                         , doCompress = False )
crossValStatsJobParser.delete_arguments( 'binFilters', 'discrFiles', 'refFile')
ioGridParser.delete_arguments( 'grid__inDS', 'grid__reusableSecondary'
                             , 'grid__nFiles', 'grid__antiMatch'
                             )
ioGridParser.suppress_arguments( grid__outputs = GridOutputCollection()
                               , grid__match = False
                               , grid__writeInputToTxt = 'IN:input.csv'
                               , grid__allowTaskDuplication = True
                               , grid__nFiles = None
                               , grid__nFilesPerJob = 1
                               , grid__nJobs = 1
                               , grid__maxNFilesPerJob = 1
                               , grid__forceStaged = True
                               , grid__forceStagedSecondary = True
                               , grid__crossSite = 1
                               , grid__secondaryDS = SecondaryDatasetCollection()
                               )

## Create our paser
# Add base parser options (this is just a wrapper so that we can have this as
# the first options to show, as they are important options)
parentParser = ArgumentParser(add_help = False)
parentReqParser = parentParser.add_argument_group("required arguments", '')
parentReqParser.add_argument('-d','--discrFilesDS', 
    required = True, metavar='DATA', action='store', dest = 'grid_inDS',
    help = "The dataset with the tuned discriminators.")
parentOptParser = parentParser.add_argument_group("optional arguments", '')
parentOptParser.add_argument('-r','--refFileDS', metavar='REF_FILE', 
    required = False,  default = None, action='store',
    help = """Input dataset to loop upon files to retrieve configuration. There
              will be one job for each file on this container.""")

parser = ArgumentParser(description = 'Retrieve performance information from the Cross-Validation method on the GRID.',
                        parents = [crossValStatsJobParser, parentParser, ioGridParser, loggerParser],
                        conflict_handler = 'resolve')
parser.make_adjustments()

emptyArgumentsPrintHelp(parser)

# Retrieve parser args:
args = parser.parse_args( namespace = TuningToolGridNamespace('prun') )
args.setBExec('source ./buildthis.sh --grid --with-scipy --no-color || source ./buildthis.sh --grid --with-scipy --no-color')
mainLogger = Logger.getModuleLogger( __name__, args.output_level )
printArgs( args, mainLogger.debug )

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
refStr = ""
if args.refFileDS:
  args.append_to_job_submission_option( 'secondaryDSs', SecondaryDataset( key = "REF_FILE", nFilesPerJob = 1, container = args.refFileDS[0], reusable = True) )
  refStr = "%REF_FILE"

# Set output:
args.append_to_job_submission_option('outputs', GridOutputCollection(
                                                  [ GridOutput('pic','crossValStat.pic')
                                                  , GridOutput('mat','crossValStat.mat')
                                                  ]
                                                )
                                    )
# FIXME The default is to create the root files. Change this to a more automatic way.
if args._doMonitoring is NotSet or BooleanStr.retrieve( args._doMonitoring ):
  args.append_to_job_submission_option('outputs', GridOutput('root','crossValStat_monitoring.root'))

startBin = True
for jobFiles, nFiles, jobFilter in zip(jobFileCollection, nFilesCollection, jobFilters):
  if startBin:
    if args.get_job_submission_option('outTarBall') is None and not args.get_job_submission_option('inTarBall'):
      args.set_job_submission_option('outTarBall', 'workspace.tar')
    startBin = False
  else:
    if args.get_job_submission_option('outTarBall') is not None:
      # Swap outtar with intar
      args.set_job_submission_option('inTarBall', args.get_job_submission_option('outTarBall') )
      args.set_job_submission_option('outTarBall', None )
  ## Now set information to grid argument
  if args.get_job_submission_option('debug') != '--skipScout':
    args.set_job_submission_option('nFiles', 1)
    args.set_job_submission_option('nFilesPerJob', 1)
  else:
    args.set_job_submission_option('nFilesPerJob', nFiles)
  args.set_job_submission_option( 'match', '"' + jobFilter + '"')
  # Set execute:
  args.setExec("""source ./setrootcore.sh --grid --no-color;
                  {tuningJob} 
                    -d @input.csv
                    {REF_PERF}
                    {OPERATION}
                    {DO_MONITORING}
                    {DO_MATLAB}
                    {DO_COMPRESS}
                    {DEBUG}
                    {OUTPUT_LEVEL}
               """.format( tuningJob = "\$ROOTCOREBIN/user_scripts/TuningTools/standalone/crossValStatAnalysis.py" ,
                           REF_PERF      = conditionalOption("--refFile",      refStr             ) ,
                           OPERATION     = conditionalOption("--operation",    args.operation     ) ,
                           DO_MONITORING = conditionalOption("--doMonitoring", args._doMonitoring ) if args._doMonitoring is not NotSet else '',
                           DO_MATLAB     = conditionalOption("--doMatlab",     args.doMatlab      ) if args.doMatlab is not NotSet else '',
                           DO_COMPRESS   = conditionalOption("--doCompress",   args._doCompress   ) ,
                           OUTPUT_LEVEL  = conditionalOption("--output-level", args.output_level  ) if args.output_level is not LoggingLevel.INFO else '',
                           DEBUG         = "--test" if ( args.gridExpand_debug != "--skipScout" ) or args.test else '',
                         )
              )
  # And run
  args.run()
  # FIXME We should want something more sofisticated
  if args.get_job_submission_option('debug') != '--skipScout':
    break
# Finished running all bins
