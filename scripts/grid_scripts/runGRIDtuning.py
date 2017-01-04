#!/usr/bin/env python

from TuningTools.parsers import ( ArgumentParser, ioGridParser, loggerParser
                                , createDataParser, TuningToolGridNamespace
                                , tuningJobParser )
from RingerCore import ( printArgs, NotSet, conditionalOption, Holder
                       , MatlabLoopingBounds, Logger, LoggingLevel
                       , SecondaryDatasetCollection, SecondaryDataset
                       , GridOutputCollection, GridOutput, emptyArgumentsPrintHelp )
from RingerCore import argparse, BooleanStr, NotSet 

tuningJobParser.delete_arguments('outputFileBase', 'data', 'crossFile', 'confFileList'
                                , 'neuronBounds', 'sortBounds', 'initBounds', 'ppFile'
                                , 'ppFile', 'no_compress')
tuningJobParser.suppress_arguments(compress = 'False')

ioGridParser.delete_arguments('grid__inDS', 'grid__nJobs')
ioGridParser.suppress_arguments( compress = False
                               , grid__mergeOutput = True
                               , grid_CSV__outputs = GridOutputCollection(GridOutput('td','tunedDiscr*.pic'))
                               , grid__nFiles = 1
                               , grid__nFilesPerJob = 1
                               , grid__forceStaged = True
                               , grid__forceStagedSecondary = True
                               )

## Create our paser
# Add base parser options (this is just a wrapper so that we can have this as
# the first options to show, as they are important options)
parentParser = ArgumentParser(add_help = False)
# WARNING: Groups can be used to replace conflicting options -o/-d and so on
parentReqParser = parentParser.add_argument_group("required arguments", '')
parentReqParser.add_argument('-d','--dataDS', required = True, metavar='DATA',
    action='store', nargs='+',
    help = "The dataset with the data for discriminator tuning.")

# New Param
parentReqParser.add_argument('-r','--refDS', required = False, metavar='REF',
    action='store', nargs='+', default = None, 
    help = "The reference values used to tuning all discriminators.")
parentLoopParser = parentParser.add_argument_group("Looping configuration", '')
parentLoopParser.add_argument('-c','--configFileDS', metavar='Config_DS', 
    required = True, action='store', nargs='+', dest = 'grid__inDS',
    help = """Input dataset to loop upon files to retrieve configuration. There
              will be one job for each file on this container.""")
parentPPParser = parentParser.add_argument_group("Pre-processing configuration", '')
parentPPParser.add_argument('-pp','--ppFileDS', 
    metavar='PP_DS', required = True, action='store', nargs='+',
    help = """The pre-processing files container.""")
parentCrossParser = parentParser.add_argument_group("Cross-validation configuration", '')
parentCrossParser.add_argument('-x','--crossValidDS', 
    metavar='CrossValid_DS', required = True, action='store', nargs='+',
    help = """The cross-validation files container.""")


# New param
parentCrossParser.add_argument('-xs','--subsetDS', default = None, 
    metavar='subsetDS', required = False, action='store', nargs='+',
    help = """The cross-validation subset file container.""")

parentBinningParser = parentParser.add_argument_group("Binning configuration", '')
parentBinningParser.add_argument('--et-bins', nargs='+', default = None, type = int,
        help = """ The et bins to use within this job. 
            When not specified, all bins available on the file will be tuned
            in a single job in the GRID, otherwise each bin available is
            submited separately.
            If specified as a integer or float, it is assumed that the user
            wants to run a single job using only for the specified bin index.
            In case a list is specified, it is transformed into a
            MatlabLoopingBounds, read its documentation on:
              http://nbviewer.jupyter.org/github/wsfreund/RingerCore/blob/master/readme.ipynb#LoopingBounds
            for more details.
        """)
parentBinningParser.add_argument('--eta-bins', nargs='+', default = None, type = int,
        help = """ The eta bins to use within grid job. Check et-bins
            help for more information.  """)
## The main parser
parser = ArgumentParser(description = 'Tune discriminators using input data on the GRID',
                        parents = [tuningJobParser, parentParser, ioGridParser, loggerParser],
                        conflict_handler = 'resolve')
parser.make_adjustments()

emptyArgumentsPrintHelp(parser)

# Retrieve parser args:
args = parser.parse_args( namespace = TuningToolGridNamespace('prun') )

mainLogger = Logger.getModuleLogger( __name__, args.output_level )
mainLogger.write = mainLogger.info
printArgs( args, mainLogger.debug )

if args.get_job_submission_option('debug') != '--skipScout':
  args.set_job_submission_option('nFiles', 1)

# Fix secondaryDSs string:
args.append_to_job_submission_option( 'secondaryDSs'
                                    , SecondaryDatasetCollection ( 
                                      [ SecondaryDataset( key = "DATA", nFilesPerJob = 1, container = args.dataDS[0], reusable = True)
                                      , SecondaryDataset( key = "PP", nFilesPerJob = 1, container = args.ppFileDS[0], reusable = True)
                                      , SecondaryDataset( key = "CROSSVAL", nFilesPerJob = 1, container = args.crossValidDS[0], reusable = True)
                                      ] ) 
                                    )
refStr = subsetStr = ''
if not args.refDS is None:
  args.append_to_job_submission_option( 'secondaryDSs', SecondaryDataset( key = "REF", nFilesPerJob = 1, container = args.refDS[0], reusable = True) )
  refStr = "%REF"
if not args.subsetDS is None:
  args.append_to_job_submission_option( 'secondaryDSs', SecondaryDataset( key = "SUBSET", nFilesPerJob = 1, container = args.subsetDS[0], reusable = True) )
  subsetStr = "%SUBSET"

# Binning
if args.et_bins is not None:
  if len(args.et_bins)  == 1: args.et_bins  = args.et_bins[0]
  if type(args.et_bins) in (int,float):
    args.et_bins = [args.et_bins, args.et_bins]
  args.et_bins = MatlabLoopingBounds(args.et_bins)
  args.set_job_submission_option('allowTaskDuplication', True)
else:
  args.et_bins = Holder([ args.et_bins ])
if args.eta_bins is not None:
  if len(args.eta_bins) == 1: args.eta_bins = args.eta_bins[0]
  if type(args.eta_bins) in (int,float):
    args.eta_bins = [args.eta_bins, args.eta_bins]
  args.eta_bins = MatlabLoopingBounds(args.eta_bins)
  args.set_job_submission_option('allowTaskDuplication', True)
else:
  args.eta_bins = Holder([ args.eta_bins ])

args.setMergeExec("""source ./setrootcore.sh --grid;
                     {fileMerging}
                      -i %IN
                      -o %OUT
                      {OUTPUT_LEVEL}
                  """.format( 
                              fileMerging  = r"\\\$ROOTCOREBIN/user_scripts/TuningTools/standalone/fileMerging.py" ,
                              OUTPUT_LEVEL = conditionalOption("--output-level",   args.output_level   ) \
                                  if LoggingLevel.retrieve( args.output_level ) is not LoggingLevel.INFO else '',
                            )
                 )

def has_subsetDS(args, key):
  return any([secondaryDS == key for secondaryDS in args.get_job_submission_option('secondaryDSs')])

# Prepare to run
from itertools import product
startBin = True
for etBin, etaBin in product( args.et_bins(), 
                              args.eta_bins() ):
  # When running multiple bins, dump workspace to a file and re-use it:
  if etBin is not None or etaBin is not None:
    if startBin:
      if args.get_job_submission_option('outTarBall') is None and not args.get_job_submission_option('inTarBall'):
        args.set_job_submission_option('outTarBall', 'workspace.tar')
      startBin = False
    else:
      if args.get_job_submission_option('outTarBall') is not None:
        # Swap outtar with intar
        args.set_job_submission_option('inTarBall', args.get_job_submission_option('outTarBall') )
        args.set_job_submission_option('outTarBall', None )
  args.setExec("""source ./setrootcore.sh --grid;
                  {tuningJob} 
                    --data %DATA 
                    --confFileList %IN 
                    --ppFile %PP 
                    --crossFile %CROSSVAL 
                    --outputFileBase tunedDiscr 
                    --no-compress
                    {SUBSET}
                    {REF}
                    {SHOW_EVO}
                    {MAX_FAIL}
                    {EPOCHS}
                    {DO_PERF}
                    {BATCH_SIZE}
                    {BATCH_METHOD}
                    {ALGORITHM_NAME}
                    {NETWORK_ARCH}
                    {COST_FUNCTION}
                    {SHUFFLE}
                    {SEED}
                    {DO_MULTI_STOP}
                    {OPERATION}
                    {ET_BINS}
                    {ETA_BINS}
                    {OUTPUT_LEVEL}
               """.format( tuningJob = "\$ROOTCOREBIN/user_scripts/TuningTools/standalone/runTuning.py" ,
                           SUBSET         = conditionalOption("--clusterFile",    subsetStr           ) ,
                           REF            = conditionalOption("--refFile",        refStr              ) ,
                           SHOW_EVO       = conditionalOption("--show-evo",       args.show_evo       ) ,
                           MAX_FAIL       = conditionalOption("--max-fail",       args.max_fail       ) ,
                           EPOCHS         = conditionalOption("--epochs",         args.epochs         ) ,
                           DO_PERF        = conditionalOption("--do-perf",        args.do_perf        ) ,
                           BATCH_SIZE     = conditionalOption("--batch-size",     args.batch_size     ) ,
                           BATCH_METHOD   = conditionalOption("--batch-method",   args.batch_method   ) ,
                           ALGORITHM_NAME = conditionalOption("--algorithm-name", args.algorithm_name ) ,
                           NETWORK_ARCH   = conditionalOption("--network-arch",   args.network_arch   ) ,
                           COST_FUNCTION  = conditionalOption("--cost-function",  args.cost_function  ) ,
                           SHUFFLE        = conditionalOption("--shuffle",        args.shuffle        ) ,
                           SEED           = conditionalOption("--seed",           args.seed           ) ,
                           DO_MULTI_STOP  = conditionalOption("--do-multi-stop",  args.do_multi_stop  ) ,
                           OPERATION      = conditionalOption("--operation",      args.operation      ) ,
                           ET_BINS        = conditionalOption("--et-bin",         etBin               ) ,
                           ETA_BINS       = conditionalOption("--eta-bin",        etaBin              ) ,
                           OUTPUT_LEVEL   = conditionalOption("--output-level",   args.output_level   ) \
                               if LoggingLevel.retrieve( args.output_level ) is not LoggingLevel.INFO else '',
                         )
              )
  # And run
  args.run()
  # FIXME We should want something more sofisticated
  if args.get_job_submission_option('debug') != '--skipScout':
    break
# Finished submitting all bins
