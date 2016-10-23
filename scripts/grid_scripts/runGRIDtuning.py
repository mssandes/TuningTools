#!/usr/bin/env python

from TuningTools.parsers import argparse, ioGridParser, loggerParser, \
                                createDataParser, TuningToolGridNamespace, tuningJobParser
from RingerCore import printArgs, NotSet, conditionalOption, Holder, \
                       MatlabLoopingBounds, Logger, LoggingLevel

## Create our paser
# Add base parser options (this is just a wrapper so that we can have this as
# the first options to show, as they are important options)
parentParser = argparse.ArgumentParser(add_help = False)
# WARNING: Groups can be used to replace conflicting options -o/-d and so on
parentReqParser = parentParser.add_argument_group("Required arguments", '')
parentParser.add_argument_group("Optional arguments", '')
parentReqParser.add_argument('-d','--dataDS', required = True, metavar='DATA',
    action='store', nargs='+',
    help = "The dataset with the data for discriminator tuning.")

# New Param
parentReqParser.add_argument('-r','--refDS', required = False, metavar='REF',
    action='store', nargs='+', default = None, 
    help = "The reference values used to tuning all discriminators.")



parentLoopParser = parentParser.add_argument_group("Looping configuration", '')
parentLoopParser.add_argument('-c','--configFileDS', metavar='Config_DS', 
    required = True, action='store', nargs='+', dest = 'grid_inDS',
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
parser = argparse.ArgumentParser(description = 'Tune discriminators using input data on the GRID',
                                 parents = [tuningJobParser, parentParser, ioGridParser, loggerParser],
                                 conflict_handler = 'resolve')
parser.add_argument('--outputFileBase', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
# Remove tuningJob options:
parser.add_argument('--data', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
parser.add_argument('--crossFile', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
parser.add_argument('--confFileList', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
parser.add_argument('--neuronBounds', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
parser.add_argument('--sortBounds', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
parser.add_argument('--initBounds', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
parser.add_argument('--ppFile', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
parser.add_argument('--no-compress', action='store_const',
    required = False, default = None, const = None,
    help = argparse.SUPPRESS)
# Force merging:
parser.add_argument('--mergeOutput', action='store_const',
    required = False, default = True, const = True, 
    dest = 'grid_mergeOutput',
    help = argparse.SUPPRESS)
# Force secondary to be reusable:
parser.add_argument('--reusableSecondary', action='store_const',
    required = False, default = 'DATA,PP,CROSSVAL', const = 'DATA,PP,CROSSVAL', 
    dest = 'grid_reusableSecondary',
    help = argparse.SUPPRESS)
# Make inDS point to inDS-SGN if used
parser.add_argument('--inDS','-i', action='store', nargs='?',
    required = False, default = False,  dest = 'grid_inDS',
    help = argparse.SUPPRESS)
# Make outputs not usable by user
parser.add_argument('--outputs', action='store_const',
    required = False, default = '"tunedDiscr*"', const = '"tunedDiscr*"', 
    dest = 'grid_outputs',
    help = argparse.SUPPRESS )
# Make nFiles not usable by user
parser.add_argument('--nFiles', action='store_const',
    required = False, default = False, const = False, dest = 'grid_nFiles',
    help = argparse.SUPPRESS)
# Make nFilesPerJob not usable by user
parser.add_argument('--nFilesPerJob', action='store_const',
    required = False, default = 1, const = 1, dest = 'grid_nFilesPerJob',
    help = argparse.SUPPRESS)
# Make nJobs not usable by user
parser.add_argument('--nJobs', action='store_const',
    required = False, default = None, const = None, dest = 'grid_nJobs',
    help = argparse.SUPPRESS)
# Hide forceStaged and make it always be true
parser.add_argument('--forceStaged', action='store_const',
    required = False,  dest = 'grid_forceStaged', default = True, 
    const = True, help = argparse.SUPPRESS)
# Hide forceStagedSecondary and make it always be true
parser.add_argument('--forceStagedSecondary', action='store_const',
    required = False, dest = 'grid_forceStagedSecondary', default = True,
    const = True, help = argparse.SUPPRESS)
#parser.add_argument('--long', action='store_const',
#    required = False, dest = 'grid_long', default = True,
#    const = True, help = argparse.SUPPRESS)
parser.add_argument('--compress', action='store_const', 
    default = 0, const = 0, required = False, 
    help = argparse.SUPPRESS)

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

# Retrieve parser args:
args = parser.parse_args( namespace = TuningToolGridNamespace('prun') )

if args.gridExpand_debug != '--skipScout':
  args.grid_nFiles = 1

# Fix secondaryDSs string:
args.grid_secondaryDS = "DATA:1:%s,PP:1:%s,CROSSVAL:1:%s" % (args.dataDS[0], 
                                                             args.ppFileDS[0],
                                                             args.crossValidDS[0])

if not args.refDS is None:
  args.grid_secondaryDS+=(":REF:1:%s")%(args.refDS[0])
if not args.subsetDS is None:
  args.grid_secondaryDS+=(":SUBSET:1:%s")%(args.subsetDS[0])


# Binning
if args.et_bins is not None:
  if len(args.et_bins)  == 1: args.et_bins  = args.et_bins[0]
  if type(args.et_bins) in (int,float):
    args.et_bins = [args.et_bins, args.et_bins]
  args.et_bins = MatlabLoopingBounds(args.et_bins)
  args.grid_allowTaskDuplication = True
else:
  args.et_bins = Holder([ args.et_bins ])
if args.eta_bins is not None:
  if len(args.eta_bins) == 1: args.eta_bins = args.eta_bins[0]
  if type(args.eta_bins) in (int,float):
    args.eta_bins = [args.eta_bins, args.eta_bins]
  args.eta_bins = MatlabLoopingBounds(args.eta_bins)
  args.grid_allowTaskDuplication = True
else:
  args.eta_bins = Holder([ args.eta_bins ])

mainLogger = Logger.getModuleLogger( __name__, args.output_level )
printArgs( args, mainLogger.debug )

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


# Prepare to run
from itertools import product
startBin = True
for etBin, etaBin in product( args.et_bins(), 
                              args.eta_bins() ):
  # When running multiple bins, dump workspace to a file and re-use it:
  if etBin is not None or etaBin is not None:
    if startBin:
      if args.grid_outTarBall is None and not args.grid_inTarBall:
        args.grid_outTarBall = 'workspace.tar'
      startBin = False
    else:
      if args.grid_outTarBall is not None:
        # Swap outtar with intar
        args.grid_inTarBall = args.grid_outTarBall
        args.grid_outTarBall = None

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
                           SUBSET         = conditionalOption("--clusterFile",    "%SUBSET"           ) ,
                           REF            = conditionalOption("--refFile",        "%REF"              ) ,
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
                           OUTPUT_LEVEL   = conditionalOption("--output-level",   args.output_level   ) if args.output_level is not LoggingLevel.INFO else '',
                         )
              )
  # And run
  args.run_cmd()
  # FIXME We should want something more sofisticated
  if args.gridExpand_debug != '--skipScout':
    break
# Finished submitting all bins
