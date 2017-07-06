#!/usr/bin/env python

from TuningTools.parsers import ArgumentParser, loggerParser, tuningExpertParser
from RingerCore import emptyArgumentsPrintHelp

parser = ArgumentParser(description = 'Tune expert discriminator based on calorimeter and trackin data.',
                        parents = [tuningExpertParser, loggerParser])
parser.make_adjustments()

emptyArgumentsPrintHelp( parser )

# Retrieve parser args:
args = parser.parse_args()

# FIXME: The core configuration is not being automaticaly set to keras

## Treating special args:
# Configuration
conf_kw = {}
if args.neuronBounds is not None: conf_kw['neuronBoundsCol'] = args.neuronBounds
if args.sortBounds   is not None: conf_kw['sortBoundsCol']   = args.sortBounds
if args.initBounds   is not None: conf_kw['initBoundsCol']   = args.initBounds
if args.confFileList is not None: conf_kw['confFileList']    = args.confFileList
# Binning
from RingerCore import printArgs, NotSet, Logger, LoggingLevel
if not(args.et_bins is NotSet) and len(args.et_bins)  == 1: args.et_bins  = args.et_bins[0]
if not(args.eta_bins is NotSet) and len(args.eta_bins) == 1: args.eta_bins = args.eta_bins[0]

logger = Logger.getModuleLogger( __name__, args.output_level )

printArgs( args, logger.debug )

## Data organization
data = [args.data_calo, args.data_track]

## Neural Networks
# FIXME: Need to find out how to obtain the name of the operation point with the number given
#        For tests purposes I am using the hardcoded name.
from RingerCore import load
# from TuningTools.dataframe import RingerOperation
# args.operation = RingerOperation.retrieve(args.operation)
# print('Operation: '+str(opPoint))
opName = "Offline_LH_Medium"
references = ['Pd','Pf','SP']

nnList_calo = {}
nnList_track = {}

def extractBins( binsEntries ):
  if isinstance(binsEntries, (int, float)): return [binsEntries]
  from RingerCore.LoopingBounds import MatlabLoopingBounds
  return MatlabLoopingBounds(binsEntries).list()

## Retrieving Calorimeter Networks
logger.info("Retrieving Calorimeter Networks")
filelist = args.network_calo
if isinstance(filelist, (str)): filelist = [filelist]
elif isinstance(filelist, (tuple)): filelist = list(filelist)

for et in extractBins( args.et_bins ):
  nnList_calo[et] = {}
  for eta in extractBins( args.eta_bins ):
    tmp = load( filelist.pop(0) )
    nnList_calo[et][eta] = {}
    for x in references:
      nnList_calo[et][eta][x] = {}
      hn = tmp['OperationPoint_%s_%s'%(opName,x)]['infoTstBest']['neuron']
      logger.debug("Reference %s: %i neurons in the hidden layer"%(x,hn))
      for sort in tmp['infoPPChain'].keys():
        # TODO: Add a progressbar to the loop
        nnList_calo[et][eta][x][sort] = tmp['OperationPoint_%s_%s'%(opName,x)]['config_%1.3i'%(hn)][sort]['infoOpBest']['discriminator']

## Retrieving Tracking Networks
logger.info("Retrieving Tracking Networks")
tmp = load( args.network_track )
x='SP'
nnList_track[et] = {}
nnList_track[et][eta] = {}
nnList_track[et][eta][x] = {}
hn = tmp['OperationPoint_%s_%s'%(opName,x)]['infoTstBest']['neuron']
logger.debug("Reference %s: %i neurons in the hidden layer"%(x,hn))
for sort in tmp['infoPPChain'].keys():
  # TODO: Add a progressbar to the loop
  nnList_track[et][eta][x][sort] = tmp['OperationPoint_%s_%s'%(opName,x)]['config_%1.3i'%(hn)][sort]['infoOpBest']['discriminator']

from TuningTools import TuningJob
tuningJob = TuningJob()
tuningJob( data,
           merged            = True,
           networks          = [nnList_calo, nnList_track],
           level             = args.output_level,
					 compress          = args.compress,
					 outputFileBase    = args.outputFileBase,
           outputDirectory   = args.outputDir,
           operationPoint    = args.operation,
           refFile           = args.refFile,
           clusterFile       = args.clusterFile,
           # Cross validation args
					 crossValidFile    = args.crossFile,
					 crossValidMethod  = args.crossValidMethod,
					 crossValidShuffle = args.crossValidShuffle,
           # Pre Processing
           ppFile            = args.ppFile,
           # Binning configuration
           etBins            = args.et_bins,
           etaBins           = args.eta_bins,
					 # Tuning CORE args
           showEvo           = args.show_evo,
           maxFail           = args.max_fail,
           epochs            = args.epochs,
           doPerf            = args.do_perf,
           batchSize         = args.batch_size,
           batchMethod       = args.batch_method,
           seed              = args.seed,
           doMultiStop       = args.do_multi_stop,
           # ExMachina CORE args
           algorithmName     = args.algorithm_name,
           networkArch       = args.network_arch,
           costFunction      = args.cost_function,
           shuffle           = args.shuffle,
					 # Looping configuration args
           **conf_kw
				 )
           

