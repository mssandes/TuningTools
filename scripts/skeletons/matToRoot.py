#!/usr/bin/env python

from RingerCore import ( csvStr2List, emptyArgumentsPrintHelp
                       , expandFolders, Logger, keyboard
                       , progressbar, LoggingLevel, BooleanStr, appendToFileName )
import ROOT, array

from TuningTools.parsers import ArgumentParser, loggerParser, LoggerNamespace

mainParser = ArgumentParser(description = 'Merge files into unique file.',
                                     add_help = False)
mainMergeParser = mainParser.add_argument_group( "Required arguments", "")
mainMergeParser.add_argument('-i','--inputFiles', action='store', 
    metavar='InputFiles', required = True, nargs='+',
    help = "Matlab input files that will be used to generate the root file")
mainMergeParser.add_argument('-c','--change-output-folder', action='store', 
    required = False, default=None,
    help = "Change output folder to be in the specified path instead using the same input dir as input file.")
mainMergeParser.add_argument('--compress', action='store',  default=True, type= BooleanStr,
                              help="Whether to compress file with scipy.savemat")
mainLogger = Logger.getModuleLogger(__name__)
parser = ArgumentParser(description = 'Save files on matlab format.',
                        parents = [mainParser, loggerParser],
                        conflict_handler = 'resolve')
parser.make_adjustments()

emptyArgumentsPrintHelp( parser )

import numpy as np

# cannot make it work, plenty of bugs
#try:
#  import root_numpy as rnp
#except ImportError:
#  raise ImportError("root_numpy is not available. Please install it following the instructions at https://rootpy.github.io/root_numpy/install.html")

try:
  import scipy.io
except ImportError:
  raise ImportError("scipy.io is not available.")

## Retrieve parser args:
args = parser.parse_args( namespace = LoggerNamespace() )
mainLogger.setLevel( args.output_level )
if mainLogger.isEnabledFor( LoggingLevel.DEBUG ):
  from pprint import pprint
  pprint(args.inputFiles)
## Treat special arguments
if len( args.inputFiles ) == 1:
  args.inputFiles = csvStr2List( args.inputFiles[0] )
args.inputFiles = expandFolders( args.inputFiles )
mainLogger.verbose("All input files are:")
if mainLogger.isEnabledFor( LoggingLevel.VERBOSE ):
  pprint(args.inputFiles)

for inFile in progressbar(args.inputFiles, len(args.inputFiles),
                          logger = mainLogger, prefix = "Processing files "):
  from RingerCore import checkExtension, changeExtension, load, save
  cOutputName = changeExtension( inFile, '.root', knownFileExtensions = ['mat'] )
  #cOutputName = appendToFileName( cOutputName, args.field )
  if args.change_output_folder:
    import os.path
    cOutputName = os.path.join( os.path.abspath(args.change_output_folder) , os.path.basename(cOutputName) )
  rfile = ROOT.TFile(cOutputName,"recreate")
  data = scipy.io.loadmat(inFile)
  fields = data['cfields'][0]
  t = ROOT.TTree('transfer_tree','')
  codebook = data['codebook'].astype('float64')
  weights = data['weights_bmu'].astype('float64')
  allA = []
  for f in fields:
    a = array.array('d',[0])
    allA.append(a)
    t.Branch( f[0], a, f[0] + '/D' )
  a = array.array('d',[0])
  t.Branch( "weight", a, 'weight/D' )
  allA.append(a)
  for c, w in zip(codebook,weights.T):
    for i, a in enumerate(allA[:-1]):
      a[0] = c[i]
    allA[-1][0] = w[0]
    t.Fill()
  rfile.Write()
  rfile.Close()
  mainLogger.info("Successfully created root file: %s", cOutputName)
# end of (for fileCollection)
