#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

parser = argparse.ArgumentParser(description = 'Generate input file for FastNet on GRID')
parser.add_argument('-s','--inDS-SGN', action='store', 
    metavar='SignalInputDataset', required = True, nargs='+',
    help = "The signal files that will be used to tune the discriminators")
parser.add_argument('-b','--inDS-BKG', action='store', 
    metavar='BackgroundInputDataset', required = True, nargs='+',
    help = "The background files that will be used to tune the discriminators")
parser.add_argument('-o','--outDS', action='store', 
    metavar='BackgroundInputDataset', required = True,
    help = "The background files that will be used to tune the discriminators")
#mutuallyEx1 = parser.add_mutually_exclusive_group(required=True)
#mutuallyEx1.add_argument('-itar','--inTarBall', 
#    metavar='InTarBall', 
#    help = "The environemnt tarball for posterior usage.")
#mutuallyEx1.add_argument('-otar','--outTarBall', 
#    metavar='OutTarBall', 
#    help = "The environemnt tarball for posterior usage.")
parser.add_argument('-op','--operation', action='store', required = True, 
    help = "The operation environment for the algorithm")
parser.add_argument('--reference', action='store', nargs='+',
    metavar='(BOTH | SGN BKG)_REFERENCE', default = ['Truth'], choices = ('Truth','Off_CutID','Off_Likelihood'),
    help = """
      The reference used for filtering datasets. It needs to be set
      to a value on the Reference enumeration on FilterEvents file.
      You can set only one value to be used for both datasets, or one
      value first for the Signal dataset and the second for the Background
      dataset.
          """)
parser.add_argument('-t','--treePath', metavar='TreePath', action = 'store', 
    default = None, type=str,
    help = """The Tree path to be filtered on the files.
        This argument is required if operation isn't set to Offline!!
      """)
import logging
parser.add_argument('--output-level', default = logging.INFO, 
    help = "The output level for the main logger")
parser.add_argument('--debug', const='--express --debugMode --allowTaskDuplication', 
    help = "The output level for the main logger",action='store_const')
import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()
if args.debug:
  args.output_level = logging.DEBUG
logging.basicConfig( level = args.output_level )
# Treat special argument
if len(args.reference) > 2:
  raise ValueError("--reference set to multiple values: %r" % args.reference)
if len(args.reference) is 1:
  args.reference.append( args.reference[0] )
if args.operation != 'Offline' and not args.treePath:
  raise ValueError("If operation is not set to Offline, it is needed to set the TreePath manually.")
logger = logging.getLogger(__name__)
#logger.setLevel( args.output_level )

from FastNetTool.util import printArgs, conditionalOption
printArgs( args, logger.debug )

if ( len(args.inDS_BKG) > 1 or len(args.inDS_SGN) > 1):
  raise NotImplementedError("Cannot use multiple datasets in this version.")

from subprocess import check_output

# TODO For now we only support one container for signal and one for background.
# We need to change it so that we can input multiple containers. This can be
# simply done by adding them as secondary datasets, copying the idea as done
# below (and looping through the input arguments:
def getNFiles( ds ):
  if ds[-1] != '/': ds += '/'
  logger.info("Retriving %s number of files...", ds)
  output=check_output('dq2-ls -n %s | cut -f1' % ds, shell=True)
  logger.info("Retrieved output: %s", output[:-1])
  nFiles=int(output)
  if nFiles < 1:
    raise RuntimeError(("Couldn't retrieve the number of files on the output "
      "dataset %s. The output retrieven was: %s.") % ( ds , output ) )
  if args.debug and nFiles > 3:
    nFiles = 3
  return nFiles

nSgnFiles = getNFiles( args.inDS_SGN[0])
nBkgFiles = getNFiles( args.inDS_BKG[0])

# We need this to avoid being banned from grid:
import os
if not os.path.isfile(os.path.expandvars("$ROOTCOREBIN/../FastNetTool/cmt/boost_1_58_0.tar.gz")):
  logger.info('Downloading boost to avoid doing it on server side.')
  import urllib
  urllib.urlretrieve("http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz", 
                     filename=os.path.expandvars("$ROOTCOREBIN/../FastNetTool/cmt/boost_1_58_0.tar.gz"))
else:
  logger.info('Boost already downloaded.')

workDir=os.path.expandvars("$ROOTCOREBIN/..")
os.chdir(workDir) # We need to cd to this dir so that prun accepts the submission
exec_str = """\
            prun --bexec "source ./buildthis.sh" \\
                 --exec \\
                    "source ./setrootcore.sh; 
                    {gridCreateData} \\
                      --sgnInputFiles %IN \\
                      --bkgInputFiles %BKG \\
                      --operation {operation} \\
                      --output fastnet.pic \\
                      --reference {referenceSgn} {referenceBkg} \\
                      {treePath}; " \\
                 --inDS={inDS_SGN} \\
                 --nFilesPerJob={nSgnFiles} \\
                 --nFiles={nSgnFiles} \\
                 --nJobs=1 \\
                 --forceStaged \\
                 --forceStagedSecondary \\
                 --excludeFile "*.o,*.so,*.a,*.gch" \\
                 --extFile "FastNetTool/cmt/boost_1_58_0.tar.gz" \\
                 --secondaryDSs=BKG:{nBkgFiles}:{inDS_BKG}  \\
                 --outDS={outDS} \\
                 --disableAutoRetry \\
                 --outputs=fastnet.pic \\
                 --maxNFilesPerJob={nInputFiles} \\
                 --nGBPerJob=10000 \\
                 {extraFlags}
          """.format(gridCreateData = "\$ROOTCOREBIN/user_scripts/FastNetTool/run_on_grid/gridCreateData",
                     inDS_SGN='%s' % ' '.join(args.inDS_SGN),
                     inDS_BKG='%s' % ' '.join(args.inDS_BKG),
                     operation=args.operation,
                     outDS=args.outDS,
                     nSgnFiles=nSgnFiles, # it seems that setting nJobs=1 doesn't work... but why?
                     nBkgFiles=nBkgFiles,
                     nInputFiles=nSgnFiles+nBkgFiles,
                     #tarBallArg=conditionalOption('--inTarBall=', args.inTarBall) + conditionalOption('--outTarBall=', args.outTarBall),
                     referenceSgn=args.reference[0],
                     referenceBkg=args.reference[1],
                     treePath = conditionalOption('--treePath ', args.treePath),
                     extraFlags = args.debug if args.debug else '--skipScout',
                     )
logger.info("Executing following command:\n%s", exec_str)
import re
exec_str = re.sub('\\\\ *\n','', exec_str )
exec_str = re.sub(' +',' ', exec_str)
logger.info("Command without spaces:\n%s", exec_str)
os.system(exec_str)
