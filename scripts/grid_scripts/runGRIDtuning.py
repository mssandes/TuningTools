#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from RingerCore import argparse

parser = argparse.ArgumentParser(description = 'Run tuning job on grid')

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()

from RingerCore.util import printArgs
from RingerCore.Logger import Logger
logger = Logger.getModuleLogger(__name__)
printArgs( args, logger.debug )

exec_str = textwrap.dedent(r"""\
                 --inDS={configFileDS} \
                 --secondaryDSs=DATA:1:{data}  \
                 --reusableSecondary=DATA \
                 --outDS={outDS} \
                 --workDir={workDir} \
                 --nFilesPerJob=1 \
                 --outputs="fastnet.tuned*.pic" \
                 --forceStaged \
                 --forceStagedSecondary \
                 --excludeFile "*.o,*.so,*.a,*.gch" \
                 --extFile "FastNetTool/cmt/boost_1_58_0.tar.gz" \
                 --skipScout \
                 --tmpDir=/tmp \
                 --long \
                 {site} \
                 {extraFlags}
          """.format(tuningJob="\$ROOTCOREBIN/user_scripts/FastNetTool/run_on_grid/tuningJob.py",
                     configFileDS=args.configFileDS,
                     data=args.dataDS,
                     outDS=args.outDS,
                     workDir=workDir,
                     site = '--site=' + args.site,
                     excludedSite = '--excludedSite=' + args.excludedSite,
                     extraFlags = args.debug if args.debug else '',
                     ))
