
#!/usr/bin/env python

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

import sys
parser = argparse.ArgumentParser(description = '')
parser.add_argument('-s','--sgnInputFiles', action='store', 
    metavar='SignalInputFiles', required = True, nargs='+',
    help = "The signal files that will be used to tune the discriminators")
parser.add_argument('-b','--bkgInputFiles', action='store', 
    metavar='BackgroundInputFiles', required = True, nargs='+',
    help = "The background files that will be used to tune the discriminators")
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
# Retrieve parser args:
args = parser.parse_args()


from FastNetTool.CreateData import CreateData
from FastNetTool.CrossValid import CrossValid

sys.mkdir('createDataOutput')
sys.cd('createDataOutput')


createData( sgnFileList = args.sgnInputFiles, 
            bkgFileList = args.bkgInputFiles,
            ringerOperation = 'L2',
            referenceSgn = 'Off_CutID',
            referenceBkg = 'Truth',
            treePath = 'Trigger/HLT/Egamma/e24_medium_L1EM18VH',
            output = 'ringerTrainData.pic'
            l1EmClusCut = 20 )
import pickle

objLoad_target = pickle.load( open( 'trainData.pic', "rb" ) )[1]

nMaxSorts=50
cross = CrossValid(objLoad_target, nSort=nMaxSorts, nBoxes=10, nTrain=6, nValid=4 )


for h in range(nMaxLayers):
  for s in range(nMaxSorts):
    jobName = 'jobTrainConfig.h'+str(h)+'.s'+str(s)+'.pic'
    objSave = [h, s, cross]
    pickle.dump( objSave, open( jobName, "wb" ) )




