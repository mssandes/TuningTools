

from FastNetTool.CrossValidStat  import *
import os

basepath = '/afs/cern.ch/work/j/jodafons/public'
data ='user_jodafons_nn_mc14_13TeV_147406_129160_sgn_Off_CitID_bkg_truth_e24_medium_L1EM20VH_t0008_fastnet_tunedXYZ/'
dirtouse=basepath+'/'+data
finallist=[]

while( dirtouse.endswith('/') ) :
  dirtouse= dirtouse.rstrip('/')
  listfiles=os.listdir(dirtouse)
  for ll in listfiles:
    finallist.append(dirtouse+'/'+ll)
print len(finallist)

stat = CrossValidStatAnalysis( finallist,[5,20], 50 )
stat('sp')

