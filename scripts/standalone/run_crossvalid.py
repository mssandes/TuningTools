

from TuningTools.CrossValidStat  import *
import os
import sys

basepath = '/afs/cern.ch/work/j/jodafons/public'
data ='user_jodafons_nn_mc14_13TeV_147406_129160_sgn_Off_CitID_bkg_truth_e24_medium_L1EM20VH_t0008_fastnet_tunedXYZ'
dirtouse=basepath+'/'+data+'/'
finallist=[]

while( dirtouse.endswith('/') ) :
  dirtouse= dirtouse.rstrip('/')
  listfiles=os.listdir(dirtouse)
  for ll in listfiles:
    finallist.append(dirtouse+'/'+ll)

stat = CrossValidStatAnalysis( finallist,[5,20], 50 )
#stat('sp', ref_det=0.9816, ref_fa=0.1269)
stat.save_network('sp',5,7,71,-0.77000,'network.tight.n100_5_1.'+data+'.pic')
stat.save_network('sp',18,15,45,-0.03000,'network.medium.n100_18_1.'+data+'.pic')
stat.save_network('sp',7,39,6,-0.3250,'network.loose.n100_7_1.'+data+'.pic')

