

from FastNetTool.CrossValidStat  import CrossValidStat
import os

dirtouse='./networks/'

finallist=[]
while( dirtouse.endswith('/') ) :
  dirtouse= dirtouse.rstrip('/')
  listfiles=os.listdir(dirtouse)
  for ll in listfiles:
    finallist.append(dirtouse+'/'+ll)


stat = CrossValidStat( inputFiles=finallist )
stat(prefix='fig', criteria=0)

