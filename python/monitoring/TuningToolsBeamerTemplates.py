
from RingerCore import checkForUnusedVars, Logger, LoggingLevel, EnumStringification

class BeamerPerformanceTable( BeamerTableSlide ):
  # TODO Still working on this

  def add(self, obj):
    #Extract the information from obj
    refDict   = obj.getRef()
    values    = obj.getPerf()    
    #Clean the name
    reference = refDict['reference']
    bname     = obj.name().replace('OperationPoint_','')
    #Make color vector, depends of the reference
    color=['','','']#For SP
    if reference == 'Pd': color = ['\\cellcolor[HTML]{9AFF99}','','']
    elif reference == 'Pf': color = ['','','\\cellcolor[HTML]{BBDAFF}']
    #[1] Make perf values stringfication
    val= {'name': bname,
          'det' : ('%s%.2f$\\pm$%.2f')%(color[0],values['detMean'] ,values['detStd'] ),
          'sp'  : ('%s%.2f$\\pm$%.2f')%(color[1],values['spMean']  ,values['spStd']  ),
          'fa'  : ('%s%.2f$\\pm$%.2f')%(color[2],values['faMean']  ,values['faStd']  ) }
    #[2] Make perf values stringfication
    ref  = {'name': bname,
            'det' : ('%s%.2f')%(color[0],refDict['det']  ),
            'sp'  : ('%s%.2f')%(color[1],refDict['sp']   ),
            'fa'  : ('%s%.2f')%(color[2],refDict['fa']   ) }

    #Make latex line stringfication
    self._tline.append( ('%s & %s & %s & %s & %s & %s\\\\\n') % (bname.replace('_','\\_'),val['det'],val['sp'],\
                         val['fa'],ref['det'],ref['fa']) ) 
    
    opDict = obj.rawOp()
    op = {'name': bname,
           'det' : ('%s%.2f')%(color[0],opDict['det']*100  ),
           'sp'  : ('%s%.2f')%(color[1],opDict['sp']*100   ),
           'fa'  : ('%s%.2f')%(color[2],opDict['fa']*100   ),
          }

    #Make latex line stringfication
    self._oline.append( ('%s & %s & %s & %s & %s & %s\\\\\n') % (bname.replace('_','\\_'),op['det'],op['sp'],\
                         op['fa'],ref['det'],ref['fa']) ) 
    #Concatenate all line tables
    line = str(); pos=0
    if self.switch: #Is operation (True)
      for l in self._oline: 
        line += l
        pos=1
      self.switch=False
    else: # (False)
      for l in self._tline: 
        line += l
        pos=0
      self.switch=True
 
