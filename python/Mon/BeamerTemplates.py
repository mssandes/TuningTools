#Beamer Templates
class BeamerConstants:

  beginDocument = "\\documentclass{beamer}\n"+\
           "% For more themes, color themes and font themes, see:\n"+\
           "\\mode<presentation>\n"+\
           "{\n"+\
           "  \\usetheme{Madrid}       % or try default, Darmstadt, Warsaw, ...\n"+\
           "  \\usecolortheme{default} % or try albatross, beaver, crane, ...\n"+\
           "  \\usefonttheme{serif}    % or try default, structurebold, ...\n"+\
           "  \\setbeamertemplate{navigation symbols}{}\n"+\
           "  \\setbeamertemplate{caption}[numbered]\n"+\
           "} \n"+\
           "\n"+\
           "\\usepackage[english]{babel}\n"+\
           "\\usepackage[utf8x]{inputenc}\n"+\
           "\\usepackage{chemfig}\n"+\
           "\\usepackage[version=3]{mhchem}\n"+\
           "\\usepackage{xcolor}\n"+\
           "\\usepackage{graphicx} % Allows including images\n"+\
           "\\usepackage{booktabs} % Allows the use of \\toprule, \midrule and \\bottomrule in tables\n"+\
           "%\usepackage[table,xcdraw]{xcolor}\n"+\
           "\\usepackage{colortbl}\n"+\
           "\n"+\
           "% On Overleaf, these lines give you sharper preview images.\n"+\
           "% You might want to comment them out before you export, though.\n"+\
           "\\usepackage{pgfpages}\n"+\
           "\\pgfpagesuselayout{resize to}[%\n"+\
           "  physical paper width=8in, physical paper height=6in]\n"+\
           "\n"+\
           "% Here's where the presentation starts, with the info for the title slide\n"

  beginHeader = ("\\title[%s]{%s}\n\\author{%s}\n\\institute{%s}\n\date{\\today}\n")

  beginTitlePage = \
           "\n"+\
           "\\begin{document}\n"+\
           "\n"+\
           "\\begin{frame}\n"+\
           "  \\titlepage\n"+\
           "\\end{frame}\n"

  endDocument= "\end{document}"

  line = "%--------------------------------------------------------------\n"


#Beamer slide for blocks
class BeamerBlocks:

  def __init__(self, frametitle, msgblocks):
    self._msgblocks = msgblocks
    self.frametitle = frametitle

  def tolatex(self, pfile):

    from BeamerTemplates import BeamerConstants as bconst
    strblock = str()
    for block in self._msgblocks:
      strblock += ('\\begin{block}{%s}\n') % (block[0]) +\
                  ('%s\n')%(block[1]) +\
                   '\\end{block}\n'

    frame = bconst.line +\
             "\\begin{frame}\n"+\
            ("\\frametitle{%s}\n")%(self.frametitle) +\
             strblock +\
             "\\end{frame}\n" + bconst.line
    pfile.write(frame)





#Beamer slide for only one center figure
class BeamerFigure:

  def __init__(self, figure, size, **kw):
    self.frametitle = kw.pop('frametitle', 'This is the title of your slide')
    self.caption = kw.pop('caption', 'This is the legend of the table' )
    self._size = size
    self._figure = figure

  def tolatex(self, pfile):
    from BeamerTemplates import BeamerConstants as bconst
    frame = bconst.line +\
            "\\begin{frame}\n"+\
            ("\\frametitle{%s}\n")%(self.frametitle.replace('_','\_')) +\
            "\\begin{center}\n"+\
            ("\\includegraphics[width=%s\\textwidth]{%s}\n")%(self._size,self._figure)+\
            "\\end{center}\n"+\
            "\\end{frame}\n" + bconst.line
    pfile.write(frame)


#Beamer slides for table
class BeamerPerfTables:

  def __init__(self, **kw):
    #Options to the frame
    self.frametitle = kw.pop('frametitle', ['This is the title of your slide','title 2'])
    self.caption = kw.pop('caption', ['This is the legend of the table','caption 2'] )
    self._tline = list()
    self._oline = list() 
    self.switch = False

  def tolatex(self, pfile, **kw):
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

    from BeamerTemplates import BeamerConstants as bconst

    frame = bconst.line +\
            "\\begin{frame}\n" +\
           ("\\frametitle{%s}\n") % (self.frametitle[pos].replace('_','\_')) +\
            "\\begin{table}[h!]\\scriptsize\n" +\
            "\\begin{tabular}{l l l l l l}\n" +\
            "\\toprule\n" +\
            "\\textbf{benchmark} & DET [\%] & SP [\%] & FA [\%] & DET & FA\\\\\n" +\
            "\\midrule\n" +\
            line +\
            "\\bottomrule\n" +\
            "\\end{tabular}\n" +\
            ("\\caption{%s}\n")%(self.caption[pos].replace('_','\_')) +\
            "\\end{table}\n" +\
            "\\end{frame}\n"+\
            bconst.line
    #Save into a txt file
    pfile.write(frame)
    

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
 

