__all__ = ['CrossValidPlots']

from RingerCore import checkForUnusedVars, Logger, LoggingLevel, EnumStringification
from pprint import pprint
from ROOT import TGraph, TParameter, TCanvas, TColor, kBlue, kRed, kBlack, kCyan
import os, copy


def plot_curves(tpad, curves, y_limits, **kw):

  title         = kw.pop('title'   , ''         )
  xlabel        = kw.pop('xlabel'  , ''         )
  ylabel        = kw.pop('ylabel'  , ''         )
  bestIdx       = kw.pop('bestIdx' , None       )
  worstIdx      = kw.pop('worstIdx', None       )
  ccurves       = kw.pop('ccurves' , kCyan      )
  cbest         = kw.pop('cbest'   , kBlack     )
  cworst        = kw.pop('cworst'  , kRed       )

  #create dummy graph
  x_max = 0; dummy = None
  for i in range(len(curves)):
    curves[i].SetLineColor(ccurves)
    x = curves[i].GetXaxis().GetXmax()
    if x > x_max: x_max = x; dummy = curves[i]
  
  dummy.SetTitle( title )
  dummy.GetXaxis().SetTitle(xlabel)
  dummy.GetYaxis().SetTitle(ylabel)
  dummy.GetHistogram().SetAxisRange(y_limits[0],y_limits[1],'Y' )
  dummy.Draw('AL')

  #Plot curves
  for c in curves:  c.Draw('same')
  if bestIdx:   
    curves[bestIdx].SetLineColor(cbest)
    curves[bestIdx].Draw('same')
  if worstIdx:  
    curves[worstIdx].SetLineColor(cworst)
    curves[worstIdx].SetLineStyle(7)
    curves[worstIdx].Draw('same')
  
  tpad.Modified()
  tpad.Update()







class Reference(EnumStringification):
  SP  = 0,
  Pd  = 1,
  Pf  = 2

class CurveType(EnumStringification):
  Train  = 'trn',
  Val    = 'val',
  Test   = 'tst'


class CrossValidInfo(Logger):

  def __init__(self, benchmarkName, obj, **kw):
    Logger.__init__(self, kw) 
    self._summary         = obj
    self._benchmarkName= benchmarkName
    self._rawBenchmark = obj['rawBenchmark']
    for configName in obj.keys():
      if 'config_' in configName:
        for sortName in obj[configName].keys():
          if 'sort_' in sortName:
            summaryInfo = obj[configName][sortName]['summaryInfoTst']
            initSize = len(summaryInfo['idx'])
            self.add( configName, sortName, initSize, obj)
    self._logger.info(('Create CrossValidInfo with benchmark name: %s')\
                       %(self._benchmarkName))
    #pprint(self._summary)

  #Get benchmark label
  def name(self):
    return self._benchmarkName

  def add(self, configName, sortName, initSize, obj):
    self._summary[configName][sortName]['initSize'] = initSize

  def getSortInfo( self, configName, sortName, infoName):
    return self._summary[configName][sortName][infoName]

  def getConfigInfo(self, configName, infoName):
    return self._summary[configName][infoName]

  def getRawBenchmark(self):
    return self._rawBenchmark

  def reference(self):
    return Reference.fromstring(self._rawBenchmark['reference'])


  def getConfigBoundName(self):
    configBounds = []
    for key in self._summary:
      if 'config_' in key:  configBounds.append(key)
    configBounds.sort()
    return configBounds

  def getSortBoundName(self, configName):
    sortBounds = []
    for key in self._summary[configName]:
      if 'sort_' in key:  sortBounds.append(key)
    sortBounds.sort()
    return sortBounds

  def getInitBoundName(self, configName, sortName):
    initBounds = []
    for init in range(self._summary[configName][sortName]['initSize']):
      initBounds.append('init_'+str(init))
    return initBounds

  def getRawObj(self):
    return self._summary


#Helper class for plot
class PlotsHolder:

  _plot = dict()
 
  _paramNames = [
          'mse_stop',
          'sp_stop',
          'det_stop',
          'fa_stop',
        ]
  _graphNames = [
         'mse_trn',
         'mse_val',
         'mse_tst',
         'sp_val',
         'sp_tst',
         'det_val',
         'det_tst',
         'fa_val',
         'fa_tst',
         'det_fitted',
         'fa_fitted',
         'roc_tst',
         'roc_op',
         'roc_tst_cut',
         'roc_op_cut'
         ] 

  def __init__(self, store, fileList, best, worst):

    self._bestIdx  = best
    self._worstIdx = worst

    for fileName in fileList:
      for graphName in self._graphNames:
        #Check if key exist into plot holder dict
        if not graphName in self._plot.keys():  self._plot[graphName] = list()
        self.addGraph( store, fileName, graphName )
      #Loop over graphs
      for paramName in self._paramNames:
        if not paramName in self._plot.keys():  self._plot[paramName] = list()
        self.addParam(store, fileName, paramName )
    #Loop over file list


  def addGraph(self, store, fileName, graphName ):
    obj = TGraph()
    #print 'adding ',fileName
    store.GetObject( fileName+'/'+graphName, obj)
    self._plot[graphName].append( obj )
    
  def addParam(self, store, fileName, paramName ):
    obj = TParameter("double")()
    store.GetObject( fileName+'/'+paramName, obj)
    self._plot[paramName].append( int(obj.GetVal()) )

  def getBounds(self):
    return range( len(self._plot[self._paramNames[0]]) )

  def get(self, key, idx = None):
    if idx:  return self._plot[key][idx]
    else: return self._plot[key]

  def best(self, key):
    return self._plot[key][self._bestIdx]

  def worst(self, key):
    return self._plot[key][self._worstIdx]

  

class CrossValidPlots( Logger ):

  _info = list()

  def __init__(self, summaryFileName, monitoringFileName, **kw):
    Logger.__init__(self, kw) 

    from ROOT import TFile
    self._monitoring = TFile(monitoringFileName, 'read')

    from RingerCore import load
    summary = load(summaryFileName)
    for benchmark in summary.keys():
      if type(summary[benchmark]) is dict:
        self._info.append( CrossValidInfo(benchmark, summary[benchmark] )) 


  def __call__(self):
    self.loop()


    beamer.frame_section('Section 1')
    #beamer.frame_center_figure('test', '3.0in', 'test.pdf')

    t = LatexPerfTable('test')
    for i in self._info:
      t.add(i.name(), i.getConfigInfo('config_5', 'summaryInfoTst'), i.getRawBenchmark() )

    beamer.frame_table('test', 'test', t)
    beamer.save()

  def mkdir(self, directory):
    if not os.path.exists(directory):
      os.makedirs(directory)

  def loop(self): 
    
    beamer  §= BeamerMaker('test')
    table   = dict()
    figures = dict()


    #Loop over benchmarks
    for crossvalInfo in self._info:
      benchmarkName = crossvalInfo.name()
      ref = crossvalInfo.reference()
      for config in crossvalInfo.getConfigBoundName():
        plotSortList = list()
        currentPath =  ('figures/%s/%s')%(benchmarkName,config)
        self.mkdir( currentPath )
        for sort in crossvalInfo.getSortBoundName(config):

          paths = [('trainEvolution/%s/%s/%s/%s')%(benchmarkName,config,sort,init) \
            for init in crossvalInfo.getInitBoundName(config, sort)]

          #Get best and worst position for each sort
          bestIdx  = crossvalInfo.getSortInfo(config, sort, 'infoTstBest' )['init']
          worstIdx = crossvalInfo.getSortInfo(config, sort, 'infoTstWorst')['init']
          #Create plots holder class (Helper)
          plotSortList.append( PlotsHolder( self._monitoring, paths, bestIdx, worstIdx ) )
          
        #Loop over sort
        infoTstBest =  crossvalInfo.getConfigInfo(config, 'infoTstBest' )
        infoTstWorst = crossvalInfo.getConfigInfo(config, 'infoTstWorst') 
        cname = ('%s/plot_evol_%s_%s')%(currentPath,benchmarkName,config)
        #Plot train evolution for all sorts
        self.plotSorts( cname, plotSortList,CurveType.Val[0],ref,  infoTstBest, infoTstWorst)
      #Loop over config
    #Loop over benchmark


  def __bestInits(self, plotSortList, graphName):
    graphs = list()
    for plot in plotSortList:  graphs.append( plot.best(graphName) )
    return graphs


  def plotSorts(self, cname, obj, curveType, ref, infoBest, infoWorst):

    canvas = TCanvas('canvas', 'canvas', 2000, 1300)
    canvas.Divide(1,4) 
    
    mse_curves = self.__bestInits(obj, 'mse_val')
    sp_curves = self.__bestInits(obj, 'sp_'+ curveType)

    if ref is Reference.SP:
      det_curves = self.__bestInits(obj, 'det_'+ curveType)
      fa_curves  = self.__bestInits(obj, 'fa_' + curveType)
    elif ref is Reference.Pd:
      det_curves = self.__bestInits(obj, 'det_fitted'     )
      fa_curves  = self.__bestInits(obj, 'fa_' + curveType)
    elif ref is Reference.Pf:
      det_curves = self.__bestInits(obj, 'det_' + curveType)
      fa_curves  = self.__bestInits(obj, 'fa_fitted'       )
    
    plot_curves( canvas.cd(1), mse_curves,
                 [0.0, 1.0],
                 xlabel = 'Epoch',
                 ylabel = ('MSE (%s)')%(curveType),
                 bestIdx = infoBest['sort'],
                 worstIdx = infoWorst['sort'])

    plot_curves( canvas.cd(2), sp_curves,
                 [0.7, 1.0],
                 xlabel = 'Epoch',
                 ylabel = ('SP (%s)')%(curveType),
                 bestIdx = infoBest['sort'],
                 worstIdx = infoWorst['sort'])
 
    plot_curves( canvas.cd(3), det_curves,
                 [0.7, 1.0],
                 xlabel = 'Epoch',
                 ylabel = ('Detection (%s)')%(curveType) if ref is Reference.Pf else 'Detection (Reference)',
                 bestIdx = infoBest['sort'],
                 worstIdx = infoWorst['sort'])
 
    plot_curves( canvas.cd(4), fa_curves,
                 [0.0, 0.5],
                 xlabel = 'Epoch',
                 ylabel = ('False Alarm (%s)')%(curveType) if ref is Reference.Pd else 'False Alarm (Reference)',
                 bestIdx = infoBest['sort'],
                 worstIdx = infoWorst['sort'])
 
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(cname+'.pdf')




class LatexPerfTable:
  def __init__(self,  name):
    self._name = name
    self.table = dict()
  def name(self):
    return self._name

  def latex(self):
    table = str()
    for line in self.table:
      table+=self.table[line]
    return table
      
  def add(self, name, values, benchmark):
    refName = benchmark['reference']
    sgnRef  = benchmark['signal_efficiency']['efficiency']
    bkgRef  = benchmark['background_efficiency']['efficiency']
    color=['','','']
    if refName == 'Pd': color = ['\\cellcolor[HTML]{9AFF99}','','']
    elif refName == 'Pf': color = ['','','\\cellcolor[HTML]{BBDAFF}']

    val= {'name': name,
          'det' : ('%s%.2f$\\pm$%.2f')%(color[0],values['detMean']*100 ,values['detStd']*100 ),
          'sp'  : ('%s%.2f$\\pm$%.2f')%(color[1],values['spMean'] *100 ,values['spStd'] *100 ),
          'fa'  : ('%s%.2f$\\pm$%.2f')%(color[2],values['faMean'] *100 ,values['faStd'] *100 ) }

    ref  = {'name': benchmark['signal_efficiency']['name'].replace('Accept',''),
            'det' : ('%s%.2f')%(color[0],sgnRef  ),
            'sp'  : ('%s%.2f')%(color[1],0       ),
            'fa'  : ('%s%.2f')%(color[2],bkgRef ) }

    self.table[name] = ('%s & %s & %s & %s & %s & %s\\\\\n') % (name.replace('_','\\_'),val['det'],val['sp'],\
                        val['fa'],ref['det'],ref['fa']) 
    




class BeamerMaker:
  #Template
  _beginDocument = "\\documentclass{beamer}\n"+\
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

  _beginHeader = ("\\title[%s]{%s}\n\\author{%s}\n\\institute{%s}\n\date{\\today}\n")

  _beginTitlePage = \
           "\n"+\
           "\\begin{document}\n"+\
           "\n"+\
           "\\begin{frame}\n"+\
           "  \\titlepage\n"+\
           "\\end{frame}\n"

  _endDocument= "\end{document}"

  _line = "%--------------------------------------------------------------\n"

  def __init__(self, outputname, **kw):

    self.author    = kw.pop('author', 'jodafons@cern.ch' )
    self.title     = kw.pop('title' , 'TuningTools'      )
    self.institute = kw.pop('institute', 'Universidade Federal do Rio de Janeiro (UFRJ)')
    #Create output file
    self._output = open(outputname+'.tex', 'w')
    self._output.write(self._beginDocument)
    self._output.write( (self._beginHeader) % (self.title,self.title,self.author,self.institute) )
    self._output.write(self._beginTitlePage)

  def save(self):
    self._output.write(self._endDocument)
    self._output.close()

  #Set picture in center into beamer output file
  def frame_section(self, section):
    frame = self._line + ("\\section{%s}\n")%(section)+self._line
    self._output.write(frame)


  #Set picture in center into beamer output file
  def frame_center_figure(self, title, size, figure):
    frame = self._line +\
            "\\begin{frame}\n"+\
            ("\\frametitle{%s}\n")%(title) +\
            "\\begin{center}\n"+\
            ("\\includegraphics[height=%s]{%s}\n")%(size,figure)+\
            "\\end{center}\n"+\
            "\\end{frame}\n" + self._line
    self._output.write(frame)


  def frame_table(self, title, caption, table):

    frame = self._line +\
            "\\begin{frame}\n" +\
           ("\\frametitle{%s}\n") % (title) +\
            "\\begin{table}[h!]\\scriptsize\n" +\
            "\\begin{tabular}{l l l l l l}\n" +\
            "\\toprule\n" +\
            "\\textbf{benchmark} & DET [\%] & SP [\%] & FA [\%] & DET & FA\\\\\n" +\
            "\\midrule\n" +\
            table.latex() +\
            "\\bottomrule\n" +\
            "\\end{tabular}\n" +\
            ("\\caption{%s}\n")%(caption) +\
            "\\end{table}\n" +\
            "\\end{frame}\n"+\
            self._line
    self._output.write(frame)




