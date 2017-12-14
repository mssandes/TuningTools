
__all__ = ['TuningMonitoringTool']

from RingerCore           import calcSP, save, load, Logger, mkdir_p, progressbar
from pprint               import pprint
import os

#Main class to plot and analyser the crossvalidStat object
#created by CrossValidStat class from tuningTool package
class TuningMonitoringTool( Logger ):
  """
  Main class to plot and analyser the crossvalidStat object
  created by CrossValidStat class from tuningTool package
  """  
  #Init class
  def __init__(self, crossvalFileName, monFileName, **kw):
    
    from ROOT import TFile, gROOT
    gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;");
    #Set all global setting from ROOT style plot!
    Logger.__init__(self, kw)
    #Hold all information abount the monitoring root file
    self._infoObjs = list()
    try:#Protection
      self._logger.info('Reading monRootFile (%s)',monFileName)
      self._rootObj = TFile(monFileName, 'read')
    except RuntimeError:
      self._logger.fatal('Could not open root monitoring file.')
    from RingerCore import load
    try:#Protection
      self._logger.info('Reading crossvalFile (%s)',crossvalFileName)
      crossvalObj = load(crossvalFileName)
    except RuntimeError:
      self._logger.fatal('Could not open pickle summary file.')
    #Loop over benchmarks

    from TuningTools.monitoring import Summary
    for benchmarkName in crossvalObj.keys():
      #Must skip if ppchain collector
      if benchmarkName == 'infoPPChain':  continue
      #Add summary information into MonTuningInfo helper class
      self._infoObjs.append( Summary( benchmarkName, crossvalObj[benchmarkName] ) ) 
      self._logger.info('Creating MonTuningInfo for %s and the iterator object [et=%d, eta=%d]',
                         benchmarkName,self._infoObjs[-1].etBinIdx(), self._infoObjs[-1].etaBinIdx())
    #Loop over all benchmarks

    # Eta bin
    self._etaBinIdx = self._infoObjs[0].etaBinIdx()
    # Et bin
    self._etBinIdx = self._infoObjs[0].etBinIdx()
    # Eta bin
    self._etaBin = self._infoObjs[0].etaBin()
    # Et bin
    self._etBin = self._infoObjs[0].etBin()


    #Reading the data rings from path or object
    dataPath = kw.pop('dataPath', None)
    if dataPath:
      from TuningTools import TuningDataArchieve
      self._logger.info(('Reading data tuning file with name %s')%(dataPath))
      TDArchieve = TuningDataArchieve.load(dataPath, etBinIdx = self._etBinIdx, etaBinIdx = self._etaBinIdx, 
                                           loadEfficiencies = False)
      self._data = (TDArchieve.signalPatterns, TDArchieve.backgroundPatterns)
    else:
      self._logger.warning('signal/backgorund patterns not passed...')


  def etBinIdx(self):
    return self._etBinIdx


  def etaBinIdx(self):
    return self._etaBinIdx


  def summary(self):
    summary=dict()
    for info in self._infoObjs:
      summary[info.name()]=info.summary()
    return summary


  #Main method to execute the monitoring 
  def __call__(self, **kw):
    
    import gc
    from TuningTools.monitoring import PlotObjects, Performance, PlotTrainingCurves, \
                                       PlotDiscriminants, PlotRocs, PlotInits
 
    #from scipy.io import loadmat

    output       = kw.pop('output'      , 'Mon'          ) 
    tuningReport = kw.pop('tuningReport', 'tuningReport' ) 
    doBeamer     = kw.pop('doBeamer'    , True           )
    shortSlides  = kw.pop('shortSlides' , False          )
    debug        = kw.pop('debug'       , False          )
    overwrite    = kw.pop('overwrite'   , False          )
    
    basepath=output
    basepath+=('_et%d_eta%d')%(self._etBinIdx,self._etaBinIdx)
    
    if not overwrite and os.path.isdir( basepath ):
      self._logger.warning("Monitoring output path already exists!")
      return 

    wantedPlotNames = { 'allBestTstSorts',
                        'allBestOpSorts',
                        'allWorstTstSorts', 
                        'allWorstOpSorts',
                        'allBestTstNeurons',
                        'allBestOpNeurons', 
                        'allWorstTstNeurons', 
                        'allWorstOpNeurons'} 

    perfBenchmarks = dict()
    pathBenchmarks = dict()

 
    # create strings 
    #binBounds = {}

    #if len(self._etBin) > 0 :
    #  binBounds['etbinstr'] = r'$%d < E_{T} \text{[Gev]}<%d$'%self._etBin
    #else:
    #  binBounds['etbinstr'] = r'\text{etBin[%d]}' % self._etBinIdx

    #if len(self._etaBin) > 0 :
    #  binBounds['etabinstr'] = r'$%.2f<\eta<%.2f$'%self._etaBin
    #else:
    #  binBounds['etabinstr'] = r'\text{etaBin[%d]}' % self._etaBinIdx


    #Loop over benchmarks
    for infoObj in self._infoObjs:
      
      # Initialize all plots
      plotObjects = dict()
      perfObjects = dict()
      infoObjects = dict()
      pathObjects = dict()
      
      # Init PlotsHolder 
      for plotname in wantedPlotNames:  
        plotObjects[plotname] = PlotObjects('Sort') if 'Sorts' in plotname else PlotObjects('Neuron')
      
      # Retrieve benchmark name
      benchmarkName = infoObj.name()
      # Retrieve reference name
      reference = infoObj.reference()
      # summary
      csummary = infoObj.summary()
      # benchmark object
      cbenchmark = infoObj.rawBenchmark()
      
      # reference value
      refVal = infoObj.rawBenchmark()['refVal']

      if infoObj.etaBinIdx() != self._etaBinIdx or infoObj.etBinIdx() != self._etBinIdx:
        self._logger.fatal("Benchmark dictionary is not compatible with the et/eta Indexs")

      self._logger.info(('Start loop over the benchmark: %s and etaBin = %d etBin = %d')%
          (benchmarkName,self._etaBinIdx, self._etBinIdx)  )
      import copy
       
      self._logger.info('Creating plots...')
      # Creating plots
      for neuron in progressbar(infoObj.neuronBounds(), len(infoObj.neuronBounds()), 'Loading : ', 60, False, logger=self._logger):
        # Figure path location
        currentPath =  '{}/figures/{}/{}'.format(basepath,benchmarkName,'neuron_'+str(neuron))
        # Config name 
        neuronName = 'config_'+str(neuron).zfill(3)
        # Create folder to store all plot objects
        mkdir_p(currentPath)
        #Clear all hold plots stored
        plotObjects['allBestTstSorts'].clear()
        plotObjects['allBestOpSorts'].clear()
        infoObjects['allInfoOpBest_'+neuronName] = list()
        #plotObjects['allWorstTstSorts'].clear()
        #plotObjects['allWorstOpSorts'].clear()

        self._logger.info( infoObj.sortBounds(neuron) )
        for sort in infoObj.sortBounds(neuron):

          sortName = 'sort_'+str(sort).zfill(3)
          #Init bounds 
          initBounds = infoObj.initBounds(neuron,sort)
          #Create path list from initBound list          
          initPaths = ['{}/{}/{}/init_{}'.format(benchmarkName,neuronName,sortName,init) for init in initBounds]
          self._logger.info('Creating init plots into the path: %s, (neuron_%s,sort_%s)', \
                              benchmarkName, neuron, sort)
          obj = PlotObjects('Init')
          try: #Create plots holder class (Helper), store all inits
            obj.retrieve(self._rootObj, initPaths)
          except RuntimeError:
            self._logger.fatal('Can not create plot holder object')
          #Hold all inits from current sort
          obj.setBoundValues(initBounds)
          obj.best = csummary[neuronName][sortName]['infoTstBest']['init']  
          obj.worst = csummary[neuronName][sortName]['infoTstWorst']['init'] 
          #PlotInits(obj,obj.best,obj.worst,reference=reference,
          #    outname='{}/plot_{}_neuron_{}_sorts_{}_mse_allInits_val.pdf'.format(currentPath,benchmarkName,neuron,sort))
          #PlotInits(obj,obj.best,obj.worst,reference=reference,
          #    outname='{}/plot_{}_neuron_{}_sorts_{}_det_allInits_val.pdf'.format(currentPath,benchmarkName,neuron,sort),key='det')
          #PlotInits(obj,obj.best,obj.worst,reference=reference,
          #    outname='{}/plot_{}_neuron_{}_sorts_{}_fa_allInits_val.pdf'.format(currentPath,benchmarkName,neuron,sort),key='fa')
          #PlotInits(obj,obj.best,obj.worst,reference=reference,
          #    outname='{}/plot_{}_neuron_{}_sorts_{}_sp_allInits_val.pdf'.format(currentPath,benchmarkName,neuron,sort),key='sp')

          plotObjects['allBestTstSorts'].append(  obj.getBestObject() )
          obj.best =  csummary[neuronName][sortName]['infoOpBest']['init']   
          obj.worst = csummary[neuronName][sortName]['infoOpWorst']['init']  
          #PlotInits(obj,obj.best,obj.worst,reference=reference,
          #    outname='{}/plot_{}_neuron_{}_sorts_{}_mse_allInits_op.pdf'.format(currentPath,benchmarkName,neuron,sort))
          #PlotInits(obj,obj.best,obj.worst,reference=reference,
          #    outname='{}/plot_{}_neuron_{}_sorts_{}_det_allInits_op.pdf'.format(currentPath,benchmarkName,neuron,sort),key='det')
          #PlotInits(obj,obj.best,obj.worst,reference=reference,
          #    outname='{}/plot_{}_neuron_{}_sorts_{}_fa_allInits_op.pdf'.format(currentPath,benchmarkName,neuron,sort),key='fa')
          #PlotInits(obj,obj.best,obj.worst,reference=reference,
          #    outname='{}/plot_{}_neuron_{}_sorts_{}_sp_allInits_op.pdf'.format(currentPath,benchmarkName,neuron,sort),key='sp')

          plotObjects['allBestOpSorts'].append( obj.getBestObject() )
          #plotObjects['allWorstTstSorts'].append( copy.deepcopy(tstObj.getBest() )
          #plotObjects['allWorstOpSorts'].append(  copy.deepcopy(opObj.getBest()  )
          infoObjects['allInfoOpBest_'+neuronName].append( copy.deepcopy(csummary[neuronName][sortName]['infoOpBest']) )
          #Release memory
          del obj

        #Loop over sorts
        gc.collect()
        
        plotObjects['allBestTstSorts'].setBoundValues(  infoObj.sortBounds(neuron) )
        plotObjects['allBestOpSorts'].setBoundValues(   infoObj.sortBounds(neuron) )
        #plotObjects['allWorstTstSorts'].setIdxCorrection( infoObj.sortBounds(neuron) )
        #plotObjects['allWorstOpSorts'].setIdxCorrection(  infoObj.sortBounds(neuron) )

        # Best and worst sorts for this neuron configuration
        plotObjects['allBestTstSorts'].best =   csummary[neuronName]['infoTstBest']['sort']  
        plotObjects['allBestTstSorts'].worst =  csummary[neuronName]['infoTstWorst']['sort'] 
        plotObjects['allBestOpSorts'].best =    csummary[neuronName]['infoOpBest']['sort']   
        plotObjects['allBestOpSorts'].worst =   csummary[neuronName]['infoOpWorst']['sort']  

        # Hold the information from the best and worst discriminator for this neuron 
        infoObjects['infoOpBest_'+neuronName] = copy.deepcopy(csummary[neuronName]['infoOpBest'])
        infoObjects['infoOpWorst_'+neuronName] = copy.deepcopy(csummary[neuronName]['infoOpWorst'])
 
        # Debug information
        self._logger.info(('Crossval indexs: (bestSort = %d, bestInit = %d) (worstSort = %d, bestInit = %d)')%\
              (plotObjects['allBestTstSorts'].best, plotObjects['allBestTstSorts'].getBestObject()['bestInit'],
               plotObjects['allBestTstSorts'].worst, plotObjects['allBestTstSorts'].getWorstObject()['bestInit']))
        
        self._logger.info(('Operation indexs: (bestSort = %d, bestInit = %d) (worstSort = %d, bestInit = %d)')%\
              (plotObjects['allBestOpSorts'].best, plotObjects['allBestOpSorts'].getBestObject()['bestInit'],
               plotObjects['allBestOpSorts'].worst, plotObjects['allBestOpSorts'].getWorstObject()['bestInit']))


        # Best and worst neuron sort for this configuration
        plotObjects['allBestTstNeurons'].append(  plotObjects['allBestTstSorts'].getBestObject()  )
        plotObjects['allBestOpNeurons'].append(   plotObjects['allBestOpSorts'].getBestObject()   )
        plotObjects['allWorstTstNeurons'].append( plotObjects['allBestTstSorts'].getWorstObject() )
        plotObjects['allWorstOpNeurons'].append(  plotObjects['allBestOpSorts'].getWorstObject()  )
        

        #NOTE: Hold all performance values to build the tables
        perfObjects[neuronName] = Performance( csummary[neuronName]['summaryInfoTst'],csummary[neuronName]['infoOpBest'],cbenchmark)
        #trnData, valData = self._crossValid(self._signalPatterns, sort)
        
        
        label = ('#splitline{#splitline{Total sorts: %d}{etaBin: %d, etBin: %d}}'+\
                 '{#splitline{sBestIdx: %d iBestIdx: %d}{sWorstIdx: %d iBestIdx: %d}}') % \
                  (plotObjects['allBestTstSorts'].size(),self._etaBinIdx, self._etBinIdx, plotObjects['allBestTstSorts'].best, \
                   plotObjects['allBestTstSorts'].getBestObject()['bestInit'], plotObjects['allBestTstSorts'].worst,\
                   plotObjects['allBestTstSorts'].getWorstObject()['bestInit'])
       
        #NOTE: plot all validation sorts for each criteria. The best color will be paint as blue and worst as red.
        fname1 = PlotTrainingCurves(plotObjects['allBestTstSorts'],
                                    outname= '{}/plot_{}_neuron_{}_sorts_val'.format(currentPath,benchmarkName,neuron),
                                    dataset = 'val',
                                    best = plotObjects['allBestTstSorts'].best, 
                                    worst = plotObjects['allBestTstSorts'].worst,
                                    label = label,
                                    refValue = refVal
                                    )
        

        label = ('#splitline{#splitline{Total sorts: %d}{etaBin: %d, etBin: %d}}'+\
                 '{#splitline{sBestIdx: %d iBestIdx: %d}{sWorstIdx: %d iBestIdx: %d}}') % \
                  (plotObjects['allBestOpSorts'].size(),self._etaBinIdx, self._etBinIdx, plotObjects['allBestOpSorts'].best, \
                   plotObjects['allBestOpSorts'].getBestObject()['bestInit'], plotObjects['allBestOpSorts'].worst,\
                   plotObjects['allBestOpSorts'].getWorstObject()['bestInit'])

        #NOTE: plot all operation (train+val+tst) curves for each criteria.
        fname2 = PlotTrainingCurves(plotObjects['allBestOpSorts'],
                                    outname= '{}/plot_{}_neuron_{}_sorts_op'.format(currentPath,benchmarkName,neuron),
                                    dataset = 'val',
                                    best = plotObjects['allBestOpSorts'].best, 
                                    worst = plotObjects['allBestOpSorts'].worst,
                                    label = label,
                                    refValue = refVal
                                    )


        #NOTE: Plot the dicriminant outputs for all sorts. The best sort will be fill as blue and
        # worst as red. 
        outname= '{}/plot_{}_neuron_{}_sorts'.format(currentPath,benchmarkName,neuron)
        fname3 = PlotDiscriminants(plotObjects['allBestOpSorts'],
                                  best = plotObjects['allBestOpSorts'].best, 
                                  worst = plotObjects['allBestOpSorts'].worst,
                                  outname = outname)

        
        #NOTE: plot the roc for all validation curves
        fname4  = PlotRocs(plotObjects['allBestOpSorts'],
                           best = plotObjects['allBestOpSorts'].best, 
                           worst = plotObjects['allBestOpSorts'].worst,
                           outname = outname)
        
        #NOTE: plot the roc for all operation curves
        fname5  = PlotRocs(plotObjects['allBestOpSorts'],
                           best = plotObjects['allBestOpSorts'].best, 
                           worst = plotObjects['allBestOpSorts'].worst,
                           outname = outname,
                           key='roc_tst')



        # the path objects holder
        #pathObjects['neuron_'+str(neuron)+'_sorts_val']      = fname1 
        #pathObjects['neuron_'+str(neuron)+'_sort_op']        = fname2
        #pathObjects['neuron_'+str(neuron)+'_best_op']        = fname3
        #pathObjects['neuron_'+str(neuron)+'_best_op_output'] = pname4
        #pathObjects['neuron_'+str(neuron)+'_sorts_roc_tst']  = pname5
        #pathObjects['neuron_'+str(neuron)+'_sorts_roc_op']   = pname6

 
      #Loop over neurons

      pathBenchmarks[benchmarkName]  = pathObjects
      perfBenchmarks[benchmarkName]  = perfObjects
      
     
      #Release memory
      for xname in plotObjects.keys():
        del plotObjects[xname]

      gc.collect()
    #Loop over benchmark
          
    

    #Start beamer presentation
    if doBeamer:

      from copy import copy
      from RingerCore.tex.TexAPI import *
      from RingerCore.tex.BeamerAPI import *
      collect=[]
      title = ('Tuning Monitoring (et=%d,eta=%d)')%(self.etBinIdx(), self.etaBinIdx())
      output = ('tuningMonitoring_et_%d_eta_%d')%(self.etBinIdx(), self.etaBinIdx())
      # apply beamer
      with BeamerTexReportTemplate2( theme = 'Berlin'
                             , _toPDF = True
                             , title = title
                             , outputFile = output
                             , font = 'structurebold' ):

        for neuron in self._infoObjs[0].neuronBounds():
          
          with BeamerSection( name = 'Neuron {}'.format(neuron) ):

            neuronName = 'config_'+str(neuron).zfill(3)

            for obj in self._infoObjs:
              with BeamerSubSection (name= obj.name().replace('_','\_')):
                
                currentPath =  '{}/figures/{}/{}'.format(basepath,obj.name(),'neuron_'+str(neuron))
                
                with BeamerSubSubSection (name='Training Curves'):
                  outname= '{}/plot_{}_neuron_{}_sorts_val'.format(currentPath,obj.name(),neuron)
                  paths = [
                            outname+'_mse.pdf',
                            outname+'_sp.pdf',
                            outname+'_det.pdf',
                            outname+'_fa.pdf',
                          ]
                  #with BeamerSlide( title = "Crossvalidatixon table"  ):
                  BeamerMultiFigureSlide( title = 'All Sorts (Validation) for Each Criteria'
                      , paths = paths
                      , nDivWidth = 2  # x
                      , nDivHeight = 2 # y
                      , texts=None
                      , fortran = False
                      , usedHeight = 0.65  # altura
                      , usedWidth = 0.95 # lasgura
                      )
                  # each bench

                with BeamerSubSubSection (name='ROC Curves'):
                  outname= '{}/plot_{}_neuron_{}_sorts_roc'.format(currentPath,obj.name(),neuron)
                  paths = [
                            outname+'_operation.pdf',
                            outname+'_tst.pdf',
                          ]
                  #with BeamerSlide( title = "Crossvalidatixon table"  ):
                  BeamerMultiFigureSlide( title = 'All ROC Sorts (Validation) and Operation'
                      , paths = paths
                      , nDivWidth = 2 # x
                      , nDivHeight = 1 # y
                      , texts=None
                      , fortran = False
                      , usedHeight = 0.6  # altura
                      , usedWidth = 0.9 # lasgura
                      )
                  # each bench

                with BeamerSubSubSection (name='Distributions (Discriminant)'):
                  outname= '{}/plot_{}_neuron_{}_sorts'.format(currentPath,obj.name(),neuron)
                  paths = [
                            outname+'_sgn_dist.pdf',
                            outname+'_bkg_dist.pdf',
                            outname+'_both_best_dist.pdf',
                          ]
                  #with BeamerSlide( title = "Crossvalidatixon table"  ):
                  BeamerMultiFigureSlide( title = 'Districiminant Distributions'
                      , paths = paths
                      , nDivWidth = 2 # x
                      , nDivHeight = 2 # y
                      , texts=None
                      , fortran = False
                      , usedHeight = 0.9  # altura
                      , usedWidth = 0.9 # lasgura
                      )
                  # each bench


            with BeamerSubSection (name='Summary'):
              lines1 = []
              lines1 += [ HLine(_contextManaged = False) ]
              lines1 += [ HLine(_contextManaged = False) ]
              lines1 += [ TableLine(    columns = ['Criteria', 'Pd []','SP []', 'Fa []'], _contextManaged = False ) ]
              lines1 += [ HLine(_contextManaged = False) ]
              lines2 = copy(lines1)

              for obj in self._infoObjs:
              
                perf = perfBenchmarks[obj.name()]['config_{}'.format(str(neuron).zfill(3))]
                # Crossvalidation values with error bar
                c1 = '\\cellcolor[HTML]{9AFF99}' if 'Pd' in obj.name() else ''
                c2 = '\\cellcolor[HTML]{BBDAFF}' if 'Pf' in obj.name() else ''
                lines1 += [ TableLine(    columns = [obj.name().replace('_','\_'), 
                                                     (c1+'%1.2f $\\pm$%1.2f')% (perf.getValue('detMean'),perf.getValue('detStd')),
                                                     ('%1.2f $\\pm$%1.2f')% (perf.getValue('spMean'),perf.getValue('spStd')),
                                                     (c2+'%1.2f $\\pm$%1.2f')% (perf.getValue('faMean'),perf.getValue('faStd')),
                                                     ],
                                      _contextManaged = False ) ]
                lines1 += [ HLine(_contextManaged = False) ]

                # Operation values
                lines2 += [ TableLine(    columns = [obj.name().replace('_','\_'), 
                                                     ('%1.2f')% (perf.getValue('det')),
                                                     ('%1.2f')% (perf.getValue('sp')),
                                                     ('%1.2f')% (perf.getValue('fa')),
                                                     ],
                                      _contextManaged = False ) ]
                lines2 += [ HLine(_contextManaged = False) ]

              lines1 += [ TableLine(    columns = ['References', 
                                                   ('\\cellcolor[HTML]{9AFF99}%1.2f')% (perf.getValue('det_target')),
                                                   ('%1.2f')% (perf.getValue('sp')),
                                                   ('\\cellcolor[HTML]{BBDAFF}%1.2f')% (perf.getValue('fa_target')),
                                                   ],
                                    _contextManaged = False ) ]
              lines1 += [ HLine(_contextManaged = False) ]

              with BeamerSlide( title = "Crossvalidation table ("+str(neuron)+")"  ):
                
                with Table( caption = 'Cross validation efficiencies for validation set.') as table:
                  with ResizeBox( size =  1.) as rb:
                    with Tabular( columns = 'l' + 'c' * 4) as tabular:
                      tabular = tabular
                      for line in lines1:
                        if isinstance(line, TableLine):
                          tabular += line
                        else:
                          TableLine(line, rounding = None)
 
                with Table( caption = 'Operation efficiencies for the best model.') as table:
                  with ResizeBox( size =  0.7) as rb:
                    with Tabular( columns = 'l' + 'c' * 4) as tabular:
                      tabular = tabular
                      for line in lines2:
                        if isinstance(line, TableLine):
                          tabular += line
                        else:
                          TableLine(line, rounding = None)
 




    self._logger.info('Done! ')

  #End of loop()





