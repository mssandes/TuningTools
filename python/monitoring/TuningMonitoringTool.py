#Author: Joao Victo da Fonseca Pinto
#Analysis framework

__all__ = ['TuningMonitoringTool']

#Import necessary classes
from TuningMonitoringInfo import TuningMonitoringInfo
from TuningStyle          import SetTuningStyle
from RingerCore           import calcSP, save, load, Logger, mkdir_p
from pprint               import pprint
import os

#Main class to plot and analyser the crossvalidStat object
#created by CrossValidStat class from tuningTool package
class TuningMonitoringTool( Logger ):
  """
  Main class to plot and analyser the crossvalidStat object
  created by CrossValidStat class from tuningTool package
  """  
  #Hold all information abount the monitoring root file
  _infoObjs = list()
  #Init class
  def __init__(self, crossvalFileName, monFileName, **kw):
    from ROOT import TFile
    #Set all global setting from ROOT style plot!
    SetTuningStyle()
    Logger.__init__(self, kw)

    try:#Protection
      self._logger.info('Reading monRootFile (%s)',monFileName)
      self._rootObj = TFile(monFileName, 'read')
    except RuntimeError:
      raise RuntimeError('Could not open root monitoring file.')
    from RingerCore import load
    try:#Protection
      self._logger.info('Reading crossvalFile (%s)',crossvalFileName)
      crossvalObj = load(crossvalFileName)
    except RuntimeError:
      raise RuntimeError('Could not open pickle summary file.')
    #Loop over benchmarks

    for benchmarkName in crossvalObj.keys():
      #Must skip if ppchain collector
      if benchmarkName == 'infoPPChain':  continue

      #Add summary information into MonTuningInfo helper class
      self._logger.info('Creating MonTuningInfo for %s and the iterator object',benchmarkName)
      self._infoObjs.append( TuningMonitoringInfo( benchmarkName, crossvalObj[benchmarkName] ) ) 
    #Loop over all benchmarks

    #Reading the data rings from path or object
    perfFile = kw.pop('perfFile', None)
    if perfFile:
      if type(perfFile) is str:
        from TuningTools import TuningDataArchieve
        TDArchieve = TuningDataArchieve(perfFile)
        self._logger.info(('Reading perf file with name %s')%(perfFile))
        try:
          with TDArchieve as data:
            #Always be the same bin for all infoObjs  
            etabin = self._infoObjs[0].etabin()
            etbin = self._infoObjs[0].etbin()
            self._data = (data['signal_patterns'][etbin][etabin], data['background_patterns'][etbin][etabin])
        except RuntimeError:
          raise RuntimeError('Could not open the patterns data file.')
      else:
        self._data = None


  #Main method to execute the monitoring 
  def __call__(self, **kw):
    """
      Call the Monitoring tool analysis, the args can be:
        basePath: holt the location where all plots and files will
                  be saved. (defalt is Mon/)
        doBeamer: Start beamer Latex presentation maker (Default is True)
        shortSliedes: Presentation only with tables (Default is False)
    """
    self.loop(**kw)

  #Loop over 
  def loop(self, **kw): 

    basepath    = kw.pop('basePath', 'Mon') 
    tuningReport= kw.pop('tuningReport', 'tuningReport') 
    doBeamer    = kw.pop('doBeamer', True)
    shortSlides = kw.pop('shortSlides', False)

    if shortSlides:
      self._logger.warning('Short slides enabled! Doing only tables...')

    wantedPlotNames = {'allBestTstSorts','allBestOpSorts','allWorstTstSorts', 'allWorstOpSorts',\
                       'allBestTstNeurons','allBestOpNeurons', 'allWorstTstNeurons', 'allWorstOpNeurons'} 

    perfBenchmarks = dict()
    pathBenchmarks = dict()

    from PlotHolder import PlotHolder
    from PlotHelper import plot_4c, plot_rocs, plot_nnoutput
    from TuningMonitoringInfo import MonitoringPerfInfo

    basepath+=('_et%d_eta%d')%(self._infoObjs[0].etbin(),self._infoObjs[0].etabin())
    
    #Loop over benchmarks
    for infoObj in self._infoObjs:
      #Initialize all plos
      plotObjects = dict()
      perfObjects = dict()
      pathObjects = dict()
      #Init PlotsHolder 
      for plotname in wantedPlotNames:  
        if 'Sorts' in plotname:
          plotObjects[plotname] = PlotHolder(label = 'Sort')
        else:
          plotObjects[plotname] = PlotHolder(label = 'Neuron')

      #Retrieve benchmark name
      benchmarkName = infoObj.name()
      #Retrieve reference name
      reference = infoObj.reference()
      #summary
      csummary = infoObj.summary()
      #benchmark object
      cbenchmark = infoObj.rawBenchmark()
      # reference value
      refVal = infoObj.rawBenchmark()['refVal']
      #Eta bin
      etabin = infoObj.etabin()
      #Et bin
      etbin = infoObj.etbin()


      self._logger.info(('Start loop over the benchmark: %s and etaBin = %d etBin = %d')%(benchmarkName,etabin, etbin)  )
      import copy
      #Loop over neuron, sort, inits. Creating plot objects
      for neuron, sort, inits in infoObj.iterator():
       
        sortName = 'sort_'+str(sort)
        #Create path list from initBound list          
        initPaths = [('%s/config_%s/sort_%s/init_%s')%(benchmarkName,\
                                                       neuron,sort,init) for init in inits]
        self._logger.debug('Creating init plots into the path: %s, (neuron_%s,sort_%s)', \
                            benchmarkName, neuron, sort)
        obj = PlotHolder(label = 'Init')
        try: #Create plots holder class (Helper), store all inits
          obj.retrieve(self._rootObj, initPaths)
        except RuntimeError:
          raise RuntimeError('Can not create plot holder object')
        #Hold all inits from current sort
        obj.set_index_correction(inits)

        neuronName = 'config_'+str(neuron);  sortName = 'sort_'+str(sort)
        obj.set_index_correction(inits)
        csummary[neuronName][sortName]['tstPlots'] = copy.deepcopy(obj)
        csummary[neuronName][sortName]['opPlots']  = copy.deepcopy(obj)

        # Hold all init plots objects
        csummary[neuronName][sortName]['tstPlots'].set_best_index( csummary[neuronName][sortName]['infoTstBest']['init'])
        csummary[neuronName][sortName]['tstPlots'].set_worst_index(csummary[neuronName][sortName]['infoTstWorst']['init'])
        csummary[neuronName][sortName]['opPlots' ].set_best_index( csummary[neuronName][sortName]['infoOpBest']['init'])
        csummary[neuronName][sortName]['opPlots' ].set_worst_index(csummary[neuronName][sortName]['infoOpWorst']['init'])
      #Loop over neuron, sort

      # Creating plots
      for neuron in infoObj.neuronBounds():

        # Figure path location
        currentPath =  ('%s/figures/%s/%s') % (basepath,benchmarkName,'neuron_'+str(neuron))
        neuronName = 'config_'+str(neuron)
        # Create folder to store all plot objects
        mkdir_p(currentPath)
        #Clear all hold plots stored
        plotObjects['allBestTstSorts'].clear()
        plotObjects['allBestOpSorts'].clear()
        #plotObjects['allWorstTstSorts'].clear()
        #plotObjects['allWorstOpSorts'].clear()

        for sort in infoObj.sortBounds(neuron):
          sortName = 'sort_'+str(sort)
          plotObjects['allBestTstSorts'].append(  copy.deepcopy(csummary[neuronName][sortName]['tstPlots'].get_best() ) )
          plotObjects['allBestOpSorts'].append(   copy.deepcopy(csummary[neuronName][sortName]['opPlots'].get_best()  ) )
          #plotObjects['allWorstTstSorts'].append( csummary[neuronName][sortName]['tstPlots'].getBest() )
          #plotObjects['allWorstOpSorts'].append(  csummary[neuronName][sortName]['opPlots'].getBest()  )
        #Loop over sorts
        
        plotObjects['allBestTstSorts'].set_index_correction(  infoObj.sortBounds(neuron) )
        plotObjects['allBestOpSorts'].set_index_correction(   infoObj.sortBounds(neuron) )
        #plotObjects['allWorstTstSorts'].setIdxCorrection( infoObj.sortBounds(neuron) )
        #plotObjects['allWorstOpSorts'].setIdxCorrection(  infoObj.sortBounds(neuron) )

        # Best and worst sorts for this neuron configuration
        plotObjects['allBestTstSorts'].set_best_index(  csummary[neuronName]['infoTstBest']['sort']  )
        plotObjects['allBestTstSorts'].set_worst_index( csummary[neuronName]['infoTstWorst']['sort'] )
        plotObjects['allBestOpSorts'].set_best_index(   csummary[neuronName]['infoOpBest']['sort']   )
        plotObjects['allBestOpSorts'].set_worst_index(  csummary[neuronName]['infoOpWorst']['sort']  )
  
        # Best and worst neuron sort for this configuration
        plotObjects['allBestTstNeurons'].append( copy.deepcopy(plotObjects['allBestTstSorts'].get_best()  ))
        plotObjects['allBestOpNeurons'].append(  copy.deepcopy(plotObjects['allBestOpSorts'].get_best()   ))
        plotObjects['allWorstTstNeurons'].append(copy.deepcopy(plotObjects['allBestTstSorts'].get_worst() ))
        plotObjects['allWorstOpNeurons'].append( copy.deepcopy(plotObjects['allBestOpSorts'].get_worst()  ))
        
        # Create perf (tables) Objects for test and operation (Table)
        perfObjects['neuron_'+str(neuron)] =  MonitoringPerfInfo(benchmarkName, reference, 
                                                                 csummary[neuronName]['summaryInfoTst'], 
                                                                 csummary[neuronName]['infoOpBest'], 
                                                                 cbenchmark) 

        # Debug information
        self._logger.info(('Crossval indexs: (bestSort = %d, bestInit = %d) (worstSort = %d, bestInit = %d)')%\
              (plotObjects['allBestTstSorts'].best, plotObjects['allBestTstSorts'].get_best()['bestInit'],
               plotObjects['allBestTstSorts'].worst, plotObjects['allBestTstSorts'].get_worst()['bestInit']))
        self._logger.info(('Operation indexs: (bestSort = %d, bestInit = %d) (worstSort = %d, bestInit = %d)')%\
              (plotObjects['allBestOpSorts'].best, plotObjects['allBestOpSorts'].get_best()['bestInit'],
               plotObjects['allBestOpSorts'].worst, plotObjects['allBestOpSorts'].get_worst()['bestInit']))

        args = dict()
        args['reference'] = reference
        args['refVal']    = refVal
        args['eps']       = cbenchmark['eps']
       
        # Figure 1: Plot all validation/test curves for all crossval sorts tested during
        # the training. The best sort will be painted with black and the worst sort will
        # be on red color. There is a label that will be draw into the figure to show 
        # the current location (neuron, sort, init) of the best and the worst network.
        args['label']     = ('#splitline{#splitline{Total sorts: %d}{etaBin: %d, etBin: %d}}'+\
                             '{#splitline{sBestIdx: %d iBestIdx: %d}{sWorstIdx: %d iBestIdx: %d}}') % \
                            (plotObjects['allBestTstSorts'].size(),etabin, etbin, plotObjects['allBestTstSorts'].best, \
                             plotObjects['allBestTstSorts'].get_best()['bestInit'], plotObjects['allBestTstSorts'].worst,\
                             plotObjects['allBestTstSorts'].get_worst()['bestInit'])

        args['cname']        = ('%s/plot_%s_neuron_%s_sorts_val')%(currentPath,benchmarkName,neuron)
        args['set']          = 'val'
        args['operation']    = False
        args['paintListIdx'] = [plotObjects['allBestTstSorts'].best, plotObjects['allBestTstSorts'].worst]
        pname1 = plot_4c(plotObjects['allBestTstSorts'], args)

        # Figure 2: Plot all validation/test curves for all crossval sorts tested during
        # the training. The best sort will be painted with black and the worst sort will
        # be on red color. But, here the painted curves represented the best and the worst
        # curve from the operation dataset. In other words, we pass all events into the 
        # network and get the efficiencis than we choose the best operation and the worst 
        # operation network and paint the validation curve who represent these sorts.
        # There is a label that will be draw into the figure to show 
        # the current location (neuron, sort, init) of the best and the worst network.
        args['label']     = ('#splitline{#splitline{Total sorts: %d (operation)}{etaBin: %d, etBin: %d}}'+\
                            '{#splitline{sBestIdx: %d iBestIdx: %d}{sWorstIdx: %d iBestIdx: %d}}') % \
                           (plotObjects['allBestOpSorts'].size(),etabin, etbin, plotObjects['allBestOpSorts'].best, \
                            plotObjects['allBestOpSorts'].get_best()['bestInit'], plotObjects['allBestOpSorts'].worst,\
                            plotObjects['allBestOpSorts'].get_worst()['bestInit'])
        args['cname']        = ('%s/plot_%s_neuron_%s_sorts_op')%(currentPath,benchmarkName,neuron)
        args['set']          = 'val'
        args['operation']    = True
        args['paintListIdx'] = [plotObjects['allBestOpSorts'].best, plotObjects['allBestOpSorts'].worst]
        pname2 = plot_4c(plotObjects['allBestOpSorts'], args)

        # Figure 3: This figure show us in deteails the best operation network for the current hidden
        # layer and benchmark analysis. Depend on the benchmark, we draw lines who represents the 
        # stops for each curve. The current neuron will be the last position of the plotObjects
        splotObject = PlotHolder()
        opt['label']     = ('#splitline{#splitline{Best network neuron: %d}{etaBin: %d, etBin: %d}}'+\
                            '{#splitline{{sBestIdx: %d iBestIdx: %d}{}}') % \
                           (neuron,etabin, etbin, plotObjects['allBestOpSorts'].best, plotObjects['allBestOpSorts'].get_best()['bestInit'])
        args['cname']     = ('%s/plot_%s_neuron_%s_best_op')%(currentPath,benchmarkName,neuron)
        args['set']       = 'val'
        args['operation'] = True
        splotObject.append( plotObjects['allBestOpNeurons'][-1] )
        pname3 = plot_4c(splotObject, args)
        
        
        # Figure 4: Here, we have a plot of the discriminator output for all dataset. Black histogram
        # represents the signal and the red onces represent the background. TODO: Apply this outputs
        # using the feedfoward manual method to generate the network outputs and create the histograms.
        args['cname']     = ('%s/plot_%s_neuron_%s_best_op_output')%(currentPath,benchmarkName,neuron)
        args['nsignal']   = self._data[0].shape[0]
        args['nbackground'] = self._data[1].shape[0]
        args['rocname'] = 'roc_op'
        pname4 = plot_nnoutput(splotObject,args)
   
        # Figure 5: The receive operation test curve for all sorts using the test dataset as base.
        # Here, we will draw the current tunnel and ref value used to set the discriminator threshold
        # when the bechmark are Pd or Pf case. When we use the SP case, this tunnel will not be ploted.
        # The black curve represents the best sort and the red onces the worst sort. TODO: Put the SP
        # point for the best and worst when the benchmark case is SP.
        args['cname']        = ('%s/plot_%s_neuron_%s_sorts_roc_tst')%(currentPath,benchmarkName,neuron)
        args['set']          = 'tst'
        args['paintListIdx'] = [plotObjects['allBestTstSorts'].best, plotObjects['allBestTstSorts'].worst]
        pname5 = plot_rocs(plotObjects['allBestTstSorts'], args)

        # Figure 6: The receive operation  curve for all sorts using the operation dataset (train+test) as base.
        # Here, we will draw the current tunnel and ref value used to set the discriminator threshold
        # when the bechmark are Pd or Pf case. When we use the SP case, this tunnel will not be ploted.
        # The black curve represents the best sort and the red onces the worst sort. TODO: Put the SP
        # point for the best and worst when the benchmark case is SP.
        args['cname']        = ('%s/plot_%s_neuron_%s_sorts_roc_op')%(currentPath,benchmarkName,neuron)
        args['set']          = 'op'
        args['paintListIdx'] = [plotObjects['allBestOpSorts'].best, plotObjects['allBestOpSorts'].worst]
        pname6 = plot_rocs(plotObjects['allBestOpSorts'], args)



        # Map names for beamer, if you add a plot, you must add into
        # the path objects holder
        pathObjects['neuron_'+str(neuron)+'_sorts_val']      = pname1 
        pathObjects['neuron_'+str(neuron)+'_sort_op']        = pname2
        pathObjects['neuron_'+str(neuron)+'_best_op']        = pname3
        pathObjects['neuron_'+str(neuron)+'_best_op_output'] = pname4
        pathObjects['neuron_'+str(neuron)+'_sorts_roc_tst']  = pname5
        pathObjects['neuron_'+str(neuron)+'_sorts_roc_op']   = pname6
  
      #Loop over neurons

      #Start individual operation plots
      #plot(plotObjects['neuronTstBest'], opt)

      #External 
      pathBenchmarks[benchmarkName]  = pathObjects
      perfBenchmarks[benchmarkName]  = perfObjects
    #Loop over benchmark


    #Start beamer presentation
    if doBeamer:
      from BeamerMonReport import BeamerMonReport
      from BeamerTemplates import BeamerPerfTables, BeamerFigure, BeamerBlocks
      #Eta bin
      etabin = self._infoObjs[0].etabin()
      #Et bin
      etbin = self._infoObjs[0].etbin()
      #Create the beamer manager
      beamer = BeamerMonReport(basepath+'/'+tuningReport, title = ('Tuning Report (et=%d, eta=%d)')%(etbin,etabin) )
      neuronBounds = self._infoObjs[0].neuronBounds()

      for neuron in neuronBounds:
        #Make the tables for crossvalidation
        ptableCross = BeamerPerfTables(frametitle= ['Neuron '+str(neuron)+': Cross Validation Performance',
                                                    'Neuron '+str(neuron)+": Operation Best Network"],
                                       caption=['Efficiencies from each benchmark.',
                                                'Efficiencies for the best operation network'])

        block = BeamerBlocks('Neuron '+str(neuron)+' Analysis', [('All sorts (validation)','All sorts evolution are ploted, each sort represents the best init;'),
                                                                 ('All sorts (operation)', 'All sorts evolution only for operation set;'),
                                                                 ('Best operation', 'Detailed analysis from the best sort discriminator.'),
                                                                 ('Tables','Cross validation performance')])
        if not shortSlides:  block.tolatex( beamer.file() )

        for info in self._infoObjs:
          #If we produce a short presentation, we do not draw all plots
          if not shortSlides:  
            bname = info.name().replace('OperationPoint_','')
            fig1 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_sorts_val'].replace(basepath+'/',''), 0.7,
                               frametitle=bname+', Neuron '+str(neuron)+': All sorts (validation)') 
            fig2 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_sorts_roc_val'].replace(basepath+'/',''), 0.8,
                               frametitle=bname+', Neuron '+str(neuron)+': All ROC sorts (validation)') 
            fig3 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_sort_op'].replace(basepath+'/',''), 0.7, 
                               frametitle=bname+', Neuron '+str(neuron)+': All sorts (operation)') 
            fig4 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_best_op'].replace(basepath+'/',''), 0.7,
                               frametitle=bname+', Neuron '+str(neuron)+': Best Network') 
            fig5 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_best_op_output'].replace(basepath+'/',''), 0.8,
                               frametitle=bname+', Neuron '+str(neuron)+': Best Network output') 
            
          
            #Draw figures into the tex file
            fig1.tolatex( beamer.file() )
            fig2.tolatex( beamer.file() )
            fig3.tolatex( beamer.file() )
            fig4.tolatex( beamer.file() )
            fig5.tolatex( beamer.file() )

          #Concatenate performance table, each line will be a benchmark
          #e.g: det, sp and fa
          ptableCross.add( perfBenchmarks[info.name()]['neuron_'+str(neuron)] ) 

        ptableCross.tolatex( beamer.file() )# internal switch is false to true: test
        ptableCross.tolatex( beamer.file() )# internal swotch is true to false: operation

      beamer.close()

  #End of loop()





