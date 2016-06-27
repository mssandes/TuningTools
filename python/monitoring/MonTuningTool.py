#Author: Joao Victo da Fonseca Pinto
#Analysis framework

__all__ = ['MonTuningTool']

#Import necessary classes
from MonTuningInfo import MonTuningInfo
from TuningStyle import SetTuningStyle
from pprint        import pprint
from RingerCore    import calcSP, save, load, Logger, mkdir_p


import os
#This function will apply a correction index
def fix_position( vec, idx ):
  """
  This function will apply a correction index
  """    
  return vec.index( idx )


#Main class to plot and analyser the crossvalidStat object
#created by CrossValidStat class from tuningTool package
class MonTuningTool( Logger ):
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
      raise RuntimeError('Could not open root monitoring file')
    from RingerCore import load
    try:#Protection
      self._logger.info('Reading crossvalFile (%s)',crossvalFileName)
      crossvalObj = load(crossvalFileName)
    except RuntimeError:
      raise RuntimeError('Could not open pickle summary file')
    #Loop over benchmarks
    for benchmarkName in crossvalObj.keys():
      #Must skip if ppchain collector
      if benchmarkName == 'infoPPChain':  continue
      #Add summary information into MonTuningInfo helper class
      self._logger.info('Creating MonTuningInfo for %s and the iterator object',benchmarkName)
      self._infoObjs.append( MonTuningInfo(benchmarkName, crossvalObj[benchmarkName] ) ) 
    #Loop over all benchmarks

    #Reading the data rings from path or object
    perfFile = kw.pop('perfFile', None)
    if perfFile:
      if type(perfFile) is str:
        from TuningTools import TuningDataArchieve
        TDArchieve = TuningDataArchieve(perfFile)
        self._logger.info(('Reading perf file with name %s')%(perfFile))
        with TDArchieve as data:
          #Always be the same bin for all infoObjs  
          etabin = self._infoObjs[0].etabin()
          etbin = self._infoObjs[0].etbin()
          self._data = (data['signal_patterns'][etbin][etabin], data['background_patterns'][etbin][etabin])
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

    plotNamesWanted = {'allBestTstSorts','allBestOpSorts',\
                       'allBestTstNeurons','allBestOpNeurons'} 

    perfBenchmarks = dict()
    pathBenchmarks = dict()

    from PlotHelper import PlotsHolder, plot_4c
    from MonTuningInfo import MonPerfInfo

    #Loop over benchmarks
    for infoObj in self._infoObjs:
      #Initialize all plos
      plotObjects = dict()
      perfObjects = dict()
      pathObjects = dict()

      #Init PlotsHolder 
      for plotname in plotNamesWanted:  plotObjects[plotname] = PlotsHolder()
      #Retrieve benchmark name
      benchmarkName = infoObj.name()
      #Retrieve reference name
      reference = infoObj.reference()
      #summary
      csummary = infoObj.summary()
      #benchmark object
      cbenchmark = infoObj.rawBenchmark()
      #Eta bin
      etabin = infoObj.etabin()
      #Et bin
      etbin = infoObj.etbin()

      self._logger.info(('Star loop for benchmark: %s and etaBin = %d etBin = %d')%(benchmarkName,etabin, etbin)  )
      #Loop over neuron, sort, inits. Creating plot objects
      for neuron, sort, inits in infoObj.iterator():
        #Create path list from initBound list          
        initPaths = [('%s/config_%s/sort_%s/init_%s')%(benchmarkName,\
                                                       neuron,sort,init) for init in inits]
        self._logger.debug('Creating init plots into the path: %s, (neuron_%s,sort_%s)', \
                            benchmarkName, neuron, sort)
        obj = PlotsHolder()
        try: #Create plots holder class (Helper), store all inits
          obj.retrieve(self._rootObj, initPaths)
        except RuntimeError:
          raise RuntimeError('Can not create plot holder object')
        #Hold all inits from current sort
        neuronName = 'config_'+str(neuron);  sortName = 'sort_'+str(sort)
        csummary[neuronName][sortName]['plots'] = obj
      #Loop over neuron, sort


      # Creating plots
      for neuron in infoObj.neuronBounds():
        # Hold the init index
        idxDict = dict()
        # Figure path location
        currentPath =  ('%s/figures/%s/%s') % (basepath,benchmarkName,'neuron_'+str(neuron))
        neuronName = 'config_'+str(neuron)
        # Create folder to store all plot objects
        mkdir_p(currentPath)
        #Clear all hold plots stored
        plotObjects['allBestTstSorts'].clear()
        plotObjects['allBestOpSorts'].clear()

        for sort in infoObj.sortBounds(neuron):
          sortName = 'sort_'+str(sort)
          ivec = infoObj.initBounds(neuron,sort)
          #Retrieve best init index from test and operation values
          bestTstInitIdx = fix_position(ivec, csummary[neuronName][sortName]['infoTstBest']['init'])
          bestOpInitIdx =  fix_position(ivec, csummary[neuronName][sortName]['infoOpBest']['init'] )
          #Retrieve plot templates
          initPlots = csummary[neuronName][sortName]['plots']
          #Get the best init for each sort
          plotObjects['allBestTstSorts'].append( initPlots.getObj(bestTstInitIdx) )
          plotObjects['allBestOpSorts'].append( initPlots.getObj(bestOpInitIdx) )
          #Hold the positions
          idxDict[sortName] = {'bestTstInitIdx':bestTstInitIdx, 'bestOpInitIdx':bestOpInitIdx}
        #Loop over sorts

        # Best and worst sorts for this neuron configuration
        idxDict['tstBestSortIdx']  = csummary[neuronName]['infoTstBest']['sort']
        idxDict['tstWorstSortIdx'] = csummary[neuronName]['infoTstWorst']['sort']
        idxDict['opBestSortIdx']   = csummary[neuronName]['infoOpBest']['sort']
        idxDict['opWorstSortIdx']  = csummary[neuronName]['infoOpWorst']['sort']
      
        # Get the best test network train object
        sortName = 'sort_'+str(idxDict['tstBestSortIdx'])
        initPlots = csummary[neuronName][sortName]['plots']
        plotObjects['allBestTstNeurons'].append( initPlots.getObj(idxDict[sortName]['bestTstInitIdx']) )
        
        # Get the best operation network train object
        sortName = 'sort_'+str(idxDict['opBestSortIdx'])
        initPlots = csummary[neuronName][sortName]['plots']
        plotObjects['allBestOpNeurons'].append( initPlots.getObj(idxDict[sortName]['bestOpInitIdx']) )
 

        # Create perf (tables) Objects for test and operation (Table)
        perfObjects['neuron_'+str(neuron)] =  MonPerfInfo(benchmarkName, reference, 
                                                          csummary[neuronName]['summaryInfoTst'], 
                                                          csummary[neuronName]['infoOpBest'], 
                                                          cbenchmark) 
        opt = dict()
        opt['reference'] = reference
        svec = infoObj.sortBounds(neuron)
        
        #Configuration of each sort val plot: (Figure 1)
        opt['label']     = ('#splitline{Sorts: %d}{etaBin: %d, etBin: %d}') % (plotObjects['allBestTstSorts'].size(),etabin, etbin)
        opt['cname']     = ('%s/plot_%s_neuron_%s_sorts_val')%(currentPath,benchmarkName,neuron)
        opt['set']       = 'val'
        opt['operation'] = False
        opt['paintListIdx'] = [fix_position(svec, idxDict['tstBestSortIdx']), fix_position(svec,  idxDict['tstWorstSortIdx'] )]
        pname1 = plot_4c(plotObjects['allBestTstSorts'], opt)

        # Configuration of each sort operation plot: (Figure 2)
        opt['label']     = ('#splitline{Sorts: %d (Operation)}{etaBin: %d, etBin: %d}') % (plotObjects['allBestOpSorts'].size(),etabin, etbin)
        opt['cname']     = ('%s/plot_%s_neuron_%s_sorts_op')%(currentPath,benchmarkName,neuron)
        opt['set']       = 'val'
        opt['operation'] = True
        opt['paintListIdx'] = [fix_position(svec, idxDict['opBestSortIdx']), fix_position(svec,  idxDict['opWorstSortIdx'] )]
        pname2 = plot_4c(plotObjects['allBestOpSorts'], opt)

        # Configuration of each best val network plot: (Figure 3)
        opt['label']     = ('#splitline{Best network, neuron: %d}{etaBin: %d, etBin: %d}') % (neuron,etabin, etbin)
        opt['cname']     = ('%s/plot_%s_neuron_%s_best_op')%(currentPath,benchmarkName,neuron)
        opt['set']       = 'val'
        opt['operation'] = True
        splotObject = PlotsHolder()
        # The current neuron will be the last position of the plotObjects
        splotObject.append( plotObjects['allBestOpNeurons'][-1] )
        pname3 = plot_4c(splotObject, opt)

        # FIXME: This plot the discrminator output from ROC curve. BEWARE! (figure 4)
        # need to add the neural network output into monitoring file.
        from PlotHelper import plot_nnoutput
        opt['cname']     = ('%s/plot_%s_neuron_%s_best_op_output')%(currentPath,benchmarkName,neuron)
        opt['nsignal']   = self._data[0].shape[0]
        opt['nbackground'] = self._data[1].shape[0]
        opt['rocname'] = 'roc_op'
        pname4 = plot_nnoutput(splotObject,opt)
    

        # Map names for beamer, if you add a plot, you must add into
        # the path objects holder
        pathObjects['neuron_'+str(neuron)+'_sorts_val']      = pname1 
        pathObjects['neuron_'+str(neuron)+'_sort_op']        = pname2
        pathObjects['neuron_'+str(neuron)+'_best_op']        = pname3
        pathObjects['neuron_'+str(neuron)+'_best_op_output'] = pname4
  
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
      beamer = BeamerMonReport(basepath+'/'+tuningReport, title = ('Tuning Report (eta=%d, et=%d)')%(etabin,etbin) )
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
            bname = info.name().replace('OperationPoint','')
            fig1 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_sorts_val'].replace(basepath+'/',''), 0.7,
                               frametitle=bname+', Neuron '+str(neuron)+': All sorts (validation)') 
            fig2 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_sort_op'].replace(basepath+'/',''), 0.7, 
                               frametitle=bname+', Neuron '+str(neuron)+': All sorts (operation)') 
            fig3 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_best_op'].replace(basepath+'/',''), 0.7,
                               frametitle=bname+', Neuron '+str(neuron)+': Best Network') 
            fig4 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_best_op_output'].replace(basepath+'/',''), 0.8,
                               frametitle=bname+', Neuron '+str(neuron)+': Best Network output') 
            
          
            #Draw figures into the tex file
            fig1.tolatex( beamer.file() )
            fig2.tolatex( beamer.file() )
            fig3.tolatex( beamer.file() )
            fig4.tolatex( beamer.file() )

          #Concatenate performance table, each line will be a benchmark
          #e.g: det, sp and fa
          ptableCross.add( perfBenchmarks[info.name()]['neuron_'+str(neuron)] ) 

        ptableCross.tolatex( beamer.file() )# internal switch is false to true: test
        ptableCross.tolatex( beamer.file() )# internal swotch is true to false: operation

      beamer.close()

  #End of loop()





