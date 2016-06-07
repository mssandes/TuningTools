
__all__ = ['MonTuningTool']
#Author: Joao Victo da Fonseca Pinto
#Analysis framework

#Import necessary classes
from MonTuningInfo import MonTuningInfo
from TuningStyle import SetTuningStyle
from pprint        import pprint
from RingerCore    import calcSP, save, load, Logger, mkdir_p


import os
#Set all global setting from ROOT style plot!
SetTuningStyle()

#This function will apply a correction index
def fix_position( vec, idx ):
  return vec.index( idx )


#Main class to plot and analyser the crossvalidStat object
#created by CrossValidStat class from tuningTool package
class MonTuningTool( Logger ):
  #Hold all information abount the monitoring root file
  _infoObjs = list()
  #Init class
  def __init__(self, crossvalFileName, monFileName, **kw):
    Logger.__init__(self, kw)
    from ROOT import TFile
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
    doBeamer    = kw.pop('doBeamer', True)
    shortSlides = kw.pop('shortSlides', False)

    plotNames = {'sortTstBest','sortOpBest','neuronTstBest','neuronOpBest'} 
    perfNames = {'tstPerf', 'opPerf'}

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
      for plotname in plotNames:  plotObjects[plotname] = PlotsHolder()
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
        initPaths = [('trainEvolution/%s/config_%s/sort_%s/init_%s')%(benchmarkName.replace('Operation','Operating'),\
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


      #Creating plots
      for neuron in infoObj.neuronBounds():
        currentPath =  ('%s/figures/%s/%s') % (basepath,benchmarkName,'neuron_'+str(neuron))
        neuronName = 'config_'+str(neuron)
        #Create folder to store all plot objects
        mkdir_p(currentPath)
        opt = dict()
        opt['reference'] = reference

        #Clear all hold plots stored in the past
        plotObjects['sortTstBest'].clear()
        plotObjects['sortOpBest'].clear()

        for sort in infoObj.sortBounds(neuron):
          sortName = 'sort_'+str(sort)
          #Retrieve best init index from test and operation values
          opt['tstBestInit'] = csummary[neuronName][sortName]['infoTstBest']['init']
          opt['opBestInit'] = csummary[neuronName][sortName]['infoOpBest']['init']
          #Retrieve plot templates
          initPlot = csummary[neuronName][sortName]['plots']
          #Add template into PlotsHolder object
          plotObjects['sortTstBest'].append( initPlot.getObj(opt['tstBestInit']) )
          plotObjects['sortOpBest'].append( initPlot.getObj(opt['opBestInit']) )
        #Loop over sorts

        opt['tstBestSort']  = csummary[neuronName]['infoTstBest']['sort']
        opt['tstWorstSort'] = csummary[neuronName]['infoTstWorst']['sort']
        opt['opBestSort']   = csummary[neuronName]['infoOpBest']['sort']
        opt['opWorstSort']  = csummary[neuronName]['infoOpWorst']['sort']
        initPlot = csummary[neuronName]['sort_'+str(opt['tstBestSort'])]['plots']
        plotObjects['neuronTstBest'].append( initPlot.getObj(opt['tstBestInit']) )
        initPlot = csummary[neuronName]['sort_'+str(opt['opBestSort'])]['plots']
        plotObjects['neuronOpBest'].append( initPlot.getObj(opt['opBestInit']) )

        #Create perf (tables) Objects for test and operation (Table)
        perfObjects['neuron_'+str(neuron)] =  MonPerfInfo(benchmarkName, reference, 
                                            csummary[neuronName]['summaryInfoTst'], 
                                            csummary[neuronName]['infoOpBest'], 
                                            cbenchmark) 

        #Configuration of each sort val plot: (Figure 1)
        rvec = infoObj.sortBounds(neuron)
        opt['label']     = ('#splitline{Sorts: %d}{etaBin: %d, etBin: %d}') % (plotObjects['sortTstBest'].size(),etabin, etbin)
        opt['cname']     = ('%s/plot_%s_neuron_%s_sorts_val')%(currentPath,benchmarkName,neuron)
        opt['set']       = 'val'
        opt['operation'] = False
        opt['paintListIdx'] = [fix_position(rvec, opt['tstBestSort']), fix_position(rvec,  opt['tstWorstSort'] )]
        pname1 = plot_4c(plotObjects['sortTstBest'], opt)

        #Configuration of each sort operation plot: (Figure 2)
        opt['label']     = ('#splitline{Sorts: %d (Operation)}{etaBin: %d, etBin: %d}') % (plotObjects['sortOpBest'].size(),etabin, etbin)
        opt['cname']     = ('%s/plot_%s_neuron_%s_sorts_op')%(currentPath,benchmarkName,neuron)
        opt['set']       = 'val'
        opt['operation'] = True
        opt['paintListIdx'] = [fix_position(rvec, opt['opBestSort']), fix_position(rvec,  opt['opWorstSort'] )]
        pname2 = plot_4c(plotObjects['sortOpBest'], opt)

        #Configuration of each best val network plot: (Figure 3)
        opt['label']     = ('#splitline{Best network, neuron: %d}{etaBin: %d, etBin: %d}') % (neuron,etabin, etbin)
        opt['cname']     = ('%s/plot_%s_neuron_%s_best_op')%(currentPath,benchmarkName,neuron)
        opt['set']       = 'val'
        opt['operation'] = True
        splotObject = PlotsHolder()
        splotObject.append( plotObjects['neuronOpBest'][fix_position(infoObj.neuronBounds(), neuron)] )
        pname3 = plot_4c(splotObject, opt)

        #Map names for beamer, if you add a plot, you must add into
        #the path objects holder
        pathObjects['neuron_'+str(neuron)+'_sorts_val'] = pname1 
        pathObjects['neuron_'+str(neuron)+'_sort_op']   = pname2
        pathObjects['neuron_'+str(neuron)+'_best_op']   = pname3

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

      beamer = BeamerMonReport(basepath+'/tuningReport.tex')
      neuronBounds = self._infoObjs[0].neuronBounds()

      for neuron in neuronBounds:
        #Make the tables for crossvalidation
        ptableCross = BeamerPerfTables(frametitle='Neuron '+str(neuron)+': Cross Validation Performance',
                                       caption='Efficiencies from each benchmark.')
        block = BeamerBlocks('Neuron '+str(neuron)+' Analysis', [('All sorts (validation)','All sorts evolution are ploted, each sort represents the best init;'),
                                                                 ('All sorts (operation)', 'All sorts evolution only for operation set;'),
                                                                 ('Best operation', 'Detailed analysis from the best sort discriminator.'),
                                                                 ('Tables','Cross validation performance')])
        if not shortSlides:  block.tolatex( beamer.file() )

        for info in self._infoObjs:
          #If we produce a short presentation, we do not draw all plots
          if not shortSlides:  
            bname = info.name().replace('OperatingPoint','')
            fig1 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_sorts_val'].replace(basepath+'/',''), 0.8,
                               frametitle=bname+', Neuron '+str(neuron)+': All sorts (validation)') 
            fig2 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_sort_op'].replace(basepath+'/',''), 0.8, 
                               frametitle=bname+', Neuron '+str(neuron)+': All sorts (operation)') 
            fig3 = BeamerFigure( pathBenchmarks[info.name()]['neuron_'+str(neuron)+'_best_op'].replace(basepath+'/',''), 0.8,
                               frametitle=bname+', Neuron '+str(neuron)+': Best Network') 
            #Draw figures into the tex file
            fig1.tolatex( beamer.file() )
            fig2.tolatex( beamer.file() )
            fig3.tolatex( beamer.file() )

          #Concatenate performance table, each line will be a benchmark
          #e.g: det, sp and fa
          ptableCross.add( perfBenchmarks[info.name()]['neuron_'+str(neuron)] ) 

        ptableCross.tolatex( beamer.file() )# internal switch is false to true: test
        ptableCross.frametitle = 'Neuron '+str(neuron)+": Operation Best Network"
        ptableCross.caption = 'Efficiencies for the best operation network'
        ptableCross.tolatex( beamer.file() )# internal swotch is true to false: operation

      beamer.close()

  #End of loop()







