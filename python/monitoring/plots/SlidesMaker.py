__all__ = ['makeSummaryMonSlides']

from RingerCore.tex.TexAPI import *
from RingerCore.tex.BeamerAPI import *
from RingerCore import load

def makeSummaryMonSlides(outputs,nbins,choicesfile,grid=False):
  from scipy.io import loadmat
  import os
  f =  loadmat(choicesfile) 
  choices = dict()
  choices['Pd'] = f['choices']['Pd'][0][0]
  choices['Pf'] = f['choices']['Pf'][0][0]
  choices['SP'] = f['choices']['SP'][0][0]

  nbinchoices = 0
  for etchoices in choices['Pd']:
    nbinchoices += len(etchoices)
  if not nbins == nbinchoices:
    raise NameError('Choices Archieve Error')

  net = len(choices['Pd'])
  neta = len (choices['Pd'][0])
  unbinned = False
  if net == 1 and neta == 1:
    unbinned = True
  if grid:
    f = load([s for s in os.listdir('.') if '_et0_eta0' in s][0]+'/perfBounds.pic.gz')
  else:
    f = load('{}/perfBounds.pic.gz'.format(outputs+'_et0_eta0'))
  benchmarkNames = f['perf'].keys()
   
  for benchmarkName in benchmarkNames:
    slideAnex = []
    for et in xrange(net):
      etlist = []
      for eta in xrange( neta):
        etaDict = dict()
        neuron = choices[benchmarkName.split('_')[-1]][et][eta]
        basepath=outputs
        basepath+=('_et%d_eta%d')%(et,eta)
        if grid:
          basepath=('_et%d_eta%d')%(et,eta)
          basepath=[s for s in os.listdir('.') if basepath in s][0]
          f = load(basepath+'/perfBounds.pic.gz')
        else:
          f =  load ('{}/perfBounds.pic.gz'.format(basepath))
        etstr =  f['bounds']['etbinstr']
        etastr =  f['bounds']['etabinstr']
        perfs  = f['perf'][benchmarkName]

        refBench = perfs['config_'+str(neuron).zfill(3)].getRef()
        detR = r'{:.2f}'.format(refBench['det'])
        spR  = r'{:.2f}'.format(refBench['sp'])
        faR  = r'{:.2f}'.format(refBench['fa'])
        refBench = [detR,spR,faR]

        perfBench = perfs['config_'+str(neuron).zfill(3)].getPerf()
        detP = r'{:.2f}$\pm${:.2f}'.format(perfBench['detMean'],perfBench['detStd'])
        spP  = r'{:.2f}$\pm${:.2f}'.format(perfBench['spMean'],perfBench['spStd'])
        faP  = r'{:.2f}$\pm${:.2f}'.format(perfBench['faMean'],perfBench['faStd'])
        perfBench = [detP,spP,faP]

        bestNetBench = perfs['config_'+str(neuron).zfill(3)].rawOp()
        detB = r'{:.2f}'.format(bestNetBench['det']*100)
        spB  = r'{:.2f}'.format(bestNetBench['sp']*100)
        faB  = r'{:.2f}'.format(bestNetBench['fa']*100)
        bestNetBench = [ detB,spB,faB]
        perfs = [refBench,perfBench,bestNetBench]  
      
        graphSections = [
              'All Sorts(Validation)' ,
              'All ROC Sorts(Validation)',
              'All Sorts(Operation)' ,  
              'All ROC Sorts(Operation)',
              'Best Network',  
              'Best Operation Output',  
              ]

        figures = [
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_sorts_val.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron),
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_sorts_roc_tst.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron),
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_sorts_op.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron)      ,  
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_sorts_roc_op.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron),
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_best_op.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron)       ,  
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_best_op_output.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron),  
                 ]
        figuresDict= dict(zip(graphSections,figures))
        etaDict['neuron'] = neuron
        etaDict['figures'] = figuresDict
        etaDict['graphSections'] = graphSections
        etaDict['perfs'] = perfs 
        etaDict['etastr'] = etastr 
        etaDict['etstr'] = etstr 

        etlist.append(etaDict)
        # for eta
      slideAnex.append(etlist)
      #for et
    with BeamerTexReportTemplate1( theme = 'Berlin'
                                 , _toPDF = True
                                 , title = benchmarkName
                                 , outputFile = benchmarkName
                                 , font = 'structurebold' ):
      with BeamerSection(name = 'Performance'):
        if not unbinned:
          l1 = ['','']
          l2 = ['','']
          sideline =  '{|c|c|'
          for et in xrange(net):
            sideline += 'ccc|'
            l1.extend(['',slideAnex[et][0]['etstr'],''])
            l2.extend(['Pd','SP','PF'])
          sideline +='}'
          etlines = []
          for eta in xrange(neta):
            la = [r'\hline'+ slideAnex[0][eta]['etastr'],'CrossValidation' ]
            lb = [r'','Reference']
            lc = [r'', 'bestNetBench']
            for et in xrange(net):
              prfs = slideAnex [et][eta]['perfs']
              refBench = prfs[0]
              perfBench = prfs[1]
              bestNetBench = prfs[2]

              la.extend(perfBench)
              lb.extend(refBench)
              lc.extend(bestNetBench)
            etlines.extend([la,lb,lc])
          linhas=[]
          linhas.append(l1)
          linhas.append(l2)
          linhas.extend(etlines)
          with BeamerSubSection(name= 'All Bins Performance'):
            BeamerTableSlide(title =  'All Bins Performance',
                           linhas = linhas,
                           sideline = sideline,
                           caption = 'Efficiences',
                           )
            
          


        for et in xrange(net):
          l1 = ['','']
          l2 = ['','']
          sideline =  '{|c|c|ccc|}'
          l1.extend(['',slideAnex[et][0]['etstr'],''])
          l2.extend(['Pd','SP','PF'])
          etlines = []
          for eta in xrange(neta):
            if unbinned:
              la = [r'\hline','CrossValidation' ]
            else:
              la = [r'\hline'+ slideAnex[0][eta]['etastr'],'CrossValidation' ]
            lb = [r'Neuron','Reference']
            lc = [r'', 'bestNetBench']
            neuron = slideAnex [et][eta]['neuron']
            prfs = slideAnex [et][eta]['perfs']
            lc[0] = '%d'%neuron
            refBench = prfs[0]

            perfBench = prfs[1]

            bestNetBench = prfs[2]

            la.extend(perfBench)
            lb.extend(refBench)
            lc.extend(bestNetBench)
            etlines.extend([la,lb,lc])
          linhas=[]
          if not unbinned:
            linhas.append(l1)
          linhas.append(l2)
          linhas.extend(etlines)
          if unbinned:
            BeamerTableSlide(title =  'Performance',
                           linhas = linhas,
                           sideline = sideline,
                           caption = 'Efficiences',
                           )
          else:
            with BeamerSubSection (name= slideAnex[et][0]['etstr']):
              BeamerTableSlide(title =  'Performance',
                           linhas = linhas,
                           sideline = sideline,
                           caption = 'Efficiences',
                           )

      with BeamerSection( name = 'Figures' ):
        if unbinned:
          neuron = slideAnex [0][0]['neuron']
          graphSections = slideAnex [0][0]['graphSections']
          figures = slideAnex [0][0]['figures']
          for graph in graphSections:
            BeamerFigureSlide(title = graph + ' Neuron: '+ str(neuron) ,
                          path = figures[graph]
                               )
        else:
          for et in xrange(net):
            with BeamerSubSection (name= slideAnex[et][0]['etstr']):
              for eta in xrange(neta):
                neuron = slideAnex [et][eta]['neuron']
                graphSections = slideAnex [et][eta]['graphSections']
                figures = slideAnex [et][eta]['figures']

                with BeamerSubSubSection (name= slideAnex[et][eta]['etastr'] + ', Neuron: {}'.format(neuron)):
                  for graph in graphSections:
                    BeamerFigureSlide(title = graph,
                                       path = figures[graph]
                                      )
  return

