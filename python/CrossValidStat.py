#!/usr/bin/env python
from FastNetTool.util   import sourceEnvFile, checkForUnusedVars
from FastNetTool.Logger import Logger
import numpy as np
from ROOT import TCanvas,TGraph,gROOT
import pickle
import os
 
class EnumStringification:
  "Adds 'enum' static methods for conversion to/from string"
  @classmethod
  def tostring(cls, val):
    "Transforms val into string."
    for k,v in vars(cls).iteritems():
      if v==val:
        return k

  @classmethod
  def fromstring(cls, str):
    "Transforms string into enumeration."
    return getattr(cls, str, None)

class Criteria(EnumStringification):
  """
    Select which framework ringer will operate
  """
  default = 0
  DET     = 1
  FA      = 2

class CrossValidStat(Logger):
  def __init__(self, inputFiles, **kw):
    Logger.__init__(self,**kw)

    self._doPlots   = kw.pop('doPlots',False)
    self._dataLabel = kw.pop('dataLabel','#scale[.8]{MC14 Zee 13TeV, #mu = 20}')
    self._logoLabel = kw.pop('logoLabel','#it{#bf{Fastnet}} train')
    checkForUnusedVars( kw, self._logger.warning )
    
    self._dict = {}
    self._mapping = []
    
    if inputFiles.find('.pic') > -1:
      self.load(inputFiles)
    else:
      self._mapping = self.__get_information_from_string_and_alloc(os.listdir(inputFiles))
      self.__read(inputFiles)
  

  def __get_information_from_string_and_alloc(self, inputFiles):
    self._dict['prefix'] = inputFiles[0][0:inputFiles[0].find('.n')]
    self._dict['.n'] = 0
    self._dict['.s'] = 0
    for file in inputFiles:
      offset = file.find('.n')
      n = int(file[offset+2:offset+6])
      offset = file.find('.s')
      s = int(file[offset+2:offset+6])
      if n > self._dict['.n']:  self._dict['.n'] = n
      if s > self._dict['.s']:  self._dict['.s'] = s
    return self.__alloc_space()


  def __alloc_space( self ):
    mapping = (self._dict['.n']+1)*[None]
    for n in range(self._dict['.n']+1):
      mapping[n] = (self._dict['.s']+1) *[None]
      for s in range(self._dict['.s']+1): mapping[n][s]=[]
    self._logger.info('space allocated with size: %dX%d',\
                       self._dict['.n'],self._dict['.s'])
    return mapping

  def __read(self, inputFiles):
    for file in os.listdir(inputFiles):
      loadObjs = pickle.load(open(inputFiles+'/'+file))
      self._logger.info('reading %s... attach position: (%d,%d)',file,loadObjs[0], loadObjs[1])
      for nets in loadObjs[3]:
        self._mapping[loadObjs[0]][loadObjs[1]].append(nets)


  def save(self, outputName):
    filehandler = open(outputName, 'wb')
    objSave = [self._mapping, self._dict]
    self._logger.info('saving...')
    pickle.dump(objSave, filehandler, protocol=2)


  def load(self, inputFiles):
    import pickle
    self._logger.info('loading...')
    [self._mapping, self._dict] = pickle.load(open(inputFiles, 'rb'))


  def __call__(self, **kw):

    self._criteria  = kw.pop('criteria',Criteria.default)
    self._ref       = kw.pop('ref', 0.95) 
    checkForUnusedVars( kw, self._logger.warning )
    
    neurons = [None]*len(self._mapping)
    for n in range(len(self._mapping)):
      if not self._mapping[n][0]:  continue
      objs = self.best_sort(n)
      neurons[n] = objs




  def best_sort(self, neuron):

    sort = [None]*len(self._mapping[0])
    graphs = []
    tst_sp = []
    tst_fa = []
    tst_det= [] 
    op_sp  = []
    op_fa  = []
    op_det = []
    for s in range(len(self._mapping[0])):
      self._logger.info('calculating neuron %d and sort %d', neuron, s)
      sort[s] = self.best_initialization( neuron, s) 
      mse     = np.array(sort[s][self._criteria][0].dataTrain.mse_val,dtype='float_')
      epochs  = np.array( range(len(mse)) ,dtype='float_')
      graphs.append( TGraph(len(epochs), epochs, mse) )
 
      tst_sp.append(  sort[s][self._criteria][1].sp)
      tst_det.append( sort[s][self._criteria][1].det)
      tst_fa.append(  sort[s][self._criteria][1].fa)
      op_sp.append(   sort[s][self._criteria][2].sp)
      op_det.append(  sort[s][self._criteria][2].det)
      op_fa.append(   sort[s][self._criteria][2].fa)

    if self._doPlots and self._mapping[neuron][0]:
      name = ('fig_best_sort.criteria_%s.n%04d.pdf') % (self._criteria,neuron)
      title = ('criteria %s, neuron: %d , best inits') % (self._criteria,neuron)
      self.__plot_train_evolution(graphs, graphs[0], title, name)

    return [sort, (tst_sp,tst_det,tst_fa), (op_sp,op_det,op_fa)]


  def best_initialization(self, neuron, sort):
    train  = self._mapping[neuron][sort]
    graphs = []
    value  = 0
    idx=0
    if self._criteria == Criteria.FA: value = 999
    for i in range(len(self._mapping[neuron][sort])):
      mse     = np.array(train[i][self._criteria][0].dataTrain.mse_val,dtype='float_')
      epochs  = np.array( range(len(mse)) ,dtype='float_')
      graphs.append( TGraph(len(epochs), epochs, mse) )
      [roc_tst, roc_op] = self.__adapt_threshold(train[i][self._criteria][1], 
                                                 train[i][self._criteria][2],
                                                 self._ref,
                                                 self._criteria)
      train[i][self._criteria][1] = roc_tst
      train[i][self._criteria][2] = roc_op
      if self._criteria == Criteria.default and roc_tst.sp > value:
        value = roc_tst.sp
        idx = i
      if self._criteria == Criteria.DET and roc_tst.det > value:
        value = roc_tst.det
        idx = i
      if self._criteria == Criteria.FA and roc_tst.fa < value:
        value = roc_tst.fa
        idx = i
    print idx
    if self._doPlots:
      name = ('fig_best_initialization.criteria_%s.n%04d.s%04d.pdf') % (self._criteria,neuron, sort)
      title = ('criteria %s, neuron: %d , sort = %d, flutuation for %d inits') % (self._criteria,neuron,sort,len(train))
      self.__plot_train_evolution(graphs, graphs[idx], title, name)

    return train[idx]
   


  def __adapt_threshold(self, roc_tst, roc_op, ref, criteria ):
    if criteria is Criteria.default:  return (roc_tst, roc_op)
    if criteria is Criteria.DET:  pos = np.where(np.array(obj.detVec) > ref)[0][0]-1
    if criteria is Criteria.FA:   pos = np.where(np.array(obj.faVec)  > ref)[0][0]-1
    roc_tst.sp   = roc_tst.spVec[pos]
    roc_tst.det  = roc_tst.detVec[pos]
    roc_tst.fa   = roc_tst.faVec[pos]
    roc_tst.cut  = roc_tst.cutVec[pos]
    roc_op.sp    = roc_op.spVec[pos]
    roc_op.det   = roc_op.detVec[pos]
    roc_op.fa    = roc_op.faVec[pos]
    roc_op.cut   = roc_op.cutVec[pos]
    return (roc_tst,roc_op)


  def __plot_train_evolution( self, plot_list, best_plot , title, output_name):
    import ROOT
    from ROOT import gROOT
    gROOT.Reset()
    canvas = ROOT.TCanvas('canvas','canvas',1000,600)
    canvas.SetGrid()
    for i in range(len(plot_list)): 
      plot_list[i].SetLineColor(16)
      if i == 0:  
        plot_list[i].SetTitle(title)
        plot_list[i].GetXaxis().SetTitle('Epochs')
        plot_list[i].GetYaxis().SetTitle('MSE')
        plot_list[0].Draw('AL')
      else: plot_list[i].Draw('same')

    best_plot.SetLineColor(2) #red
    #best_plot.SetLineColor(4) #blue
    best_plot.SetLineStyle(9) 
    best_plot.Draw('same')
    
    logoLabel_obj   = ROOT.TLatex(.15,.85, self._logoLabel)
    logoLabel_obj.SetNDC(1)
    logoLabel_obj.Draw()

    leg = ROOT.TLegend(0.68,0.65,0.85,0.80)
    leg.AddEntry(plot_list[0],'flutuation','lp')
    leg.AddEntry(best_plot,'best init','lp')
    leg.Draw()
    canvas.Modified();
    canvas.Update();
    canvas.SaveAs(output_name)















