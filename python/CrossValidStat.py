#!/usr/bin/env python
from FastNetTool.util   import sourceEnvFile, checkForUnusedVars
from FastNetTool.Logger import Logger
import ROOT
import numpy as np
import pickle
 
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
  SPProduct       = 0
  Detection       = 1
  FalseAlarm      = 2

class Model(EnumStringification):
  """
    Select which framework ringer will operate
  """
  NeuralNetwork         = 0
  ValidPerformance      = 1
  OperationPerformance  = 2



class CrossData(Logger):
  def __init__(self, neuronsBound, sortBound, **kw):
    Logger.__init__(self, **kw)
    self.sortBound    = sortBound
    self.neuronsBound  = neuronsBound
    rowSize  = (neuronsBound[1]-neuronsBound[0]) + 1
    self._data = rowSize * [None]
    for row in range(rowSize):
      self._data[row] = sortBound * [None]
      for col in range(sortBound):
        self._data[row][col] = []

    self.shape = (rowSize,sortBound)
    self._logger.info('space allocated with size: %dX%d',rowSize,sortBound)
 
  def append(self, neuron, sort, object):
    self._data[neuron - self.neuronsBound[0]][sort].append( object )

  def __call__(self, neuron, sort):
    return self._data[neuron - self.neuronsBound[0]][sort]

  def neuronsBoundLooping(self):
    return range(*[self.neuronsBound[0], self.neuronsBound[1]+1])
    
  def sortBoundLooping(self):
    return range(self.sortBound)

  def initBoundLooping(self, neuron, sort):
    return range(len(self._data[neuron - self.neuronsBound[1]][sort]))




class CrossValidStat(Logger):

  def __init__(self, inputFiles, **kw):
    Logger.__init__(self,**kw)
    
    self._prefix        = inputFiles[0][0:inputFiles[0].find('.n')]
    self._neuronsBound  = [999,0]
    self._sortBound     = list()
    self._doFigure      = True
    self.canvas=None

    for file in inputFiles:
      offset = file.find('.n')
      n_min = n_max = int(file[offset+2:offset+6])

      offset = file.find('.s')
      s = self._sortBound = int(file[offset+2:offset+6])
      if n_min < self._neuronsBound[0]:  self._neuronsBound[0] = n_min
      if n_max > self._neuronsBound[1]:  self._neuronsBound[1] = n_max
      if s > self._sortBound:  self._sortBound = s

    self._data = CrossData( self._neuronsBound, self._sortBound+1)
    for file in inputFiles:
      objects = pickle.load(open(file))
      self._logger.info('reading %s... attach position: (%d,%d)',file,objects[0], objects[1])

      for train in objects[3]:
        self._data.append(objects[0], objects[1], train)


  def __call__(self, **kw):

    self._criteria  = kw.pop('criteria',Criteria.SPProduct)
    self._doPlots   = kw.pop('doFigure',True)
    self._prefix    = kw.pop('prefix','crossvalstat')
    self._dataLabel = kw.pop('dataLabel','#scale[.8]{MC14 13TeV, #mu = 20}')
    self._logoLabel = kw.pop('logoLabel','#it{#bf{Fastnet}} train')
    self._ref       = kw.pop('ref', 0.95) 
    checkForUnusedVars( kw, self._logger.warning )
    networks_by_neuron  = self.best_sort(self._criteria)
    filehandler         = open('save_networks.pic','wb')
    pickle.dump(networks_by_neuron,filehandler)

  '''
    Private function using to adapt and plot figures
  '''

  def __adapt_cut(self, valid, operation, referency_value, criteria):

    if criteira is Criteria.SPProduct: return (test,operation)
    if criteria is Criteria.Detection:  pos =np.where(np.array(valid.detVec) >referency_value)[0][0]-1
    if criteria is Criteria.FalseAlarm: pos =np.where(np.array(valid.faVec) > referency_value)[0][0]-1
    
    valid.sp  = valid.spVec[pos]; valid.det = valid.detVec[pos] ;
    valid.fa  = valid.faVec[pos]; valid.cut = valid.cutVec[pos]
    operation.sp  = operation.spVec[pos]; operation.det = operation.detVec[pos]
    operation.fa  = operation.faVec[pos]; operation.cut = operation.cutVec[pos]
    
    return (valid, operation) 

  '''
    Figures and plots
  '''
  def __boxplot( self, canvas, th2f, y_axis_limits, **kw):
    title         = kw.pop('title', '')
    xlabel        = kw.pop('xlabel','x axis')
    ylabel        = kw.pop('ylabel','y axis')
    color_curves  = kw.pop('color_curves',ROOT.kBlack)
    th2f.SetTitle(title)
    th2f.SetStats(0)
    th2f.GetXaxis().SetTitle(xlabel)
    th2f.GetYaxis().SetTitle(ylabel)
    th2f.GetYaxis().SetRangeUser(y_axis_limits[0],y_axis_limits[1])
    th2f.Draw('CANDLE')
    canvas.Modified()
    canvas.Update()

  def __plot_boxes( self, th2f_sp, th2f_det, th2f_fa, outputname ):
    canvas = ROOT.TCanvas('canvas', 'canvas', 1200, 800)
    canvas.Divide(1,3)
    self.__boxplot( canvas.cd(1), th2f_sp,  [0.94,0.975],xlabel='#neurons',ylabel='SP')
    self.__boxplot( canvas.cd(2), th2f_det, [0.95,0.99], xlabel='#neurons',ylabel='Detection')
    self.__boxplot( canvas.cd(3), th2f_fa,  [0.0,0.1],  xlabel='#neurons',ylabel='False alarm')
    canvas.cd(1)
    logoLabel_obj   = ROOT.TLatex(.65,.65,self._logoLabel);
    logoLabel_obj.SetNDC(ROOT.kTRUE);
    logoLabel_obj.SetTextSize(.20)
    logoLabel_obj.Draw()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(outputname)

  def __plot_evol( self, canvas, curves, y_axis_limits, **kw):
    title         = kw.pop('title', '')
    xlabel        = kw.pop('xlabel','x axis')
    ylabel        = kw.pop('ylabel','y axis')
    select_pos    = kw.pop('select_pop',None)
    color_curves  = kw.pop('color_curves',ROOT.kBlack)
    color_select  = kw.pop('color_select',ROOT.kRed)

    #create dummy graph
    x_max = 0; dummy = None
    for i in range(len(curves)):
      curves[i].SetLineColor(color_curves)
      x = curves[i].GetXaxis().GetXmax()
      if x > x_max: x_max = x; dummy = curves[i]
    
    dummy.SetTitle( title )
    dummy.GetXaxis().SetTitle(xlabel)
    #dummy.GetYaxis().SetTitleSize( 0.4 ) 
    dummy.GetYaxis().SetTitle(ylabel)
    #dummy.GetYaxis().SetTitleSize( 0.4 )

    #change the axis range for y axis
    dummy.GetHistogram().SetAxisRange(y_axis_limits[0],y_axis_limits[1],'Y' )
    dummy.Draw('AL')

    for c in curves:  c.Draw('same')
    if select_pos:  curves[select_pos].SetLineColor(color_select);  curves[select_pos].Draw('same')
    
    canvas.Modified()
    canvas.Update()


  def __plot_train(self, mse, sp, det, fa, best_id, worse_id, outputname):
    
    red   = ROOT.kRed+2
    blue  = ROOT.kAzure+6
    black = ROOT.kBlack
    canvas = ROOT.TCanvas('canvas', 'canvas', 1200, 800)
    canvas.Divide(1,4)

    self.__plot_evol(canvas.cd(1),mse,[0,.3],title='Mean Square Error Evolution',
                                              xlabel='epoch #', ylabel='MSE',
                                              select_pop=best_id,
                                              color_curves=blue,
                                              color_select=black)
    self.__plot_evol(canvas.cd(2),sp,[.93,.97],title='SP Evolution',
                                              xlabel='epoch #', ylabel='SP',
                                              select_pop=best_id,
                                              color_curves=blue,
                                              color_select=black)
    self.__plot_evol(canvas.cd(3),det,[.95,1],title='Detection Evolution',
                                              xlabel='epoch #',
                                              ylabel='Detection',
                                              select_pop=best_id,
                                              color_curves=blue,
                                              color_select=black)
    self.__plot_evol(canvas.cd(4),fa,[0,.3],title='False alarm evolution',
                                              xlabel='epoch #', ylabel='False alarm',
                                              select_pop=best_id,
                                              color_curves=blue,
                                              color_select=black)
    
    canvas.cd(1)
    logoLabel_obj   = ROOT.TLatex(.65,.65,self._logoLabel);
    logoLabel_obj.SetNDC(ROOT.kTRUE);
    logoLabel_obj.SetTextSize(.25)
    logoLabel_obj.Draw()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(outputname)

  '''
    Analysis
  '''
  def best_sort(self, criteria ):
   
    network_by_neuron = list()
    
    #boxplot graphics
    x_min       = self._neuronsBound[0]-0.5
    x_max       = self._neuronsBound[1]+0.5
    x_bins      = int(x_max-x_min) 
    th2f_sp     = ROOT.TH2F('','', x_bins, x_min, x_max,  100,0.5,1)
    th2f_det    = ROOT.TH2F('','', x_bins, x_min, x_max, 100,0.5,1)
    th2f_fa     = ROOT.TH2F('','', x_bins, x_min, x_max, 100,0,0.5)
 
    #loop over neurons
    for n in self._data.neuronsBoundLooping():
      best_value        = worse_value = 0
      best_sort         = worse_sort  = 0
      mse_val_graph     = list()
      sp_val_graph      = list()
      fa_val_graph      = list()
      det_val_graph     = list() 
      
      if criteria is Criteria.FalseAlarm: best_value = worse_value = 99
      
      #loop over sorts
      for s in self._data.sortBoundLooping():  

        self._logger.info('looking for the pair (%d, %d)', n,s)
        object = self.best_init(n,s,criteria)
        best_network = object[0]
  
        train_evolution  = best_network[0].dataTrain
        roc_val          = best_network[1]
        roc_operation    = best_network[2]
        
        th2f_sp.Fill (n, roc_operation.sp ) 
        th2f_det.Fill(n, roc_operation.det)
        th2f_fa.Fill (n, roc_operation.fa )
 
        if self._doFigure:
          epochs_val        = np.array( range(len(train_evolution.mse_val)),  dtype='float_')
          mse_val           = np.array( train_evolution.mse_val,              dtype='float_')
          sp_val            = np.array( train_evolution.sp_val,               dtype='float_')
          det_val           = np.array( train_evolution.det_val,              dtype='float_')
          fa_val            = np.array( train_evolution.fa_val,               dtype='float_') 
          mse_val_graph.append( ROOT.TGraph(len(epochs_val), epochs_val, mse_val ))
          sp_val_graph.append(  ROOT.TGraph(len(epochs_val), epochs_val, sp_val  ))
          det_val_graph.append( ROOT.TGraph(len(epochs_val), epochs_val, det_val ))
          fa_val_graph.append(  ROOT.TGraph(len(epochs_val), epochs_val, fa_val  ))
          
        #choose best sort
        if criteria is Criteria.SPProduct  and roc_operation.sp   > best_value:  best_value = roc_operation.sp;  best_sort  = s
        if criteria is Criteria.Detection  and roc_operation.det  > best_value:  best_value = roc_operation.det; best_sort  = s
        if criteria is Criteria.FalseAlarm and roc_operation.fa   < best_value:  best_value = roc_operation.fa;  best_sort  = s

        #choose worse sort
        if criteria is Criteria.SPProduct  and roc_operation.sp   < worse_value: worse_value = roc_operation.sp;  worse_sort = s
        if criteria is Criteria.Detection  and roc_operation.det  < worse_value: worse_value = roc_operation.det; worse_sort = s
        if criteria is Criteria.FalseAlarm and roc_operation.fa   > worse_value: worse_value = roc_operation.fa;  worse_sort = s
  
      if self._doFigure:
        outputname = ('%s_neuron_%d_criteria_%s.pdf') % (self._prefix, n, criteria)
        self.__plot_train(mse_val_graph, sp_val_graph, det_val_graph, fa_val_graph, best_sort, worse_sort, outputname)

      network_by_neuron.append( (n, self._data(n, best_sort), self._data(n,worse_sort)) )

    if self._doFigure:  
      outputname = ('%s_boxplot_crit_%s.pdf') % (self._prefix, criteria)
      self.__plot_boxes(th2f_sp,th2f_det,th2f_fa, outputname)

    return network_by_neuron
   

  def best_init(self, n, s, criteria):

    train_objs        = self._data(n,s)
    best_value        = worse_value = 0
    best_pos          = worse_pos   = 0
    best_network      = None 
    worse_network     = None
    mse_val_graph     = list()
    sp_val_graph      = list()
    fa_val_graph      = list()
    det_val_graph     = list() 

    if criteria is Criteria.FalseAlarm: best_value = worse_value = 99

    for pos in self._data.initBoundLooping(n,s):

      train_evolution   = self._data(n,s)[pos][criteria][Model.NeuralNetwork].dataTrain
      network           = self._data(n,s)[pos][criteria][Model.NeuralNetwork]
      roc_val           = self._data(n,s)[pos][criteria][Model.ValidPerformance]
      roc_operation     = self._data(n,s)[pos][criteria][Model.OperationPerformance]
      objects           = (network, roc_val, roc_operation, pos)

      if self._doFigure:
        epochs_val        = np.array( range(len(train_evolution.mse_val)),  dtype='float_')
        mse_val           = np.array( train_evolution.mse_val,              dtype='float_')
        sp_val            = np.array( train_evolution.sp_val,               dtype='float_')
        det_val           = np.array( train_evolution.det_val,              dtype='float_')
        fa_val            = np.array( train_evolution.fa_val,               dtype='float_')
        
        mse_val_graph.append( ROOT.TGraph(len(epochs_val), epochs_val, mse_val ))
        sp_val_graph.append(  ROOT.TGraph(len(epochs_val), epochs_val, sp_val  ))
        det_val_graph.append( ROOT.TGraph(len(epochs_val), epochs_val, det_val ))
        fa_val_graph.append(  ROOT.TGraph(len(epochs_val), epochs_val, fa_val  ))

      #choose best init 
      if criteria is Criteria.SPProduct  and roc_val.sp  > best_value: best_pos= pos; best_value = roc_val.sp; best_network = objects
      if criteria is Criteria.Detection  and roc_val.det > best_value: best_pos= pos; best_value = roc_val.det; best_network = objects
      if criteria is Criteria.FalseAlarm and roc_val.fa  < best_value: best_pos= pos; best_value = roc_val.fa; best_network = objects

      #choose worse init 
      if criteria is Criteria.SPProduct  and roc_val.sp  < worse_value: worse_pos= pos; worse_value = roc_val.sp; worse_network = objects
      if criteria is Criteria.Detection  and roc_val.det < worse_value: worse_pos= pos; worse_value = roc_val.det; worse_network = objects 
      if criteria is Criteria.FalseAlarm and roc_val.fa  > worse_value: worse_pos= pos; worse_value = roc_val.fa; worse_network = objects

    #plot figure
    if self._doFigure:
      outputname = ('%s_neuron_%d_sort_%d_inits_criteria_%s.pdf') % (self._prefix, n, s,criteria)
      self.__plot_train(mse_val_graph, sp_val_graph, det_val_graph, fa_val_graph, best_pos, worse_pos, outputname)

    return (best_network, worse_network)


