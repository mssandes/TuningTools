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
    self._dataLabel = kw.pop('dataLabel','#scale[.8]{MC14 Zee 13TeV, #mu = 20}')
    self._logoLabel = kw.pop('logoLabel','#it{#bf{Fastnet}} train')
    self._ref       = kw.pop('ref', 0.95) 
    checkForUnusedVars( kw, self._logger.warning )
    
    networks_by_neuron = self.best_sort(self._criteria)


  def best_sort(self, criteria ):
   
    network_by_neuron = list()
    
    #boxplot graphics
    x_min       = self._neuronsBound[0]
    x_max       = self._neuronsBound[1]
    x_bins      = x_max-x_min 
    th2f_sp     = ROOT.TH2F('','',x_bins,x_min,x_max, 100,0,1)
    th2f_det    = ROOT.TH2F('','',x_bins,x_min, x_max, 100,0,1)
    th2f_fa     = ROOT.TH2F('','', x_bins,x_min, x_max, 100,0,1)
 
    #loop over neurons
    for n in self._data.neuronsBoundLooping():

      best_value        = worse_value = 0
      best_sort         = worse_sort  = 0
      mse_val_graph     = list()
      sp_val_graph      = list()
      fa_val_graph      = list()
      det_val_graph     = list() 
      epochs_max        = 0
      
      if criteria is Criteria.FalseAlarm: best_value = worse_value = 99
      
      #loop over sorts
      for s in self._data.sortBoundLooping():  

        self._logger.info('looking for the pair (%d, %d)', n,s)
        object = self.best_init(n,s,criteria)
        best_network = object[0]
  
        train_evolution  = best_network[0].dataTrain
        roc_val          = best_network[1]
        roc_operation    = best_network[2]
        
        th2f_sp.Fill(n,  roc_operation.sp)
        th2f_det.Fill(n, roc_operation.det)
        th2f_fa.Fill(n,  roc_operation.fa)
 
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
          
          if epochs_val.shape[0] > epochs_max:  epochs_max = epochs_val.shape[0]

        #choose best sort
        if criteria is Criteria.SPProduct  and roc_operation.sp   > best_value:  best_value = roc_operation.sp;  best_sort  = s
        if criteria is Criteria.Detection  and roc_operation.det  > best_value:  best_value = roc_operation.det; best_sort  = s
        if criteria is Criteria.FalseAlarm and roc_operation.fa   < best_value:  best_value = roc_operation.fa;  best_sort  = s

        #choose worse sort
        if criteria is Criteria.SPProduct  and roc_operation.sp   < worse_value: worse_value = roc_operation.sp;  worse_sort = s
        if criteria is Criteria.Detection  and roc_operation.det  < worse_value: worse_value = roc_operation.det; worse_sort = s
        if criteria is Criteria.FalseAlarm and roc_operation.fa   > worse_value: worse_value = roc_operation.fa;  worse_sort = s
  
      if self._doFigure:
        '''
        outname= ('%s_neuron_%d_sorts.pdf') % (self._prefix, n)
        self.__plot_train(epochs_max,mse_val_graph,sp_val_graph,det_val_graph,\
                                          fa_val_graph, best_sort, worse_sort,\
                                          outputName=outname) 
        '''

      network_by_neuron.append( (n, self._data(n, best_sort), self._data(n,worse_sort)) )

    #if self._doFigure:  self.__plot_boxplot(th2f_sp,th2f_det,th2f_fa)

    return network_by_neuron
   



  def best_init(self, n, s, criteria):

    train_objs        = self._data(n,s)
    best_value        = worse_value = 0
    best_pos          = worse_pos = 0
    best_network      = worse_network = None
    mse_val_graph     = list()
    sp_val_graph      = list()
    fa_val_graph      = list()
    det_val_graph     = list() 
    epochs_max        = 0

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
        
        if epochs_val.shape[0] > epochs_max:  epochs_max = epochs_val.shape[0]
    

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
      
      outputname = ('%s_neuron_%d_sort_%d_inits.pdf') % (self._prefix, n, s)
      '''
      self.__plot_train(epochs_max, mse_val_graph,sp_val_graph,det_val_graph,\
                                           fa_val_graph, best_pos, worse_pos,\
                                           outputName='best_init.pdf') 
      '''
      self.__plot_train([0,epochs_max], mse_val_graph, sp_val_graph, det_val_graph, fa_val_graph, best_pos, worse_pos, outputname)

        
    return (best_network, worse_network)



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


  def __plot_dummy( self, x_axis_limits, y_axis_limits, **kw):
   
    title       = kw.pop('title', '')
    xlabel      = kw.pop('xlabel','x axis')
    ylabel      = kw.pop('ylabel','y axis')
    
    #create dummy graph
    x_axis = range(*x_axis_limits)
    dummy  = ROOT.TGraph( len(x_axis),np.array(x_axis,dtype='float_'),np.zeros(len(x_axis),dtype='float_'))
    dummy.SetTitle( title )
    dummy.GetXaxis().SetTitle(xlabel)
    #dummy.GetYaxis().SetTitleSize( 0.4 ) 
    dummy.GetYaxis().SetTitle(ylabel)
    dummy.GetYaxis().SetTitleSize( 0.4 )

    #change the axis range for y axis
    dummy.GetHistogram().SetAxisRange(y_axis_limits[0],y_axis_limits[1],'Y' )
    return dummy

  def __plot_train(self, x_limits, mse, sp, det, fa, best_id, worse_id, outputname):
    
    red   = ROOT.kRed
    blue  = ROOT.kBlue-3
    black = ROOT.kBlack
    dummy = list()
    dummy.append(self.__plot_dummy(x_limits,[0,.3],title='Mean Square Error',xlabel='epoch #', ylabel='MSE'))
    dummy.append(self.__plot_dummy(x_limits,[.94,.98],title='SP Product',xlabel='epoch #', ylabel='SP'))
    dummy.append(self.__plot_dummy(x_limits,[.94,.98],title='Detection',xlabel='epoch #', ylabel='DET'))
    dummy.append(self.__plot_dummy(x_limits,[0,.3],title='False alarm',xlabel='epoch #', ylabel='FA'))


    canvas = ROOT.TCanvas('canvas', 'canvas', 1200, 800)
    canvas.Divide(1,4)

    canvas.cd(1)
    dummy[0].Draw('AL')
    for curve in mse:  curve.SetLineColor(blue);  curve.Draw('same')
    mse[best_id].SetLineColor(red); mse[best_id].Draw('same')

    canvas.cd(2)
    dummy[1].Draw('AL')
    for curve in sp:  curve.SetLineColor(blue);  curve.Draw('same')
    sp[best_id].SetLineColor(red); sp[best_id].Draw('same')

    canvas.cd(3)
    dummy[2].Draw('AL')
    for curve in det:  curve.SetLineColor(blue);  curve.Draw('same')
    det[best_id].SetLineColor(red); det[best_id].Draw('same')

    canvas.cd(4)
    dummy[3].Draw('AL')
    for curve in fa:  curve.SetLineColor(blue);  curve.Draw('same')
    fa[best_id].SetLineColor(red); fa[best_id].Draw('same')

    canvas.SaveAs(outputname)

