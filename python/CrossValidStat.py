#!/usr/bin/env python
from FastNetTool.util         import sourceEnvFile, checkForUnusedVars
from FastNetTool.Logger       import Logger
from FastNetTool.util         import calcSP
import ROOT
import numpy as np
import pickle

def plot_topo(canvas, obj, var, y_limits, title, xlabel, ylabel):
  
  x_axis = range(*[y_limits[0],y_limits[1]+1])
  x_axis_values = np.array(x_axis,dtype='float_')
  inds = x_axis_values.astype('int_')
  x_axis_error   = np.zeros(x_axis_values.shape,dtype='float_')
  y_axis_values  = obj[var+'_mean'].astype('float_')
  y_axis_error   = obj[var+'_std'].astype('float_')
  graph = ROOT.TGraphErrors(len(x_axis_values),x_axis_values,y_axis_values[inds], x_axis_error, y_axis_error[inds])
  graph.Draw('ALP')
  graph.SetTitle(title)
  graph.SetMarkerColor(4); graph_sp.SetMarkerStyle(21)
  graph.GetXaxis().SetTitle('neuron #')
  graph.GetYaxis().SetTitle('SP')
  canvas.Modified()
  canvas.Update()


def plot_evol( canvas, curves, y_axis_limits, **kw):
  title         = kw.pop('title', '')
  xlabel        = kw.pop('xlabel','x axis')
  ylabel        = kw.pop('ylabel','y axis')
  select_pos1   = kw.pop('select_pop1',-1)
  select_pos2   = kw.pop('select_pop2',-1)
  color_curves  = kw.pop('color_curves',ROOT.kBlue)
  color_select1 = kw.pop('color_select1',ROOT.kBlack)
  color_select2 = kw.pop('color_select2',ROOT.kRed)

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
  if select_pos1 > -1:  curves[select_pos1].SetLineColor(color_select1); curves[select_pos1].Draw('same')
  if select_pos2 > -1:  curves[select_pos2].SetLineColor(color_select2); curves[select_pos2].Draw('same')
  
  canvas.Modified()
  canvas.Update()


class DataReader(Logger):
  def __init__(self, rowBounds, ncol, **kw):
    Logger.__init__(self, **kw)

    self.size_row    = (rowBounds[1]-rowBounds[0]) + 1
    self.size_col    = ncol
    self.rowBounds  = rowBounds
    self._data       = self.size_row * [None]

    for row in range(self.size_row):
      self._data[row] = self.size_col * [None]
      for col in range(self.size_col):
        self._data[row][col] = None

    self.shape = (self.size_row, self.size_col)
    self._logger.info('space allocated with size: %dX%d',self.size_row,self.size_col)
 
  def append(self, row, col, filename):
    self._data[row - self.rowBounds[0]][col]=filename 

  def __call__(self, row, col):
    filename=self._data[row - self.rowBounds[0]][col]
    return pickle.load(open(filename))[3]

  def rowBoundLooping(self):
    return range(*[self.rowBounds[0], self.rowBounds[1]+1])
    
  def colBoundLooping(self):
    return range(self.size_col)


class PerfValues:
  def __init__(self, spVec, detVec, faVec, cutVec):
    self.spVec  = spVec
    self.detVec = detVec
    self.faVec  = faVec
    self.cutVec = cutVec
    self.sp     = spVec[np.argmax(spVec)]
    self.det    = detVec[np.argmax(spVec)]
    self.fa     = faVec[np.argmax(spVec)]
    self.cut    = cutVec[np.argmax(spVec)]
    self.cut_id = np.argmax(spVec)
  
  def searchRefPoint(self,ref,criteria):
    i=-1
    if criteria is 'det': i =np.where(self.detVec<ref)[0][0]
    if criteria is 'fa':  i =np.where(self.faVec<ref)[0][0]
    return i

  def setValues(self, cut_id):
    self.sp  = self.spVec[cut_id]
    self.det = self.detVec[cut_id]
    self.fa  = self.faVec[cut_id]
    self.cut = self.cutVec[cut_id]
    self.cut_id = cut_id
    #print '(',self.det,',',self.fa,')'


class CrossValidStatAnalysis(Logger):

  def __init__(self, inputFiles, neuronsBound, size_sort, **kw):

    Logger.__init__(self,**kw)    
    filename            = kw.pop('filename', 'file_crossvalidstat.root')
    self._ref           = kw.pop('reference',None)

    self._neuronsBound  = neuronsBound
    self._sorts     = size_sort

    count=0
    self._dataReader = DataReader( self._neuronsBound, self._sorts)
    for file in inputFiles:
      offset = file.find('.n'); n = int(file[offset+2:offset+6])
      offset = file.find('.s'); s = int(file[offset+2:offset+6])
      self._logger.info('reading %s... attach position: (%d,%d) / count= %d',file,n, s,count)
      self._dataReader.append(n, s, file)
      count+=1
      
    self._logger.info('There is a totol of %d jobs into your directory',count)


  def __call__(self, stop_criteria, selected_criteria, **kw):
    self._detRef   = kw.pop('ref_det',0.9956)
    self._faRef    = kw.pop('ref_det',0.2869)
    self._logoLabel = kw.pop('logoLabel','Fastnet')
    outputname      = kw.pop('outputname','crossvalidStatAnalysis.pic')
    self._criteria_network = stop_criteria
    self._criteria_thebest_theworse = selected_criteria
    obj=self.loop()
    filehandler = open(outputname,'w')
    pickle.dump(obj,filehandler)


  def loop(self):

    crit_mapping      = {'sp':0,'det':1, 'fa':2}
    outputObj=dict()
    for s in ['sp_val','det_val','fa_val','sp_op','det_op','fa_op']:
      outputObj[s]  = np.zeros((self._neuronsBound[1]+1,self._sorts))
    outputObj['best_networks']=(self._neuronsBound[1]+1)*[None]


    for n in self._dataReader.rowBoundLooping():
      bucket_sorts=list()
      for s in self._dataReader.colBoundLooping():   
        self._logger.info('reading information from pait (%d, %d)',n,s)
        train_data   = self._dataReader(n,s)
        bucket_inits = list()
        count=0
        for train in train_data:
          obj=dict()

          #self._logger.info('count is: %d',count )
          criteria = crit_mapping[self._criteria_network]
          #variables to hold all vectors 
          train_evolution   = train[criteria][0].dataTrain
          network           = train[criteria][0]
          roc_val           = train[criteria][1]
          roc_operation     = train[criteria][2]

          epoch             = np.array( range(len(train_evolution.epoch)),  dtype='float_')
          mse_trn           = np.array( train_evolution.mse_trn,              dtype='float_')
          mse_val           = np.array( train_evolution.mse_val,              dtype='float_')
          sp_val            = np.array( train_evolution.sp_val,               dtype='float_')
          det_val           = np.array( train_evolution.det_val,              dtype='float_')
          fa_val            = np.array( train_evolution.fa_val,               dtype='float_')
          mse_tst           = np.array( train_evolution.mse_tst,              dtype='float_')
          sp_tst            = np.array( train_evolution.sp_tst,               dtype='float_')
          det_tst           = np.array( train_evolution.det_tst,              dtype='float_')
          fa_tst            = np.array( train_evolution.fa_tst,               dtype='float_')
          roc_val_det       = np.array( roc_val.detVec,                       dtype='float_')
          roc_val_fa        = np.array( roc_val.faVec,                        dtype='float_')
          roc_val_cut       = np.array( roc_val.cutVec,                       dtype='float_')
          roc_op_det        = np.array( roc_operation.detVec,                 dtype='float_')
          roc_op_fa         = np.array( roc_operation.faVec,                  dtype='float_')
          roc_op_cut        = np.array( roc_operation.cutVec,                 dtype='float_')

          #self._logger.info('dump into the store')

          obj['mse_trn']=ROOT.TGraph(len(epoch),epoch, mse_val )
          obj['mse_val']=ROOT.TGraph(len(epoch),epoch, mse_val )
          obj['sp_val' ]=ROOT.TGraph(len(epoch),epoch, sp_val )
          obj['det_val']=ROOT.TGraph(len(epoch),epoch, det_val)
          obj['fa_val' ]=ROOT.TGraph(len(epoch),epoch, fa_val )        
          obj['mse_tst']=ROOT.TGraph(len(epoch),epoch, mse_tst )
          obj['sp_tst' ]=ROOT.TGraph(len(epoch),epoch, sp_tst )
          obj['det_tst']=ROOT.TGraph(len(epoch),epoch, det_tst)
          obj['fa_tst' ]=ROOT.TGraph(len(epoch),epoch, fa_tst )        
          obj['roc_val']=ROOT.TGraph(len(roc_val_fa),roc_val_fa, roc_val_det )
          obj['roc_val_cut']=ROOT.TGraph(len(roc_val_cut),np.array(range(len(roc_val_cut)),'float_'),roc_val_cut )
          obj['roc_op']=ROOT.TGraph(len(roc_op_fa),roc_op_fa, roc_op_det )
          obj['roc_op_cut']=ROOT.TGraph(len(roc_op_cut),np.array(range(len(roc_op_cut)),'float_'), roc_op_cut )
          self.add_performance(obj)           
          tmp=dict(); 
          tmp['neuron']=n; tmp['sort']=s; tmp['init']=count; tmp['train']=obj
          bucket_inits.append(tmp)
          count+=1
         
        [thebest_id_init, theworse_id_init] = self.find_thebest_theworse(bucket_inits,'perf_val')
        self.plot_evol(bucket_inits, thebest_id_init, theworse_id_init, ('neuron_%d_sort_%d_inits_evol.pdf')%(n,s))
        objsaved_thebest=bucket_inits[thebest_id_init]
        bucket_sorts.append(objsaved_thebest)

        outputObj['sp_val'][n][s]  = objsaved_thebest['train']['perf_val'].sp
        outputObj['det_val'][n][s] = objsaved_thebest['train']['perf_val'].det
        outputObj['fa_val'][n][s]  = objsaved_thebest['train']['perf_val'].fa
        outputObj['sp_op'][n][s]   = objsaved_thebest['train']['perf_op'].sp
        outputObj['det_op'][n][s]  = objsaved_thebest['train']['perf_op'].det
        outputObj['fa_op'][n][s]   = objsaved_thebest['train']['perf_op'].fa
      #end of sorts
      self._logger.info('find the best sort')
      [thebest_id_sort, theworse_id_sort] = self.find_thebest_theworse(bucket_sorts,'perf_op')
      self.plot_evol(bucket_sorts, thebest_id_sort,  theworse_id_sort, ('neuron_%d_sort_evol.pdf')%(n))
      outputObj['best_networks'][n]=bucket_sorts[thebest_id_sort]
     
    for s in ['sp_val','det_val','fa_val','sp_op','det_op','fa_op']:
      outputObj[s+'_mean'] = np.mean(outputObj[s],axis=1)
      outputObj[s+'_std']  = np.std(outputObj[s],axis=1)
    outputname = ('topo_fluctuation.net_stopby_%s.selection_criteria_%s.pdf')%(self._criteira_network,
  		  self._criteira_thebest_theworse)
    self.plot_topo(outputObj,self._neuronsBound, outputname)
    return outputObj

  def getXarray(self, graph):
    bufferx=graph.GetX()
    bufferx.SetSize(graph.GetN())
    return np.array(bufferx,'float_')

  def getYarray(self, graph):
    buffery=graph.GetY()
    buffery.SetSize(graph.GetN())
    return np.array(buffery,'float_')

  def add_performance( self, obj):

    faVec     = self.getXarray(obj['roc_val'])
    detVec    = self.getYarray(obj['roc_val'])
    cutVec    = self.getYarray(obj['roc_val_cut'])
    spVec     = calcSP(detVec,1-faVec)
    #default is sp max
    perf_val  = PerfValues(spVec,detVec,faVec,cutVec) 
    
    faVec     = self.getXarray(obj['roc_op'])
    detVec    = self.getYarray(obj['roc_op'])
    cutVec    = self.getYarray(obj['roc_op_cut'])
    spVec     = calcSP(detVec,1-faVec)
    #default is sp max
    perf_op  = PerfValues(spVec,detVec,faVec,cutVec) 

    #set by sp point
    if self._criteria_thebest_theworse is 'sp':
      perf_op.setValues(perf_val.cut_id)

    #set by detection reference point
    if self._criteria_thebest_theworse is 'det':
      cut_id = perf_val.searchRefPoint(self._detRef,'det')
      perf_val.setValues(cut_id)
      perf_op.setValues(cut_id)

    #set by false alarm point
    if self._criteria_thebest_theworse is 'fa':
      cut_id = perf_val.searchRefPoint(self._faRef,'fa')
      perf_val.setValues(cut_id)
      perf_op.setValues(cut_id)
    
    obj['perf_val'] = perf_val
    obj['perf_op']  = perf_op



  def find_thebest_theworse(self, bucket, key):
    thebest_value   = 0
    theworse_value  = 99
    thebest_idx     = -1
    theworse_idx    = -1
    if self._criteria_thebest_theworse is 'det':
      thebest_value = 99; theworse_value = 0

    for i in range(len(bucket)):
      obj=bucket[i]['train'][key]
      #the best
      if self._criteria_thebest_theworse is 'sp' and obj.sp > thebest_value:
        thebest_value=obj.sp; thebest_idx=i
      if self._criteria_thebest_theworse is 'fa' and obj.det > thebest_value:
        thebest_value=obj.det; thebest_idx=i
      if self._criteria_thebest_theworse is 'det' and obj.fa < thebest_value:
        thebest_value=obj.fa; thebest_idx=i
      #the worse

      if self._criteria_thebest_theworse is 'sp' and obj.sp < theworse_value:
        theworse_value=obj.sp; theworse_idx=i
      if self._criteria_thebest_theworse is 'fa' and obj.det < theworse_value:
        theworse_value=obj.det; theworse_idx=i
      if self._criteria_thebest_theworse is 'det' and obj.fa > theworse_value:
        theworse_value=obj.fa; theworse_idx=i

    return (thebest_idx, theworse_idx)

  def plot_topo(self, obj, y_limits, outputname)
    
    canvas = ROOT.TCanvas('c1','c1',2000,1300)
    canvas.Divide(1,3) 
    plot_topo(canvas.cd(1), obj, 'sp_op', y_limits, 'SP fluctuation', '# neuron', 'SP'):
    plot_topo(canvas.cd(2), obj, 'det_op', y_limits, 'Detection fluctuation', '# neuron', 'Detection'):
    plot_topo(canvas.cd(3), obj, 'fa_op', y_limits, 'False alarm fluctuation', '# neuron', 'False alarm'):
    canvas.SaveAs(outputname)

  def plot_evol(self, bucket, best_id, worse_id, outputname):
    
    red   = ROOT.kRed+2
    blue  = ROOT.kAzure+6
    black = ROOT.kBlack
    canvas = ROOT.TCanvas('c1','c1',2000,1300)
    canvas.Divide(1,4) 
    mse=list();sp=list();det=list();fa=list()
    roc_val=list();roc_op=list()

    for graphs in bucket:
      mse.append( graphs['train']['mse_val'] )
      sp.append( graphs['train']['sp_val'] )
      det.append( graphs['train']['det_val'] )
      fa.append( graphs['train']['fa_val'] )
      roc_val.append( graphs['train']['roc_val'] )
      roc_op.append( graphs['train']['roc_op'] )

    plot_evol(canvas.cd(1),mse,[0,.3],title='Mean Square Error Evolution',
                                       xlabel='epoch #', ylabel='MSE',
                                       select_pos1=best_id,
                                       select_pos2=worse_id,
                                       color_curves=blue,
                                       color_select1=black,
                                       color_select2=red)
    plot_evol(canvas.cd(2),sp,[.93,.97],title='SP Evolution',
                                       xlabel='epoch #', ylabel='SP',
                                       select_pos1=best_id,
                                       select_pos2=worse_id,
                                       color_curves=blue,
                                       color_select1=black,
                                       color_select2=red)
    plot_evol(canvas.cd(3),det,[.95,1],title='Detection Evolution',
                                       xlabel='epoch #',
                                       ylabel='Detection',
                                       select_pos1=best_id,
                                       select_pos2=worse_id,
                                       color_curves=blue,
                                       color_select1=black,
                                       color_select2=red)
    plot_evol(canvas.cd(4),fa,[0,.3],title='False alarm evolution',
                                       xlabel='epoch #', ylabel='False alarm',
                                       select_pos1=best_id,
                                       select_pos2=worse_id,
                                       color_curves=blue,
                                       color_select1=black,
                                       color_select2=red)
     
    canvas.cd(1)
    logoLabel_obj   = ROOT.TLatex(.65,.65,self._logoLabel);
    logoLabel_obj.SetTextSize(.25)
    logoLabel_obj.Draw()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(outputname)
    del canvas 
    canvas = ROOT.TCanvas('c2','c2',2000,1300)
    canvas.Divide(2,1)
    plot_evol(canvas.cd(1),roc_val,[.80,1],title='ROC (Validation)',
                                       xlabel='false alarm',
                                       ylabel='detection',
                                       select_pos1=best_id,
                                       select_pos2=worse_id,
                                       color_curves=blue,
                                       color_select1=black,
                                       color_select2=red)
    plot_evol(canvas.cd(2),roc_op,[.80,.1],title='ROC (Operation)',
                                       xlabel='false alarm', 
                                       ylabel='detection',
                                       select_pos1=best_id,
                                       select_pos2=worse_id,
                                       color_curves=blue,
                                       color_select1=black,
                                       color_select2=red)
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs('roc_'+outputname)
        
















