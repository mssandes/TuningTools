from RingerCore.Logger        import Logger
from RingerCore.util          import checkForUnusedVars, calcSP, percentile
from TuningTools.Neural       import Neural
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
  graph.SetMarkerColor(4); graph.SetMarkerStyle(21)
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
    #return pickle.load(open(filename))["tunedDiscriminators"]
    

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

  def __call__(self):
    obj=dict()
    obj['sp']=self.sp
    obj['det']=self.det
    obj['fa']=self.fa
    obj['cut']=self.cut
    obj['cut_id']=self.cut_id
    return obj


class CrossValidStatAnalysis(Logger):

  def __init__(self, inputFiles, neuronsBound, size_sort, **kw):

    Logger.__init__(self,**kw)    
    self._neuronsBound  = neuronsBound
    self._sorts     = size_sort

    count=0
    self._dataReader = DataReader( self._neuronsBound, self._sorts)
    for file in inputFiles:
      #fixes for location
      tmp=file.split('/')
      file_tmp=tmp[len(tmp)-1]
      offset = file_tmp.find('.n'); n = int(file_tmp[offset+2:offset+6])
      offset = file_tmp.find('.s'); s = int(file_tmp[offset+2:offset+6])
      self._logger.info('reading %s... attach position: (%d,%d) / count= %d',file,n, s,count)
      self._dataReader.append(n, s, file)
      count+=1
      
    self._logger.info('There is a totol of %d jobs into your directory',count)


  def __call__(self, stop_criteria, **kw):
    self._detRef    = kw.pop('ref_det',0.9259)
    self._faRef     = kw.pop('ref_det',0.1259)
    self._logoLabel = kw.pop('logoLabel','Fastnet')
    to_matlab       = kw.pop('to_matlab',True)
    outputname      = kw.pop('outputname','crossvalidStatAnalysis.pic')
    self._criteria_network = stop_criteria
    obj=self.loop()
    filehandler = open(outputname,'w')
    self.single_objects(obj)
    pickle.dump(obj,filehandler)
    if to_matlab:
      import scipy.io
      scipy.io.savemat(outputname,mdict={'fastnet':obj})
      self._logger.info('beware! in python the fist index is 0, in matlab is 1!\
          so, when you start your analysis, you must know that in matlab i=1\
          means i=0 in python! e.g. for neuron = 1 the index will be 2 in matlab!!!!')

  def loop(self):

    crit_map          = {'sp':0,'det':1, 'fa':2}
    outputObj=dict()
    #initialize all variables
    for tighteness in ['loose','medium','tight']:
      outputObj[tighteness]=dict()
      for s in ['sp_val','det_val','fa_val','sp_op','det_op','fa_op']:
        outputObj[tighteness][s]  = np.zeros((self._neuronsBound[1]+1,self._sorts))
      outputObj[tighteness]['best_networks']=(self._neuronsBound[1]+1)*[None]


    for n in self._dataReader.rowBoundLooping():
      bucket_sorts=dict(); 
      bucket_sorts['loose']=list();
      bucket_sorts['medium']=list(); 
      bucket_sorts['tight']=list()

      for s in self._dataReader.colBoundLooping():   
        self._logger.info('reading information from pait (%d, %d)',n,s)
        train_data   = self._dataReader(n,s)
        bucket_inits = list()
        count=0
        for train in train_data:
          obj=dict()
          #self._logger.info('count is: %d',count )
          criteria = crit_map[self._criteria_network]
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
          
          tmp=dict(); tmp['perf']=dict() 
          for tighteness in ['loose','medium','tight']: tmp['perf'][tighteness]=self.add_performance(obj,tighteness)           
          tmp['neuron']=n; tmp['sort']=s; tmp['init']=count; tmp['train']=obj
          tmp['network']=dict()
          tmp['network']['nodes']=network.nNodes
          tmp['network']['weights']=network.get_w_array()
          tmp['network']['bias']=network.get_b_array()
          bucket_inits.append(tmp)
          count+=1
        #end of inits
      
        tmp=dict()
        for tighteness in ['loose','medium','tight']:
          [thebest_id_init, theworse_id_init] = self.find_thebest_theworse(bucket_inits,tighteness,'validation')
          self.plot_evol(bucket_inits, thebest_id_init, theworse_id_init, 
              ('%s_neuron_%d_sort_%d_inits_evol.pdf')%(tighteness,n,s))
          objsaved_thebest=bucket_inits[thebest_id_init]
          outputObj[tighteness]['sp_val'][n][s]  = objsaved_thebest['perf'][tighteness]['validation']['sp']
          outputObj[tighteness]['det_val'][n][s] = objsaved_thebest['perf'][tighteness]['validation']['det']                         
          outputObj[tighteness]['fa_val'][n][s]  = objsaved_thebest['perf'][tighteness]['validation']['fa']                               
          outputObj[tighteness]['sp_op'][n][s]   = objsaved_thebest['perf'][tighteness]['operation']['sp']                            
          outputObj[tighteness]['det_op'][n][s]  = objsaved_thebest['perf'][tighteness]['operation']['det']
          outputObj[tighteness]['fa_op'][n][s]   = objsaved_thebest['perf'][tighteness]['operation']['fa']
          bucket_sorts[tighteness].append(objsaved_thebest)
      #end of sorts

      self._logger.info('find the best sort')
      for tighteness in ['loose','medium','tight']:
        [thebest_id_sort, theworse_id_sort] = self.find_thebest_theworse(bucket_sorts[tighteness],tighteness,'operation',
                                                                         remove_outliers=True)
        self.plot_evol(bucket_sorts[tighteness], thebest_id_sort,  theworse_id_sort,('%s_neuron_%d_sort_evol.pdf')%(tighteness,n))
        thebest_network = dict(bucket_sorts[tighteness][thebest_id_sort])
        thebest_network['perf']=thebest_network['perf'][tighteness]
        outputObj[tighteness]['best_networks'][n]=thebest_network
    #end of neurons

    for tighteness in ['loose','medium','tight']:
      for key in ['sp_val','det_val','fa_val','sp_op','det_op','fa_op']:
        outputObj[tighteness][key+'_mean'] = np.mean(outputObj[tighteness][key],axis=0)
        outputObj[tighteness][key+'_std']  = np.std(outputObj[tighteness][key],axis=0)
      outputname  = ('%s_topo_fluctuation.net_stopby_%s.pdf')%(tighteness,self._criteria_network)
      self.plot_topo(outputObj[tighteness],self._neuronsBound, outputname)

    return outputObj

  def getXarray(self, graph):
    bufferx=graph.GetX()
    bufferx.SetSize(graph.GetN())
    return np.array(bufferx,'float_')

  def getYarray(self, graph):
    buffery=graph.GetY()
    buffery.SetSize(graph.GetN())
    return np.array(buffery,'float_')

  def add_performance( self, obj, tighteness):

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
    if tighteness is 'medium':
      perf_op.setValues(perf_val.cut_id)

    #set by detection reference point
    if tighteness is 'loose':
      cut_id = perf_val.searchRefPoint(self._detRef,'det')
      perf_val.setValues(cut_id)
      perf_op.setValues(cut_id)

    #set by false alarm point
    if tighteness is 'tight':
      cut_id = perf_val.searchRefPoint(self._faRef,'fa')
      perf_val.setValues(cut_id)
      perf_op.setValues(cut_id)
    
    tmp=dict()
    tmp['validation'] = perf_val()
    tmp['operation']  = perf_op()
    return tmp



  def find_thebest_theworse(self, bucket, tighteness, key, **kw):

    remove_outliers = kw.pop('remove_outliers',False)
    thebest_value   = 0
    theworse_value  = 99
    thebest_idx     = -1
    theworse_idx    = -1
    outlier_lower = outlier_higher = q1 = q2 = q3 =  0

    if tighteness is 'tight':
      thebest_value = 99; theworse_value = 0

    if remove_outliers:
      data=list()
      for i in range(len(bucket)):
        obj=bucket[i]['perf'][tighteness][key]
        if tighteness is 'medium': data.append(obj['sp'])
        if tighteness is 'loose': data.append(obj['det'])
        if tighteness is 'tight': data.append(obj['fa'])

      q1=self.percentile(data,25.0)
      q2=self.percentile(data,50.0)
      q3=self.percentile(data,75.0)
      outlier_higher = q3+1.5*(q3-q1)
      outlier_lower  = q1-1.5*(q3-q1)

    for i in range(len(bucket)):
      obj=bucket[i]['perf'][tighteness][key]

      if remove_outliers:
        value=0
        if tighteness is 'medium': value = obj['sp']
        if tighteness is 'loose':  value = obj['det']
        if tighteness is 'tight':  value = obj['fa']
        if value > outlier_higher or value < outlier_lower: continue

      #the best
      if tighteness is 'medium' and obj['sp'] > thebest_value:
        thebest_value=obj['sp']; thebest_idx=i
      if tighteness is 'loose' and obj['det'] > thebest_value:
        thebest_value=obj['det']; thebest_idx=i
      if tighteness is 'tight' and obj['fa'] < thebest_value:
        thebest_value=obj['fa']; thebest_idx=i
      #the worse

      if tighteness is 'medium' and obj['sp'] < theworse_value:
        theworse_value=obj['sp']; theworse_idx=i
      if tighteness is 'loose' and obj['det'] < theworse_value:
        theworse_value=obj['det']; theworse_idx=i
      if tighteness is 'tight' and obj['fa'] > theworse_value:
        theworse_value=obj['fa']; theworse_idx=i

    return (thebest_idx, theworse_idx)


  def percentile(self, data, score):
    data = np.sort(data).tolist()
    for i in range(len(data)):
      x=percentile(data, data[i],kind='mean')
      if x == score:  return data[i]
      if x >  score:  return (data[i]+data[i-1])/float(2)

  def plot_topo(self, obj, y_limits, outputname):
    
    canvas = ROOT.TCanvas('c1','c1',2000,1300)
    canvas.Divide(1,3) 
    plot_topo(canvas.cd(1), obj, 'sp_op', y_limits, 'SP fluctuation', '# neuron', 'SP')
    plot_topo(canvas.cd(2), obj, 'det_op', y_limits, 'Detection fluctuation', '# neuron', 'Detection')
    plot_topo(canvas.cd(3), obj, 'fa_op', y_limits, 'False alarm fluctuation', '# neuron', 'False alarm')
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
        


  def single_objects(self, obj):
    #remove tgraph from all objects
    for tighteness in ['loose','medium','tight']:
      for s in range(len(obj[tighteness]['best_networks'])):
        tmp1=obj[tighteness]['best_networks'][s]
        if tmp1 is None: 
          obj[tighteness]['best_networks'][s]=0
          continue
        tmp1=tmp1['train']; tmp=dict()
        for key in ['mse_trn','mse_val','sp_val','det_val','fa_val','mse_tst','sp_tst',
          'det_tst','fa_tst','roc_val_cut','roc_op_cut']: 
           tmp[key]= self.getYarray(tmp1[key])

        tmp['roc_val_fa']=self.getXarray(tmp1['roc_val'])
        tmp['roc_val_det']=self.getYarray(tmp1['roc_val'])
        tmp['roc_op_fa']=self.getXarray(tmp1['roc_op'])
        tmp['roc_op_det']=self.getYarray(tmp1['roc_op'])
        obj[tighteness]['best_networks'][s]['train']=tmp


  def save_network(self, stop_criteria, neuron, sort, init, threshold, outputname):
    crit_map     = {'sp':0,'det':1, 'fa':2}
    train_objs   = self._dataReader(neuron,sort)[init]
    network      = train_objs[crit_map[stop_criteria]][0]
    net = dict()
    net['nodes']      = network.nNodes
    net['threshold']  = threshold
    net['bias']       = network.get_b_array()
    net['weights']    = network.get_w_array()
    pickle.dump(net,open(outputname,'wb'))











