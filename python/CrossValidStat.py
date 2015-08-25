#!/usr/bin/env python
from FastNetTool.util         import sourceEnvFile, checkForUnusedVars
from FastNetTool.Logger       import Logger
from FastNetTool.plots.macros import boxplot, plot_evol
from ROOT                     import AddressOf, std
import ROOT
import numpy as np
import pickle
 

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


class Storage:
  def __init__(self, filename):  
    self._file        = ROOT.TFile(filename,'recreate')
    self.basepath     = 'root'
    self._current_dir = self.basepath
    self._file.mkdir(self.basepath)
    self._data = dict()

  def save(self):
    self._file.Write()
    self._file.Close()

  def mkdir( self, path ):
    fullpath=self.basepath+'/'+path
    self._current_dir = fullpath
    self._file.mkdir(fullpath)
    self._file.cd(fullpath)

  def cd( self, path = ''):
    if path is '':  self._current_dir = self.basepath; self._file.cd()
    else: 
      fullpath = self.basepath+'/'+path
      self._current_dir=fullpath; self._file.cd(fullpath)

  def addGraph(self, feature, graph):
    graph.SetName(feature)
    fullpath=self._current_dir+'/'+feature
    self._data[fullpath]=graph
    #bug:https://root.cern.ch/root/roottalk/roottalk01/0662.html
    graph.Write(feature)

  def addTree(self, tree):
    fullpath=self._current_dir+'/'+tree.GetName()
    self._data[fullpath]=tree
    #tree.Write(tree.GetName())

  def addHistogram(self, hist):
    fullpath=self._current_dir+'/'+hist.GetName()
    self._data[fullpath]=hist

  def data(self):
    return self._data




class CrossValidStatReader(Logger):

  def __init__(self, inputFiles, neuronsBound, size_sort, **kw):

    Logger.__init__(self,**kw)    
    filename            = kw.pop('filename', 'file_crossvalidstat.root')
    self._ref           = kw.pop('reference',None)

    self._neuronsBound  = neuronsBound
    self._size_sort     = size_sort
    self._storage	= Storage(filename)

    count=0
    self._dataReader = DataReader( self._neuronsBound, self._size_sort)
    for file in inputFiles:
      offset = file.find('.n'); n = int(file[offset+2:offset+6])
      offset = file.find('.s'); s = int(file[offset+2:offset+6])
      self._logger.info('reading %s... attach position: (%d,%d) / count= %d',file,n, s,count)
      self._dataReader.append(n, s, file)
      count+=1
      
    self._logger.info('There is a totol of %d jobs into your directory',count)
    networks = self.execute()
    filehandler = open('networks.pic','w')
    pickle.dump(networks,filehandler,protocol=2)
    self._storage.save()

  '''
    Analysis
  '''
  def best_init(self, n, s):

    train_data   = self._dataReader(n,s)
    networks	   = dict()
    mapping      = {'sp':0,'det':1, 'fa':2}
    best_value    = {'sp':0,'det':0,'fa':99} 
    worse_value   = {'sp':99,'det':99,'fa':0} 
    best_pos      = {'sp':0,'det':0,'fa':0}
    worse_pos     = {'sp':0,'det':0,'fa':0}
    best_network  = {'sp':None,'det':None,'fa':None} 
    worse_network = {'sp':None,'det':None,'fa':None} 

    pos=0
    for train in train_data:
      for criteria in mapping:
        #variables to hold all vectors 
        train_evolution   = train[mapping[criteria]][0].dataTrain
        network           = train[mapping[criteria]][0]
        roc_val           = train[mapping[criteria]][1]
        roc_operation     = train[mapping[criteria]][2]

        if self._ref:
          (roc_val,roc_operation) = self.__adapt_cut(self, roc_val, roc_operation, self._ref[criteria], criteria)

        objects           = (network, roc_val, roc_operation, pos)
        epochs_val        = np.array( range(len(train_evolution.mse_val)),  dtype='float_')
        mse_val           = np.array( train_evolution.mse_val,              dtype='float_')
        sp_val            = np.array( train_evolution.sp_val,               dtype='float_')
        det_val           = np.array( train_evolution.det_val,              dtype='float_')
        fa_val            = np.array( train_evolution.fa_val,               dtype='float_')
        roc_val_det       = np.array( roc_val.detVec,                       dtype='float_')
        roc_val_fa        = np.array( roc_val.faVec,                        dtype='float_')
        roc_op_det        = np.array( roc_operation.detVec,                 dtype='float_')
        roc_op_fa         = np.array( roc_operation.faVec,                  dtype='float_')

        self._storage.mkdir(('networks_%s/neuron_%d/sort_%d/init_%d')%(criteria,n,s,pos))           
        self._storage.addGraph('mse_val',ROOT.TGraph(len(epochs_val),epochs_val,mse_val ))
        self._storage.addGraph('sp_val' ,ROOT.TGraph(len(epochs_val),epochs_val, sp_val ))
        self._storage.addGraph('det_val',ROOT.TGraph(len(epochs_val),epochs_val, det_val))
        self._storage.addGraph('fa_val' ,ROOT.TGraph(len(epochs_val),epochs_val, fa_val ))        
        self._storage.addGraph('roc_val',ROOT.TGraph(len(roc_val_fa),roc_val_fa, roc_val_det ))
        self._storage.addGraph('roc_op',ROOT.TGraph(len(roc_op_fa),roc_op_fa, roc_op_det ))
 
        #choose best init 
        if criteria is 'sp'  and roc_val.sp  > best_value[criteria]:
          best_pos[criteria]= pos; best_value[criteria] = roc_val.sp; best_network[criteria] = objects
        if criteria is 'det'  and roc_val.det  > best_value[criteria]:
          best_pos[criteria]= pos; best_value[criteria] = roc_val.det; best_network[criteria] = objects
        if criteria is 'fa'  and roc_val.fa  < best_value[criteria]:
          best_pos[criteria]= pos; best_value[criteria] = roc_val.fa; best_network[criteria] = objects
  
        #choose worse init 
        if criteria is 'sp'  and roc_val.sp  < worse_value[criteria]:
          worse_pos[criteria]= pos; worse_value[criteria] = roc_val.sp; worse_network[criteria] = objects
        if criteria is 'det'  and roc_val.det  < worse_value[criteria]:
          worse_pos[criteria]= pos; worse_value[criteria] = roc_val.det; worse_network[criteria] = objects
        if criteria is 'fa'  and roc_val.fa  > worse_value[criteria]:
          worse_pos[criteria]= pos; worse_value[criteria] = roc_val.fa; worse_network[criteria] = objects
 
      pos+=1
      #loop over networks
   
    for criteria in mapping:
      self._storage.cd(('networks_%s/neuron_%d/sort_%d')%(criteria,n,s))           
      self.__fill_metadata(best_network[criteria],worse_network[criteria],criteria) 

    return (best_network, worse_network)

  def execute(self):
    
    networks     = list()
    mapping      = {'sp':0,'det':1, 'fa':2}
    
    #loop over neurons
    for n in self._dataReader.rowBoundLooping():

      best_value    = {'sp':0,'det':0,'fa':99} 
      worse_value   = {'sp':99,'det':99,'fa':0} 
      best_pos      = {'sp':0,'det':0,'fa':0}
      worse_pos     = {'sp':0,'det':0,'fa':0}
      best_network  = {'sp':None,'det':None,'fa':None} 
      worse_network = {'sp':None,'det':None,'fa':None} 
      
      pos=0
      #loop over sorts
      for s in self._dataReader.colBoundLooping():  

        self._logger.info('looking for the pair (%d, %d)',n,s)
        object_mapping = self.best_init(n,s)[0]
        for criteria in mapping:
          network       = object_mapping[criteria][0]
          roc_val       = object_mapping[criteria][1]
          roc_operation = object_mapping[criteria][2]
          objects = (network, roc_val, roc_operation, pos)

          #choose best init 
          if criteria is 'sp'  and roc_operation.sp  > best_value[criteria]:
            best_pos[criteria]= pos; best_value[criteria] = roc_operation.sp; best_network[criteria] = objects
          if criteria is 'det'  and roc_operation.det  > best_value[criteria]:
            best_pos[criteria]= pos; best_value[criteria] = roc_operation.det; best_network[criteria] = objects
          if criteria is 'fa'  and roc_operation.fa  < best_value[criteria]:
            best_pos[criteria]= pos; best_value[criteria] = roc_operation.fa; best_network[criteria] = objects
  
          #choose worse init 
          if criteria is 'sp'  and roc_operation.sp  < worse_value[criteria]:
            worse_pos[criteria]= pos; best_value[criteria] = roc_operation.sp; worse_network[criteria] = objects
          if criteria is 'det'  and roc_operation.det  < worse_value[criteria]:
            worse_pos[criteria]= pos; best_value[criteria] = roc_operation.det; worse_network[criteria] = objects
          if criteria is 'fa'  and roc_operation.fa  > worse_value[criteria]:
            worse_pos[criteria]= pos; best_value[criteria] = roc_operation.fa; worse_network[criteria] = objects
 
        pos+=1
        #loop over sorts

      for criteria in mapping: 
        self._storage.cd(('networks_%s/neuron_%d')%(criteria,n))           
        self.__fill_metadata(best_network[criteria],worse_network[criteria],criteria) 

      networks.append((best_network, worse_network))

    return networks


  def __fill_metadata(self, best_network, worse_network, criteria):
    ref=0
    if self._ref: ref=self._ref[criteria] 
    else: ref=-1
    md = {'position'   : ([best_network[3]],   'int'),
          'sp_val'     : ([best_network[1].sp], 'float'),
          'fa_val'     : ([best_network[1].fa], 'float'),
          'det_val'    : ([best_network[1].det],'float'),
          'sp_op'      : ([best_network[2].sp], 'float'),
          'fa_op'      : ([best_network[2].fa], 'float'),
          'det_op'     : ([best_network[2].det],'float'),
          'criteria'   : ([criteria],   'string'),
          'ref'        : ([ref],'int')}
    self._storage.addTree(self.__metadata('metadata_best_network',md))

    md = {'position'   : ([worse_network[3]],   'int'),
          'sp_val'     : ([worse_network[1].sp], 'float'),
          'fa_val'     : ([worse_network[1].fa], 'float'),
          'det_val'    : ([worse_network[1].det],'float'),
          'sp_op'      : ([worse_network[2].sp], 'float'),
          'fa_op'      : ([worse_network[2].fa], 'float'),
          'det_op'     : ([worse_network[2].det],'float'),
          'criteria'   : ([criteria],   'string'),
          'ref'        : ([ref],'int')}
    self._storage.addTree(self.__metadata('metadata_worse_network',md))


  def __metadata(self,name,metadata ):
    t = ROOT.TTree(name, 'tree')
    local=dict()
    for var in metadata:  
      local[var]=std.vector(metadata[var][1])()
      t.Branch(var,local[var])
      for value in metadata[var][0]:  local[var].push_back(value)
    t.Fill()
    return t



  def __adapt_cut(self, valid, operation, referency_value, criteria):

    if criteira is 'sp' : return (test,operation)
    if criteria is 'det': pos =np.where(np.array(valid.detVec) >referency_value)[0][0]-1
    if criteria is 'fa' : pos =np.where(np.array(valid.faVec) > referency_value)[0][0]-1
    
    valid.sp  = valid.spVec[pos]; valid.det = valid.detVec[pos] ;
    valid.fa  = valid.faVec[pos]; valid.cut = valid.cutVec[pos]
    operation.sp  = operation.spVec[pos]; operation.det = operation.detVec[pos]
    operation.fa  = operation.faVec[pos]; operation.cut = operation.cutVec[pos]
    
    return (valid, operation) 


