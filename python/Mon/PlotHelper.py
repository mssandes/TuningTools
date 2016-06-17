#Author: Joao Victor da Fonseca Pinto
#Helper class for plot
from pprint import pprint
from ROOT   import TEnv, TGraph, TCanvas, TLine, TParameter
from ROOT   import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta
from RingerCore import Logger

#Class to hold all objects from monitoring file
class PlotsHolder(Logger):
  #Helper names 
  _paramNames = [ 'mse_stop', 'sp_stop', 'det_stop', 'fa_stop' ]
  _graphNames = [ 'mse_trn' , 'mse_val', 'mse_tst' , 'sp_val',
                  'sp_tst'  , 'det_val', 'det_tst' , 'fa_val',
                  'fa_tst'  , 'det_fitted', 'fa_fitted', 'roc_tst', 
                  'roc_op'  , 'roc_tst_cut', 'roc_op_cut' ] 


  def __init__(self, logger = None):
    #Retrive python logger  
    Logger.__init__( self, logger = logger)  
    self._obj = []
     
  def retrieve(self, rawObj, pathList):

    #Loop to retrieve objects from root rawObj
    self._obj = [dict()]*len(pathList)
    for idx, path in enumerate(pathList):
      for graphName in self._graphNames:
        #Check if key exist into plot holder dict
        self.__retrieve_graph( rawObj, idx, path, graphName )
      #Loop over graphs
      for paramName in self._paramNames:
        self.__retrieve_param(rawObj, idx, path, paramName )
    #Loop over file list
    #pprint(self._obj)

  #Private method:
  def __retrieve_graph(self, rawObj, idx, path, graphName ):
    obj = TGraph()
    rawObj.GetObject( path+'/'+graphName, obj)
    self._obj[idx][graphName] = obj 
    
  #Private method:
  def __retrieve_param(self, rawObj, idx, path, paramName ):
    obj = TParameter("double")()
    rawObj.GetObject( path+'/'+paramName, obj)
    self._obj[idx][paramName] = int(obj.GetVal()) 

  #Public method:
  #Return the object store into obj dict. Can be a list of objects
  #Or a single object
  def getObj(self, idx):
    return self._obj[idx]

  def rawObj(self):
    return self._obj

  def setRawObj(self, obj):
    self._obj = obj

  def append(self, obj):
    self._obj.append(obj)

  def size(self):
    return len(self._obj)

  def clear(self):
    self._obj = []

  def __getitem__(self, idx):
    return self._obj[idx]

  def keys(self):
    return self._graphNames+self._paramNames




#*******************************************************************
#Hemper function: This method is usefull to retrieve
#the min and max value found into a list of TGraphs
#afterIdx is a param that can be set to count after this values
#default is 0. You can adjust this values using the perrncent 
#param default is 0.
#*******************************************************************
def getminmax( curves, idx = 0, percent=0):
  cmin = 999;  cmax = -999
  for g in curves:
    y = [g.GetY()[i] for i in range(g.GetN())] #Get the list of Y values
    if max(y[idx::]) > cmax:  cmax = max(y[idx::])
    if min(y[idx::]) < cmin:  cmin = min(y[idx::])
  return cmin*(1-percent), cmax*(1+percent)

#*******************************************************************
class pair: #Put into RingerCore for future
  def __init__(self, a, b):
    self.first = a
    self.second = b

#*******************************************************************
def plot_curves(tpad, curves, y_limits, **kw):

  title       = kw.pop('title'       , ''    )
  xlabel      = kw.pop('xlabel'      , ''    )
  ylabel      = kw.pop('ylabel'      , ''    )
  paintCurves = kw.pop('paintCurves' , None  )
  colorCurves = kw.pop('colorCurves' , kCyan )
  lines       = kw.pop('lines'       , []    )

  #create dummy graph
  x_max = 0; dummy = None
  for i in range(len(curves)):
    curves[i].SetLineColor(colorCurves)
    x = curves[i].GetXaxis().GetXmax()
    if x > x_max: x_max = x; dummy = curves[i]
  
  dummy.SetTitle( title )
  dummy.GetXaxis().SetTitle(xlabel)
  dummy.GetYaxis().SetTitle(ylabel)
  dummy.GetHistogram().SetAxisRange(y_limits[0],y_limits[1],'Y' )
  dummy.Draw('AL')



  #Plot curves
  for c in curves:  c.Draw('same')

  #Paint a specifical curve
  if paintCurves:
    if len(paintCurves) > len(curves):
      for idx, c in enumerate(curves):
        c.SetLineColor(paintCurves[idx].second)
        c.Draw('same')
    else:  
      for pair in paintCurves:
        curves[pair.first].SetLineColor(pair.second)
        curves[pair.first].Draw('same')

  #Plot lines
  for l in lines:  l.Draw()

  #Update TPad
  tpad.Modified()
  tpad.Update()

 
#*******************************************************************
def line(x1,y1,x2,y2,color,style,width,text=''):
  l = TLine(x1,y1,x2,y2)
  l.SetNDC(False)
  l.SetLineColor(color)
  l.SetLineWidth(width)
  l.SetLineStyle(style)
  return l
#*******************************************************************
#opt is a dict with all option needed to config the figure and the
#curves. The options will be:
# reference: Pd, SP or Pf
# operation: True or False
# set: tst or val
#plot 4 curves
def plot_4c(plotObjects, opt):
  Colors = [kBlue, kRed, kMagenta, kBlack, kCyan, kGreen]

  ref         = opt['reference']
  dset        = opt['set'] 
  isOperation = opt['operation']
  detailed    = True if plotObjects.size() == 1 else False
  percent     = 0.03 #(default for now)
  savename    = opt['cname']+'.pdf'
  lines       = []

  #Some protection
  if not ('val' in dset or 'tst' in dset):
    raise ValueError('Option set must be: tst (test) or val (validation)')
  if not ('SP' in ref or  'Pd' in ref or 'Pf' in ref):
    raise ValueError('Option reference must be: SP, Pd or Pf')

  ylabel = {'mse':'MSE ('+dset+')', 'sp':'SP ('+dset+')',
            'det': 'Det ('+dset+')', 
            'fa':'FA ('+dset+')'}

  #Create dict to hold all list plots
  curves = dict()
  #list of dicts to dict of lists
  for name in plotObjects.keys():
    curves[name] = [plotObjects[idx][name] for idx in range(plotObjects.size())]

  #Adapt reference into the validation set
  if ref == 'Pd':
    curves['det_val'] = curves['det_fitted']
  elif ref == 'Pf':
    curves['fa_val']  = curves['fa_fitted']

  #check if the test set is zeros
  hasTst = True if curves['mse_tst'][0].GetMean(2) > 0 else False


  if dset == 'tst' and not hasTst:
    print 'We dont have test set into plotObjects, abort plot!'
    return False

  #If only one plot per key, enabled analysis using all sets
  if detailed:
    #train, validation and test
    paint_curves  = [pair(i,Colors[i]) for i in range(3)]
    curves['mse'] = [curves['mse_trn'][0], curves['mse_val'][0]]
    curves['sp']  = [curves['sp_val'][0]] 
    curves['det'] = [curves['det_val'][0]]
    curves['fa']  = [curves['fa_val'][0]]
    if hasTst:
      for key in ['mse', 'sp', 'det', 'fa']:
        curves[key].append( curves[key+'_tst'][0])
    ylabel = {'mse':'MSE', 'sp':'SP','det': 'Det', 'fa':'FA'}


  else:#Do analysis for each set type
    paintIdx = opt['paintListIdx']# [best, worst]
    paint_curves  = [ pair(paintIdx[0],kBlack), pair(paintIdx[1], kRed) ]
    curves['mse'] = curves['mse_'+dset]
    curves['sp']  = curves['sp_'+dset]
    curves['det'] = curves['det_'+dset]
    curves['fa']  = curves['fa_'+dset]

  #Adapt the legend and percent vec
  pmask = [1,1,1,1]
  if ref == 'Pd':
    ylabel['det'] = ylabel['det']+' [Reference]'
    ylabel['fa'] = ylabel['fa']+' [benchmark]'
    pmask = [1,1,0,1]
  elif ref == 'Pf':
    ylabel['det'] = ylabel['det']+' [benchmark]'
    ylabel['fa'] = ylabel['fa']+' [Reference]'
    pmask = [1,1,1,0]
  else:
    ylabel['sp'] = ylabel['sp']+' [benchmark]'


  #Build lines 
  lines = {'mse':[],'sp':[],'det':[],'fa':[]}
  if detailed:# Hard code setting lines
    y=dict();  x=dict()
    for idx, key in enumerate(['mse','sp','det','fa']):
      y[key] = getminmax( curves[key], 8, pmask[idx]*percent)
      x[key] = curves[key+'_stop'][0]
    #Colors = [kBlue, kRed, kMagenta, kBlack, kCyan, kGreen]
    lines['mse'].append( line(x['mse'],y['mse'][0],x['mse'],y['mse'][1], Colors[3],1,4) )
    lines['sp'].append(  line(x['sp'] ,y['sp'][0] ,x['sp'] ,y['sp'][1] , Colors[3],1,4) )
    if ref == 'Pd':
      lines['det'].append(line(x['det'],y['det'][0],x['det'],y['det'][1], Colors[2],1,5))
      lines['fa'].append( line(x['det'],y['fa'][0] ,x['det'],y['fa'][1] , Colors[2],2,5))
    elif ref == 'Pf':
      lines['det'].append(line(x['fa'],y['det'][0],x['fa'],y['det'][1], Colors[2],2,5))
      lines['fa'].append(line(x['fa'] ,y['fa'][0] ,x['fa'],y['fa'][1] , Colors[2],1,5))


  #Start to build all ROOT objects
  canvas = TCanvas('canvas', 'canvas', 1600, 1300)
  canvas.Divide(1,4) 

  for idx, key in enumerate(['mse','sp','det','fa']):
    #There are more plots
    plot_curves( canvas.cd(idx+1), curves[key],
                 getminmax( curves[key], 8, pmask[idx]*percent),
                 xlabel       = 'Epoch',
                 ylabel       = ylabel[key],
                 paintCurves  = paint_curves,
                 lines        = lines[key])
  #Loop over plots

  #Check if there is any label
  if 'label' in opt.keys():
    canvas.cd(1)
    from TuningStyle import Label
    Label(0.6,0.8,opt['label'],1,0.15)

  canvas.Modified()
  canvas.Update()
  canvas.SaveAs(savename)
  del canvas

  return savename
#*********************************************************************************
def plot_nnoutput( plotObject, opt):
  
  savename = opt['cname']+'.pdf'
  from ROOT import TH1F, TCanvas
  from RingerCore.util import Roc_to_histogram
  curve = plotObject[0][opt['rocname']]
  signal, background = Roc_to_histogram(curve, opt['nsignal'], opt['nbackground'])
  hist_signal = TH1F('Discriminator output','Discriminator output;output;count',100,-1,1)
  hist_background = TH1F('','',100,-1,1)
  for out in signal:  hist_signal.Fill(out)
  for out in background:  hist_background.Fill(out)
  canvas = TCanvas('canvas','canvas', 800, 600)
  canvas.SetLogy()
  hist_signal.SetStats(0)
  hist_background.SetStats(0)
  hist_signal.SetLineColor( kBlack )
  hist_background.SetLineColor( kRed )

  #hist_signal.GetXaxis().SetTitleSize(0.05);
  #hist_signal.GetYaxis().SetTitleSize(0.05);
  #hist_background.GetXaxis().SetTitleSize(0.05);
  #hist_background.GetYaxis().SetTitleSize(0.05);

  hist_signal.Draw()
  hist_background.Draw('same')
  canvas.SaveAs(savename)
  return savename






