# Plot helper functions


class std__Pair( object ): # TODO Should be on RingerCore
  def __init__(self, a, b):
    self.first = a
    self.second = b


def line(x1,y1,x2,y2,color,style,width,text=''):
  from ROOT import TLine
  l = TLine(x1,y1,x2,y2)
  l.SetNDC(False)
  l.SetLineColor(color)
  l.SetLineWidth(width)
  l.SetLineStyle(style)
  return l


def getminmax( curves, idx = 0, percent=0):
  """
  Helper function: This method is usefull to retrieve
  the min and max value found into a list of TGraphs
  afterIdx is a param that can be set to count after this values
  default is 0. You can adjust this values using the perrncent 
  param default is 0.
  """
  cmin = 999;  cmax = -999
  for g in curves:
    y = [g.GetY()[i] for i in range(g.GetN())] #Get the list of Y values
    if max(y[idx::]) > cmax:  cmax = max(y[idx::])
    if min(y[idx::]) < cmin:  cmin = min(y[idx::])
  return cmin*(1-percent), cmax*(1+percent)


def plot_4c(plotObjects, opt):
  """
  opt is a dict with all option needed to config the figure and the
  curves. The options will be:
   reference: Pd, SP or Pf
   operation: True or False
   set: tst or val
  plot 4 curves
  """
  from ROOT import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta
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
    curves[name] = plotObjects.tolist(name) 

  #Adapt reference into the validation set
  #mse_trn, mse_val, mse_tst
  if ref == 'Pd':
    curves['sp_val']  = curves['det_point_sp_val']
    curves['det_val'] = curves['det_point_det_val'] # det_fitted
    curves['fa_val']  = curves['det_point_fa_val']
    curves['sp_tst']  = curves['det_point_sp_tst']
    curves['det_tst'] = curves['det_point_det_tst']
    curves['fa_tst']  = curves['det_point_fa_tst']
  elif ref == 'Pf':
    curves['sp_val']  = curves['fa_point_sp_val']
    curves['det_val'] = curves['fa_point_det_val'] 
    curves['fa_val']  = curves['fa_point_fa_val'] # fa_fitted
    curves['sp_tst']  = curves['fa_point_sp_tst']
    curves['det_tst'] = curves['fa_point_det_tst']
    curves['fa_tst']  = curves['fa_point_fa_tst']
  else:# ref == 'SP'
    curves['sp_val']  = curves['bestsp_point_sp_val'] # best SP curve
    curves['det_val'] = curves['bestsp_point_det_val'] 
    curves['fa_val']  = curves['bestsp_point_fa_val'] 
    curves['sp_tst']  = curves['bestsp_point_sp_tst']
    curves['det_tst'] = curves['bestsp_point_det_tst']
    curves['fa_tst']  = curves['bestsp_point_fa_tst']

  #check if the test set is zeros
  hasTst = True if curves['mse_tst'][0].GetMean(2) > 0 else False

  if dset == 'tst' and not hasTst:
    print 'We dont have test set into plotObjects, abort plot!'
    return False

  #If only one plot per key, enabled analysis using all sets
  if detailed:
    #train, validation and test
    paint_curves  = [std__Pair(i,Colors[i]) for i in range(3)]
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
    paint_curves  = [ std__Pair(plotObjects.index_correction(paintIdx[0]),kBlack), 
                      std__Pair(plotObjects.index_correction(paintIdx[1]), kRed) ]
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
  from ROOT import TCanvas
  canvas = TCanvas('canvas', 'canvas', 1600, 1300)
  canvas.Divide(1,4) 

  def __plot_curves(tpad, curves, y_limits, **kw):
    from ROOT import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta, kGray
    title       = kw.pop('title'       , ''    )
    xlabel      = kw.pop('xlabel'      , ''    )
    ylabel      = kw.pop('ylabel'      , ''    )
    paintCurves = kw.pop('paintCurves' , None  )
    colorCurves = kw.pop('colorCurves' , kGray )
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
    for c in curves:  
      c.SetLineWidth(1)
      c.SetLineStyle(3)
      c.Draw('same')
    #Paint a specifical curve
    if paintCurves:
      if len(paintCurves) > len(curves):
        for idx, c in enumerate(curves):
          c.SetLineWidth(1)
          c.SetLineColor(paintCurves[idx].second)
          c.SetLineStyle(1)
          c.Draw('same')
      else:  
        for pair in paintCurves:
          curves[pair.first].SetLineWidth(1)
          curves[pair.first].SetLineStyle(1)
          curves[pair.first].SetLineColor(pair.second)
          curves[pair.first].Draw('same')
    #Plot lines
    for l in lines:  l.Draw()
    #Update TPad
    tpad.Modified()
    tpad.Update()
  #__plot_curves end

  for idx, key in enumerate(['mse','sp','det','fa']):
    #There are more plots
    __plot_curves( canvas.cd(idx+1), curves[key],
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
    Label(0.6,0.7,opt['label'],1,0.15)

  canvas.Modified()
  canvas.Update()
  canvas.SaveAs(savename)
  del canvas

  return savename

def plot_nnoutput( plotObject, opt):
  
  savename = opt['cname']+'.pdf'
  from ROOT import TH1F, TCanvas
  from ROOT import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta
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


def plot_rocs(plotObjects, opt):

  from ROOT import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta, kGray, kWhite, kYellow
  Colors = [kBlue, kRed, kMagenta, kBlack, kCyan, kGreen]

  dset        = opt['set'] 
  ref         = opt['reference']
  refVal      = opt['refVal']
  corredorVal = opt['corredorVal']
  savename    = opt['cname']+'.pdf'

  #Some protection
  if not ('val' in dset or 'tst' in dset):
    raise ValueError('Option set must be: tst (test) or val (validation)')
  if not ('SP' in ref or  'Pd' in ref or 'Pf' in ref):
    raise ValueError('Option reference must be: SP, Pd or Pf')

  #Create dict to hold all list plots
  curves = dict()
  #list of dicts to dict of lists
  for name in plotObjects.keys():
    curves[name] = plotObjects.tolist(name)

  paintIdx = opt['paintListIdx']# [best, worst] 
  paintCurves  = [ std__Pair(plotObjects.index_correction(paintIdx[0]),kBlack), 
                   std__Pair(plotObjects.index_correction(paintIdx[1]), kRed) ]
  curves['roc'] = curves['roc_'+dset]


  #Start to build all ROOT objects
  from ROOT import TCanvas
  canvas = TCanvas('canvas', 'canvas', 1600, 1300)
 
  x_limits = [0,0.2]
  y_limits = [0.7,1.1]

  #create dummy graph
  dummy = curves['roc'][0]
  dummy.SetTitle( 'Receive Operation Curve' )
  dummy.GetXaxis().SetTitle('False Alarm')
  dummy.GetYaxis().SetTitle('Detection')
  dummy.GetHistogram().SetAxisRange(y_limits[0],y_limits[1],'Y' )
  dummy.GetHistogram().SetAxisRange(x_limits[0],x_limits[1],'X' )
  dummy.Draw('AL')

  corredor = None; target = None
  from ROOT import TBox
  if ref == 'Pf':
    corredor = TBox( refVal - corredorVal, y_limits[0], refVal + corredorVal, y_limits[1])
    target = line(refVal,y_limits[0],refVal,y_limits[1],kBlack,2,1,'')
  elif ref == 'Pd':
    corredor = TBox( x_limits[0], refVal - corredorVal, x_limits[1], refVal + corredorVal)
    target = line( x_limits[0],refVal,x_limits[1], refVal,kBlack,2,1,'')
  
  
  if ref != 'SP':
    corredor.SetFillColor(kMagenta+5)
    corredor.Draw('same')
    target.Draw('same')
    canvas.Modified()
    canvas.Update()

  #Plot curves
  for c in curves['roc']:  
    c.SetLineColor(kGray-1)
    #c.SetMarkerStyle(7)
    #c.SetMarkerColor(kBlue)
    c.SetLineWidth(1)
    c.SetLineStyle(3)
    #c.Draw('PLsame')
    c.Draw('same')

  marker=None
  #Paint a specifical curve
  for pair in paintCurves:
    curves['roc'][pair.first].SetLineWidth(1)
    curves['roc'][pair.first].SetLineStyle(1)
    #curves['roc'][pair.first].SetMarkerStyle(7)
    #curves['roc'][pair.first].SetMarkerColor(kBlue)
    curves['roc'][pair.first].SetLineColor(pair.second)
    #curves['roc'][pair.first].Draw('PLsame')
    curves['roc'][pair.first].Draw('same')

    if ref == 'SP':
      faVec = curves['roc'][pair.first].GetX()
      detVec = curves['roc'][pair.first].GetY()
      from RingerCore import calcSP
      spVec = [calcSP(detVec[i], 1-faVec[i]) for i in range(curves['roc'][pair.first].GetN())]
      imax = spVec.index(max(spVec))
      from ROOT import TMarker
      marker = TMarker(faVec[imax],detVec[imax],29)
      marker.Draw('same')

  #Update Canvas
  canvas.Modified()
  canvas.Update()
  canvas.SaveAs(savename)
  del canvas

  return savename


