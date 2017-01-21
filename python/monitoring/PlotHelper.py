# Plot helper functions

def plot_4c(plotObjects, kwargs):
  """
  kwargs is a dict with all kwargsion needed to config the figure and the
  curves. The kwargsions will be:
   reference: Pd, SP or Pf
   operation: True or False
   set: tst or val
  plot 4 curves
  """
  from ROOT import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta, kGray
  Colors = [kBlue, kRed, kMagenta, kBlack, kCyan, kGreen]
  from RingerCore import StdPair as std_pair
  from util import line

  ref         = kwargs['reference']
  refVal      = kwargs['refVal']
  dset        = kwargs['set'] 
  isOperation = kwargs['operation']
  detailed    = True if plotObjects.size() == 1 else False
  percent     = 0.03 #(default for now)
  savename    = kwargs['cname']+'.pdf'
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
  
  from util import minmax

  #check if the test set is zeros
  hasTst = True if curves['mse_tst'][0].GetMean(2) > 1e-10 else False


  if dset == 'tst' and not hasTst:
    print 'We dont have test set into plotObjects, abort plot!'
    return False

  #If only one plot per key, enabled analysis using all sets
  if detailed:
    #train, validation and test
    paint_curves  = [std_pair(i,Colors[i]) for i in range(3)]
    curves['mse'] = [curves['mse_trn'][0], curves['mse_val'][0]]
    curves['sp']  = [curves['sp_val'][0]] 
    curves['det'] = [curves['det_val'][0]]
    curves['fa']  = [curves['fa_val'][0]]
    if hasTst:
      for key in ['mse', 'sp', 'det', 'fa']:
        curves[key].append( curves[key+'_tst'][0])
    ylabel = {'mse':'MSE', 'sp':'SP','det': 'Det', 'fa':'FA'}


  else:#Do analysis for each set type
    paintIdx = kwargs['paintListIdx']# [best, worst]
    paint_curves  = [ std_pair(plotObjects.index_correction(paintIdx[0]),kBlack), 
                      std_pair(plotObjects.index_correction(paintIdx[1]), kRed) ]
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
  lines = {'mse':[],'sp':[],'det':[],'fa':[],'ref':None}
  if detailed:# Hard code setting lines
    y=dict();  x=dict()
    for idx, key in enumerate(['mse','sp','det','fa']):
      y[key] = minmax( curves[key], 8, pmask[idx]*percent)
      x[key] = curves[key+'_stop'][0]
    #Colors = [kBlue, kRed, kMagenta, kBlack, kCyan, kGreen]
    lines['mse'].append( line(x['mse'],y['mse'][0],x['mse'],y['mse'][1], Colors[3],1,2) )
    lines['sp'].append(  line(x['sp'] ,y['sp'][0] ,x['sp'] ,y['sp'][1] , Colors[3],1,2) )
    if ref == 'Pd':
      lines['det'].append(line(x['det'],y['det'][0],x['det'],y['det'][1], Colors[2],1,2))
      lines['fa'].append( line(x['det'],y['fa'][0] ,x['det'],y['fa'][1] , Colors[2],2,2))
    if ref == 'Pf':
      lines['det'].append(line(x['fa'],y['det'][0],x['fa'],y['det'][1], Colors[2],2,2))
      lines['fa'].append(line(x['fa'] ,y['fa'][0] ,x['fa'],y['fa'][1] , Colors[2],1,2))



  #Start to build all ROOT objects
  from ROOT import TCanvas, gROOT, kTRUE
  gROOT.SetBatch(kTRUE)
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
    return x_max
  #__plot_curves end

  xlimits = list()
  for idx, key in enumerate(['mse','sp','det','fa']):
    #There are more plots
    x_max = __plot_curves( canvas.cd(idx+1), curves[key],
                 minmax( curves[key], 8, pmask[idx]*percent),
                 xlabel       = 'Epoch',
                 ylabel       = ylabel[key],
                 paintCurves  = paint_curves,
                 colorCurves  = kGray+1,
                 lines        = lines[key])
    xlimits.append(x_max)
  #Loop over plots

  #Check if there is any label
  if 'label' in kwargs.keys():
    tpad = canvas.cd(1)
    from TuningStyle import Label
    Label(0.6,0.7,kwargs['label'],1,0.15)
    tpad.Modified(); tpad.Update()
 

  # Reference base line 
  if ref == 'Pd':
    tpad = canvas.cd(3)
    lines['ref'] = line(0.0, refVal, xlimits[2], refVal, kGreen, 2,1)
    lines['ref'].Draw()
    tpad.Modified(); tpad.Update()
  if ref == 'Pf':
    tpad = canvas.cd(4)
    lines['ref'] = line(0.0, refVal, xlimits[3], refVal, kGreen, 2,1)
    lines['ref'].Draw()
    tpad.Modified(); tpad.Update()

  canvas.Modified()
  canvas.Update()
  canvas.SaveAs(savename)
  del canvas
  return savename

def plot_nnoutput( plotObject, kwargs):
  
  savename = kwargs['cname']+'.pdf'
  cut = kwargs['cut']
 
  from ROOT import TH1F, TCanvas, gROOT, gStyle, kTRUE
  gROOT.SetBatch(kTRUE)
  from ROOT import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta,gPad
  from RingerCore.util import Roc_to_histogram
  from util import setBox, line
  
  gStyle.SetOptStat(1111)
  curve = plotObject[0][kwargs['rocname']]
  signal, background = Roc_to_histogram(curve, kwargs['nsignal'], kwargs['nbackground'])
  hist_signal = TH1F('Signal','dist output;output;count',100,-1,1)
  hist_background = TH1F('Background','dist output;output;count',100,-1,1)
  for out in signal:  hist_signal.Fill(out)
  for out in background:  hist_background.Fill(out)
  canvas = TCanvas('canvas','canvas', 800, 600)

  hist_signal.SetStats(1)
  hist_background.SetStats(1)
  hist_signal.SetLineColor( kBlack )
  hist_background.SetLineColor( kRed )
  #hist_signal.GetXaxis().SetTitleSize(0.05);
  #hist_signal.GetYaxis().SetTitleSize(0.05);
  #hist_background.GetXaxis().SetTitleSize(0.05);
  #hist_background.GetYaxis().SetTitleSize(0.05);
  if hist_signal.GetEntries() > hist_background.GetEntries():
    hist_signal.Draw()
    hist_background.Draw('sames')
  else:
    hist_background.Draw()
    hist_signal.Draw('sames')
  
  canvas.SetLogy()
  setBox(gPad,[hist_signal, hist_background])
  l = line(cut, 0, cut ,1000, kBlue, 2,2)
  l.Draw()
  canvas.SaveAs(savename)
  return savename


def plot_rocs(plotObjects, kwargs):

  from ROOT import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta, kGray, kWhite, kYellow
  Colors = [kBlue, kRed, kMagenta, kBlack, kCyan, kGreen]
  from RingerCore import StdPair as std_pair
  from util import line, minmax

  dset        = kwargs['set'] 
  ref         = kwargs['reference']
  refVal      = kwargs['refVal']
  eps         = kwargs['eps']
  savename    = kwargs['cname']+'.pdf'

  #Some protection
  if not ('operation' in dset or 'tst' in dset):
    raise ValueError('Option set must be: tst (test) or val (validation)')
  if not ('SP' in ref or  'Pd' in ref or 'Pf' in ref):
    raise ValueError('Option reference must be: SP, Pd or Pf')

  #Create dict to hold all list plots
  curves = dict()
  #list of dicts to dict of lists
  for name in plotObjects.keys():
    curves[name] = plotObjects.tolist(name)

  paintIdx = kwargs['paintListIdx']# [best, worst] 
  paintCurves  = [ std_pair(plotObjects.index_correction(paintIdx[0]),kBlack), 
                   std_pair(plotObjects.index_correction(paintIdx[1]), kRed) ]
  curves['roc'] = curves['roc_'+dset]


  #Start to build all ROOT objects
  from ROOT import TCanvas, gROOT, kTRUE
  gROOT.SetBatch(kTRUE)
  canvas = TCanvas('canvas', 'canvas', 1600, 1300)
 
  x_limits = [0.00,0.40]
  y_limits = [0.6 ,1.03]

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
    corredor = TBox( refVal - eps, y_limits[0], refVal + eps, y_limits[1])
    target = line(refVal,y_limits[0],refVal,y_limits[1],kBlack,2,1,'')
  elif ref == 'Pd':
    corredor = TBox( x_limits[0], refVal - eps, x_limits[1], refVal + eps)
    target = line( x_limits[0],refVal,x_limits[1], refVal,kBlack,2,1,'')
   
  if ref != 'SP':
    corredor.SetFillColor(kYellow-9)
    corredor.Draw('same')
    target.Draw('same')
    canvas.Modified()
    canvas.Update()

  #Plot curves
  for c in curves['roc']:  
    c.SetLineColor(kGray+1)
    #c.SetMarkerStyle(7)
    #c.SetMarkerColor(kBlue)
    c.SetLineWidth(1)
    c.SetLineStyle(3)
    #c.Draw('PLsame')
    c.Draw('same')

  marker=list()
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
      marker.append( TMarker(faVec[imax],detVec[imax],4) )
      marker[-1].SetMarkerColor(pair.second)
      marker[-1].Draw('same')

  

  #Update Canvas
  canvas.Modified()
  canvas.Update()
  canvas.SaveAs(savename)
  del canvas

  return savename



