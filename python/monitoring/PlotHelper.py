
__all__ = ["PlotTrainingCurves", "PlotDiscriminants", "PlotRocs", "PlotInits"]

from plots.PlotFunctions import *
from plots.TAxisFunctions import *
from ROOT import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta, kGray
from ROOT import TCanvas, gROOT, kTRUE, TColor, gStyle

class PlotHelper(object):
  
  _keys = ['sp_val','det_val','fa_val','sp_tst','det_tst','fa_tst']

  def __init__( self, objects ):
    self._rawObjects = objects
    self._curves = {}
    # get all grpahs inside of external meta object
    for key in objects.keys():
      self._curves[key] = objects.tolist(key)
    #from pprint import pprint
    #pprint(self._curves)
    # set default
    self.setSP()
    # check if we uses the test dataset
    self._tst_empty = False if self._curves['mse_tst'][0].GetMean(2) > 1e-10 else True

  def setSP(self):
    for key in self._keys:
      self._curves[key] = self._curves['bestsp_point_'+key]

  def setPD(self):
    for key in self._keys:
      self._curves[key] = self._curves['det_point_'+key]

  def setFA(self):
    for key in self._keys:
      self._curves[key] = self._curves['fa_point_'+key]

  def getBoundValues(self):
    return self._rawObjects.getBoundValues()

  def getCurve(self, key, idx):
    idx = self._rawObjects.getBoundValues().index(idx)
    return self._curves[key][idx]

  def getStops(self, idx):
    return {'mse':self._rawObjects.getObject(idx)['mse_stop'],
            'det':self._rawObjects.getObject(idx)['det_stop'],
            'sp' :self._rawObjects.getObject(idx)['sp_stop'],
            'fa' :self._rawObjects.getObject(idx)['fa_stop']}


  def test(self):
    return not self._tst_empty

  # Set option from string
  def setReference( self, ref ):
    if 'Pf' in ref:
      self.setFA()
    elif 'Pd' in ref:
      self.setPD()
    elif 'SP' in ref:
      self.setSP()
    else: # Exception will happing if ref not attend to these options
      raise ValueError('Option reference must be: SP, PD or PF.')

def GetTransparent(color,factor=.5):
  return TColor.GetColorTransparent(color,factor)
 


def GetMinMax( curves, idx = 0, percent=0):
  cmin = 999;  cmax = -999
  for g in curves:
    y = [g.GetY()[i] for i in range(g.GetN())] #Get the list of Y values
    if idx > len(y):
      idx=0
    if len(y)>0:
      if max(y[idx::]) > cmax:  cmax = max(y[idx::])
      if min(y[idx::]) < cmin:  cmin = min(y[idx::])
  return cmin*(1-percent), cmax*(1+percent)


def MakeYLegend( dataset='val', reference='Pd', refValue=None ):
  # Set all ylabels names here
  ylabel = {
            'mse':'MSE ('+dataset+')', 
            'sp' :'SP ('+dataset+')',
            'det': 'Det ('+dataset+')', 
            'fa':'FA ('+dataset+')'
            }
  ref = {'Pd':None,'Pf':None,'SP':None}
  if reference == 'Pd':
    ylabel['det'] += ' [Reference]'
    ylabel['fa']  += ' [benchmark]'
    ref['Pd']=refValue;
  elif reference == 'Pf':
    ylabel['det'] += ' [benchmark]'
    ylabel['fa']  += ' [Reference]'
    ref['Pf']=refValue;
  else:
    ylabel['sp'] += ' [benchmark]'
  return ylabel, ref


def PlotInits( objects, best, worst, reference='Pd',outname=None,key='mse'):

  collect=[]
  gROOT.SetBatch(kTRUE)
  these_colors = [kBlue+1, kRed+2, kGray+1, kMagenta, kBlack, kCyan, kGreen+2]
  these_transcolors=[TColor.GetColorTransparent(c,.5) for c in these_colors]
  plots = PlotHelper( objects )
  plots.setReference( reference )
  drawopt='L' 
  canvas = TCanvas('canvas', 'canvas', 1000, 500)
  FormatCanvasAxes(canvas)
 
  # plot all cunves from cross validation method in gray
  graph_list=[]
  for idx in plots.getBoundValues():
    graph = plots.getCurve(key+'_val',idx)
    graph.SetTitle('')
    graph.SetName(graph.GetName()+'_ShadedProfile')
    graph.SetLineColor(these_transcolors[2]) # gray
    AddHistogram(canvas,graph,drawopt=drawopt)
    graph_list.append(graph)

  # plot the best curve
  graph = plots.getCurve(key+'_val',best)
  graph.SetLineColor(these_colors[0]) # blue
  AddHistogram(canvas,graph,drawopt=drawopt)
  # plot the worst curve
  graph = plots.getCurve(key+'_val',worst)
  graph.SetLineColor(these_colors[1]) # red
  AddHistogram(canvas,graph,drawopt=drawopt)

  #AddTopLabels(canvas, ['Validation', 'Best'])
  #DrawText(can,text_lines,.15,.68,.47,.93,totalentries=4)
  SetAxisLabels(canvas, 'Epoch', key)
  AutoFixAxes(canvas)
  canvas.SaveAs(outname)




def PlotCurves( objects, best, worst, reference='Pd', refValue=None, dataset='val', 
                drawtrn=False, drawtst=False, outname=None, label=None, ylabel='Mean Square Error',key='mse'):
  

  collect=[]
  gROOT.SetBatch(kTRUE)
  these_colors = [kBlue+1, kRed+2, kGray+1, kMagenta, kBlack, kCyan, kGreen+2]
  these_transcolors=[TColor.GetColorTransparent(c,.5) for c in these_colors]
  plots = PlotHelper( objects )
  plots.setReference( reference )
  drawopt='L' 
  canvas = TCanvas('canvas', 'canvas', 1000, 500)
  FormatCanvasAxes(canvas)
 
  # plot all cunves from cross validation method in gray
  graph_list=[]
  for idx in plots.getBoundValues():
    graph = plots.getCurve(key+'_val',idx)
    graph.SetTitle('')
    graph.SetName(graph.GetName()+'_ShadedProfile')
    graph.SetLineColor(these_transcolors[2]) # gray
    AddHistogram(canvas,graph,drawopt=drawopt)
    graph_list.append(graph)

  # plot the best curve
  graph = plots.getCurve(key+'_val',best)
  graph.SetLineColor(these_colors[0]) # blue
  max_best_epoch = graph.GetN()
  AddHistogram(canvas,graph,drawopt=drawopt)
  # plot the worst curve
  graph = plots.getCurve(key+'_val',worst)
  graph.SetLineColor(these_colors[1]) # red
  AddHistogram(canvas,graph,drawopt=drawopt)

  # this only be used for MSE plots
  if drawtrn:
    graph = plots.getCurve(key+'_trn', best)
    graph.SetLineColor(these_transcolors[0]) # blue
    graph.SetMarkerColor(these_transcolors[0]) # blue
    AddHistogram(canvas,graph,drawopt='Lp')

  #AddTopLabels(canvas, ['Validation', 'Best'])
  #DrawText(can,text_lines,.15,.68,.47,.93,totalentries=4)
  SetAxisLabels(canvas, 'Epoch', ylabel)
  if drawtrn:
    AutoFixAxes(canvas)
    ymin, ymax = GetYaxisRanges(canvas,check_all=True,ignorezeros=False,ignoreErrors=True)
  else:   
    ymin,ymax=GetMinMax(graph_list,5,0.05)
    SetYaxisRanges(canvas,ymin,ymax)


  from ROOT import TLine
  stops = plots.getStops(best)
  colors = [kGreen+4,kGreen+3,kGreen+2,kGreen+1]
  for idx, s in enumerate(['mse','det','fa','sp']):
    l = TLine(stops[s],ymin,stops[s],ymax)
    if s == key:
      l.SetLineColor(kGreen+2)
      l.SetLineWidth(2)
    else:
      l.SetLineColor(these_transcolors[6])
    l.Draw()
    collect.append(l)

  xmin, xmax = GetXaxisRanges(canvas,check_all=True)
  
  if xmax > max_best_epoch+150:
    xmax_real=xmax
    xmax=max_best_epoch+150
    SetXaxisRanges(canvas,xmin,xmax)
    from ROOT import TArrow, TLatex
    ar = TArrow(xmax-0.3,(ymax/2.)+ymin,xmax-0.08,(ymax/2.)+ymin,0.02,"|>")
    ar.SetLineColor(kRed)
    ar.SetFillColor(kRed)
    ar.SetAngle(40)
    ar.Draw()
    collect.append(ar)
    lx = TLatex( xmax-0.15, (ymax/2.)+ymin+0.02, ('%d')%(xmax_real))
    lx.SetTextAngle(90)
    lx.Draw()
    collect.append(lx)
  else:
    SetXaxisRanges(canvas,xmin,xmax)

  if refValue:
    l = TLine(xmin,refValue,xmax,refValue)
    l.SetLineColor(kBlack)

    l.SetLineStyle(3)
    l.Draw()
    collect.append(l)

  canvas.SaveAs(outname)




def PlotTrainingCurves( objects, best, worst, reference='Pd', refValue=None,outname=None, label=None, dataset='val'):
 
  ylabel, ref = MakeYLegend( reference=reference, dataset=dataset, refValue=refValue )
  
  # plot all curves for each case
  PlotCurves(objects,best,worst,dataset=dataset,reference=reference,outname=outname+'_mse.pdf',
      label=label,drawtrn=True,ylabel=ylabel['mse'], key='mse')
  PlotCurves(objects,best,worst,dataset=dataset,reference=reference,outname=outname+'_det.pdf',
      label=label,ylabel=ylabel['det'],key='det', refValue=ref[reference])
  PlotCurves(objects,best,worst,dataset=dataset,reference=reference,outname=outname+'_sp.pdf' ,
      label=label,ylabel=ylabel['sp'],key='sp')
  PlotCurves(objects,best,worst,dataset=dataset,reference=reference,outname=outname+'_fa.pdf' ,
      label=label,ylabel=ylabel['fa'],key='fa', refValue=ref[reference])



 
def PlotDiscriminants( objects, best=0, worst=0, outname=None, nsgn=2500,nbkg=1000 ):

  collect=[]
  gROOT.SetBatch(kTRUE) 
  gStyle.SetOptStat(1111)

  plots = PlotHelper( objects )
  drawopt='hist' 
  canvas1 = TCanvas('canvas1', 'canvas1', 500, 500)
  canvas2 = TCanvas('canvas2', 'canvas2', 500, 500)
  canvas3 = TCanvas('canvas3', 'canvas3', 500, 500)
  
  FormatCanvasAxes(canvas1)
  FormatCanvasAxes(canvas2)
  FormatCanvasAxes(canvas3)
 
  canvas1.SetLogy()
  canvas2.SetLogy()
  canvas3.SetLogy()

  from RingerCore.util import Roc_to_histogram
  from ROOT import TH1F
  for idx in plots.getBoundValues():
    roc = plots.getCurve('roc_operation',idx)
    sgn, bkg = Roc_to_histogram(roc, nsgn, nbkg)
    h_sgn = TH1F('Signal',"Signal Distribution;Discriminant;Count",100,-1,1)    
    h_bkg = TH1F('Background',"Background Distribution;Discriminant;Count",100,-1,1)    
    for o in sgn: h_sgn.Fill(o)
    for o in bkg: h_bkg.Fill(o)
    h_sgn.SetLineColor(kBlack)
    h_sgn.SetFillColor(GetTransparent(kGray+2))
    AddHistogram(canvas1, h_sgn,drawopt=drawopt)
    h_bkg.SetLineColor(kRed-3)
    h_bkg.SetFillColor(GetTransparent(kGray+2))
    AddHistogram(canvas2, h_bkg,drawopt=drawopt)
    collect.append((h_sgn,h_bkg))

  # plot signal
  best_objs = collect[objects.getBoundValues().index(best)]
  best_objs[0].SetFillColor(GetTransparent(kBlue))
  best_objs[1].SetFillColor(GetTransparent(kRed))
  
  # clean all
  best_objs[0].SetTitle("")
  best_objs[1].SetTitle("")
  AddHistogram(canvas1,best_objs[0],drawopt=drawopt)
  AddHistogram(canvas2,best_objs[1],drawopt=drawopt)

  AddHistogram(canvas3,best_objs[0],drawopt=drawopt)
  AddHistogram(canvas3,best_objs[1],drawopt=drawopt)
  
  SetAxisLabels(canvas1, 'Discriminant', 'Count')
  SetAxisLabels(canvas2, 'Discriminant', 'Count')
  SetAxisLabels(canvas3, 'Discriminant', 'Count')
  
  AutoFixAxes(canvas1)
  AutoFixAxes(canvas2)
  AutoFixAxes(canvas3)
  
  # Save!
  if outname:
    canvas1.SaveAs(outname+'_sgn_dist.pdf')
    canvas2.SaveAs(outname+'_bkg_dist.pdf')
    canvas3.SaveAs(outname+'_both_best_dist.pdf')


def PlotRocs( objects, best=0, worst=0, reference=None, eps=.05, outname=None,
    xmin=0.0, xmax=0.07, ymin=0.6, ymax=1.05, key='roc_operation'):

  plots = PlotHelper( objects )
  drawopt='L' 
  
  canvas = TCanvas('canvas', 'canvas', 500, 500)
  FormatCanvasAxes(canvas)
 
  for idx in plots.getBoundValues():
    roc = plots.getCurve(key,idx)
    roc.SetTitle("")
    roc.SetLineColor(kGray)
    AddHistogram(canvas,roc,drawopt=drawopt)


  roc_best = plots.getCurve(key,best)
  roc_best.SetLineColor(kBlue+2)
  roc_worst = plots.getCurve(key,worst)
  roc_worst.SetLineColor(kRed+2)
  
  AddHistogram(canvas,roc_best,drawopt=drawopt)
  AddHistogram(canvas,roc_worst,drawopt=drawopt)
  SetAxisLabels(canvas, 'False Alarm', 'Detection')
  SetXaxisRanges(canvas,xmin,xmax)
  SetYaxisRanges(canvas,ymin,ymax)
  canvas.SaveAs(outname+'_'+key+'.pdf')
     




#def plot_rocs(plotObjects, kwargs):
#
#  from ROOT import kCyan, kRed, kGreen, kBlue, kBlack, kMagenta, kGray, kWhite, kYellow
#  Colors = [kBlue, kRed, kMagenta, kBlack, kCyan, kGreen]
#  from RingerCore import StdPair as std_pair
#  from util import line, minmax
#
#  dset        = kwargs['set'] 
#  ref         = kwargs['reference']
#  refVal      = kwargs['refVal']
#  eps         = kwargs['eps']
#  savename    = kwargs['cname']+'.pdf'
#
#  #Some protection
#  if not ('operation' in dset or 'tst' in dset):
#    raise ValueError('Option set must be: tst (test) or val (validation)')
#  if not ('SP' in ref or  'Pd' in ref or 'Pf' in ref):
#    raise ValueError('Option reference must be: SP, Pd or Pf')
#
#  #Create dict to hold all list plots
#  curves = dict()
#  #list of dicts to dict of lists
#  for name in plotObjects.keys():
#    curves[name] = plotObjects.tolist(name)
#
#  paintIdx = kwargs['paintListIdx']# [best, worst] 
#  paintCurves  = [ std_pair(plotObjects.index_correction(paintIdx[0]),kBlack), 
#                   std_pair(plotObjects.index_correction(paintIdx[1]), kRed) ]
#  curves['roc'] = curves['roc_'+dset]
#
#
#  #Start to build all ROOT objects
#  from ROOT import TCanvas, gROOT, kTRUE
#  gROOT.SetBatch(kTRUE)
#  canvas = TCanvas('canvas', 'canvas', 1600, 1300)
# 
#  x_limits = [0.00,0.40]
#  y_limits = [0.6 ,1.03]
#
#  #create dummy graph
#  dummy = curves['roc'][0]
#  dummy.SetTitle( 'Receive Operation Curve' )
#  dummy.GetXaxis().SetTitle('False Alarm')
#  dummy.GetYaxis().SetTitle('Detection')
#  dummy.GetHistogram().SetAxisRange(y_limits[0],y_limits[1],'Y' )
#  dummy.GetHistogram().SetAxisRange(x_limits[0],x_limits[1],'X' )
#  dummy.Draw('AL')
#
#  corredor = None; target = None
#  from ROOT import TBox
#  if ref == 'Pf':
#    corredor = TBox( refVal - eps, y_limits[0], refVal + eps, y_limits[1])
#    target = line(refVal,y_limits[0],refVal,y_limits[1],kBlack,2,1,'')
#  elif ref == 'Pd':
#    corredor = TBox( x_limits[0], refVal - eps, x_limits[1], refVal + eps)
#    target = line( x_limits[0],refVal,x_limits[1], refVal,kBlack,2,1,'')
#   
#  if ref != 'SP':
#    corredor.SetFillColor(kYellow-9)
#    corredor.Draw('same')
#    target.Draw('same')
#    canvas.Modified()
#    canvas.Update()
#
#  #Plot curves
#  for c in curves['roc']:  
#    c.SetLineColor(kGray+1)
#    #c.SetMarkerStyle(7)
#    #c.SetMarkerColor(kBlue)
#    c.SetLineWidth(1)
#    c.SetLineStyle(3)
#    #c.Draw('PLsame')
#    c.Draw('same')
#
#  marker=list()
#  #Paint a specifical curve
#  for pair in paintCurves:
#    curves['roc'][pair.first].SetLineWidth(1)
#    curves['roc'][pair.first].SetLineStyle(1)
#    #curves['roc'][pair.first].SetMarkerStyle(7)
#    #curves['roc'][pair.first].SetMarkerColor(kBlue)
#    curves['roc'][pair.first].SetLineColor(pair.second)
#    #curves['roc'][pair.first].Draw('PLsame')
#    curves['roc'][pair.first].Draw('same')
#
#    if ref == 'SP':
#      faVec = curves['roc'][pair.first].GetX()
#      detVec = curves['roc'][pair.first].GetY()
#      from RingerCore import calcSP
#      spVec = [calcSP(detVec[i], 1-faVec[i]) for i in range(curves['roc'][pair.first].GetN())]
#      imax = spVec.index(max(spVec))
#      from ROOT import TMarker
#      marker.append( TMarker(faVec[imax],detVec[imax],4) )
#      marker[-1].SetMarkerColor(pair.second)
#      marker[-1].Draw('same')
#
#  
#
#  #Update Canvas
#  canvas.Modified()
#  canvas.Update()
#  canvas.SaveAs(savename)
#  del canvas
#
#  return savename



