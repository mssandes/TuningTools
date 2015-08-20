import ROOT

def boxplot( canvas, th2f, y_axis_limits, **kw):
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


