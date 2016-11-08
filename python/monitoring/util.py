
def setLabels(histo, labels):
	if ( len(labels)>0 ):
	  for i in range( min( len(labels), histo.GetNbinsX() ) ):
	    bin = i+1
	    histo.GetXaxis().SetBinLabel(bin, labels[i])
	
	  for i in range( histo.GetNbinsX(), min( len(labels), histo.GetNbinsX()+histo.GetNbinsY() ) ):
	    bin = i+1-histo.GetNbinsX()
	    histo.GetYaxis().SetBinLabel(bin, labels[i])


def setBox(pad, hists):
    pad.Update();
    x_begin = 1.
    x_size = .18
    x_dist = .03; 
    for hist in hists:
      histStats = hist.FindObject("stats")
      histStats.SetX1NDC(x_begin-x_dist); histStats.SetX2NDC(x_begin-x_size-x_dist);
      histStats.SetTextColor(hist.GetLineColor())
      x_begin-=x_dist+x_size;


def line(x1,y1,x2,y2,color,style,width, text=''):
  from ROOT import TLine
  l = TLine(x1,y1,x2,y2)
  l.SetNDC(False)
  l.SetLineColor(color)
  l.SetLineWidth(width)
  l.SetLineStyle(style)
  return l


def minmax( curves, idx = 0, percent=0):
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
    if idx > len(y):
      idx=0
    if len(y)>0:
      if max(y[idx::]) > cmax:  cmax = max(y[idx::])
      if min(y[idx::]) < cmin:  cmin = min(y[idx::])
  return cmin*(1-percent), cmax*(1+percent)


