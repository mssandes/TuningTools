
def SetTuningStyle ():
  print "\nApplying TuningTool style settings..."
  tuningStyle = TuningStyle()
  from ROOT import gROOT
  gROOT.SetStyle("Tuning")
  gROOT.ForceStyle()

def TuningStyle():
  from ROOT import TStyle
  tuningStyle = TStyle("Tuning","Tuning style")
  # use plain black on white colors
  icol=0 # WHITE
  tuningStyle.SetFrameBorderMode(icol)
  tuningStyle.SetFrameFillColor(icol)
  tuningStyle.SetCanvasBorderMode(icol)
  tuningStyle.SetCanvasColor(icol)
  tuningStyle.SetPadBorderMode(icol)
  tuningStyle.SetPadColor(icol)
  tuningStyle.SetStatColor(icol)
  #tuningStyle.SetFillColor(icol) # don't use: white fill color for *all* objects
  # set the paper & margin sizes
  tuningStyle.SetPaperSize(20,26)

  # set margin sizes
  tuningStyle.SetPadTopMargin(0.05)
  tuningStyle.SetPadRightMargin(0.05)
  tuningStyle.SetPadBottomMargin(0.18)
  tuningStyle.SetPadLeftMargin(0.10)

  # set title offsets (for axis label)
  #tuningStyle.SetTitleXOffset(1.4)
  tuningStyle.SetTitleXOffset(0.4)
  #tuningStyle.SetTitleYOffset(1.4)
  tuningStyle.SetTitleYOffset(0.6)

  # use large fonts
  #Int_t font=72 # Helvetica italics
  font=42 # Helvetica
  tsize=0.05
  tuningStyle.SetTextFont(font)

  tuningStyle.SetTextSize(tsize)
  tuningStyle.SetLabelFont(font,"x")
  tuningStyle.SetTitleFont(font,"x")
  tuningStyle.SetLabelFont(font,"y")
  tuningStyle.SetTitleFont(font,"y")
  tuningStyle.SetLabelFont(font,"z")
  tuningStyle.SetTitleFont(font,"z")
  
  tuningStyle.SetLabelSize(tsize,"x")
  tuningStyle.SetTitleSize(tsize,"x")
  tuningStyle.SetLabelSize(tsize,"y")
  tuningStyle.SetTitleSize(tsize,"y")
  tuningStyle.SetLabelSize(tsize,"z")
  tuningStyle.SetTitleSize(tsize,"z")

  # use bold lines and markers
  tuningStyle.SetMarkerStyle(20)
  tuningStyle.SetMarkerSize(1.2)
  tuningStyle.SetHistLineWidth(2)
  tuningStyle.SetLineStyleString(2,"[12 12]") # postscript dashes

  # get rid of X error bars 
  #tuningStyle.SetErrorX(0.001)
  # get rid of error bar caps
  tuningStyle.SetEndErrorSize(0.)

  # do not display any of the standard histogram decorations
  tuningStyle.SetOptTitle(0)
  #tuningStyle.SetOptStat(1111)
  tuningStyle.SetOptStat(0)
  #tuningStyle.SetOptFit(1111)
  tuningStyle.SetOptFit(0)

  # put tick marks on top and RHS of plots
  tuningStyle.SetPadTickX(1)
  tuningStyle.SetPadTickY(1)
  tuningStyle.SetPalette(1)
  return tuningStyle


def Label(x,y,text="",color=1,tsize=0.045):   
  from ROOT import TLatex, gPad
  l = TLatex()
  l.SetNDC();
  l.SetTextFont(42);
  l.SetTextSize(tsize);
  l.SetTextColor(color);
  dely = 0.115*472*gPad.GetWh()/(506*gPad.GetWw());
  l.DrawLatex(x,y-dely,text);


