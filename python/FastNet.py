
import sys
sys.path.append('../../RootCoreBin/lib/x86_64-slc6-gcc48-opt/')
from libFastNetTool import FastnetPyWrapper


class FastNet(FastnetPyWrapper)
  def __init__(self, trfData, trfTarget, valData, valTarget, tstData)
    FastnetPyWrapper.__init__(self)
    self.trfData   = trfData
    self.trfTarget = trfTarget
    self.valData   = valData
    self.valTarget = valTarget
    self.tstData   = tstData




