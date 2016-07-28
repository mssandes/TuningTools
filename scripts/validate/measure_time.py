#!/usr/bin/env python
import cProfile, pstats, StringIO
pr = cProfile.Profile()
pr.enable()
from TuningTools import TunedDiscrArchieve
for _ in range(10):
  a  = TunedDiscrArchieve.load('/afs/cern.ch/work/j/jodafons/public/Tuning201603XX/user.wsfreund.nn.norm1.newstop.mc14_13TeV.147406.129160.sgn.offLH.bkg.truth.trig.ef.e24_lhmedium_nod0_tunedDiscrXYZ.tgz/user.wsfreund.8194608._000995.tunedDiscrXYZ.tgz')
pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
