
import ROOT 
import pickle
from Event import *


fileName = 'mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.e3084_s2045_s2008_r5989_ntupleEGamma_20.1.4.1_v02.root'
location = 'NavNtuple/e24_medium_L1EM20VH'

data = Event(fileName, location)
data.initialize()
data.normalize()

#print data.get_target()
print data.get_rings()
print data.event[0].showInfo()

