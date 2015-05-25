
import pickle
from ROOT import *

#Load EventModel.h
gROOT.ProcessLine("#include <IEventModel.h>")

list_of_branchs = [ 'el_pt',
                    'el_eta',
                    'el_phi',
                    'el_loose',
                    'el_medium',
                    'el_tight',
                    'el_lhLoose',
                    'el_lhMedium',
                    'el_lhTight',
                    'trig_L1_emClus',
                    'trig_L1_accept',
                    'trig_L2_calo_rings',
                    'trig_L2_calo_accept',
                    'trig_L2_el_accept',
                    'trig_EF_calo_accept',
                    'trig_EF_el_accept',
                    'mc_hasMC',
                    'mc_isElectron',
                    'mc_hasZMother']

def find_higher( vec, value ):
  return [i for i,x in enumerate(vec) if x > value]
#end

def find_less( vec, value ):
  return [i for i,x in enumerate(vec) if x < value]
#end

#Helper function
def setBranchAddress( tree, varname, var ):
  tree.SetBranchAddress(varname, AddressOf(var,varname) )  
#end

def stdvector_to_list(vec):
    size = vec.size()
    l = size*[0]
    for i in range(size):
      l[i] = vec[i]
    return l
#end


class EventModel:
  def __init__( self, trig, event, doTruth ):

    self.idSample       = 0
    #self.runNumber      = event.RunNumber
    #self.eventNumber    = event.EventNumber
    self.el_pt          = event.el_pt
    self.el_eta         = event.el_eta
    self.el_phi         = event.el_phi
    self.el_loose       = event.el_loose
    self.el_medium      = event.el_medium
    self.el_tight       = event.el_tight
    self.el_lhLoose     = event.el_lhLoose
    self.el_lhMedium    = event.el_lhMedium
    self.el_lhTight     = event.el_lhTight
    self.trig_L2_calo_rings = stdvector_to_list(event.trig_L2_calo_rings)
    
    tgt = -1
    if trig.find('lh'):
      if self.el_lhLoose == True: tgt = 0
      if self.el_lhTight == True: tgt = 1
    else:
      if self.el_loose == True: tgt = 0
      if self.el_tight == True: tgt = 1
 
    self.el_target = tgt
    self.mc_isElectron = event.mc_isElectron
    self.mc_hasZMother = event.mc_hasZMother

    if self.mc_isElectron and self.mc_hasZMother: self.mc_target = 1
    else: self.mc_target = 0

    if doTruth: self.target = self.mc_target
    else: self.target = self.el_target

    #Trigger levels
    self.trig_L1_accept      = event.trig_L1_accept
    self.trig_L2_calo_accept = event.trig_L2_calo_accept
    self.trig_L2_el_accept   = event.trig_L2_el_accept
    self.trig_EF_calo_accept = event.trig_EF_calo_accept 
    self.trig_EF_el_accept   = event.trig_EF_el_accept
  #end

  def showInfo(self):
    print '======= Event ========'
    print 'el_pt        = ', self.el_pt
    print 'el_eta       = ', self.el_eta
    print 'el_phi       = ', self.el_phi
    print 'el_loose     = ', self.el_loose
    print 'el_medium    = ', self.el_medium
    print 'el_tight     = ', self.el_tight
    print 'el_lhLoose   = ', self.el_lhLoose
    print 'el_lhMedium  = ', self.el_lhMedium
    print 'el_lhTight   = ', self.el_lhTight
    print 'mc_target    = ', self.el_target
    print 'el_target    = ', self.mc_target
    print 'target       = ', self.target
  #end    
  


#Main class
class Event:
  def __init__(self, fileName, trig):

    #List of events
    self.event  = []
    self.rings  = []
    self.target = []

    #Open root file
    f  = TFile.Open(fileName, 'read')
    t  = f.Get(trig)

    #Hold the address of all brachs
    event = IEventModel()
    #Connect variables
    for var in list_of_branchs:
      setBranchAddress(t,var,event)
    
    for entry in range(t.GetEntries()):
      t.GetEntry(entry)
      self.event.append( EventModel(trig, event, True) )

    f.Close()
    del f,t
  #end

  def initialize(self):
    eventList = []
    for event in self.event:
      if not event.trig_L1_accept : continue
      if event.target == -1:        continue
      self.rings.append(event.trig_L2_calo_rings)
      self.target.append(event.target)
      eventList.append(event)
    #Update list
    self.event = eventList
  #end

  def get_rings(self, target = None):
    if target == None: 
      return self.rings
    else:
      signal = []
      noise  = []
      indexs = [ find_higher(target,0), find_less(target,0) ]
      indexs[0]
      for i in indexs[0]:
         signal.append(self.rings[i])
      for i in indexs[1]:
         noise.append(self.rings[i])
      return [signal,noise]

  def get_target(self):
    return self.target
  #end

  def normalize(self):
    for i in range(len(self.rings)):
      sum_rings = sum(self.rings[i])
      for j in range(len(self.rings[i])):
        self.rings[i][j] = self.rings[i][j]/float(sum_rings)
  #end


