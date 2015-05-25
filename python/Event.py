
import pickle
from ROOT import *
from defines import *
from util import *

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


