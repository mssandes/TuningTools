
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


class DataTrain:
  def __init__(self, train):
    #Train evolution information
    self.epoch = []
    self.mse_trn = []
    self.mse_val = []
    self.sp_val = []
    self.mse_tst = []
    self.sp_tst = []
    self.is_best_mse = []
    self.is_best_sp = []
    self.num_fails_mse = []
    self.num_fails_sp = []
    self.stop_mse = []
    self.stop_sp = []   
    #Get train evolution information from TrainDatapyWrapper
    for i in range(len(train)):
      self.epoch.append(train[i].getEpoch())
      self.mse_trn.append(train[i].getMseTrn())
      self.mse_val.append(train[i].getMseVal())
      self.sp_val.append(train[i].getSPVal())
      self.mse_tst.append(train[i].getMseTst())
      self.sp_tst.append(train[i].getSPTst())
      self.is_best_mse.append(train[i].getIsBestMse())
      self.is_best_sp.append(train[i].getIsBestSP())
      self.num_fails_mse.append(train[i].getNumFailsMse())
      self.num_fails_sp.append(train[i].getNumFailsSP())
      self.stop_mse.append(train[i].getStopMse())
      self.stop_sp.append(train[i].getStopSP())

  def showInfo(self, i = 0):
    print 'epoch          =', self.epoch[i]
    print 'mse_trn        =', self.mse_trn[i]
    print 'mse_val        =', self.mse_val[i]
    print 'sp_val         =', self.sp_val[i]
    print 'mse_tst        =', self.mse_tst[i]
    print 'sp_tst         =', self.sp_tst[i]
    print 'is_best_mse    =', self.is_best_mse[i]
    print 'is_best_sp     =', self.is_best_sp[i]
    print 'num_fails_mse  =', self.num_fails_mse[i]
    print 'num_fails_sp   =', self.num_fails_sp[i]
    print 'stop_mse       =', self.stop_mse[i]
    print 'stop_sp        =', self.stop_sp[i]   
 

class Performance:
  def __init__(self, spVec, detVec, faVec, cutVec):
    self.spVec  = spVec;
    self.cutVec = cutVec;
    self.detVec = detVec;
    self.faVec  = faVec;
    idx = spVec.index(max(spVec));
    self.sp  = spVec[idx];
    self.det = detVec[idx];
    self.fa  = faVec[idx];
    self.cut = cutVec[idx];

  def showInfo(self):
    print 'sp  = ',self.sp
    print 'det = ',self.det
    print 'fa  = ',self.fa
    print 'cut = ',self.cut


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
 
