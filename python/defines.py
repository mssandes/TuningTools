
from util import *

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
      self.epoch.append(train[i].epoch)
      self.mse_trn.append(train[i].mseTrn)
      self.mse_val.append(train[i].mseVal)
      self.sp_val.append(train[i].SPVal)
      self.mse_tst.append(train[i].mseTst)
      self.sp_tst.append(train[i].SPTst)
      self.is_best_mse.append(train[i].isBestMse)
      self.is_best_sp.append(train[i].isBestSP)
      self.num_fails_mse.append(train[i].numFailsMse)
      self.num_fails_sp.append(train[i].numFailsSP)
      self.stop_mse.append(train[i].stopMse)
      self.stop_sp.append(train[i].stopSP)

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


