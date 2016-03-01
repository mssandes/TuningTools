#ifndef TUNINGTOOLS_RINGERPHYSVAL_H
#define TUNINGTOOLS_RINGERPHYSVAL_H
#include <vector>

#include "Rtypes.h"
//#include "TObject.h"

struct RingerPhysVal /*: public TObject*/ {

  UInt_t             RunNumber;

  // Rings!
  std::vector<Float_t> *el_ringsE;
  std::vector<Float_t> *trig_L2_calo_rings;

  // Offline electron cluster
  Float_t         el_et;
  Float_t         el_eta;
  Float_t         el_phi;
  Float_t         el_ethad1;
  Float_t         el_ethad;
  Float_t         el_ehad1;
  Float_t         el_f1;
  Float_t         el_f3;
  Float_t         el_f1core;
  Float_t         el_f3core;
  Float_t         el_weta1;
  Float_t         el_weta2;
  Float_t         el_wtots1;
  Float_t         el_fracs1;
  Float_t         el_Reta;
  Float_t         el_Rphi;
  Float_t         el_Eratio;
  Float_t         el_Rhad;
  Float_t         el_Rhad1;
  // Track combined
  Float_t         el_deta1;
  Float_t         el_deta2;
  Float_t         el_dphi2;
  Float_t         el_dphiresc;

  // Pure track
  Float_t         el_pt;
  Float_t         el_d0;
  Float_t         el_eprobht;
  Float_t         el_charge;
  uint8_t       el_nblayerhits;
  uint8_t       el_nblayerolhits;
  uint8_t       el_npixhits;
  uint8_t       el_npixolhits;
  uint8_t       el_nscthits;
  uint8_t       el_nsctolhits;
  uint8_t       el_ntrthighreshits;
  uint8_t       el_ntrthits;
  uint8_t       el_ntrthighthresolhits;
  uint8_t       el_ntrtolhits;
  uint8_t       el_ntrtxenonhits;
  uint8_t       el_expectblayerhit;

  int           trk_nPileupPrimaryVtx;
  Int_t         el_nPileupPrimaryVtx;

  // Selector decision
  Bool_t       el_loose;
  Bool_t       el_medium;
  Bool_t       el_tight;
  Bool_t       el_lhLoose;
  Bool_t       el_lhMedium;
  Bool_t       el_lhTight;
  Bool_t       el_multiLepton;

  // Trigger info
  Float_t         trig_L1_emClus;
  Bool_t          trig_L1_accept;
  Float_t         trig_L2_calo_et;
  Float_t         trig_L2_calo_eta;
  Bool_t          trig_L2_calo_accept;
  Bool_t          trig_L2_el_accept;
  Bool_t          trig_EF_calo_accept;
  Bool_t          trig_EF_el_accept;

  Bool_t          mc_hasMC;
  Bool_t          mc_isElectron;
  Bool_t          mc_hasZMother;
  Bool_t          mc_hasWMother;

  //ClassDef(RingerPhysVal,1);
};

#endif // TUNINGTOOLS_RINGERPHYSVAL_H

