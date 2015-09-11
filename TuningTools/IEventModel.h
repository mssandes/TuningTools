#ifndef FASTNETTOOL_IEVENTMODEL_H
#define FASTNETTOOL_IEVENTMODEL_H
#include <vector>

#include "Rtypes.h"
//#include "TObject.h"

struct IEventModel /*: public TObject*/ {

  int             RunNumber;

  // Rings!
  std::vector<float> *el_ringsE;
  std::vector<float> *trig_L2_calo_rings;

  // Offline electron cluster
  float         el_et;
  float         el_eta;
  float         el_phi;
  float         el_ethad1;
  float         el_ethad;
  float         el_ehad1;
  float         el_f1;
  float         el_f3;
  float         el_f1core;
  float         el_f3core;
  float         el_weta1;
  float         el_weta2;
  float         el_wtots1;
  float         el_fracs1;
  float         el_Reta;
  float         el_Rphi;
  float         el_Eratio;
  float         el_Rhad;
  float         el_Rhad1;
  // Track combined
  float         el_deta1;
  float         el_deta2;
  float         el_dphi2;
  float         el_dphiresc;

  // Pure track
  float         el_pt;
  float         el_d0;
  float         el_eprobht;
  float         el_charge;
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

  // Selector decision
  bool       el_loose;
  bool       el_medium;
  bool       el_tight;
  bool       el_lhLoose;
  bool       el_lhMedium;
  bool       el_lhTight;
  bool       el_multiLepton;

  // Trigger info
  float         trig_L1_emClus;
  bool          trig_L1_accept;
  bool          trig_L2_calo_accept;
  bool          trig_L2_el_accept;
  bool          trig_EF_calo_accept;
  bool          trig_EF_el_accept;

  bool          mc_hasMC;
  bool          mc_isElectron;
  bool          mc_hasZMother;
  bool          mc_hasWMother;

  //ClassDef(IEventModel,1);
};

#endif // FASTNETTOOL_IEVENTMODEL_H

