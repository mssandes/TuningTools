
#include <vector>

struct IEventModel {

  int             RunNumber;
  Float_t         el_pt;
  Float_t         el_eta;
  Float_t         el_phi;
  bool            el_loose;
  bool            el_medium;
  bool            el_tight;
  bool            el_lhLoose;
  bool            el_lhMedium;
  bool            el_lhTight;
  bool            el_multiLepton;
  bool            trk_nPileupPrimaryVtx;
  bool            trig_L1_emClus;
  bool            trig_L1_accept;
  std::vector<float>  *trig_L2_calo_rings;
  bool            trig_L2_calo_accept;
  bool            trig_L2_el_accept;
  bool            trig_EF_calo_accept;
  bool            trig_EF_el_accept;
  bool            mc_hasMC;
  bool            mc_isElectron;
  bool            mc_hasZMother;
};


