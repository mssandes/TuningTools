


runGRIDtuning.py \
 -c user.jodafons.Config.n5to20.JK.inits_10by10 \
 -d user.jodafons.data17_13TeV.AllPeriods.sgn.probes_EGAM1.bkg.vetoProbes_EGAM7.npz  \
 -p user.jodafons.PreProc.norm1.pic.gz \
 -x user.jodafons.CrossValid.pic.gz \
 -r user.jodafons.data17_13TeV.allPeriods.tight_effs.npz \
 -o user.jodafons.nn.data17_13TeV.AllPeriods.sgn_Zee_EGAM1.bkg_EGAM7.tight_v7.Norm1.t0002 \
 --eta-bin 0  \
 --et-bin 4  \
 --excludedSite ANALY_DESY-HH_UCORE ANALY_BNL_MCORE ANALY_MWT2_SL6 ANALY_MWT2_HIMEM  ANALY_DESY-HH ANALY_FZK_UCORE ANALY_FZU DESY-HH_UCORE FZK-LCG2_UCORE \
 --multi-files \
 -mt \
 #--cloud US \

 #-cloud CERN \
 #--memory 17010 \
 #--dry-run



