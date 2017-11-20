
runGRIDtuning.py \
 -c user.jodafons.config.n5to20.JK.inits_100by100 \
 -d user.jodafons.data17_13TeV.allPeriods.sgn.probes_EGAM1.bkg.vetoProbes_EGAM7.npz  \
 -p user.jodafons.ppFile_norm1.pic.gz \
 -x user.jodafons.CrossValid.pic.gz \
 -r user.jodafons.data17_13TeV.allPeriods.tight_effs.npz \
 -o user.jodafons.nn.data17_13TeV.allPeriods.sgn_Zee_EGAM1.bkg_EGAM7.tight_v7.t0002 \
 --eta-bin 0 4 \
 --et-bin 0 4 \
 --excludedSite ANALY_BNL_MCORE \
 --multi-files \
 #-mt \
 #--memory 17010 \
 #--dry-run

