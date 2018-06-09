


runGRIDtuning.py \
 --do-multi-stop 0 \
 -c user.jodafons.configs.n5to10.jk.inits100_20by20 \
 -d user.jodafons.data17_13TeV.AllPeriods.sgn.probes_EGAM2.bkg.VProbes_EGAM7.GRL_v97.npz  \
 -p user.jodafons.preproc_norm1.pic.gz \
 -x user.jodafons.crossValid_10sorts.pic.gz \
 -r user.jodafons.data17_13TeV.AllPeriods.sgn.probes_EGAM2.bkg.VProbes_EGAM7.GRL_v97-eff.npz \
 -o user.jodafons.nn.data17_13TeV.AllPeriods.sgn_Jpsi_EGAM2.bkg_EGAM7.bestSP.Norm1.v9.t0002 \
 --excludedSite ANALY_DESY-HH_UCORE ANALY_BNL_MCORE ANALY_MWT2_SL6 ANALY_MWT2_HIMEM  ANALY_DESY-HH ANALY_FZK_UCORE ANALY_FZU DESY-HH_UCORE FZK-LCG2_UCORE \
 --et-bins 0 2 \
 --eta-bins 0 4 \
 #--dry-run
 




