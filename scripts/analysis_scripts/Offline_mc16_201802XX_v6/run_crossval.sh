
crossValStatAnalysis.py \
  --output-level INFO \
  -d data/tuning/user.jodafons.nn.mc16a.zee.20M.jf17.20M.offline.binned.track.wdatadrivenlh.v6.t0002_td/ \
  --data data/files/mc16track_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.track.wdatadrivenlh.npz \
  -r data/files/mc16track_lhgrid_v3/mc16a.zee.20M.jf17.20M.offline.binned.track.wdatadrivenlh_eff.npz \
  --crossFile data/files/user.jodafons.crossValid.10sorts.pic.gz/crossValid.10sorts.pic.gz \
  --binFilters GridJobFilter \
  --operation Offline_LH_DataDriven2016_Rel21_Medium \
  --always-use-SP-network \
  --pile-up-ref "nvtx" \



