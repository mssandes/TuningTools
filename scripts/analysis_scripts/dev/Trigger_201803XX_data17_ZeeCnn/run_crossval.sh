
python run_popen_jobs.py -c "crossValStatAnalysis.py --doMatlab -r data/files/reference-vloose.npz --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0" -d /home/jodafons/CERN-DATA/rDevBook/tuning/trigger/Zee/data17_201801XX_v8/tuning/user.jodafons.nn.data17_13TeV.AllPeriods.sgn_Zee_EGAM1.bkg_EGAM7.bestSP.Norm1.v8.t0001_td  -n 20

#python run_popen_jobs.py -c "crossValStatAnalysis.py --doMatlab -r data/files/reference-tight.npz --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0" -d /home/jodafons/CERN-DATA/rDevBook/tuning/trigger/Zee/data17_201801XX_v8/tuning/user.jodafons.nn.data17_13TeV.AllPeriods.sgn_Zee_EGAM1.bkg_EGAM7.bestSP.Norm1.v8.t0001_td  -n 20



#crossValStatAnalysis.py --doMatlab -r data/files/reference-tight.npz --expandOP --output-level INFO -d data/tuning/user.jodafons.cnn.data17_13TeV.AllPeriods.sgn_Zee_EGAM1.bkg_EGAM7.bestSP.Norm1Vortex.v10.t0001_td  -rocm 0 -modelm AUC --overwrite --doMatlab 0  --binFilters StandaloneJobBinnedFilter


#crossValStatAnalysis.py --doMatlab -r data/files/reference-vloose.npz --expandOP --output-level INFO -d data/tuning/user.jodafons.cnn.data17_13TeV.AllPeriods.sgn_Zee_EGAM1.bkg_EGAM7.bestSP.Norm1Vortex.v10.t0001_td  -rocm 0 -modelm AUC --overwrite --doMatlab 0  --binFilters StandaloneJobBinnedFilter


