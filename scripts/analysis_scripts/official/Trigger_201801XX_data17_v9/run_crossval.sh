
#python run_popen_jobs.py -c "crossValStatAnalysis.py -r data_jpsi/files/reference-vloose.npz --doMatlab --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0 -op L2Calo" -d data_jpsi/tuning/user.jodafons.nn.data17_13TeV.AllPeriods.sgn_Jpsi_EGAM2.bkg_EGAM7.Rel21.bestSP.Norm1.v9.t0001_td  -n 8
#
#mkdir crossval_vloose
#mv crossVal* crossval_vloose
#
#
#python run_popen_jobs.py -c "crossValStatAnalysis.py -r data_jpsi/files/reference-loose.npz --doMatlab --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0 -op L2Calo" -d data_jpsi/tuning/user.jodafons.nn.data17_13TeV.AllPeriods.sgn_Jpsi_EGAM2.bkg_EGAM7.Rel21.bestSP.Norm1.v9.t0001_td  -n 8
#
#mkdir crossval_loose
#mv crossVal* crossval_loose
#
#
#python run_popen_jobs.py -c "crossValStatAnalysis.py -r data_jpsi/files/reference-medium.npz --doMatlab --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0 -op L2Calo" -d data_jpsi/tuning/user.jodafons.nn.data17_13TeV.AllPeriods.sgn_Jpsi_EGAM2.bkg_EGAM7.Rel21.bestSP.Norm1.v9.t0001_td  -n 8
#
#mkdir crossval_medium
#mv crossVal* crossval_medium
#
#
#python run_popen_jobs.py -c "crossValStatAnalysis.py -r data_jpsi/files/reference-tight.npz --doMatlab --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0 -op L2Calo" -d data_jpsi/tuning/user.jodafons.nn.data17_13TeV.AllPeriods.sgn_Jpsi_EGAM2.bkg_EGAM7.Rel21.bestSP.Norm1.v9.t0001_td  -n 8
#
#mkdir crossval_tight
#mv crossVal* crossval_tight



python run_popen_jobs.py -c "crossValStatAnalysis.py -r data_jpsi/files/reference-vloose.npz --doMatlab --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0 -op L2Calo" -d data_jpsi/tuning/user.mverissi.nn.data17_13TeV.AllPeriods.sgn_Jpsi_EGAM2.bkg_EGAM7.Rel21.bestSP.Norm1.v9.t0002_td  -n 15

mkdir crossval_vloose
mv crossVal* crossval_vloose


python run_popen_jobs.py -c "crossValStatAnalysis.py -r data_jpsi/files/reference-loose.npz --doMatlab --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0 -op L2Calo" -d data_jpsi/tuning/user.mverissi.nn.data17_13TeV.AllPeriods.sgn_Jpsi_EGAM2.bkg_EGAM7.Rel21.bestSP.Norm1.v9.t0002_td  -n 15

mkdir crossval_loose
mv crossVal* crossval_loose


python run_popen_jobs.py -c "crossValStatAnalysis.py -r data_jpsi/files/reference-medium.npz --doMatlab --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0 -op L2Calo" -d data_jpsi/tuning/user.mverissi.nn.data17_13TeV.AllPeriods.sgn_Jpsi_EGAM2.bkg_EGAM7.Rel21.bestSP.Norm1.v9.t0002_td  -n 15

mkdir crossval_medium
mv crossVal* crossval_medium


python run_popen_jobs.py -c "crossValStatAnalysis.py -r data_jpsi/files/reference-tight.npz --doMatlab --expandOP --output-level INFO -rocm 0 -modelm AUC --overwrite --doMatlab 0 -op L2Calo" -d data_jpsi/tuning/user.mverissi.nn.data17_13TeV.AllPeriods.sgn_Jpsi_EGAM2.bkg_EGAM7.Rel21.bestSP.Norm1.v9.t0002_td  -n 15

mkdir crossval_tight
mv crossVal* crossval_tight





