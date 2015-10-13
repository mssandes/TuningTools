

X=$1
Y=$2

python runGRIDtuning.py \
  -d user.jodafons.mc14_13TeV.147406.129160.sgn.offLH.bkg.truth.trig.l1cluscut_20.l2etcut_19.e24_medium_L1EM18VH_etBin_${X}_etaBin_${Y} \
  -pp user.wsfreund.Norm1  \
  -c user.wsfreund.config.nn5to20_sorts50_1by1_inits100_100by100 \
  -x user.wsfreund.CrossValid.50Sorts.seed_0 \
  -o user.jodafons.nn.mc14_13TeV.147406.sgn.Off_LH.129160.bkg.truth.l1_20.l2_19.e24_medium_etBin_${X}_etaBin_${Y}.t0002  \
  --cloud="US" \
  #--dry-run
