
#BASEPATH=/afs/cern.ch/work/j/jodafons/public
#DATA=user.wsfreund.nn.norm1.newstop.mc14_13TeV.147406.129160.sgn.offLH.bkg.truth.trig.ef.e24_lhmedium_nod0_tunedDiscrXYZ.tgz
#REFBASEPATH=/afs/cern.ch/work/j/jodafons/public/Tuning2016/TuningConfig
#REFDATA=mc14_13TeV.147406.129160.sgn.offLikelihood.bkg.truth.trig.e24_lhmedium_nod0_l1etcut20_l2etcut19_efetcut24_binned.pic.npz

DATA=/afs/cern.ch/work/w/wsfreund/public/Online/user.wsfreund.nn.norm1.batHfSigSz.mc14_13TeV.147406.129160.sgn.offLH.bkg.truth.trig.ef.e24_lhmedium_nod0_t001_tunedDiscrXYZ.tgz
REFBASEPATH=/afs/cern.ch/work/j/jodafons/public/Tuning201607XX
REFDATA=mc14_13TeV.147406.129160.sgn.offLikelihood.bkg.truth.trig.e24_lhmedium_nod0_l1etcut20_l2etcut19_efetcut24_binned2.pic.npz

crossValStatAnalysis.py -d $DATA --output-level DEBUG -r $REFBASEPATH/$REFDATA --doMonitoring 1 --output-level DEBUG
                                            #--binFilters GridJobFilter --binFilterIdx 4 

m run
