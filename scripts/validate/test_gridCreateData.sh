

SIG="/tmp/jodafons/mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.e3059_s1982_s2008_r5993_rr0008.0001/sample.user.jodafons.mc14_13TeV.147406.PowhegPythia8_AZNLO_Zee.recon.RDO.e3059_s1982_s2008_r5993_rr0008.0001_PhysVal._*"
BKG="/tmp/jodafons/mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.e3084_s2044_s2008_r5988_rr0001/sample.user.jodafons.2.mc14_13TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.RDO.e3084_s2044_s2008_r5988_rr0001.0001_PhysVal._*"


gridCreateJob -ns 50 -nb 10 -ntr 6 -nval 4 --initBounds 100 --nMaxLayers 20 --nSortsPerJob 2  -out job.mc14_13TeV.129160.147406

#gridCreateData -s $SIG -b $BKG --reference Off_CutID Truth -t Trigger/HLT/Egamma/Ntuple/e24_medium_L1EM18VH --l1EmClusCut 20 -op L2
#./gridCreateData -s $SIG -b $BKG --reference Truth Truth -t Offline/Egamma/Ntuple/electron -op Offline



