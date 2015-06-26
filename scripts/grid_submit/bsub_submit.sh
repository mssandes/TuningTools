#!/bin/sh
datasetPlace=$1

startNeurons=5
endNeurons=6

#nSorts = 50
#nInits = 100

nSorts=2
nInits=2

for Sort in $nSorts
do
  echo "Sort is $Sort"
  for neuron in `seq $startNeurons $endNeurons`
  do
    echo "Neuron is $neuron"
    ~wsfreund/public/TrigCaloRingerAnalysisPackages/root/FastNetTool/scripts/grid_submit/bsub_script.sh \
      "--datasetPlace" $datasetPlace \
      "--neuron" $neuron \
      "--sort" $Sort \
      "--inits" $nInits \
      "--output" "NN.mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.l2trig.e24_medium_L1EM20VH" \
      "--outputPlace" "lxplus0010:/tmp/wsfreund/test/"
  done
done
