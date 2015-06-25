#!/bin/sh
datasetPlace=$1

startNeurons=5
endNeurons=20

#nSorts = 50
#nInits = 100

nSorts=1
nInits=1

for Sort in $nSorts
do
  echo "Sort is $Sort"
  #foreach neuron in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
  for neuron in 5
  do
    echo "Neuron is $neuron"
    ~wsfreund/public/TrigCaloRingerAnalysisPackages/root/FastNetTool/scripts/grid_submit/bsub_script.sh \
      "--datasetPlace" $datasetPlace \
      "--neuron" $neuron \
      "--sort" $Sort \
      "--init" $nInits \
      "--outputPlace" "lxplus0010:/tmp/wsfreund/test/"
  done
done
