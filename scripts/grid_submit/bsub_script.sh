#!/bin/sh


# Default args:
Inits=100

while true
do
  echo "reading $1 $2"
  if test "$1" == "--datasetPlace"
  then
    DatasetPlace=$2
    echo "Setting DatasetPlace to $DatasetPlace"
    Dataset=`basename $DatasetPlace`
    shift 2
  elif test "$1" == "--neuron"
  then
    Neuron=$2
    echo "Setting Neuron to $Neuron"
    shift 2
  elif test "$1" == "--sort"
  then
    Sort=$2
    echo "Setting Sort to $Sort"
    shift 2
  elif test "$1" == "--inits"
  then
    Inits=$2
    echo "Setting Inits to $Inits"
    shift 2
  elif test "$1" == "--outputPlace"
  then
    outputPlace="$2"
    echo "Setting outputPlace to $outputPlace"
    shift 2
  elif test "$1" == "--output"
  then
    output="$2"
    echo "Setting output to $output"
    shift 2
  else
    break
  fi
done

basePath=$PWD

# Check arguments
test "x$Neuron" = "x" && echo "Missing arg neuron" && exit 1;
test "x$Sort" = "x" && echo "Missing arg sort" && exit 1;
test "x$DatasetPlace" = "x" -o ! -f "$DatasetPlace" && echo "DatasetPlace \"$DatasetPlace\" doesn't exist" && exit 1;

# Retrieve package and compile
git clone https://github.com/joaoVictorPinto/TrigCaloRingerAnalysisPackages.git
sleep 1
rootFolder=$basePath/TrigCaloRingerAnalysisPackages/root
cd $rootFolder
source ./buildthis.sh

# Go to job path:
gridSubFolder=$rootFolder/FastNetTool/scripts/grid_submit
cd $gridSubFolder

# Retrieve dataset
rsync -rvhzP $DatasetPlace .

# Run the job
./bsub_job.py $Dataset $Neuron $Sort $output $Inits || { echo "Couldn't run job!" && return 1;}

# Copy output to outputPlace
ssh mkdir -p $outputPlace$Dataset
echo "ssh mkdir -p $outputPlace$Dataset"

ls 

echo "rsync -rvhzP \"$output*\" \"$outputPlace$Dataset\""
rsync -rvhzP $output* "$outputPlace$Dataset"

