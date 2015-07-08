#!/bin/sh

# Default args:
Inits=100
debug=0

while true
do
  echo "reading $1 $2"
  if test "$1" == "--datasetPlace"
  then
    DatasetPlace=$2
    echo "Setting DatasetPlace to $DatasetPlace"
    Dataset=`basename $DatasetPlace`
    shift 2
  elif test "$1" == "--jobConfig"
  then
    jobConfig="$2"
    jobFile=$(basename $jobConfig)
    echo "Setting jobConfig to $jobConfig"
    echo "Setting jobFile to $jobFile"
    shift 2
  elif test "$1" == "--outputPlace"
  then
    outputPlace="$2"
    outputDestination=${outputPlace%%:*}
    outputFolder=${outputPlace#*:}
    echo "Setting outputPlace to $outputPlace: destination is $outputDestination and folder is $outputFolder"
    shift 2
  elif test "$1" == "--output"
  then
    output="$2"
    echo "Setting output to $output"
    shift 2
  elif test "$1" == "--debug"
  then
    debug=1
    shift
  else
    break
  fi
done

test $debug -eq 1 && set -x

basePath=$PWD

# Check arguments
test "x$DatasetPlace" = "x" -o ! -f "$DatasetPlace" && echo "DatasetPlace \"$DatasetPlace\" doesn't exist" && exit 1;
test "x$jobConfig" = "x" -o ! -f "$jobConfig" && echo "JobConfig file \"$jobConfig\" doesn't exist" && exit 1;

# Retrieve package and compile
git clone https://github.com/joaoVictorPinto/TrigCaloRingerAnalysisPackages.git
rootFolder=$basePath/TrigCaloRingerAnalysisPackages/root
cd $rootFolder
git checkout `git tag | tail -n 1`
rm -rf ./CaloRingerAnalysis
source ./setrootcore.sh

# Build and set env:
export OMP_NUM_THREADS=$((`cat /proc/cpuinfo | grep processor | tail -n 1 | cut -f2 -d " "`+1))
source ./buildthis.sh
source FastNetTool/cmt/new_env_file.sh

# Retrieve dataset and job config
rsync -rvhzP $DatasetPlace .
rsync -rvhzP $jobConfig .

# Job path:
gridSubFolder=$ROOTCOREBIN/user_scripts/FastNetTool/run_on_grid
# Run the job
$gridSubFolder/tuningJob.py $Dataset $jobFile $output || { echo "Couldn't run job!" && exit 1;}

# Copy output to outputPlace
ssh $outputDestination mkdir -p $outputFolder

rsync -rvhzP $output* "$outputPlace"
