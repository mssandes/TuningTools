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
rm -rf ./CaloRingerAnalysis
source ./setrootcore.sh
# Add new gcc to path if we have cvmfs and it is not set to it:
#gccPath=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/x86_64/Gcc/gcc481_x86_64_slc6/slc6/gcc48/bin
#test ${ROOTCOREDIR#*cvmfs} != $ROOTCOREDIR; echo "cvmfs in rootcore: $? (0 equal true)"
#echo $(dirname $(which gcc))
#echo $gccPath
#test "$(dirname $(which gcc))" != "$gccPath"; echo "gcc isn't same: $? (0 equal true)"
#if test ${ROOTCOREDIR#*cvmfs} != $ROOTCOREDIR -a "$(dirname $(which gcc))" != "$gccPath"
#then
#  echo $PATH
#  export PATH=$gccPath:$PATH
#  echo $PATH
#else
#  echo "Gcc wasn't overriden"
#fi

# Build and set env:
source ./buildthis.sh
source FastNetTool/cmt/new_env_file.sh
#echo "ROOTCOREBIN=$ROOTCOREBIN"
#basePlace=$(dirname $ROOTCOREBIN)
#echo "basePlace=$basePlace"
#source $basePlace/setrootcore.sh

# Go to job path:
gridSubFolder=$ROOTCOREBIN/user_scripts/FastNetTool/grid_submit

# Retrieve dataset
rsync -rvhzP $jobConfig .

rsync -rvhzP $DatasetPlace .

# Run the job
$gridSubFolder/bsub_job.py $Dataset $jobFile $output || { echo "Couldn't run job!" && exit 1;}

# Copy output to outputPlace
ssh $outputDestination mkdir -p $outputFolder

rsync -rvhzP $output* "$outputPlace"

