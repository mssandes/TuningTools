
INPUT_FILE="/afs/cern.ch/user/j/jodafons/public/valid_ringer_sample.pic"
OUTPUT_NAME="networks"


bsub -J "E01" -q 8nh -u "" bsub_exec.csh $INPUT_FILE $OUTPUT_NAME 5

