
DATA=data/tuned/
REFDATA=data/tuningData_citest1.npz
crossValStatAnalysis.py -d $DATA  -r $REFDATA --output-level DEBUG  --doMonitoring 1 
