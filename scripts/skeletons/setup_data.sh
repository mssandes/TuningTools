
python createTuningFiles.py
python createDataFiles.py  
mkdir files/
#mv summary.log files
mv  config.* files
mv *.pic.npz files
mv *.pic.gz files
mv *.mat files

python run_tuning.py
cd files
mkdir tuned
mv ../../*.gz .
cd ../..
python createCrossValidFiles.py
