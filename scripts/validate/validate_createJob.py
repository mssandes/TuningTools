#!/usr/bin/env python

from TuningTools.CreateJob  import CreateJob
#sourceEnvFile()

OutputFolder              = 'mc14_13TeV.147406.129160.sgn.offCutID.bkg.truth.trig.e24_medium_L1EM20VH.job'

genJob = CreateJob()
genJob( OutputFolder=OutputFolder,
        nInits=100,
        nBoxes=10,
        nTrain=6,
        nValid=4,
        nSorts=50,
        nSortsPerJob=5,
        nInitsPerJob=100,
        neurons=[5,20])



