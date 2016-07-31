

from TuningTools.CreateTuningJobFiles import createTuningJobFiles
createTuningJobFiles( outputFolder   = 'config_citest0',
                      neuronBounds   = [2,2],
                      sortBounds     = 10,
                      nInits         = 2,
                      nNeuronsPerJob = 1,
                      nInitsPerJob   = 2,
                      nSortsPerJob   = 10,
                      compress       = True )

from TuningTools.CrossValid import CrossValid, CrossValidArchieve
crossValid = CrossValid(nSorts = 50,
                        nBoxes = 10,
                        nTrain = 6, 
                        nValid = 4,
                        #nTest=args.nTest,
                        #seed=args.seed,
                        #level=args.output_level
                        )
place = CrossValidArchieve( 'crossValid_citest0', 
                            crossValid = crossValid,
                            ).save( True )


from TuningTools.PreProc import *
#ppCol = PreProcCollection( PreProcChain( MapStd() ) )
ppCol = PreProcChain( Norm1() )
from TuningTools.TuningJob import fixPPCol
ppCol = fixPPCol(ppCol)
place = PreProcArchieve( 'ppFile_citest0', ppCol = ppCol ).save()
import sys,os
sys.exit(os.EX_OK) # code 0, all ok

