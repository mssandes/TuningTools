

from TuningTools.CreateTuningJobFiles import createTuningJobFiles
createTuningJobFiles( outputFolder   = 'config.n5to10.JK.inits_10by10',
                      neuronBounds   = [5,10],
                      sortBounds     = 10,
                      nInits         = 100,
                      nNeuronsPerJob = 1,
                      nInitsPerJob   = 10,
                      nSortsPerJob   = 1,
                      prefix         = 'job_slim',
                      compress       = True )



from TuningTools.CrossValid import CrossValid, CrossValidArchieve
crossValid = CrossValid(nSorts = 10,
                        nBoxes = 10,
                        nTrain = 9, 
                        nValid = 1,
                        )
place = CrossValidArchieve( 'crossValid', 
                            crossValid = crossValid,
                            ).save( True )


from TuningTools.PreProc import *
ppCol = PreProcChain( Norm1() ) 

#ppCol = PreProcChain( RingerEtaMu() ) 
#etbins = [15,20,30,40,50,50000]
#etabins = [0,0.8,1.37,1.54,2.37,2.5]
#ppCol = [[None for _ in range(5)] for __ in range(5)]
#for etBinIdx in range(len(etbins)-1):
#  for etaBinIdx in range(len(etabins)-1):
#    ppCol[etBinIdx][etaBinIdx] = PreProcChain( RingerEtaMu(pileupThreshold=100, etamin=etabins[etaBinIdx], etamax=etabins[etaBinIdx+1]) ) 

from TuningTools.TuningJob import fixPPCol
ppCol = fixPPCol(ppCol)
place = PreProcArchieve( 'ppFile', ppCol = ppCol ).save()

