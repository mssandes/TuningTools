#!/usr/bin/env python

import logging

try:
  import argparse
except ImportError:
  from FastNetTool import argparse

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--sgnInputFiles', action='append', required = True,
    help = "The signal files that will be used to tune the discriminators")
parser.add_argument('--bkgInputFiles', action='append', required = True, 
    help = "The background files that will be used to tune the discriminators")
parser.add_argument('--operation', action='store', required = True, 
    help = "The operation for the ")
parser.add_argument('--pickleTmpFile', default = 'fastnet.pic', 
    help = "The pickle intermediate file that will be used to train the datasets")

# Retrieve parser args:
args = parser.parse_args()
try:
  from pprint import pprint
  pprint([(key, args[key]) for key in sorted(args.keys())])
except ImportError:
  print args

createData = CreateData( sgnFileList = args.sgnInputFiles, 
                         bkgFileList = args.bkgInputFiles,
                         ringerOperation = 

class CreateData():

  def __init__( self, logger = None ):
    from FastNetTool.FilterEvents import filterEvents
    self._filter = filterEvents
    self._logger = logger or logging.getLogger(__name__)

  def __call__(self, sgnFileList, bkgFileList, ringerOperation, **kw):
    """
      Creates a pickle file ntuple with rings and its targets
      Arguments:
        - sgnFileList: A python list or a comma separated list of the root files
            containing the FastNet TTree for the signal dataset
        - bkgFileList: A python list or a comma separated list of the root files
            containing the FastNet TTree for the background dataset
        - ringerOperation: Set Operation type to be used by the filter
      Optional arguments:
        - filterType [None]: whether to filter. Use FilterType enumeration
        - reference [Truth]: set reference for targets. Use Reference enumeration
        - treeName ['CollectionTree']: set tree name on file
        - l1EmClusCut [None]: Set L1 cluster energy cut if operating for the trigger
    """

    import pickle

    npBkg = self._filter(bkgFileList, 
                         ringerOperation,
                         filterType = FilterType.Background, 
                         reference = Reference.Truth )

    self._logger('Extracted background rings with size: %r',[npBkg[0].shape])

    npSgn  = self._filter(sgnFileList,
                          ringerOperation,
                          filterType = FilterType.Signal,
                          reference = Reference.Truth )

    self._logger('Extracted signal rings with size: %r',[npSgn[0].shape])

    rings = np.concatenate( (data_zee[0],data_jf17[0]), axis=0)
    target = np.concatenate( (data_zee[1],data_jf17[1]), axis=0)


    self._logger('Total rings size is: %r | target size is: %r', 
        rings.shape,
        target.shape)

    objSave = [rings, target]
    filehandler = open(, 'w')
    pickle.dump(objSave, filehandler)


