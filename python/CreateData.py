#!/usr/bin/env python

import logging

class CreateData():

  def __init__( self, logger = None ):
    from FastNetTool.FilterEvents import filterEvents
    self._filter = filterEvents
    import logging
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
        - output ['fastnet.pic']: Name for the output file
        - referenceSgn [Reference.Truth]: Filter reference for signal dataset
        - referenceBkg [Reference.Truth]: Filter reference for background dataset
        - treePath: Sets tree path on file to be used as the TChain. The default
            value depends on the operation. If set to None, it will be set to 
            the default value.
    """
    from FastNetTool.FilterEvents import FilterType, Reference
    import numpy as np 
    import pickle

    output = kw.pop('output', 'fastnet.pic' )
    referenceSgn = kw.pop('referenceSgn', Reference.Truth )
    referenceBkg = kw.pop('referenceBkg', Reference.Truth )
    treePath = kw.pop('treePath', None )
    l1EmClusCut = kw.pop('l1EmClusCut', None )
    print "Creating np signal"
    
    npSgn  = self._filter(sgnFileList,
                          ringerOperation,
                          filterType = FilterType.Signal,
                          reference = referenceSgn, 
                          treePath = treePath,
                          l1EmClusCut = l1EmClusCut)
  
    print '=====> ' , npSgn[0].shape
    print '=====> ' , npSgn[1].shape


    print "Created np signal"
    self._logger.info('Extracted signal rings with size: %r',[npSgn[0].shape])

    print "Creating np bkg"
    npBkg = self._filter(bkgFileList, 
                         ringerOperation,
                         filterType = FilterType.Background, 
                         reference = referenceBkg,
                         treePath = treePath,
                         l1EmClusCut = l1EmClusCut)
    print "Created np bkg"

    self._logger.info('Extracted background rings with size: %r',[npBkg[0].shape])

    print "Concatenating!!!"
    rings = np.concatenate( (npSgn[0],npBkg[0]), axis=0)
    target = np.concatenate( (npSgn[1],npBkg[1]), axis=0)
    print "Concatenated!!!"

    self._logger.info('Total rings size is: %r | target size is: %r', 
        rings.shape,
        target.shape)

    print "Saving!!!"
    objSave = [rings, target]
    filehandler = open(output, 'w')
    pickle.dump(objSave, filehandler)
    print "Saved!!!"

createData = CreateData()

try:
  parseOpts
except NameError,e:
  parseOpts = False

if __name__ == "__main__" or parseOpts:
  try:
    import argparse
  except ImportError:
    from FastNetTool import argparse

  parser = argparse.ArgumentParser(description = '')
  parser.add_argument('-s','--sgnInputFiles', action='store', 
      metavar='SignalInputFiles', required = True, nargs='+',
      help = "The signal files that will be used to tune the discriminators")
  parser.add_argument('-b','--bkgInputFiles', action='store', 
      metavar='BackgroundInputFiles', required = True, nargs='+',
      help = "The background files that will be used to tune the discriminators")
  parser.add_argument('-op','--operation', action='store', required = True, 
      help = "The operation environment for the algorithm")
  parser.add_argument('-o','--output', default = 'fastnet.pic', 
      help = "The pickle intermediate file that will be used to train the datasets.")
  parser.add_argument('--reference', action='store', nargs='+',
      metavar='(BOTH | SGN BKG)_REFERENCE', default = ['Truth'], choices = ('Truth','Off_CutID','Off_Likelihood'),
      help = """
        The reference used for filtering datasets. It needs to be set
        to a value on the Reference enumeration on FilterEvents file.
        You can set only one value to be used for both datasets, or one
        value first for the Signal dataset and the second for the Background
        dataset.
            """)
  parser.add_argument('--output-level', default = logging.INFO, 
      help = "The output level for the main logger")
  parser.add_argument('-t','--treePath', metavar='TreePath', action = 'store', 
      default = None, type=str,
      help = "The Tree path to be filtered on the files.")
  parser.add_argument('-l1','--l1EmClusCut', default = None, 
      help = "The L1 cut threshold")

  import sys
  if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
  # Retrieve parser args:
  args = parser.parse_args()
  # Treat special argument
  if len(args.reference) > 2:
    raise ValueError("--reference set to multiple values: %r", args.reference)
  if len(args.reference) is 1:
    args.reference.append( args.reference[0] )
  logger = logging.getLogger(__name__)
  logger.setLevel( logging.DEBUG ) # args.output_level)
  if args.operation != 'Offline' and not args.treePath:
    ValueError("If operation is not set to Offline, it is needed to set the TreePath manually.")

  from FastNetTool.util import printArgs
  printArgs( args, logger.debug )


  createData( sgnFileList     = args.sgnInputFiles, 
              bkgFileList     = args.bkgInputFiles,
              ringerOperation = args.operation,
              referenceSgn    = args.reference[0],
              referenceBkg    = args.reference[1],
              treePath        = args.treePath,
              output          = args.output,
              l1EmClusCut     = args.l1EmClusCut )
    
