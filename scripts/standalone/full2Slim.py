#!/usr/bin/env python
from RingerCore import Logger,  LoggingLevel

# Local class to extract the ntuple from the list of files 
class CopyTree( Logger ):

  def __init__(self, outputDS):
    Logger.__init__(self)
    from ROOT import TFile
    # Check if the type is correct
    if not '.root' in outputDS:  outputDS+='.root'
    self._file = TFile( outputDS, 'recreate' )
    self._outputDS=outputDS
    self._logger.info( ('Create root file with name: %s') \
                        %(outputDS) )

  def __call__(self, inputDS_list, path, treeName_list):
    
    if not type(inputDS_list) is list:  inputDS_list = [inputDS_list]
    if not type(treeName_list) is list: treeName_list = [treeName_list]

    self._file.cd();  self._file.mkdir( path )
    from ROOT import TChain, TObject
    for treeName in treeName_list:
      cobject = TChain()
      for inputDS in inputDS_list:
        self._logger.info( ('Copy tree name %s in %s to %s')%(treeName,\
                             inputDS, self._outputDS) )
        location = inputDS+'/'+path+'/'+treeName+'/trigger'
        cobject.Add( location )

      if cobject.GetEntries() == 0:
        self._logger.warning(('There is no events into this path: %s')%(location))
        import os
        os.system( ('rm -rf %s')%(self._outputDS) )
      else:
        self._logger.info(('Copy %d events...')%(cobject.GetEntries()))
        copy_cobject = cobject.CloneTree(-1)
        copy_cobject.Write("", TObject.kOverwrite)
        del copy_cobject
      del cobject
        
  def save(self):
    self._logger.info( ('Saving file %s') % (self._outputDS) )
    self._file.Close()




######################### __main__ ############################
from RingerCore import expandFolders, csvStr2List
from pprint import pprint
import argparse

mainFilterParser = argparse.ArgumentParser()

mainFilterParser.add_argument('-i','--inputFiles', action='store', 
                               metavar='InputFiles', required = True, nargs='+',
                               help = "The input files that will be used to generate a extract file")

mainFilterParser.add_argument('-t', '--trigger', action='store', default='e0_perf_L1EM15',
                               required = True,
                               help = "Trigger list to keep on the filtered file.")

mainFilterParser.add_argument('--path', action='store', default='HLT/Egamma/Expert',
                               help = "Trigger tuple path")

mainFilterParser.add_argument('-o','--output', action='store', default='NTUPLE.*.root',
                               help = "output file name.")


import sys, os
if len(sys.argv)==1:
  mainFilterParser.print_help()
  sys.exit(1)

mainLogger = Logger.getModuleLogger( __name__, LoggingLevel.INFO )
mainLogger.info('Start ntuple extraction...')
# retrieve args
args=mainFilterParser.parse_args()

# Treat special arguments
if len( args.inputFiles ) == 1:
  args.inputFiles = csvStr2List( args.inputFiles[0] )

args.inputFiles = expandFolders( args.inputFiles )
mainLogger.verbose("All input files are:")
pprint(args.inputFiles)

if '*' in args.output:
  output = args.output.replace('*', args.trigger.replace('HLT_',''))

for inputname in args.inputFiles:
  try: # protection 
    obj  = CopyTree( output )
    obj( inputname, args.path, args.trigger) 
    obj.save()
    del obj
  except: 
    if os.path.exists( output ):
      os.system( ('rm -rf %s')%(output ) )
    mainLogger.error( ('Can not extract the file %s')%(inputname))
    break #stop the trigger loop












