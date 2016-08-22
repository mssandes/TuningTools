#!/usr/bin/env python

from RingerCore import Logger,  LoggingLevel

class CopyTree( Logger ):

  def __init__(self, outputDS):
    Logger.__init__(self)
    from ROOT import TFile
    # Check if the type is correct
    if not '.root' in outputDS:
      outputDS+='.root'
    self._file = TFile( outputDS, 'recreate' )
    self._outputDS=outputDS
    self._logger.info( ('Create root file with name: %s') \
                        %(outputDS) )

  def __call__(self, inputDS_list, path, treeName_list):
    
    if not type(inputDS_list) is list:
      inputDS_list = [inputDS_list]
    
    if not type(treeName_list) is list:
      treeName_list = [treeName_list]

    self._file.cd()
    self._file.mkdir( path )
    from ROOT import TChain, TObject

    for treeName in treeName_list:

      cobject = TChain()
      for inputDS in inputDS_list:
        self._logger.info( ('Copy tree name %s in %s to %s')%(treeName,\
                             inputDS, self._outputDS) )
        location = inputDS+'/'+path+'/'+treeName+'/trigger'
        self._logger.info(('Location: %s')%(location))
        cobject.Add( location )
      self._logger.info(('Copy %d events')%(cobject.GetEntries()))
      copy_cobject = cobject.CloneTree(-1)
      copy_cobject.Write("", TObject.kOverwrite)
      del cobject, copy_cobject
        
  def save(self):
    self._logger.info( ('Saving file %s') % (self._outputDS) )
    self._file.Close()




# __main__
mainLogger = Logger.getModuleLogger( __name__, LoggingLevel.INFO )

from RingerCore import expandFolders
import argparse
from pprint import pprint


triggerList = ['HLT_e24_lhmedium_nod0_iloose', 'HLT_e28_lhtight_nod0_iloose', 'HLT_e0_perf_L1EM15']

parser = argparse.ArgumentParser()

parser.add_argument('--inDS', action='store',
        help = "Whether the dataset contains (TP)Ntuple")
#parser.add_argument('--filemerged', action='store', default="mergedOutput.root",
#        help = "Name of the output file")
parser.add_argument('--trigger', nargs='+', default=triggerList,
        help = "Trigger list to keep on the filtered file.")
parser.add_argument('--path', action='store', default='HLT/Egamma/Expert',
        help = "Trigger tuple path")
args=parser.parse_args()


mainLogger.info('Start ntuple extraction using rucio as client')
mainLogger.info( ('The Dataset is: %s')%(args.inDS) )
#mainLogger.info('output merged file is: ', args.filemerged)

from RingerCore import RucioTools
rucio = RucioTools()

try:# protection for rucio get-list command
  list_files = rucio.get_list_files( args.inDS )
  pprint( list_files )
except:
  mainLogger.fatal( ('Can not get list of files for this datase: %s')%(args.inDS) )
  raise RuntimeError('error in rucio get-list')


import os
for fullname in list_files:

  rucio.download( fullname )
  fullname = rucio.noUsername( fullname ) 

  for chain in args.trigger:
    newname=fullname
    try:
      obj  = CopyTree( newname.replace('.root', ('%s.root')%(chain)) )
      obj( fullname, args.path, chain) 
      obj.save()
      del obj
    except:
      mainLogger.error( ('Can not extract the file: %s')%(fullname))
  
  os.system( ('rm %s')%(fullname) )












