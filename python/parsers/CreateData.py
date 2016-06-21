__all__ = ['createDataParser']

from RingerCore import argparse, get_attributes, BooleanStr, \
                       NotSet, LoggerNamespace

from TuningTools.FilterEvents import RingerOperation

###############################################################################
# Create data related objects
###############################################################################
createDataParser = argparse.ArgumentParser(add_help = False, 
                                           description = 'Create TuningTool data from PhysVal.')
from TuningTools.FilterEvents import Reference
mainCreateData = createDataParser.add_argument_group( "Required arguments", "")
mainCreateData.add_argument('-s','--sgnInputFiles', action='store', 
    metavar='SignalInputFiles', required = True, nargs='+',
    help = "The signal files that will be used to tune the discriminators")
mainCreateData.add_argument('-b','--bkgInputFiles', action='store', 
    metavar='BackgroundInputFiles', required = True, nargs='+',
    help = "The background files that will be used to tune the discriminators")
mainCreateData.add_argument('-op','--operation', default = None, 
                     help = """The Ringer operation determining in each Trigger 
                     level or what is the offline operation point reference.
                     Possible options are: """ \
                     + str(get_attributes( RingerOperation, onlyVars = True, getProtected = False)) )
mainCreateData.add_argument('-t','--treePath', metavar='TreePath', action = 'store', 
    default = NotSet, type=str, nargs='+',
    help = """The Tree path to be filtered on the files. It can be a value for
    each dataset.""")
optCreateData = createDataParser.add_argument_group( "Extra-configuration arguments", "")
optCreateData.add_argument('--reference', action='store', nargs='+',
    default = ['Truth'], choices = get_attributes( Reference, onlyVars = True, getProtected = False),
    help = """
      The reference used for filtering datasets. It needs to be set
      to a value on the Reference enumeration on FilterEvents file.
      You can set only one value to be used for both datasets, or one
      value first for the Signal dataset and the second for the Background
      dataset.
          """)
optCreateData.add_argument('-tEff','--efficiencyTreePath', metavar='EfficienciyTreePath', action = 'store', 
    default = NotSet, type=str, nargs='+',
    help = """The Tree path to calculate efficiency. 
    If not specified, efficiency is calculated upon treePath.""")
optCreateData.add_argument('-l1','--l1EmClusCut', default = NotSet, 
    type=float, help = "The L1 cut threshold")
optCreateData.add_argument('-l2','--l2EtCut', default = NotSet, 
    type=float, help = "The L2 Et cut threshold")
optCreateData.add_argument('-off','--offEtCut', default = NotSet, 
    type=float, help = "The Offline Et cut threshold")
optCreateData.add_argument('--getRatesOnly', default = NotSet, 
    action='store_true', help = """Don't save output file, just print benchmark 
                                   algorithm operation reference.""")
optCreateData.add_argument('--etBins', action='store', nargs='+',
    default = NotSet, type=float,
    help = "E_T bins (GeV) where the data should be segmented.")
optCreateData.add_argument('--etaBins', action='store', nargs='+',
    default = NotSet, type=float,
    help = "eta bins where the data should be segmented.")
optCreateData.add_argument('--ringConfig', action='store', nargs='+',
    type=int, default = NotSet, 
    help = "Number of rings for each eta bin segmentation.")
optCreateData.add_argument('-nC','--nClusters', 
    default = NotSet, type=int,
    help = "Maximum number of events to add to each dataset.")
optCreateData.add_argument('-o','--output', default = 'tuningtoolData', 
    help = "The pickle intermediate file that will be used to train the datasets.")
optCreateData.add_argument('--crossFile', 
    default = NotSet, type=str,
    help = """Cross-Validation file which will be used to tune the Ringer
    Discriminators.""")
################################################################################

