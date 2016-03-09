try:
  import argparse
except ImportError:
  from RingerCore import argparse

from RingerCore.util import get_attributes, EnumStringification, NotSet
from RingerCore.Parser import GridNamespace

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
mainCreateData.add_argument('-op','--operation', action='store', required = True, 
    help = "The operation environment for the algorithm")
mainCreateData.add_argument('-t','--treePath', metavar='TreePath', action = 'store', 
    default = None, type=str, nargs='+',
    help = """The Tree path to be filtered on the files. It can be a value for
    each dataset.""")
optCreateData = createDataParser.add_argument_group( "Extra-configuration arguments", "")
optCreateData.add_argument('--reference', action='store', nargs='+',
    default = ['Truth'], choices = get_attributes( Reference, onlyVars = True),
    help = """
      The reference used for filtering datasets. It needs to be set
      to a value on the Reference enumeration on FilterEvents file.
      You can set only one value to be used for both datasets, or one
      value first for the Signal dataset and the second for the Background
      dataset.
          """)
optCreateData.add_argument('-tEff','--efficiencyTreePath', metavar='EfficienciyTreePath', action = 'store', 
    default = None, type=str, nargs='+',
    help = """The Tree path to calculate efficiency. 
    If not specified, efficiency is calculated upon treePath.""")
optCreateData.add_argument('-l1','--l1EmClusCut', default = None, 
    type=float, help = "The L1 cut threshold")
optCreateData.add_argument('-l2','--l2EtCut', default = None, 
    type=float, help = "The L2 Et cut threshold")
optCreateData.add_argument('-off','--offEtCut', default = None, 
    type=float, help = "The Offline Et cut threshold")
optCreateData.add_argument('--getRatesOnly', default = False, 
    action='store_true', help = """Don't save output file, just print benchmark 
                                   algorithm operation reference.""")
optCreateData.add_argument('--etBins', action='store', nargs='+',
    default = None, type=float,
    help = "E_T bins where the data should be segmented.")
optCreateData.add_argument('--etaBins', action='store', nargs='+',
    default = None, type=float,
    help = "eta bins where the data should be segmented.")
optCreateData.add_argument('--ringConfig', action='store', nargs='+',
    type=int, default = None, 
    help = "Number of rings for each eta bin segmentation.")
optCreateData.add_argument('-nC','--nClusters', 
    default = None, type=int,
    help = "Maximum number of events to add to each dataset.")
optCreateData.add_argument('-o','--output', default = 'tuningtoolData', 
    help = "The pickle intermediate file that will be used to train the datasets.")
optCreateData.add_argument('--crossFile', 
    default = None, type=str,
    help = """Cross-Validation file which will be used to tune the Ringer
    Discriminators.""")
################################################################################

################################################################################
# Create tuningJob file related objects
################################################################################
class JobFileTypeCreation( EnumStringification ):
  """
    The possible file creation options
  """
  all = 0,
  ConfigFiles = 1,
  CrossValidFile = 2,
  ppFile = 3
tuningJobFileParser = argparse.ArgumentParser( add_help = False,
                        description = 'Create files used by TuningJob.' )
tuningJobFileParser.add_argument('fileType', 
                     choices = get_attributes(JobFileTypeCreation, onlyVars = True),
                     nargs='+',
                     help = """Which kind of files to create. You can choose one
                     or more of the available choices, just don't use all with
                     the other available choices.""")
tuningJobFileParser.add_argument('--compress',  type=int, 
    default = 1, nargs='?',
    help = "Whether to compress files or not.")
from RingerCore.Logger import Logger, LoggingLevel
tuningJobFileParser.add_argument('--output-level', 
    default = LoggingLevel.tostring( LoggingLevel.INFO ), 
    type=str, help = "The logging output level.")
################################################################################
jobConfig = tuningJobFileParser.add_argument_group( "JobConfig Files Creation Options", 
                                       """Change configuration for
                                       job config files creation.""")
jobConfig.add_argument('-outJobConfig', '--jobConfiFilesOutputFolder', 
                       default = 'jobConfig', 
                       help = "The job config files output folder.")
jobConfig.add_argument('--neuronBounds', nargs='+', type=int, default = [5,20],  
                        help = """
                            Input a sequential bounded list to be used as the
                            neuron job range, the arguments should have the
                            same format from the seq unix command or as the
                            Matlab format. If not specified, the range will
                            start from 1.  I.e 5 2 9 leads to [5 7 9] and 50
                            leads to 1:50
                               """)
jobConfig.add_argument('--sortBounds', nargs='+', type=int, default = [50],  
                       help = """
                          Input a sequential bounded list using seq format to
                          be used as the sort job range, but the last bound
                          will be opened just as happens when using python
                          range function. If not specified, the range will
                          start from 0.  I.e. 5 2 9 leads to [5 7] and 50 leads
                          to range(50)
                              """)
jobConfig.add_argument('--nInits', nargs='?', type=int, default = 100,
                       help = "The number of initilizations of the discriminator.")
jobConfig.add_argument('--nNeuronsPerJob', type=int, default = 1,  
                        help = "The number of hidden layer neurons per job.")
jobConfig.add_argument('--nSortsPerJob', type=int, default = 1,  
                       help = "The number of sorts per job.")
jobConfig.add_argument('--nInitsPerJob', type=int, default = 5,  
                        help = "The number of initializations per job.")
################################################################################
crossConfig = tuningJobFileParser.add_argument_group( "CrossValid File Creation Options", 
                                         """Change configuration for CrossValid
                                         file creation.""")
crossConfig.add_argument('-outCross', '--crossValidOutputFile', 
                       default = 'crossValid', 
                       help = "The cross validation output file.")
crossConfig.add_argument('-ns',  '--nSorts', type=int, default = 50, 
                         help = """The number of sort used by cross validation
                                configuration.""")
crossConfig.add_argument('-nb', '--nBoxes', type=int,  default = 10, 
                         help = """The number of boxes used by cross validation
                                 configuration.""")
crossConfig.add_argument('-ntr', '--nTrain', type=int, default = 6,  
                         help = """The number of train boxes used by cross
                                validation.""")
crossConfig.add_argument('-nval','--nValid', type=int, default = 4, 
                         help = """The number of valid boxes used by cross
                                validation.""")
crossConfig.add_argument('-ntst','--nTest',  type=int, default = 0, 
                         help = """The number of test boxes used by cross
                                validation.""")
crossConfig.add_argument('-seed', type=int, default=None,
                         help = "The seed value for generating CrossValid object.")
################################################################################
ppConfig = tuningJobFileParser.add_argument_group( "PreProc File Creation Options", 
                                      """Change configuration for pre-processing 
                                      file creation. These options will only
                                      be taken into account if job fileType is
                                      set to "ppFile" or "all".""")
ppConfig.add_argument('-outPP','--preProcOutputFile',
                      default = 'ppFile',
                      help = "The pre-processing validation output file")
ppConfig.add_argument('-ppCol', type=str,
                      default = '[[Norm1()]]',
                      help = """The pre-processing collection to apply. The
                             string will be parsed by python and created using
                             the available pre-processings on
                             TuningTools.PreProc.py file""")
################################################################################

################################################################################
# Create tuningJob file related objects
################################################################################
tuningJobParser = argparse.ArgumentParser(add_help = False, 
                                          description = 'Tune discriminator for a specific TuningTool data.',
                                          conflict_handler = 'resolve')
tuningDataArgs = tuningJobParser.add_argument_group( "Required arguments", "")
tuningDataArgs.add_argument('-d', '--data', action='store', 
    metavar='data', required = True,
    help = "The data file that will be used to tune the discriminators")
tuningOptArgs = tuningJobParser.add_argument_group( "Optional arguments", "")
tuningCrossVars = tuningJobParser.add_argument_group( "Cross-validation configuration", "")
tuningCrossVars.add_argument('-x', '--crossFile', action='store', default = NotSet, 
    help = """The cross-validation file path, pointing to a file
            created with the create tuning job files""")
tuningLoopVars = tuningJobParser.add_argument_group( "Looping configuration", "")
tuningLoopVars.add_argument('-c','--confFileList', nargs='+', default = None,
    help = """A python list or a comma separated list of the
          root files containing the configuration to run the jobs. The files can
          be generated using a CreateConfFiles instance which can be accessed via
          command line using the createTuningJobFiles.py script.""")
tuningLoopVars.add_argument('--neuronBounds', nargs='+', type=int, default = None,  
                        help = """
                            Input a sequential bounded list to be used as the
                            neuron job range, the arguments should have the
                            same format from the seq unix command or as the
                            Matlab format. If not specified, the range will
                            start from 1.  I.e 5 2 9 leads to [5 7 9] and 50
                            leads to 1:50
                               """)
tuningLoopVars.add_argument('--sortBounds', nargs='+', type=int, default = None,  
                       help = """
                          Input a sequential bounded list using seq format to
                          be used as the sort job range, but the last bound
                          will be opened just as happens when using python
                          range function. If not specified, the range will
                          start from 0.  I.e. 5 2 9 leads to [5 7] and 50 leads
                          to range(50)
                              """)
tuningLoopVars.add_argument('--initBounds', nargs='+', type=int, default = None,
                       help = """
                          Input a sequential bounded list using seq format to
                          be used as the inits job range, but the last bound
                          will be opened just as happens when using python
                          range function. If not specified, the range will
                          start from 0.  I.e. 5 2 9 leads to [5 7] and 50 leads
                          to range(50)
                              """)
tuningPPVars = tuningJobParser.add_argument_group( "Pre-processing configuration", "")
tuningPPVars.add_argument('--ppFileList', nargs='+', default = NotSet,
        help = """A list or a comma separated list of the
          file paths containing the pre-processing chain to apply in the
          input space and obtain the pattern space. The files can be generated
          using a CreateConfFiles instance which is accessed via command
          line using the createTuningJobFiles.py script.
          The ppFileList must have a file for each of the configuration list 
          defined, that is, one pre-processing chain for each one of the 
          neuron/sort/init bounds collection. When only one ppFile is defined and
          the configuration list has size greater than one, the pre-processing
          chain will be copied for being applied on the other bounds.
        """)
tuningDepVars = tuningJobParser.add_argument_group( "Binning configuration", "")
tuningDepVars.add_argument('--et-bins', nargs='+', default = NotSet, type = int,
        help = """ The et bins to use within this job. 
            When not specified, all bins available on the file will be tuned
            separately.
            If specified as a integer or float, it is assumed that the user
            wants to run the job only for the specified bin index.
            In case a list is specified, it is transformed into a
            MatlabLoopingBounds, read its documentation on:
              http://nbviewer.jupyter.org/github/wsfreund/RingerCore/blob/master/readme.ipynb#LoopingBounds
            for more details.
        """)
tuningDepVars.add_argument('--eta-bins', nargs='+', default = NotSet, type = int,
        help = """ The eta bins to use within this job. Check et-bins
            help for more information.  """)
tuningOptArgs.add_argument('--no-compress', action='store_true',
          help = """Don't compress output files.""")
tuningArgs = tuningJobParser.add_argument_group( "Tuning CORE configuration", "")
tuningArgs.add_argument('--show-evo', type=int,
          default = NotSet, 
          help = """The number of iterations where performance is shown.""")
tuningArgs.add_argument('--max-fail', type=int,
          default = NotSet, 
          help = """Maximum number of failures to imrpove performance over 
          validation dataset that is tolerated.""")
tuningArgs.add_argument('--epochs', type=int,
          default = NotSet, 
          help = """Number of iterations where the tuning algorithm can run the
          optimization.""")
tuningArgs.add_argument('--do-perf', type=int,
          default = NotSet, 
          help = """Whether we should run performance
            testing under convergence conditions, using test/validation dataset
            and also estimate operation condition.""")
tuningArgs.add_argument('--batch-size', type=int,
          default = NotSet, 
          help = """Set the batch size used during tuning.""")
exMachinaArgs = tuningJobParser.add_argument_group( "ExMachina CORE configuration", "")
exMachinaArgs.add_argument('--algorithm-name', default = NotSet, 
          help = """The tuning method to use.""")
exMachinaArgs.add_argument('--network-arch', default = NotSet, 
          help = """The neural network architeture to use.""")
exMachinaArgs.add_argument('--cost-function', default = NotSet, 
          help = """The cost function used by ExMachina.""")
exMachinaArgs.add_argument('--shuffle', default = NotSet, 
          help = """Whether to shuffle datasets while training.""")
fastNetArgs = tuningJobParser.add_argument_group( "FastNet CORE configuration", "")
fastNetArgs.add_argument('--seed', default = NotSet, 
          help = """The seed to be used by the tuning algorithm.""")
fastNetArgs.add_argument('--do-multi-stop', default = NotSet, 
          help = """Tune classifier using P_D, P_F and
          SP when set to True. Uses only SP when set to False.""")
################################################################################
## Specialization of GridNamespace for this package
# Use this namespace when parsing grid option on TuningTool package.
class TuningToolGridNamespace(GridNamespace):
  """
    Special TuningTools GridNamespace class.
  """

  def __init__(self, prog = 'prun', **kw):
    GridNamespace.__init__( self, prog, **kw )
    self.setBExec('source ./buildthis.sh --grid')

  def pre_download(self):
    import os
    # We need this to avoid being banned from grid:
    if not os.path.isfile(os.path.expandvars("$ROOTCOREBIN/../TuningTools/cmt/boost_1_58_0.tar.gz")):
      self._logger.info('Downloading boost to avoid doing it on server side.')
      import urllib
      urllib.urlretrieve("http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz", 
                         filename=os.path.expandvars("$ROOTCOREBIN/../TuningTools/cmt/boost_1_58_0.tar.gz"))
    else:
      self._logger.info('Boost already downloaded.')

  def extFile(self):
    return '"TuningTools/cmt/boost_1_58_0.tar.gz"'
################################################################################
