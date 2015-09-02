import os
import re
import textwrap
try:
  import argparse
except ImportError:
  from FastNetTool import argparse
from FastNetTool.util import EnumStringification, get_attributes

###############################################################################
# Logger related objects
###############################################################################
from FastNetTool.Logger import LoggingLevel, Logger
loggerParser = argparse.ArgumentParser(add_help = False)
loggerParser.add_argument('--output-level', 
    default = LoggingLevel.tostring( LoggingLevel.INFO ), 
    type=str, required = False, choices = get_attributes(LoggingLevel, onlyVars = True),
    help = "The output level for the main logger")
###############################################################################
## LoggerNamespace
# When using logger parser parent, make sure to use LoggerNamespace when
# retrieving arguments
class LoggerNamespace( argparse.Namespace ):
  """
    Namespace for dealing with logger parser properties
  """
  def __init__(self, **kw):
    argparse.Namespace.__init__( self, **kw )

  def __getattr__(self, attr):
    if attr == 'output_level' and type(self.output_level) == str:
      return LoggingLevel.fromstring( self.__dict__['output_level'] )
    else:
      return self.__dict__[attr]
###############################################################################

###############################################################################
# Create data related objects
###############################################################################
createDataParser = argparse.ArgumentParser(add_help = False, 
                                           description = 'Create FastNet data from PhysVal.')
from FastNetTool.FilterEvents import Reference
createDataParser.add_argument('-s','--sgnInputFiles', action='store', 
    metavar='SignalInputFiles', required = True, nargs='+',
    help = "The signal files that will be used to tune the discriminators")
createDataParser.add_argument('-b','--bkgInputFiles', action='store', 
    metavar='BackgroundInputFiles', required = True, nargs='+',
    help = "The background files that will be used to tune the discriminators")
createDataParser.add_argument('-op','--operation', action='store', required = True, 
    help = "The operation environment for the algorithm")
createDataParser.add_argument('-o','--output', default = 'fastnetData', 
    help = "The pickle intermediate file that will be used to train the datasets.")
createDataParser.add_argument('--reference', action='store', nargs='+',
    metavar='(BOTH | SGN BKG)_REFERENCE', default = ['Truth'], 
    choices = get_attributes( Reference, onlyVars = True),
    help = """
      The reference used for filtering datasets. It needs to be set
      to a value on the Reference enumeration on FilterEvents file.
      You can set only one value to be used for both datasets, or one
      value first for the Signal dataset and the second for the Background
      dataset.
          """)
createDataParser.add_argument('-t','--treePath', metavar='TreePath', action = 'store', 
    default = None, type=str,
    help = "The Tree path to be filtered on the files.")
createDataParser.add_argument('-l1','--l1EmClusCut', default = None, 
    type=int, help = "The L1 cut threshold")
createDataParser.add_argument('-nClusters','--numberOfClusters', 
    default = None, type=int,
    help = "Maximum number of events to add to each dataset.")
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
tuningJobFileParser = argparse.ArgumentParser( description = 'Create files used by TuningJob.' )
tuningJobFileParser.add_argument('fileType', 
                     choices = get_attributes(JobFileTypeCreation, onlyVars = True),
                     nargs='+',
                     help = """Which kind of files to create. You can choose one
                     or more of the available choices, just don't use all with
                     the other available choices.""")
from FastNetTool.Logger import Logger, LoggingLevel
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
jobConfig.add_argument('--nInits', nargs='+', type=int, default = 100,
                       help = """
                          Input a sequential bounded list using seq format to
                          be used as the inits job range, but the last bound
                          will be opened just as happens when using python
                          range function. If not specified, the range will
                          start from 0.  I.e. 5 2 9 leads to [5 7] and 50 leads
                          to range(50)
                              """)
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
                             FastNetTool.PreProc.py file""")
################################################################################


################################################################################
# Grid parser related objects
################################################################################
# Basic grid parser
gridParser = argparse.ArgumentParser(add_help = False)
gridParser.add_argument('--site',default = 'AUTO',
    help = "The site location where the job should run.",
    nargs='?', required = False,
    dest = 'grid_site')
gridParser.add_argument('--excludedSite', default = '', 
    help = "The excluded site location.", nargs='?',
    required = False, dest = 'grid_excludedSite')
gridParser.add_argument('--debug', 
    const='--express --debugMode --allowTaskDuplication', dest='gridExpand_debug',
    help = "Submit GRID job on debug mode.", action='store_const',
    required = False )
gridParser.add_argument('--nJobs', nargs='?', type=int,
    required = False, dest = 'grid_nJobs',
    help = """Number of jobs to submit.""")
gridParser.add_argument('--excludeFile', nargs='?', 
    required = False, default = '"*.o,*.so,*.a,*.gch"', dest = 'grid_excludeFile',
    help = """Files to exclude from environment copied to grid.""")
gridParser.add_argument('--disableAutoRetry', action='store_true',
    required = False, dest = 'grid_disableAutoRetry',
    help = """Flag to disable auto retrying jobs.""")
gridParser.add_argument('--extFile', nargs='?',
    required = False, dest = 'grid_extFile', default='',
    help = """External file to add.""")
gridParser.add_argument('--maxNFilesPerJob', nargs='?',
    required = False, dest = 'grid_maxNFilesPerJob',
    help = """Maximum number of files per job.""")
gridParser.add_argument('--cloud', nargs='?',
    required = False, default=False, dest = 'grid_cloud',
    help = """The cloud where to submit the job.""")
gridParser.add_argument('--nGBPerJob', nargs='?',
    required = False, dest = 'grid_nGBPerJob',
    help = """Maximum number of GB per job.""")
gridParser.add_argument('--skipScout', action='store_true',
    required = False, dest = 'grid_skipScout',
    help = """Flag to disable auto retrying jobs.""")
gridParser.add_argument('--dry-run', action='store_true',
    help = """Only print grid resulting command, but do not execute it.
            Used for debugging submission.""")
mutuallyEx1 = gridParser.add_mutually_exclusive_group( required=False )
mutuallyEx1.add_argument('-itar','--inTarBall', 
    metavar='InTarBall', nargs = '?', dest = 'grid_inTarBall',
    help = "The environemnt tarball for posterior usage.")
mutuallyEx1.add_argument('-otar','--outTarBall',
    metavar='OutTarBall',  nargs = '?', dest = 'grid_outTarBall',
    help = "The environemnt tarball for posterior usage.")
################################################################################
## Temporary classes only to deal with diamond inherit scheme
_inParser = argparse.ArgumentParser(add_help = False)
_inParser.add_argument('--inDS','-i', action='store', 
                       required = True, dest = 'grid_inDS',
                       help = "The input Dataset ID (DID)")
_inParser.add_argument('--secondaryDSs', action='store', nargs='+',
                       required = False, dest = 'grid_secondaryDS',
                       help = "The secondary Dataset ID (DID), in the format name:nEvents:place")
_inParser.add_argument('--forceStaged', action='store_true',
    required = False,  dest = 'grid_forceStaged', default = False,
    help = """Force files from primary DS to be staged to local
    disk, even if direct-access is possible.""")
_inParser.add_argument('--forceStagedSecondary', action='store_true',
    required = False, dest = 'grid_forceStagedSecondary',
    help = """Force files from secondary DS to be staged to local
              disk, even if direct-access is possible.""")
_inParser.add_argument('--reusableSecondary', nargs='?',
    required = False, dest = 'grid_reusableSecondary',
    help = """Allow reuse secondary dataset.""")
_inParser.add_argument('--nFiles', nargs='?', type=int,
    required = False, dest = 'grid_nFiles',
    help = """Number of files to run.""")
_inParser.add_argument('--nFilesPerJob', nargs='?', type=int,
    required = False, dest = 'grid_nFilesPerJob',
    help = """Number of files to run per job.""")
################################################################################
_outParser = argparse.ArgumentParser(add_help = False)
_outParser.add_argument('--outDS','-o', action='store', 
                        required = True, dest = 'grid_outDS',
                        help = "The output Dataset ID (DID)")
_outParser.add_argument('--outputs', required = True, dest = 'grid_outputs',
    help = """The output format.""")
gridParser.add_argument('--allowTaskDuplication', action='store_true',
    required = False, dest = 'grid_allowTaskDuplication',
    help = """Flag to disable auto retrying jobs.""")
################################################################################
## Input and output grid parser
ioGridParser = argparse.ArgumentParser(add_help = False, 
                                       parents = [_inParser, _outParser, gridParser])

## Input grid parser
inGridParser = argparse.ArgumentParser(add_help = False, 
                                       parents = [_inParser, gridParser])

## Output grid parser
outGridParser = argparse.ArgumentParser(add_help = False, 
                                        parents = [_outParser, gridParser])
# Remove temp classes
del _inParser, _outParser
################################################################################
## GridNamespace
# Make sure to use GridNamespace specialization for the used package when
# parsing arguments.
class GridNamespace( LoggerNamespace, Logger ):
  """
    Improves argparser workspace object to support creating a string object
    with the input options.
  """

  def __init__(self, prog = 'prun', **kw):
    Logger.__init__( self, kw )
    LoggerNamespace.__init__( self, **kw )
    self.prog = prog

  def __call__(self):   
    self.run_cmd()

  def setBExec(self, value):
    """
      Add a build execute command.
    """
    if len(value) > 0 and value[0] != '"':
      value = '"' + value
    if value[-1] != '"':
      value += '"'
    self.bexec = value 

  def setExec(self, value):
    """
      Add the execution command on grid.
    """
    if len(value) < 1 or value[0] != '"':
      value = '"' + value
    if len(value) < 2 or value[-1] != '"':
      value += '"'
    self.exec_ = value 

  def pre_download(self):
    """
      Packages which need special libraries downloads to install should inherit
      from this class and overload this method to download needed libraries.
    """
    pass

  def extFile(self):
    """
      Return a comma separated list of extFiles needed by this GridNamespace.
    """
    return ''

  def __run(self, str_):
    """
      Run the command
    """
    self.pre_download()
    workDir=os.path.expandvars("$ROOTCOREBIN/..")
     # We need to cd to this dir so that prun accepts the submission
    os.chdir(workDir)
    os.system(str_)

  def nSpaces(self):
    return len(self.prog) + 1
    
  def run_cmd(self):
    """
      Execute parsed arguments.
    """
    # Try to change our level if we have an output_level option:
    try:
      self.setLevel( self.output_level )
    except AttributeError:
      pass
    # Add program to exec and build exec if available
    full_cmd_str = self.prog + (' --bexec ' + self.bexec if hasattr(self,'bexec') else '') + ' \\\n'
    # The number of spaces to add to each following option to improve readability:
    nSpaces = self.nSpaces()
    # Add execute grid command if available
    if hasattr(self,'exec_'):
      full_cmd_str += (' ' * nSpaces) + '--exec' + ' \\\n'
      exec_str = [textwrap.dedent(l) for l in self.exec_.split('\n')]
      exec_str = [l for l in exec_str if l not in (';','"','')]
      if exec_str[-1][-2:] != ';"': 
        exec_str[-1] += ';"' 
      for i, l in enumerate(exec_str):
        if i == 0:
          moreSpaces = 2
        else:
          moreSpaces = 4
        full_cmd_str += (' ' * (nSpaces + moreSpaces) ) + l + ' \\\n'
    # Add needed external files:
    if self.extFile() and not self.extFile() in self.grid_extFile:
      if len(self.grid_extFile):
        self.grid_extFile += ','
      self.grid_extFile += self.extFile()
    # Add extra arguments
    for name, value in get_attributes(self):
      if 'grid_' in name:
        name = name.replace('grid_','--')
      elif 'gridExpand_' in name:
        if value:
          name = value
          value = True
        else:
          continue
      else:
        continue
      tVal = type(value)
      if tVal == bool and value:
        full_cmd_str += (' ' * nSpaces) + name + ' \\\n'
      elif value:
        if isinstance(value, list):
          full_cmd_str += (' ' * nSpaces) + name + '=' + ','.join(value) + ' \\\n'
        else:
          full_cmd_str += (' ' * nSpaces) + name + '=' + str(value) + ' \\\n'
    # Now we show command:
    self._logger.info("Command:\n%s", full_cmd_str)
    full_cmd_str = re.sub('\\\\ *\n','', full_cmd_str )
    full_cmd_str = re.sub(' +',' ', full_cmd_str)
    self._logger.debug("Command without spaces:\n%s", full_cmd_str)
    # And run it:
    if not self.dry_run:
      self.__run(full_cmd_str)
      pass
################################################################################
## Specialization of GridNamespace for this package
# Use this namespace when parsing grid option on FastNet package.
class FastNetGridNamespace(GridNamespace):
  """
    Special FastNetTool GridNamespace class.
  """

  def __init__(self, prog = 'prun', **kw):
    GridNamespace.__init__( self, prog, **kw )
    self.setBExec('source ./buildthis.sh')

  def pre_download(self):
    # We need this to avoid being banned from grid:
    if not os.path.isfile(os.path.expandvars("$ROOTCOREBIN/../FastNetTool/cmt/boost_1_58_0.tar.gz")):
      self._logger.info('Downloading boost to avoid doing it on server side.')
      import urllib
      urllib.urlretrieve("http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz", 
                         filename=os.path.expandvars("$ROOTCOREBIN/../FastNetTool/cmt/boost_1_58_0.tar.gz"))
    else:
      self._logger.info('Boost already downloaded.')

  def extFile(self):
    return '"FastNetTool/cmt/boost_1_58_0.tar.gz"'
################################################################################


