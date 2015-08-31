import os
import re
import textwrap
try:
  import argparse
except ImportError:
  from FastNetTool import argparse

####################################################
# Logger parser
from FastNetTool.util import get_attributes
from FastNetTool.Logger import LoggingLevel, Logger
loggerParser = argparse.ArgumentParser(add_help = False)
loggerParser.add_argument('--output-level', 
    default = LoggingLevel.tostring( LoggingLevel.INFO ), 
    type=str, required = False, choices = get_attributes(LoggingLevel, onlyVars = True),
    help = "The output level for the main logger")

####################################################
# Basic grid parser
gridParser = argparse.ArgumentParser(add_help = False)

gridParser.add_argument('--site',default = 'AUTO',
    help = "The site location where the job should run.",
    nargs='?', required = False,
    dest = 'grid_site')
gridParser.add_argument('--excludedSite', default = '', 
    help = "The excluded site location.", nargs='?',
    required = False, dest = 'grid_excludedSize')
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
    required = False, dest = 'grid_extFile',
    help = """External file to add.""")
gridParser.add_argument('--maxNFilesPerJob', nargs='?',
    required = False, dest = 'grid_maxNFilesPerJob',
    help = """Maximum number of files per job.""")
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

_outParser = argparse.ArgumentParser(add_help = False)
_outParser.add_argument('--outDS','-o', action='store', 
                        required = True, dest = 'grid_outDS',
                        help = "The output Dataset ID (DID)")
_outParser.add_argument('--outputs', required = True, dest = 'grid_outputs',
    help = """The output format.""")

####################################################
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

####################################################
### Workspaces
## LoggerWorkspace
class LoggerWorkspace( argparse.Namespace ):
  """
    Workspace for dealing with logger parser properties
  """
  def __init__(self, **kw):
    argparse.Namespace.__init__( self, **kw )

  def __getattr__(self, attr):
    if attr == 'output_level' and type(self.output_level) == str:
      return LoggingLevel.fromstring( self.__dict__['output_level'] )
    else:
      return self.__dict__[attr]


## GridWorkspace
class GridWorkspace( LoggerWorkspace, Logger ):
  """
    Improves argparser workspace object to support creating a string object
    with the input options.
  """

  def __init__(self, prog = 'prun', **kw):
    Logger.__init__( self, kw )
    LoggerWorkspace.__init__( self, **kw )
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
      Return a comma separated list of extFiles needed by this GridWorkspace.
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
      print exec_str
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
        name = value
        value = True
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
    self._logger.info("Command without spaces:\n%s", full_cmd_str)
    # And run it:
    try:
      if not self.dry_run:
        self.__run(full_cmd_str)
        pass
    except KeyError:
      self.__run(full_cmd_str)
      pass

class FastNetGridWorkspace(GridWorkspace):
  """
    Special FastNetTool GridWorkspace class.
  """

  def __init__(self, prog = 'prun', **kw):
    GridWorkspace.__init__( self, prog, **kw )
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


