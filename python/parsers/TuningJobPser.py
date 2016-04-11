__all__ = ['tuningJobParser', 'TuningToolGridNamespace']

try:
  import argparse
except ImportError:
  from RingerCore import argparse

from RingerCore.util import NotSet
from RingerCore.Parser import GridNamespace

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
tuningOptArgs.add_argument('--outputFileBase', action='store', default = NotSet, 
    help = """Base name for the output file.""")
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
    self.setBExec('source ./buildthis.sh --grid; source ./buildthis.sh --grid')

  def pre_download(self):
    import os
    # We need this to avoid being banned from grid:
    if not os.path.isfile(os.path.expandvars("$ROOTCOREBIN/../Downloads/boost.tgz")):
      self._logger.info('Downloading boost to avoid doing it on server side.')
      import urllib
      urllib.urlretrieve("http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz", 
                         filename=os.path.expandvars("$ROOTCOREBIN/../Downloads/boost.tgz"))
    else:
      self._logger.info('Boost already downloaded.')
    if not os.path.isfile(os.path.expandvars("$ROOTCOREBIN/../Downloads/numpy.tgz")):
      self._logger.info('Downloading numpy to avoid doing it on server side.')
      import urllib
      urllib.urlretrieve("http://sourceforge.net/projects/numpy/files/NumPy/1.10.4/numpy-1.10.4.tar.gz/download", 
                         filename=os.path.expandvars("$ROOTCOREBIN/../Downloads/numpy.tgz"))
    else:
      self._logger.info('Numpy already downloaded.')

  def extFile(self):
    from glob import glob
    #return ','.join(glob("Downloads/*.tgz"))
    return 'Downloads/numpy.tgz,Downloads/boost.tgz'
################################################################################
