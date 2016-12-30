__all__ = ['crossValStatsJobParser', 'CrossValidStatNamespace']

from RingerCore import argparse, get_attributes, BooleanStr, \
                       NotSet, LoggerNamespace

from TuningTools.dataframe.EnumCollection import RingerOperation

################################################################################
# Create cross valid stats job parser file related objects
################################################################################
crossValStatsJobParser = argparse.ArgumentParser(add_help = False, 
                                          description = 'Retrieve cross-validation information and tuned discriminators performance.',
                                          conflict_handler = 'resolve')
################################################################################
reqArgs = crossValStatsJobParser.add_argument_group( "Required arguments", "")
reqArgs.add_argument('-d', '--discrFiles', action='store', 
    metavar='data', required = True,
    help = """The tuned discriminator data files or folders that will be used to run the
          cross-validation analysis.""")
################################################################################
optArgs = crossValStatsJobParser.add_argument_group( "Optional arguments", "")
# TODO Reset this when running on the Grid to the GridJobFilter
optArgs.add_argument('--binFilters', action='store', default = NotSet, 
    help = """This option filter the files types from each job. It can be a string
    with the name of a class defined on python/CrossValidStat dedicated to automatically 
    separate the files or a comma separated list of patterns that identify unique group 
    of files for each bin. A python list can also be speficied. 
    E.g.: You can specify 'group001,group002' if you have file001.group001.pic, 
    file002.group001, file001.group002, file002.group002 available and group001 
    specifies one binning limit, and group002 another, both of them with 2 files 
    available in this case.
    When not set, all files are considered to be from the same binning. 
    """)
optArgs.add_argument('-idx','--binFilterIdx', default = None, nargs='+',type = int,
        help = """The index of the bin job to run. e.g. two bins, idx will be: 0 and 1"""
        )
optArgs.add_argument('--doMonitoring', default=NotSet, dest = '_doMonitoring',
    help = "Enable or disable monitoring file creation. Allowed options: " + \
       str( get_attributes( BooleanStr, onlyVars = True, getProtected = False ) )
       )
optArgs.add_argument('--doMatlab', default=NotSet,  dest = '_doMatlab',
    help = "Enable or disable matlab file creation. Allowed options: " + \
       str( get_attributes( BooleanStr, onlyVars = True, getProtected = False ) )
       )
optArgs.add_argument('--doCompress', default=NotSet,  dest = '_doCompress',
    help = "Enable or disable raw output file compression. Allowed options: " + \
       str( get_attributes( BooleanStr, onlyVars = True, getProtected = False ) )
       )
optArgs.add_argument('-r','--refFile', default = None,
                     help = """The performance reference file to retrieve the operation points.""")
optArgs.add_argument('-op','--operation', default = None, 
                     help = """The Ringer operation determining in each Trigger 
                     level or what is (are) the offline operation point reference(s).
                     Possible options are: """ \
                     + str(get_attributes( RingerOperation, onlyVars = True, getProtected = False)) )
optArgs.add_argument('--outputFileBase', action='store', default = NotSet, 
    help = """Base name for the output file.""")
optArgs.add_argument('--monitoringFileName', action='store', default = NotSet, 
    help = """The output name for the monitoring file.""")
optArgs.add_argument('--test', action='store_true', default = False,
    help = "Set debug mode.")

################################################################################
# Use this namespace when parsing grid CrossValidStat options
class CrossValidStatNamespace(LoggerNamespace):
  """
    Parse CrossValidStat options.
  """

  def __init__(self, **kw):
    LoggerNamespace.__init__( self, **kw )

  @property
  def doMonitoring(self):
    return BooleanStr.treatVar('_doMonitoring', self.__dict__, True)

  @property
  def doMatlab(self):
    return BooleanStr.treatVar('_doMatlab', self.__dict__, True)
  @property
  def doCompress(self):
    return BooleanStr.treatVar('_doCompress', self.__dict__, True)

