__all__ = ['JobFileTypeCreation', 'tuningJobFileParser','CreateTuningJobFilesNamespace']

from RingerCore import argparse, get_attributes, BooleanStr, \
                       NotSet, LoggerNamespace, EnumStringification

from TuningTools.CrossValid import CrossValidMethod
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
                     choices = get_attributes(JobFileTypeCreation, onlyVars = True, getProtected = False),
                     nargs='+',
                     help = """Which kind of files to create. You can choose one
                     or more of the available choices, just don't use all with
                     the other available choices.""")
tuningJobFileParser.add_argument('--compress', default='True', dest = '_compress',
    help = "Whether to compress files or not. Allowed options: " + \
       str( get_attributes( BooleanStr, onlyVars = True, getProtected = False ) )
       )
################################################################################
jobConfig = tuningJobFileParser.add_argument_group( "JobConfig Files Creation Options", 
                                       """Change configuration for
                                       job config files creation.""")
jobConfig.add_argument('-oJConf', '--jobConfiFilesOutputFolder', 
                       default = NotSet, 
                       help = "The job config files output folder.")
jobConfig.add_argument('--neuronBounds', nargs='+', type=int, default = NotSet,  
                        help = """
                            Input a sequential bounded list to be used as the
                            neuron job range, the arguments should have the
                            same format from the seq unix command or as the
                            Matlab format. If not specified, the range will
                            start from 1. I.e. 5 2 9 leads to [5 7 9] and 50
                            leads to 1:50
                               """)
jobConfig.add_argument('--sortBounds', nargs='+', type=int, default = NotSet,
                       help = """
                          Input a sequential bounded list using seq format to
                          be used as the sort job range, but the last bound
                          will be opened just as happens when using python
                          range function. If not specified, the range will
                          start from 0. I.e. 5 2 9 leads to [5 7] and 50 leads
                          to range(50)
                              """)
jobConfig.add_argument('--nInits', nargs='?', type=int, default = NotSet,
                       help = "The number of initilizations of the discriminator.")
jobConfig.add_argument('--nNeuronsPerJob', type=int, default = NotSet,
                        help = "The number of hidden layer neurons per job.")
jobConfig.add_argument('--nSortsPerJob', type=int, default = NotSet,  
                       help = "The number of sorts per job.")
jobConfig.add_argument('--nInitsPerJob', type=int, default = NotSet,  
                        help = "The number of initializations per job.")
################################################################################
crossConfig = tuningJobFileParser.add_argument_group( "CrossValid File Creation Options", 
                                         """Change configuration for CrossValid
                                         file creation.""")
crossConfig.add_argument('-outCross', '--crossValidOutputFile', 
                       default = 'crossValid', 
                       help = "The cross validation output file.")
crossConfig.add_argument('-m','--method', default = NotSet, dest = '_method',
                         help = "The Cross-Validation method. Possible options are: " + \
                             str(get_attributes( CrossValidMethod, onlyVars = True, getProtected = False))
                        )
crossConfig.add_argument('-ns',  '--nSorts', type=int, default = NotSet,
                         help = """The number of sort used by cross validation
                                configuration.""")
crossConfig.add_argument('-nb', '--nBoxes', type=int,  default = NotSet,
                         help = """The number of boxes used by cross validation
                                 configuration.""")
crossConfig.add_argument('-ntr', '--nTrain', type=int, default = NotSet,
                         help = """The number of train boxes used by cross
                                validation.""")
crossConfig.add_argument('-nval','--nValid', type=int, default = NotSet,
                         help = """The number of valid boxes used by cross
                                validation.""")
crossConfig.add_argument('-ntst','--nTest',  type=int, default = NotSet,
                         help = """The number of test boxes used by cross
                                validation.""")
crossConfig.add_argument('-seed', type=int, default = NotSet,
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
                             TuningTools.PreProc.py file.
                             
                             This string can have classes from the PreProc
                             module initialized with determined values. E.g.:

                             -ppCol "[[[Norm1(),MapStd()],[RingerRp(2.,1.3)],[MapStd]],[[Norm1(),MapStd],[Norm1],[MapStd]],[[Norm1,MapStd],[Norm1({'level' : 'VERBOSE'})],[MapStd({'d' : {'level' : 'VERBOSE'}})]]]"

                             The usage of () or empty will make no difference
                             resulting in the class instance initialization.

                             Also, a special syntax need to be used when
                             passing keyword arguments as specified in:

                             MapStd({'level' : 'VERBOSE'}) (equivalent in python) => MapStd( level = VERBOSE )

                             MapStd({'d' : {'level' : 'VERBOSE'}}) => MapStd( d = { level : VERBOSE } )
                             """)
ppConfig.add_argument('-pp_ns', '--pp_nSorts', default = NotSet, type=int,
                      help = """The number of sort used by cross validation
                             configuration. Import from nSorts if not set.""")
ppConfig.add_argument('-pp_nEt', '--pp_nEtBins', default = NotSet, type=int,
                      help = """The number of et bins.""")
ppConfig.add_argument('-pp_nEta', '--pp_nEtaBins', default = NotSet, type=int,
                      help = """The number of eta bins.""")
################################################################################

################################################################################
# Use this namespace when parsing grid CrossValidStat options
class CreateTuningJobFilesNamespace(LoggerNamespace):
  """
    Parse CreateTuningJobFiles options.
  """

  def __init__(self, **kw):
    LoggerNamespace.__init__( self, **kw )

  @property
  def compress(self):
    return BooleanStr.treatVar('_compress', self.__dict__, False)

  @property
  def method(self):
    return CrossValidMethod.retrieve(self.__dict__['_method'] ) if self.__dict__['_method'] not in (NotSet, None) \
        else self.__dict__['_method']

