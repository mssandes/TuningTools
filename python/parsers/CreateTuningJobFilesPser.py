__all__ = ['JobFileTypeCreation', 'tuningJobFileParser']

try:
  import argparse
except ImportError:
  from RingerCore import argparse

from RingerCore.util import get_attributes, EnumStringification

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
