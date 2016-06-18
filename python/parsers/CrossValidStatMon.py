__all__ = ['crossValStatsMonParser']

from RingerCore import argparse, get_attributes, BooleanStr, \
                           NotSet, LoggerNamespace

################################################################################
# Create cross valid monitoring job parser file related objects
################################################################################

crossValStatsMonParser = argparse.ArgumentParser(add_help = False, 
                                                 description = 'Retrieve cross-validation-monitoring information performance.',
                                                 conflict_handler = 'resolve')

reqArgs = crossValStatsMonParser.add_argument_group( "Required arguments", "")


reqArgs.add_argument('-f', '--file', action='store', required = True,
                     help = """The crossvalidation data files or folders that will be used to run the
                               analysis.""")

reqArgs.add_argument('-p','--perfFile', default = None, required = True,
                     help = """The performance file to retrieve the operation points.""")

optArgs = crossValStatsMonParser.add_argument_group( "Optional arguments", "")


optArgs.add_argument('--grid', default="False",
                     help = "Enable or disable the bin filter. Allowed options: " + \
                   str( get_attributes( BooleanStr, onlyVars = True, getProtected = False ) )
                          )

optArgs.add_argument('--doBeamer', default="True",
                     help = "Enable or disable the beamer creation. Allowed options: " + \
                   str( get_attributes( BooleanStr, onlyVars = True, getProtected = False ) )
                          )


optArgs.add_argument('--shortSlides', default="False",
                     help = "Enable or disable the short presentation. If True, will draw only tables performance. Allowed options: " + \
                         str( get_attributes( BooleanStr, onlyVars = True, getProtected = False ) )
                          )

optArgs.add_argument('--basePath', default="report", 
                     help = "the output file path to the data"
                     )

optArgs.add_argument('--tuningReport', default="tuningReport", 
                     help = "the output file path to the data"
                     )


