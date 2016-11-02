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

reqArgs.add_argument('-r','--refFile', default = None, required = True,
                     help = """The performance file to retrieve the operation points.""")

optArgs = crossValStatsMonParser.add_argument_group( "Optional arguments", "")

optArgs.add_argument('--debug', default=False, action='store_true',
                     help = "Debug mode")


optArgs.add_argument('--grid', default=False, action='store_true',
                     help = "Enable the grid filter tag.")


optArgs.add_argument('--doBeamer', default=False, action='store_true',
                     help = "Enable the beamer creation.")

optArgs.add_argument('--doShortSlides', default=False, action='store_true',
                     help = "Enable the beamer short slides.")


optArgs.add_argument('--output', '-o', default="report", 
                     help = "the output file path to the data"
                     )
