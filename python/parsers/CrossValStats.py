__all__ = ['crossValStatsJobParser']

from RingerCore import argparse

################################################################################
# Create tuningJob file related objects
################################################################################
crossValStatsJobParser = argparse.ArgumentParser(add_help = False, 
                                          description = 'Retrieve cross-validation information and tuned discriminators performance.',
                                          conflict_handler = 'resolve')
reqArgs = tuningJobParser.add_argument_group( "Required arguments", "")
reqArgs.add_argument('-d', '--discrFiles', action='store', 
    metavar='data', required = True,
    help = """The tuned discriminator data file that will be used to run the
          cross-validation analysis.""")
optArgs = tuningJobParser.add_argument_group( "Optional arguments", "")
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
optArgs.add_argument('--no-evol-plots', action='store_true',
          help = """Don't do tuning evolution analysis plots.""")
optArgs.add_argument('-p','--perfFile', default = None,
                     help = """The performance file to retrieve the operation points.""")
optArgs.add_argument('-op','--operation', default = None,
                     help = """The Ringer operation determining in each Trigger level
                     or what is the offline operation point reference.""")
optArgs.add_argument('-rn','--ref-name', default = "Reference",
                     help = "The reference base name.")
optARgs.add_argument('--outputFileBase', action='store', default = NotSet, 
    help = """Base name for the output file.""")

