__all__ = [ 'RetrieveCoreFramework', 'coreFrameworkParser'
          , 'RetrieveDataFramework', 'dataframeParser'
          ]

from RingerCore import argparse, ArgumentParser
from TuningTools.coreDef import coreConf, AvailableTuningToolCores, dataframeConf

class RetrieveCoreFramework( argparse.Action ):
  def __call__(self, parser, namespace, value, option_string=None):
    coreConf.set( value )

coreFrameworkParser = ArgumentParser( add_help = False)
coreFrameworkGroup = coreFrameworkParser.add_argument_group("TuningTools CORE configuration" , "")
coreFrameworkGroup.add_argument( '-core', '--core-framework',
    type = AvailableTuningToolCores, action = RetrieveCoreFramework,
    help = """ Specify which core framework should be used in the job.""" \
      + ( " Current default is: " + AvailableTuningToolCores.tostring( coreConf.default() ) ))

if not hasattr(argparse.Namespace, 'core_framework'):
  # Decorate Namespace with the TuningTools core properties.
  # We do this on the original class to simplify usage, as there will be
  # no need to specify a different namespace for parsing the arguments.
  def _getCoreFramework(self):
    if coreConf: return coreConf()
    else: return None

  def _setCoreFramework(self, val):
    coreConf.set( val )

  argparse.Namespace.data_framework = property( _getCoreFramework, _setCoreFramework )

class RetrieveDataFramework( argparse.Action ):
  def __call__(self, parser, namespace, value, option_string=None):
    dataframeConf.set( value )


from TuningTools.dataframe.EnumCollection import Dataframe
dataframeParser = ArgumentParser( add_help = False )
dataframeGroup = dataframeParser.add_argument_group("TuningTools DATA framework configuration" , "")
dataframeGroup.add_argument( '--data-framework',
    type = Dataframe, action = RetrieveDataFramework,
    help = """Specify which data framework should be used in the job.""" )


if not hasattr(argparse.Namespace, 'data_framework'):
  # Decorate Namespace with the TuningTools data properties.
  # We do this on the original class to simplify usage, as there will be
  # no need to specify a different namespace for parsing the arguments.
  def _getDataFramework(self):
    from RingerCore import keyboard
    keyboard()
    if dataframeConf: return dataframeConf()
    else: return None

  def _setDataFramework(self, val):
    from RingerCore import keyboard
    keyboard()
    dataframeConf.set( val )

  argparse.Namespace.data_framework = property( _getDataFramework, _setDataFramework )

