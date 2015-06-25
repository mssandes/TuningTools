
from FastNetTool.util import getModuleLogger
class Logger():
  """
    Simple class for giving class logging capability as well as the possibility
    for being serialized by pickle.

    The logger states are not pickled. When unpickled, it will have to be
    configured or it will use default configuration.
  """
  def __init__(self, **kw ):
    """
      Retrieve from args the logger, or create it using default configuration.
    """
    self._logger = kw.pop('logger', None)  or getModuleLogger(self.__module__)

  def __getstate__(self):
    """
      Makes logger invisible for pickle
    """
    odict = self.__dict__.copy() # copy the dict since we change it
    del odict['_logger']         # remove logger entry
    return odict

  def __setstate__(self, dict):
    """
      Add logger to object if it doesn't have one:
    """
    self.__dict__.update(dict)   # update attributes
    try: 
      self._logger
    except AttributeError:
      self._logger = getModuleLogger(self.__module__)

    if not self._logger: # Also add a logger if it is set to None
      self._logger = getModuleLogger(self.__module__)
