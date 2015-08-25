from FastNetTool.util import EnumStringification
import logging

class LoggingLevel ( EnumStringification ):
  """
    A wrapper for logging levels, which allows stringification of known log
    levels.
  """
  DEBUG = logging.DEBUG
  INFO = logging.INFO
  WARNING = logging.WARNING
  ERROR = logging.ERROR

class Logger():
  """
    Simple class for giving inherited classes logging capability as well as the
    possibility for being serialized by pickle.

    The logger states are not pickled. When unpickled, it will have to be
    manually configured or it will use default configuration.
  """

  # FIXME I'm not sure why, but the property level does not call the setter if
  # I don't call it directly (for the FastNet inherited class). Enter in deeper
  # details to fix this issue

  # FIXME: Be sure that the FastNetWrapper is able to reconfigure its inherited
  # classes to use the desired configuration level as well as the python levels

  @classmethod
  def getModuleLogger(cls, logName, logDefaultLevel = logging.INFO):
    """
      Retrieve logging stream handler using logName and add a handler
      to stdout if it does not have any handlers yet.

      Format logging stream handler to output in the same format used by Athena
      messages.
    """
    logger = logging.getLogger( logName )
    # Make sure we only add one handler:
    if not logger.handlers:
      logger.setLevel( logDefaultLevel )
      # create console handler and set level to debug
      import sys
      ch = logging.StreamHandler( sys.__stdout__ )
      ch.setLevel( logging.NOTSET ) #  Minimal level in which the ch will print
      # create formatter
      formatter = logging.Formatter("Py.%(name)-34s%(levelname)7s %(message)s")
      # add formatter to ch
      ch.setFormatter(formatter)
      # add ch to logger
      logger.addHandler(ch)
    return logger

  def __init__(self, d = {}, **kw ):
    """
      Retrieve from args the logger, or create it using default configuration.
    """
    d.update( kw )
    self._level = d.pop('level', LoggingLevel.INFO )
    self._logger = d.pop('logger', None)  or \
        Logger.getModuleLogger(self.__module__, self._level )
    self._logger.debug('Initialiazing %s', self.__class__.__name__)

  def getLevel(self):
    return self._level

  def setLevel(self, value):
    self._level = value
    self._logger.setLevel(self._level)

  level = property( getLevel, setLevel )

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
      self._logger = Logger.getModuleLogger(self.__module__)

    if not self._logger: # Also add a logger if it is set to None
      self._logger = Logger.getModuleLogger(self.__module__)
