from RingerCore import checkForUnusedVars, Logger, LoggingLevel, EnumStringification

class BeamerMonReport( Logger ):
  """
  Main object
  """

  def __init__(self, filename, **kw):
    Logger.__init__(self,kw)
    self._title = kw.pop('title', 'Tuning Report')
    self._institute = kw.pop('institute', 'Universidade Federal do Rio de Janeiro (UFRJ)')

    checkForUnusedVars( kw, self._logger.warning )

    import socket
    self._machine = socket.gethostname()
    import getpass
    self._author = getpass.getuser()
    from time import gmtime, strftime
    self._data = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    #Create output file
    self._pfile = open(filename+'.tex','w')

    from BeamerTemplates import BeamerConstants as bconst
    self._pfile.write( bconst.beginDocument )
    pname = self._author+'$@$'+self._machine
    self._pfile.write( (bconst.beginHeader) % \
              (self._title, self._title, pname, self._institute) )

    self._pfile.write( bconst.beginTitlePage )


  def file(self):
    return self._pfile

  def close(self):
    from BeamerTemplates import BeamerConstants as bconst
    self._pfile.write( bconst.endDocument )
    self._pfile.close()
