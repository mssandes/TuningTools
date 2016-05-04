__all__ = ['TuningToolGridNamespace']

from RingerCore import GridNamespace

################################################################################
## Specialization of GridNamespace for this package
# Use this namespace when parsing grid option on TuningTool package.
class TuningToolGridNamespace(GridNamespace):
  """
    Special TuningTools GridNamespace class.
  """

  def __init__(self, prog = 'prun', **kw):
    GridNamespace.__init__( self, prog, **kw )
    self.setBExec('source ./buildthis.sh --grid; source ./buildthis.sh --grid')

  def pre_download(self):
    import os
    # We need this to avoid being banned from grid:
    from RingerCore import mkdir_p
    mkdir_p("$ROOTCOREBIN/../Downloads/")
    if not os.path.isfile(os.path.expandvars("$ROOTCOREBIN/../Downloads/boost.tgz")):
      self._logger.info('Downloading boost to avoid doing it on server side.')
      import urllib
      urllib.urlretrieve("http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz", 
                         filename=os.path.expandvars("$ROOTCOREBIN/../Downloads/boost.tgz"))
    else:
      self._logger.info('Boost already downloaded.')
    if not os.path.isfile(os.path.expandvars("$ROOTCOREBIN/../Downloads/cython.tgz")):
      self._logger.info('Downloading cython to avoid doing it on server side.')
      import urllib
      urllib.urlretrieve("http://cython.org/release/Cython-0.23.4.tar.gz", 
                         filename=os.path.expandvars("$ROOTCOREBIN/../Downloads/cython.tgz"))
    else:
      self._logger.info('Cython already downloaded.')
    if not os.path.isfile(os.path.expandvars("$ROOTCOREBIN/../Downloads/numpy.tgz")):
      self._logger.info('Downloading numpy to avoid doing it on server side.')
      import urllib
      urllib.urlretrieve("http://sourceforge.net/projects/numpy/files/NumPy/1.10.4/numpy-1.10.4.tar.gz/download", 
                         filename=os.path.expandvars("$ROOTCOREBIN/../Downloads/numpy.tgz"))
    else:
      self._logger.info('Numpy already downloaded.')

  def extFile(self):
    from glob import glob
    #return ','.join(glob("Downloads/*.tgz"))
    return 'Downloads/numpy.tgz,Downloads/boost.tgz'
################################################################################
