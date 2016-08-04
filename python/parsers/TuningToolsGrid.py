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
    GridNamespace.pre_download(self)

  def extFile(self):
    from glob import glob
    #return ','.join(glob("Downloads/*.tgz"))
    return 'Downloads/numpy.tgz,Downloads/boost.tgz,Downloads/cython.tgz'
################################################################################
