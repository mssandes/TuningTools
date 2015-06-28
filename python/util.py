#!/usr/bin/env python

import re, os, __main__
import pickle
import logging


class NullHandler(logging.Handler):
  def emit(self, record):
    pass

def getModuleLogger(logName, logDefaultLevel = logging.INFO):
  logger = logging.getLogger( logName )
  # Make sure we only add one handler:
  if not logger.handlers:
    logger.setLevel( logDefaultLevel )
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel( logDefaultLevel )
    # create formatter
    formatter = logging.Formatter("Py.%(name)-37s%(levelname)s %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
  return logger
  

def checkForUnusedVars(d, fcn = None):
  """
    Checks if dict @d has unused properties and print them as warnings
  """
  for key in d.keys():
    msg = 'Obtained not needed parameter: %s' % key
    if fcn:
      fcn(msg)
    else:
      print 'WARNING:%s' % msg

def mkdir_p(path):
  import os, errno
  try:
    os.makedirs(path)
  except OSError as exc: # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: raise


def printArgs(args, fcn = None):
  try:
    import pprint as pp
    if args:
      if not isinstance(args,dict):
        args_dict = vars(args)
      else:
        args_dict = args
      msg = 'Retrieved the following configuration:\n%s' % pp.pformat([(key, args_dict[key]) for key in sorted(args_dict.keys())])
    else:
      msg = 'Retrieved empty configuration!'
    if fcn:
      fcn(msg)
    else:
      print 'INFO:%s' % msg
  except ImportError:
    logger.info('Retrieved the following configuration: \n %r', vars(args))


def reshape( input ):
  import numpy as np
  return np.array(input.tolist())


def load(input):
  return pickle.load(open(input, 'r'))

def save(output, object):
  filehandler = open(output,"wb")
  pickle.dump(object,filehandler)
  filehandler.close()

def normalizeSumRow(data):
  import numpy as np
  for row in xrange(data.shape[0]):
    data[row] /= np.sum(data[row])
  return data

def stdvector_to_list(vec):
  size = vec.size()
  l = size*[0]
  for i in range(size):
    l[i] = vec[i]
  return l

def findFile( filename, pathlist, access ):
  """
     Find <filename> with rights <access> through <pathlist>.
     Author: Wim Lavrijsen (WLavrijsen@lbl.gov)
     Copied from 'atlas/Control/AthenaCommon/python/Utils/unixtools.py'
  """

  # special case for those filenames that already contain a path
  if os.path.dirname( filename ):
    if os.access( filename, access ):
      return filename

  # test the file name in all possible paths until first found
  for path in pathlist:
    f = os.path.join( path, filename )
    if os.access( f, access ):
      return f

  # no such accessible file avalailable
  return None  

class Include:
  def __call__(self, filename, globalz=None, localz=None, clean=False):
    "Simple routine to execute python script, possibly keeping global and local variables."
    searchPath = re.split( ',|' + os.pathsep, os.environ['PYTHONPATH'] )
    if '' in searchPath:
      searchPath[ searchPath.index( '' ) ] = str(os.curdir)
    trueName = findFile(filename, searchPath, os.R_OK )
    gworkspace = {}
    lworkspace = {}
    if globalz: gworkspace.update(globalz)
    if localz: lworkspace.update(localz)
    if not clean:
      gworkspace.update(__main__.__dict__)
      lworkspace.update(__main__.__dict__)
    if trueName: 
      try:
        execfile(trueName, gworkspace, lworkspace)
      except NameError, e:
        if e == "name 'execfile' is not defined":
          Include.xfile(trueName, globalz, localz)
        else:
          raise e
    else:
      raise ImportError("Cannot include file: %s" % filename)

  @classmethod
  def xfile(cls, afile, globalz=None, localz=None):
    "Alternative to execfile for python3.0"
    with open(afile, "r") as fh:
      exec(fh.read(), globalz, localz)


include = Include()
