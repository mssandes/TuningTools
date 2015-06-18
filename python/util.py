#!/usr/bin/env python

import re, os, __main__
import pickle
import logging

class NullHandler(logging.Handler):
  def emit(self, record):
    pass

def printArgs(args, fcn = None):
  try:
    import pprint as pp
    args_dict = vars(args)
    msg = 'Retrieved the following configuration:\n%s' % pp.pformat([(key, args_dict[key]) for key in sorted(args_dict.keys())])
    if fcn:
      fcn(msg)
    else:
      print msg
  except ImportError:
    logger.info('Retrieved the following configuration: \n %r', vars(args))

def load(input):
  return pickle.load(open(input, 'r'))

def save(output, object):
  filehandler = open(output,"wb")
  pickle.dump(object,filehandler)
  filehandler.close()

def alloc_list_space(size):
  l = []
  for i in range(size):
    l.append( None )
  return l

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
      except NameError:
        Include.xfile(trueName, globalz, localz)
    else:
      raise ImportError("Cannot include file: %s" % filename)

  @classmethod
  def xfile(cls, afile, globalz=None, localz=None):
    "Alternative to execfile for python3.0"
    with open(afile, "r") as fh:
      exec(fh.read(), globalz, localz)


include = Include()
