import re, os, __main__
import cPickle
import gzip
import inspect
import numpy as np


loadedEnvFile = False
def sourceEnvFile():
  """
    Emulate source new_env_file.sh on python environment.
  """
  try:
    from FastNetTool.Logger import Logger
    logger = Logger.getModuleLogger(__name__)
    import os, sys
    global loadedEnvFile
    if not loadedEnvFile:
      with open(os.path.expandvars('$ROOTCOREBIN/../FastNetTool/cmt/new_env_file.sh'),'r') as f:
        lines = f.readlines()
        lineparser = re.compile(r'test "\$\{(?P<shellVar>[A-Z1-9]*)#\*(?P<addedPath>\S+)\}" = "\$\{(?P=shellVar)\}" && export (?P=shellVar)=\$(?P=shellVar):(?P=addedPath) || true')
        for line in lines:
          m = lineparser.match(line)
          if m:
            shellVar = m.group('shellVar')
            if shellVar != 'PYTHONPATH':
              continue
            addedPath = os.path.expandvars(m.group('addedPath'))
            if not addedPath:
              logger.warning("Couldn't retrieve added path on line \"%s\".", line)
              continue
            if not os.path.exists(addedPath):
              logger.warning("Couldn't find following path \"%s\".", addedPath)
              continue
            if not addedPath in os.environ[shellVar]:
              sys.path.append(addedPath)
              logger.info("Successfully added path: \"%s\".", line)
      loadedEnvFile=True
  except IOError:
    raise RuntimeError("Cannot find new_env_file.sh, did you forget to set environment or compile the package?")
  
class EnumStringification:
  "Adds 'enum' static methods for conversion to/from string"
  @classmethod
  def tostring(cls, val):
    "Transforms val into string."
    for k,v in vars(cls).iteritems():
      if v==val:
        return k

  @classmethod
  def fromstring(cls, str):
    "Transforms string into enumeration."
    return getattr(cls, str, None)


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

def fixFileList( fileList ):
  """
    Return a filelist from untreated file list.
  """
  # Treat comma separated lists:
  if type(fileList) is str:
    fileList = fileList.split(',')
  # Make sure our confFileList is a list (just to be compatible for 
  if not type(fileList) is list:
    fileList = [fileList]
  import os
  for i, filePath in enumerate(fileList):
    fileList[i] = os.path.abspath(os.path.expandvars(os.path.expanduser(filePath)))
    if not os.path.isfile(fileList[i]):
      raise ValueError("Inexistent file '%s'" % filePath)
  return fileList


def get_attributes(o, **kw):
  """
    Return attributes from a class or object.
  """
  onlyVars = kw.pop('onlyVars', False)
  return [(a[0] if onlyVars else a) for a in inspect.getmembers(o, lambda a:not(inspect.isroutine(a))) \
             if not(a[0].startswith('__') and a[0].endswith('__')) ]

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
  #sourceEnvFile()
  import numpy as np
  return np.array(input.tolist())

def reshape_to_array( input ):
  import numpy as np
  return np.reshape(input, (1,np.product(input.shape)))[0]


def conditionalOption( argument, value ):
  return argument + value if value else ''

def trunc_at(s, d, n=1):
  "Returns s truncated at the n'th (1st by default) occurrence of the delimiter, d."
  return d.join(s.split(d)[:n])

def start_after(s, d, n=1):
  "Returns s after at the n'th (1st by default) occurrence of the delimiter, d."
  return d.join(s.split(d)[n:])

#def stdvector(vecType, *argl):
#  from ROOT.std import vector
#  v = vector(vecType)
#  return v(*argl)

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


def normalizeSumRow(data):
  #sourceEnvFile()
  import numpy as np
  norms = data.sum(axis=1)
  norms[norms==0] = 1
  data = data / norms[:, np.newaxis ]
  return data

def geomean(nums):
  return (reduce(lambda x, y: x*y, nums))**(1.0/len(nums))

def mean(nums):
  return (sum(nums)/len(nums))


def getEff( outSignal, outNoise, cut ):
  '''
    [detEff, faEff] = getEff(outSignal, outNoise, cut)
    Returns the detection and false alarm probabilities for a given input
    vector of detector's perf for signal events(outSignal) and for noise 
    events (outNoise), using a decision threshold 'cut'. The result in whithin
    [0,1].
  '''

  detEff = (outSignal.shape[0]- np.searchsorted(outSignal,cut))/ float(outSignal.shape[0]) 
  faEff  = (outNoise.shape[0] - np.searchsorted(outNoise, cut))/ float(outNoise.shape[0])
  return [detEff, faEff]

def calcSP( pd, pf ):
  '''
    ret  = calcSP(x,y) - Calculates the normalized [0,1] SP value.
    effic is a vector containing the detection efficiency [0,1] of each
    discriminating pattern.  
  '''
  from numpy import sqrt
  return sqrt(geomean([pd,pf]) * mean([pd,pf]))

def genRoc( outSignal, outNoise, numPts = 1000 ):
  '''
    [spVec, cutVec, detVec, faVec] = genROC(out_signal, out_noise, numPts, doNorm)
    Plots the RoC curve for a given dataset.
    Input Parameters are:
       out_signal     -> The perf generated by your detection system when
                         electrons were applied to it.
       out_noise      -> The perf generated by your detection system when
                         jets were applied to it
       numPts         -> (opt) The number of points to generate your ROC.
    
    If any perf parameters is specified, then the ROC is plot. Otherwise,
    the sp values, cut values, the detection efficiency and false alarm rate 
    are returned (in that order).
  '''
  cutVec = np.arange( -1,1,2 /float(numPts))
  cutVec = cutVec[0:numPts - 1]
  detVec = np.array(cutVec.shape[0]*[float(0)])
  faVec  = np.array(cutVec.shape[0]*[float(0)])
  spVec  = np.array(cutVec.shape[0]*[float(0)])
  signal = np.sort(np.array(outSignal), kind='heapsort')
  noise  = np.sort(np.array(outNoise) , kind='heapsort')
  for i in range(cutVec.shape[0]):
    [detVec[i],faVec[i]] = getEff( signal, noise,  cutVec[i] ) 
    spVec[i] = calcSP(detVec[i],1-faVec[i])

  return [spVec, cutVec, detVec, faVec]


'''
  Class Performance
'''
class Roc:
  def __init__(self, vecList, label):
    self.label    = label
    self.spVec    = vecList[0]
    self.detVec   = vecList[1]
    self.faVec    = vecList[2]
    self.cutVec   = vecList[3]
    self.sp       = self.spVec[ np.argmax(self.spVec) ]
    self.det      = self.detVec[ np.argmax(self.spVec) ]
    self.fa       = self.faVec[ np.argmax(self.spVec) ]
    self.cut      = self.cutVec[ np.argmax(self.spVec)]


