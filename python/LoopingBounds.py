from FastNetTool.Logger import Logger
import math

class LoopingBounds ( Logger ):
  """
    Create looping bounds using Matlab/(unix seq command) format
          firstBound:incr:lastBound
    where:
      - firstBound [default 1 if matlabFlag else 0]: always closed bounded
        value where looping will start;
      - incr [default]: the increment to firstBound which will add looping
        instances until lastBound is reached;
      - lastBound: the bound for until when it should increment firstBound.
        If set matlab flag, then it will a closed bound, otherwise opened.
        I.e, if lastBound is 10, when matlab flag is set to false, it will 
        loop on [0,9] or [0,10) otherwise on [1,10] or [1,11).
  """

  _matlabFlag = False
  _vec = []

  def __init__(self, *args, **kw):
    """
      Set values using min:incr:max and matlabFlag as explained in class
      documentation.
    """
    from FastNetTool.util import checkForUnusedVars
    Logger.__init__(self, kw)
    self._matlabFlag = kw.pop('matlabFlag', False )
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    if len(args) > 3: 
      raise ValueError("Input more than 3 values, format should be [min=0:[incr=1:]]max")

    if len(args) == 0: 
      raise ValueError("No input. Format should be [min=0,[incr=1,]]max")

    if isinstance(args[0], LoopingBounds ):
      self._vec = args[0]._vec
      oldFlag = self._matlabFlag
      self._matlabFlag = args[0]._matlabFlag
      if oldFlag != self._matlabFlag:
        if self._matlabFlag():
          self.turnMatlabFlagOn()
        else:
          self.turnMatlabFlagOff()
      return
    elif isinstance(args[0], list ):
      self._vec = args[0]
    else:
      self._vec = list(args)

    if len(self._vec) == 1:
      self._vec.append( self._vec[0] + 1 )
      if not self._matlabFlag:
        self._vec[1] -= 1
      self._vec[0] = 1 if self._matlabFlag else 0
    elif len(self._vec) == 2:
      if self._matlabFlag: 
        self._vec[1] += 1
    else:
      tmp = self._vec[1]
      if self._matlabFlag:
        if tmp > 0:
          self._vec[1] = self._vec[2] + 1
        else:
          self._vec[1] = self._vec[2] - 1
      else:
        self._vec[1] = self._vec[2]
      self._vec[2] = tmp

    if len(self._vec) == 3 and self._vec[2] == 0:
      raise ValueError("Attempted to create looping bounds without increment.")

  def range(self):
    """
      Returns range generator
    """
    for x in range(*self._vec):
      yield x

  def window(self, n):
    """
    Returns a divided window (of width n) over data from the sequence
    s -> (s0,s1,...s[n-1]), (s[n],s[n+1],...,s[2n-1]), ...
    """
    l = self.list()
    for i in xrange(0, len(l), n):
      yield l[i:i+n]

  def __call__(self):
    """
      Returns range generator
    """
    return self.range()

  def list(self):
    """
      Returns list with index range
    """
    return range(*self._vec)

  def incr(self):
    """
      Return increment value
    """
    return self._vec[2] if len(self._vec) == 3 else 1


  def lowerBound(self, openBounded = False):
    """
      Returns lower bound. If openBounded flag is set to true, return the open
      bounded limit.
    """
    l = self.list()
    if not l:
      return None
    increment = 0
    if openBounded:
      increment = -1
    if self.incr() > 0:
      tmp = l[0]
    else:
      tmp = l[-1]
    return tmp + increment

  def upperBound(self, openBounded = False):
    """
      Return upper bound. If openBounded flag is set to true, return the open
      bounded limit.
    """
    l = self.list()
    if not l:
      return None
    increment = 0
    if openBounded:
      increment = 1
    if self.incr() > 0:
      tmp = l[-1]
    else:
      tmp = l[0]
    return tmp + increment

  def startBound(self, openBounded = False):
    """
      Return first entered bound. If openBounded flag is set to true, return
      the open bounded limit.
    """
    l = self.list()
    if not l:
      return None
    deslocate = 0
    if openBounded:
      deslocate = -1 if self.incr() > 0 else +1
    return l[0] + deslocate

  def endBound(self, openBounded = False):
    """
      Return last entered bound. If openBounded flag is set to true, return
      the open bounded limit.
    """
    l = self.list()
    if not l:
      return None
    deslocate = 0
    if openBounded:
      deslocate = 1 if self.incr() > 0 else -1
    return l[-1] + deslocate

  def __getitem__(self, k):
    """
      Overload []
    """
    return self._vec[k]

  def getOriginalVec(self):
    """
      Get original _vec used at the ctor
    """
    tmp = self._vec[:] # copy list
    if len(self._vec) == 3:
      if self._matlabFlag:
        if self._vec[2] > 0:
          tmp[2] = self._vec[1] - 1
        else:
          tmp[2] = self._vec[1] + 1
      else:
        tmp[2] = self._vec[1]
      tmp[1] = self._vec[2]
      if tmp[1] == 1:
        tmp.pop(1)
    else:
      if self._matlabFlag:
        tmp[1] -= 1
        if tmp[0] == 1:
          tmp[0] = tmp.pop()
      else:
        if tmp[0] == 0:
          tmp[0] = tmp.pop()
    return tmp

  def turnMatlabFlagOn(self):
    """
      Set object matlab flag to on
    """
    if not self._matlabFlag:
      self._vec = LoopingBounds( self.getOriginalVec(), matlabFlag = True )._vec
      self._matlabFlag = True

  def turnMatlabFlagOff(self):
    """
      Set object matlab flag to off
    """
    if self._matlabFlag:
      self._vec = LoopingBounds( self.getOriginalVec(), matlabFlag = False )._vec
      self._matlabFlag = False

  def __str__(self):
    """
      String representation of the object
    """
    lb = self.lowerBound()
    ub = self.upperBound()
    nfill = math.ceil( math.log10(ub) )
    if nfill < 4:
      nfill = 4
    nfill = int(nfill)
    if lb != ub:
      return 'l%s.u%s' % ( str(lb).zfill(nfill), str(ub).zfill(nfill) )
    else:
      return 'i%s' % str(lb).zfill(nfill)

  def formattedString(self, s = '', nfill = None):
    """
      String representation of the object
    """
    # TODO Make work with negative index
    lb = self.lowerBound()
    ub = self.upperBound()
    if nfill is None:
      nfill = math.ceil(math.log10(abs(ub))) if ub else 4
      if nfill < 4:
        nfill = 4
    nfill = int(nfill)
    if lb != ub:
      return '%sl%s.%su%s' % ( s, str(lb).zfill(nfill), s, str(ub).zfill(nfill)  )
    else:
      return '%s%s' % (s, str(lb).zfill(nfill) )

  def __len__(self):
    """
      Retrieve total looping range.
    """
    # FIXME This could be improved for performance, if needed
    return len(self.list())


from FastNetTool.LimitedTypeList import LimitedTypeList
LoopingBoundsCollection = LimitedTypeList("LoopingBoundsCollection", (), \
    {'_acceptedTypes' : (LoopingBounds,)})

class PythonLoopingBounds (LoopingBounds):
  """
    As LoopingBounds, but sets matlab flag to false
  """
  def __init__(self, *args):
    """
      Set this generator min:incr:max.
    """
    LoopingBounds.__init__(self, *args, matlabFlag = False)
    self.turnMatlabFlagOff()

  def toMatlabLoopingBounds(self):
    """
      Return python loopingBounds as it would be written using Matlab notation
    """
    return MatlabLoopingBounds(self.getOriginalVec())

  def toSeqLoopingBounds(self):
    return SeqLoopingBounds(self.getOriginalVec())

  def turnMatlabFlagOn(self):
    """
      Cannot call turn matlab flags off on PythonLoopingBounds.
    """
    raise RuntimeError("Can only set matlab flag to on on LoopingBounds objects.")

PythonLoopingBoundsCollection = LimitedTypeList( \
    "PythonLoopingBoundsCollection", (), \
    {'_acceptedTypes' : (PythonLoopingBounds,)})

class MatlabLoopingBounds (LoopingBounds):
  """
    As LoopingBounds, but sets matlab flag to true
  """
  def __init__(self, *args):
    """
      Set this generator min:incr:max.
    """
    LoopingBounds.__init__(self, *args, matlabFlag = True)

  def toPythonLoopingBounds(self):
    """
      Return python loopingBounds as it would be written using Matlab notation
    """
    return PythonLoopingBounds(self.getOriginalVec())

  def turnMatlabFlagOff(self):
    """
      Cannot call turn matlab flags off on MatlabLoopingBounds.
    """
    raise RuntimeError("Can only set matlab flag to off on LoopingBounds objects.")

MatlabLoopingBoundsCollection = LimitedTypeList( \
    "MatlabLoopingBoundsCollection", (), \
    {'_acceptedTypes' : (PythonLoopingBounds,)})

# Simply redirect SeqLoopingBounds to MatlabLoopingBounds
SeqLoopingBounds = MatlabLoopingBounds
SeqLoopingBoundsCollection = MatlabLoopingBoundsCollection
