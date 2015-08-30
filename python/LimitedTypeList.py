import types

class LimitedTypeList (type):
  """
    LimitedTypeList metaclass create lists which only accept declared types.

    One LimitedTypeList class must specify _acceptedTypes property as a tuple,
    which will be the only types accepted by the list.
  """

  def __new__(cls, name, bases, dct):
    if not list in bases:
      bases = bases + (list,)
    import inspect
    import sys
    import re
    methodSearch=re.compile("_LimitedTypeList__(\S+)")
    for localFcnName, fcn in inspect.getmembers( sys.modules[__name__], \
        inspect.isfunction):
      m = methodSearch.match(localFcnName)
      if m:
        fcnName = m.group(1)
        if not fcnName in dct:
          dct[fcnName] = fcn
    return type.__new__(cls, name, bases, dct)

  def __init__(cls, name, bases, dct):
    ## Take care to _acceptedTypes be in the right specification
    if not '_acceptedTypes' in dct:
      raise TypeError("Cannot create a LimitedTypeList without a list of "
          "accepted types defined by _acceptedTypes.")
    if not type(dct['_acceptedTypes']) is tuple:
      raise ValueError("_acceptedTypes must be declared as a tuple.")
    if not dct['_acceptedTypes']:
      raise ValueError("_acceptedTypes cannot be empty.")
    return type.__init__(cls, name, bases, dct)

  def __call__(cls, *args, **kw):
    return type.__call__(cls, *args, **kw)


def _LimitedTypeList__setitem(self, k, var):
  """
    Default override setitem
  """
  # This is default overload for list setitem, checking if item is accepted
  if not isinstance(var, self._acceptedTypes):
    raise NotAllowedType(self, var, self._acceptedTypes)
  list.setitem(self, k, var)

def _LimitedTypeList__append(self, var):
  """
    Default append method
  """
  # This is default overload for list append, checking if item is accepted
  if not isinstance(var, self._acceptedTypes):
    raise NotAllowedType( self, var, self._acceptedTypes)
  list.append(self,var)

def _LimitedTypeList____add__(self, var):
  """
    Default __add__ method
  """
  if type(var) in (list, tuple, type(self)):
    for value in var:
      if not isinstance(value, self._acceptedTypes):
        raise NotAllowedType( self, value, self._acceptedTypes )
  else:
    if not isinstance(var, self._acceptedTypes):
      raise NotAllowedType( self, var, self._acceptedTypes)
    var = [ var ]
  # This is default overload for list iadd, checking if item is accepted
  copy = list.__add__(self, var)
  return copy 

def _LimitedTypeList____iadd__( self, var, *args ):
  """
    Default __iadd__ method
  """
  for arg in args:
    if not isinstance( arg, self._acceptedTypes ):
      raise NotAllowedType( self, arg, self._acceptedTypes )
  if type(var) in (list, tuple, type(self)):
    for value in var:
      if not isinstance( value, self._acceptedTypes ):
        raise NotAllowedType( self, value, self._acceptedTypes )
  else:
    if not isinstance(var, self._acceptedTypes):
      raise NotAllowedType( self, var, self._acceptedTypes)
    var = [ var ]
  # This is default overload for list iadd, checking if item is accepted
  list.__iadd__(self, var)
  if args:
    list.__iadd__(self, args)
  return self


def _LimitedTypeList____init__( self, *args ):
  """
    Default __init__ method
  """
  if args:
    self.__iadd__(*args)

def _LimitedTypeList____call__( self ):
  """
    Default __call__ method.
    Yield holden objects.
  """
  for obj in self:
    yield obj

class NotAllowedType(ValueError):
  """
    Raised by LimitedTypeList to sign that it was attempted to add an item to the
    list which is not an allowedType instance.
  """
  def __init__( self , obj, input_, allowedTypes ):
    ValueError.__init__(self, ("Attempted to add to %s an object (type=%s) which is not an "
      "instance from the allowedTypes: %s!") % (obj.__class__.__name__, type(input_),allowedTypes,) )

