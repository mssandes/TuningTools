import os

hasExmachina = True if int(os.environ.get('TUNINGTOOL_EXMACHINA',0)) else False
hasFastnet = True if int(os.environ.get('TUNINGTOOL_FASTNET',0)) else False

from RingerCore.util import EnumStringification
class TuningToolCores( EnumStringification ):
  _ignoreCase = True
  FastNet = 0
  ExMachina = 1

if hasFastnet: 
  default = TuningToolCores.retrieve('FastNet')
elif hasExmachina:
  default = TuningToolCores.retrieve('ExMachina')
else:
  raise RuntimeError("Couldn't define which core was compiled...")
#default = TuningToolCores.retrieve('ExMachina')

def __retrieve_np_exmachina():
  if not hasExmachina: 
    raise RuntimeError("Requested npExmachina but ExMachina not compiled.")
  import numpy as np
  from RingerCore.npConstants import npConstants
  # Define the exmachina numpy constants
  return npConstants( useFortran = True, 
                      fp_dtype   = np.float64,
                      int_dtype  = np.int64 )

def __retrieve_np_fastnet():
  if not hasFastnet: 
    raise RuntimeError("Requested npFastnet but FastNet not compiled.")
  import numpy as np
  from RingerCore.npConstants import npConstants
  # Define the fastnet numpy constants
  return npConstants( useFortran = False,
                      fp_dtype   = np.float32,
                      int_dtype  = np.int32 )

def retrieve_npConstants(str_ = None):
  if str_ is None:
    str_ = default
  enum = TuningToolCores.retrieve( str_ )
  if enum is TuningToolCores.ExMachina:
    return __retrieve_np_exmachina(), TuningToolCores.ExMachina
  elif enum is TuningToolCores.FastNet:
    return __retrieve_np_fastnet(), TuningToolCores.FastNet
  else:
    raise ValueError("'%s' is not a valid numpy constants option." % str_)

def retrieve_core(str_ = None):
  if str_ is None:
    str_ = default
  enum = TuningToolCores.retrieve( str_ )
  if enum is TuningToolCores.ExMachina:
    import exmachina
    return exmachina, TuningToolCores.ExMachina
  elif enum is TuningToolCores.FastNet:
    from libTuningTools import TuningToolPyWrapper as RawWrapper
    class TuningToolPyWrapper( RawWrapper, object ): 
      def __init__(self, level, seed = None):
        self._doMultiStop = False
        if seed is None:
          RawWrapper.__init__(self, level)
        else:
          RawWrapper.__init__(self, level, seed)
      @property
      def multiStop(self):
        return self._doMultiStop
      @multiStop.setter
      def multiStop(self, value):
        if value: 
          self._doMultiStop = True
          self.useAll()
        else: 
          self._doMultiStop = False
          self.useSP()
    # End of TuningToolPyWrapper
    return TuningToolPyWrapper, TuningToolCores.FastNet
  else:
    raise ValueError("'%s' is not a valid core option." % str_)

