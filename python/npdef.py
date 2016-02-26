import os

hasExmachina = True if int(os.environ.get('TUNINGTOOL_EXMACHINA',0)) else False
hasFastnet = True if int(os.environ.get('TUNINGTOOL_FASTNET',0)) else False

def __retrieve_np_exmachina():
  if not hasExmachina: 
    raise RuntimeError("Requested npExmachina but Exmachina not compiled.")
  import numpy as np
  from RingerCore.npConstants import npConstants
  # Define the exmachina numpy constants
  return npConstants( useFortran = True, 
                      fp_dtype   = np.float64,
                      int_dtype  = np.int64 )
  return npExmachina

def __retrieve_np_fastnet():
  if not hasExmachina: 
    raise RuntimeError("Requested npExmachina but Exmachina not compiled.")
  import numpy as np
  from RingerCore.npConstants import npConstants
  # Define the fastnet numpy constants
  return npConstants( useFortran = False,
                      fp_dtype   = np.float32,
                      int_dtype  = np.int32 )

def retrieve_npConstants(str_ = None):
  if str_ is not None:
    if str_.upper() == 'EXMACHINA':
      return __retrieve_np_exmachina()
    elif str.upper() == 'FASTNET':
      return __retrieve_np_fastnet()
    else:
      raise ValueError("'%s' is not a valid numpy constants option." % str_)
  else:
    if hasFastnet:
      return __retrieve_np_fastnet()
    if hasExmachina:
      return __retrieve_np_exmachina()
  raise RuntimeError("Couldn't define which numpy constants to use...")

