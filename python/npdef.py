from RingerCore.npConstants import npConstants
import numpy as np

# Define the exmachina numpy constants
npExmachina = npConstants( useFortran = True, 
                           fp_dtype   = np.float64,
                           int_dtype  = np.int64 )

# Define the fastnet numpy constants
npFastnet   = npConstants( useFortran = False,
                           fp_dtype   = np.float32,
                           int_dtype  = np.int32 )

npCurrent = npExmachina

