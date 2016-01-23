from RingerCore.npConstants import npConstants
from RingerCore.Logger import Logger
logger = Logger.getModuleLogger( __name__ )
import numpy as np

# Define the exmachina numpy constants
npExmachina = npConstants( useFortran = True, 
                           fp_dtype = np.double,
                           int_dtype = np.int64 )

# Define the fastnet numpy constants
npFastnet   = npConstants( useFortran = False,
                           fp_dtype = np.float32,
                           int_dtype = np.int32 )

npCurrent = npExmachina

logger.info( 'Using following numpy flags as default: %r', npCurrent)

