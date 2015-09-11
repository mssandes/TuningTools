
import sys

import TuningTools.CrossValid
import TuningTools.Neural
import TuningTools.PreProc


# Redirect cPickle to old module interface
sys.modules['FastNetTool.CrossValid'] = CrossValid
sys.modules['FastNetTool.Neural'] = Neural
sys.modules['FastNetTool.PreProc'] = PreProc
