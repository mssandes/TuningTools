// Boost include(s):
#include <boost/python.hpp>

#include "TuningTools/FastnetPyWrapper.h"
#include "TuningTools/system/util.h"

/// BOOST module
BOOST_PYTHON_MODULE(libTuningTools)
{

  __expose_FastnetPyWrapper__::__load_numpy();
  __expose_system_util__::__load_numpy();

  __expose_FastnetPyWrapper__::expose_exceptions();

  __expose_FastnetPyWrapper__::expose_multiply();

  __expose_FastnetPyWrapper__::expose_DiscriminatorPyWrapper();
  __expose_FastnetPyWrapper__::expose_TrainDataPyWrapper();
  __expose_FastnetPyWrapper__::expose_FastnetPyWrapper();
}
