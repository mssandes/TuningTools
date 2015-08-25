// Boost include(s):
#include <boost/python.hpp>

#include "FastNetTool/FastnetPyWrapper.h"
#include "FastNetTool/system/util.h"

/// BOOST module
BOOST_PYTHON_MODULE(libFastNetTool)
{

  __expose_FastnetPyWrapper__::__load_numpy();
  __expose_system_util__::__load_numpy();

  __expose_FastnetPyWrapper__::expose_exceptions();

  __expose_FastnetPyWrapper__::expose_multiply();

  __expose_FastnetPyWrapper__::expose_DiscriminatorPyWrapper();
  __expose_FastnetPyWrapper__::expose_TrainDataPyWrapper();
  __expose_FastnetPyWrapper__::expose_FastnetPyWrapper();
}
