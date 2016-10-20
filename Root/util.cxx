// Local include(s):
#include "TuningTools/system/util.h"
#include "TuningTools/system/defines.h"
#include "TuningTools/system/ndarray.h"

// Boost include(s):
#include <boost/python.hpp>

// Numpy include(s):
#include <numpy/ndarrayobject.h>
#include <numpy/arrayobject.h>

// STL include(s):
#include <vector>
#include <string>
#include <cmath>

namespace __expose_system_util__ {

//==============================================================================
void __load_numpy(){
  py::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();
} 

}

namespace util
{

//==============================================================================
float rand_float_range(float min, float max){
  return  (max - min) * ((((float) std::rand()) / (float) RAND_MAX)) + min ;
}

//==============================================================================
REAL get_norm_of_weight( REAL *weight , size_t size){
  
  REAL sum=0;
  for(size_t i=0; i < size; ++i){
    sum += pow(weight[i],2);
  }
  return sqrt(sum);
}

} // namespace util
