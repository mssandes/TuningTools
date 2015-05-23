

#ifndef FASTNETTOOL_UTIL_H
#define FASTNETTOOL_UTIL_H
#include <boost/python.hpp>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include "math.h"

//Define system variables
#include "FastNetTool/system/defines.h"
// Python boost
#include <boost/python/stl_iterator.hpp>


namespace py = boost::python;

namespace util{




  template< typename T >
  inline std::vector< T > to_std_vector( const py::object& iterable )
  {
      return std::vector< T >( py::stl_input_iterator< T >( iterable ), py::stl_input_iterator< T >( ) );
  }

  template<class T>
  inline py::list std_vector_to_py_list(const std::vector<T>& v)
  {
      py::object get_iter = py::iterator<std::vector<T> >();
      py::object iter = get_iter(v);
      py::list l(iter);
      return l;
  }

  ///Return a float random number between min and max value
  ///This function will be used to generate the weight random numbers
  float rand_float_range(float min = -1.0, float max = 1.0);
  ///Return the norm of the weight
  REAL get_norm_of_weight( REAL *weight , size_t size);

}

#endif
