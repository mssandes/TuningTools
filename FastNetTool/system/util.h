#ifndef FASTNETTOOL_UTIL_H
#define FASTNETTOOL_UTIL_H

#include <boost/python.hpp>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include "math.h"

// Define system variables
#include "FastNetTool/system/defines.h"

// Python boost
#include <boost/python/stl_iterator.hpp>
namespace py = boost::python;

namespace __expose_system_util__ 
{
/// This is needed by boost::python to correctly import numpy array.
void __load_numpy();
}

namespace util
{

//==============================================================================
template< typename T >
inline 
std::vector< T > to_std_vector( const py::object& iterable )
{
  return std::vector< T >( py::stl_input_iterator< T >( iterable ), 
      py::stl_input_iterator< T >( ) );
}

//==============================================================================
template< typename T >
inline 
void convert_to_array_and_copy( const py::object& iterable, T* &array )
{
  std::vector<T> aux = std::vector<T>(
      py::stl_input_iterator< T >( iterable ), 
      py::stl_input_iterator< T >( ) );
  memcpy( array, aux.data(), aux.size()*sizeof(T) );
}

//==============================================================================
template <class T>
inline
py::list std_vector_to_py_list(std::vector<T> vec) 
{
  typename std::vector<T>::iterator iter;
  boost::python::list list;
  for (iter = vec.begin(); iter != vec.end(); ++iter) {
    list.append(*iter);
  }
  return list;
}

//==============================================================================
template< typename T >
inline
void cat_std_vector( std::vector<T> a, std::vector<T> &b)
{
  b.insert( b.end(),a.begin(), a.end() );
}

/// Return a float random number between min and max value
/// This function will be used to generate the weight random numbers
float rand_float_range(float min = -1.0, float max = 1.0);

/// Return the norm of the weight
REAL get_norm_of_weight( REAL *weight , size_t size);

/// Fill roc values from target values
void genRoc( const std::vector<REAL> &signal, 
    const std::vector<REAL> &noise, 
    REAL signalTarget, REAL noiseTarget, 
    std::vector<REAL> &det,  std::vector<REAL> &fa, 
    std::vector<REAL> &sp, std::vector<REAL> &cut, 
    const REAL RESOLUTION = 0.01, 
    REAL signalWeight = 1,
    REAL noiseWeight = 1);

/// Check whether numpy array representation is correct
py::handle<PyObject> get_np_array( const py::numeric::array &pyObj, 
                                   int ndim = 2 );
 
} // namespace util


#endif
