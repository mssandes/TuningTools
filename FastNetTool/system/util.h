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
using namespace std;
namespace py = boost::python;

namespace util
{

template< typename T >
inline std::vector< T > to_std_vector( const py::object& iterable )
{
  return std::vector< T >( py::stl_input_iterator< T >( iterable ), py::stl_input_iterator< T >( ) );
}

template< typename T >
inline void convert_to_array_and_copy( const py::object& iterable, T* &array )
{
  vector<T> aux = std::vector<T>(
      py::stl_input_iterator< T >( iterable ), 
      py::stl_input_iterator< T >( ) );
  memcpy( array, aux.data(), aux.size()*sizeof(T) );
}

template <class T>
py::list std_vector_to_py_list(std::vector<T> vector) {
  typename std::vector<T>::iterator iter;
  boost::python::list list;
  for (iter = vector.begin(); iter != vector.end(); ++iter) {
    list.append(*iter);
  }
  return list;
}

template< typename T >
void cat_std_vector( vector<T> a, vector<T> &b){
  b.insert( b.end(),a.begin(), a.end() );
}

/// Return a float random number between min and max value
/// This function will be used to generate the weight random numbers
float rand_float_range(float min = -1.0, float max = 1.0);

/// Return the norm of the weight
REAL get_norm_of_weight( REAL *weight , size_t size);

void genRoc( const unsigned signalSize, const unsigned noiseSize, const REAL *signal, 
    const REAL *noise, REAL signalTarget, REAL noiseTarget, 
    vector<REAL> &det,  vector<REAL> &fa, 
    vector<REAL> &sp, vector<REAL> &cut, 
    const REAL RESOLUTION = 0.01, REAL signalWeight = 1,
    REAL noiseWeight = 1);
 
} // namespace util

#endif
