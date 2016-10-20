#ifndef TUNINGTOOLS_UTIL_H
#define TUNINGTOOLS_UTIL_H

// boost include(s):
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
namespace py = boost::python;

// STL include(s)
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>

// Define system variables
#include "TuningTools/system/defines.h"

namespace __expose_system_util__ 
{

/// This is needed by boost::python to correctly import numpy array.
void __load_numpy();

}

//==============================================================================
/**
 * @brief Overload unitary minus for std::vector
 **/
template<typename T>
std::vector<T> operator-(std::vector<T> v)
{
	for ( auto &val : v ) {
		val = -val;
	}
	return v;
}

namespace basic_paralel
{

//==============================================================================
template< class InputIt, class UnaryFunction >
UnaryFunction for_each( InputIt first, InputIt last, UnaryFunction f ){
  unsigned i(0), size(std::distance(first,last));
#ifdef USE_OMP
  #pragma omp parallel shared(input, last, size, f) private(i)
#endif
  {
#ifdef USE_OMP
    #pragma omp for schedule(auto) nowait
#endif
    for (i=0; i<size; ++i) {
      f(*first);
    }
  }
  return f;
}

}

namespace util
{

/**
 * @brief Calculate SP-index
 * @param[in] Signal detection probability
 * @param[in] Background detection probability
 * @return SP-index
 **/
inline
REAL calcSP( REAL pds, REAL pdb ){
  return std::sqrt( sqrt(pds * pdb) * ((pds + pdb) / 2) );
}

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

/// Check whether numpy array representation is correct
py::handle<PyObject> get_np_array( const py::numeric::array &pyObj, 
                                   int ndim = 2 );


/**
 * @brief Transfer ownership to a Python object.  If the transfer fails,
 *        then object will be destroyed and an exception is thrown.
 *
 * See http://stackoverflow.com/a/32291471/1162884 for more details.
 **/
template <typename T>
py::object transfer_to_python(T* t)
{
  // Transfer ownership to a smart pointer, allowing for proper cleanup
  // incase Boost.Python throws.
  std::unique_ptr<T> ptr(t);

  // Create a functor with a call policy that will have Boost.Python
  // manage the new object, then invoke it.
  py::object object = py::make_function(
    [t]() { return t; },
    py::return_value_policy<py::manage_new_object>(),
    boost::mpl::vector<T*>())();

  // As the Python object now has ownership, release ownership from
  // the smart pointer.
  ptr.release();
  return object;
}
 
/**
 * @brief Return string True or False from boolean
 **/
inline
const char* boolStr( bool boolean ) {
  return (boolean)?("True"):("False");
}

} // namespace util


#endif
