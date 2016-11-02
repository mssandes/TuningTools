#include <TuningTools/system/ndarray.h>

// explici instantiate 
template class Ndarray<REAL,1>;
template class Ndarray<REAL,2>;
template class std::vector< Ndarray<REAL,2> >;
