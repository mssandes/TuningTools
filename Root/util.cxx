

#include "FastNetTool/system/util.h"

namespace util{


  float rand_float_range(float min, float max){
    srand(time(NULL)); 
    return  (max - min) * ((((float) rand()) / (float) RAND_MAX)) + min ;
  }

  REAL get_norm_of_weight( REAL *weight , size_t size){
    
    REAL sum=0;
    for(size_t i=0; i < size; ++i){
      sum += pow(weight[i],2);
    }
    return sqrt(sum);
  }

}
