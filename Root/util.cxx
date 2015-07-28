#include "FastNetTool/system/util.h"
#include <vector>
#include <omp.h>

using namespace std;

namespace util
{

//==============================================================================
float rand_float_range(float min, float max){
  return  (max - min) * ((((float) rand()) / (float) RAND_MAX)) + min ;
}

//==============================================================================
REAL get_norm_of_weight( REAL *weight , size_t size){
  
  REAL sum=0;
  for(size_t i=0; i < size; ++i){
    sum += pow(weight[i],2);
  }
  return sqrt(sum);
}

//==============================================================================
void genRoc( const unsigned signalSize, const unsigned noiseSize, const REAL *signal, 
    const REAL *noise, REAL signalTarget, REAL noiseTarget, 
    vector<REAL> &det,  vector<REAL> &fa, 
    vector<REAL> &sp,   vector<REAL> &cut, REAL RESOLUTION, 
    REAL signalWeight,  REAL noiseWeight)
{
  const int numSignalEvents = static_cast<int>(signalSize);
  const int numNoiseEvents  = static_cast<int>(noiseSize);
  int i;
  int chunk = 10000;

  for (REAL pos = noiseTarget; pos < signalTarget; pos += RESOLUTION)
  {
    REAL sigEffic = 0.;
    REAL noiseEffic = 0.;
    unsigned se, ne;
    #pragma omp parallel shared(signal, noise, sigEffic, noiseEffic) private(i,se,ne)
    {
      se = ne = 0;
      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numSignalEvents; i++) if (signal[i] >= pos) se++;
      
      #pragma omp critical
      sigEffic += static_cast<REAL>(se);

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numNoiseEvents; i++) if (noise[i] < pos) ne++;
      
      #pragma omp critical
      noiseEffic += static_cast<REAL>(ne);
    }
    
    sigEffic /= static_cast<REAL>(numSignalEvents);
    noiseEffic /= static_cast<REAL>(numNoiseEvents);

    // Use weights for signal and noise efficiencies
    sigEffic *= signalWeight;
    noiseEffic *= noiseWeight;

    //Using normalized SP calculation.
    sp.push_back( sqrt( ((sigEffic + noiseEffic) / 2) * sqrt(sigEffic * noiseEffic) ) );
    det.push_back( sigEffic );
    fa.push_back( 1-noiseEffic );
    cut.push_back( pos );
  }
}

} // namespace util
