
#ifndef FASTNETTOOL_TRAINING_H
#define FASTNETTOOL_TRAINING_H

#include <list>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef NO_OMP
//#include <omp.h>
#endif


#include "FastNetTool/neuralnetwork/Backpropagation.h"
#include "FastNetTool/system/defines.h"
#include "FastNetTool/system/MsgStream.h"

using namespace msg;


enum ValResult{
  WORSE = -1, 
  EQUAL = 0, 
  BETTER = 1
};


//This struct will hold the training info to be ruterned to the user.
struct TrainData
{
  unsigned epoch;
  REAL mse_trn;
  REAL mse_val;
  REAL sp_val;
  REAL mse_tst;
  REAL sp_tst;
  ValResult is_best_mse;
  ValResult is_best_sp;
  unsigned num_fails_mse;
  unsigned num_fails_sp;
  bool stop_mse;
  bool stop_sp;
};

class DataManager
{
  private:
    vector<unsigned>::const_iterator pos;
    vector<unsigned> vec;
    
  public:
    DataManager(const unsigned numEvents)
    {
      for (unsigned i=0; i<numEvents; i++) vec.push_back(i);
      random_shuffle(vec.begin(), vec.end());
      pos = vec.begin();
    }
    
    inline unsigned size() const
    {
      return vec.size();
    }
    
    inline unsigned get()
    {
      if (pos == vec.end())
      {
        random_shuffle(vec.begin(), vec.end());
        pos = vec.begin();
      }
      return *pos++;
    }
};


class Training
{
  private:

    ///Name of the aplication
    string        m_appName;
    ///Hold the output level that can be: verbose, debug, info, warning or
    //fatal. This will be administrated by the MsgStream Class manager.
    Level         m_msgLevel;
    /// MsgStream for monitoring
    MsgStream     *m_log;

   
  protected:

    std::list<TrainData> trnEvolution;
    REAL bestGoal;
    FastNet::Backpropagation *mainNet;
    FastNet::Backpropagation **netVec;
    unsigned nThreads;
    unsigned batchSize;
    int chunkSize;
  
    void updateGradients()
    {
      for (unsigned i=1; i<nThreads; i++) mainNet->addToGradient(*netVec[i]);
    }
  
    virtual void updateWeights()
    {
      mainNet->updateWeights(batchSize);
      for (unsigned i=1; i<nThreads; i++) (*netVec[i]) = (*mainNet);
    };
  
  
#ifdef NO_OMP
  int omp_get_num_threads() {return 1;}
  int omp_get_thread_num() {return 0;}
#endif
  
  public:
  
    Training(FastNet::Backpropagation *n, const unsigned bSize, Level msglevel):m_msgLevel(msglevel)
    {
      m_appName = "Training";
      m_log = new MsgStream(m_appName, m_msgLevel);
      bestGoal = 10000000000.;
      batchSize = bSize;
      
      int nt;
      #pragma omp parallel shared(nt)
      {
        #pragma omp master
        nt = omp_get_num_threads();
      }
  
      nThreads = static_cast<unsigned>(nt);
      chunkSize = static_cast<int>(std::ceil(static_cast<float>(batchSize) / static_cast<float>(nThreads)));
      
      netVec = new FastNet::Backpropagation* [nThreads];
      mainNet = netVec[0] = n;
      for (unsigned i=1; i<nThreads; i++) netVec[i] = new FastNet::Backpropagation(*n);
    };
  
  
    virtual ~Training()
    {
      for (unsigned i=1; i<nThreads; i++) delete netVec[i];
      delete netVec;
      delete m_log;
    };
  
  
   /// Writes the training information of a network in a linked list.
   /**
    This method writes in a linked list in memory the information generated
    by the network during training, for improved speed. To actually stores this
    values for posterior use in matlab, you must call, at the end of the training process,
    the flushErrors method. 
    @param[in] epoch The epoch number.
    @param[in] trnError The training error obtained in that epoch.
    @param[in] valError The validation error obtained in that epoch.
   */
    virtual void saveTrainInfo(const unsigned epoch, const REAL mse_trn, const REAL mse_val, 
                                const REAL sp_val, const REAL mse_tst, const REAL sp_tst, 
                                const ValResult is_best_mse, const ValResult is_best_sp, 
                                const unsigned num_fails_mse, const unsigned num_fails_sp, 
                                const bool stop_mse, const bool stop_sp)
    {
      TrainData trainData;    
      trainData.epoch = epoch;
      trainData.mse_trn = mse_trn;
      trainData.mse_val = mse_val;
      trainData.sp_val = sp_val;
      trainData.mse_tst = mse_tst;
      trainData.sp_tst = sp_tst;
      trainData.is_best_mse = is_best_mse;
      trainData.is_best_sp = is_best_sp;
      trainData.num_fails_mse = num_fails_mse;
      trainData.num_fails_sp = num_fails_sp;
      trainData.stop_mse = stop_mse;
      trainData.stop_sp = stop_sp;
      trnEvolution.push_back(trainData);
    };
     
    virtual void showInfo(const unsigned nEpochs) const = 0;
    
    virtual void isBestNetwork(const REAL currMSEError, const REAL currSPError, ValResult &isBestMSE, ValResult &isBestSP)
    {
      if (currMSEError < bestGoal)
      {
        bestGoal = currMSEError;
        isBestMSE = BETTER;
      }
      else if (currMSEError > bestGoal) isBestMSE = WORSE;
      else isBestMSE = EQUAL;
    };
    
    virtual void showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError)
    {
      MSG_INFO(m_log, "Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << " mse (val) = " << valError);
    };
  
    virtual void showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError, const REAL tstError)
    {
      MSG_INFO(m_log, "Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << " mse (val) = " << valError<< " mse (tst) = " << tstError);
    };
  
    virtual void tstNetwork(REAL &mseTst, REAL &spTst) = 0;
  
    virtual void valNetwork(REAL &mseVal, REAL &spVal) = 0;
    
    virtual REAL trainNetwork() = 0;  
};

#endif
