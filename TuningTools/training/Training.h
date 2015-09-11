#ifndef TUNINGTOOLS_TRAINING_H
#define TUNINGTOOLS_TRAINING_H

#include <list>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>

#include "RingerCore/MsgStream.h"
#include "TuningTools/neuralnetwork/Backpropagation.h"
#include "TuningTools/neuralnetwork/RProp.h"
#include "TuningTools/system/defines.h"
#include "TuningTools/system/ndarray.h"

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
  REAL det_val;
  REAL fa_val;
  REAL mse_tst;
  REAL sp_tst;
  REAL det_tst;
  REAL fa_tst;
  ValResult is_best_mse;
  ValResult is_best_sp;
  ValResult is_best_det;
  ValResult is_best_fa;
  unsigned num_fails_mse;
  unsigned num_fails_sp;
  unsigned num_fails_det;
  unsigned num_fails_fa;
  bool stop_mse;
  bool stop_sp;
  bool stop_det;
  bool stop_fa;

};

// FIXME: Change this to be a better distribution generator
namespace {
  int rndidx (int i) { return std::rand()%i; }
}

/**
 * @brief Simple DataManager 
 **/
class DataManager
{
  private:
    std::vector<unsigned>::const_iterator pos;
    std::vector<unsigned> vec;
    unsigned numEvents;
    
  public:
    DataManager(const unsigned numEvents)
      : numEvents(numEvents)
    {
      vec.reserve(numEvents);
      for (unsigned i=0; i<numEvents; i++) {
        vec.push_back(i);
      }
      random_shuffle(vec.begin(), vec.end(), rndidx );
      pos = vec.begin();
    }
    
    inline unsigned size() const
    {
      return numEvents;
    }

    inline void print() const
    {
      std::cout << "DataManager is : [";
      for ( unsigned cPos = 0; cPos < 10; ++cPos ) {
        std::cout << vec[cPos] << ",";
      } std::cout << "]" << std::endl;
    }
    
    inline unsigned get()
    {
      if (pos == vec.end())
      {
        random_shuffle(vec.begin(), vec.end(), rndidx);
        pos = vec.begin();
      }
      return *pos++;
    }
};

class Training : public MsgService
{
  protected:

    TuningTool::Backpropagation *mainNet;
    TuningTool::Backpropagation **netVec;
    std::list<TrainData*>    trnEvolution;
    REAL bestGoal;
    unsigned nThreads;
    unsigned batchSize;
    int chunkSize;
  
    void updateGradients()
    {
      for (unsigned i=1; i<nThreads; i++) {
        mainNet->addToGradient(*netVec[i]);
      }
    }
  
    void updateWeights()
    {
      mainNet->updateWeights(batchSize);
      for (unsigned i=1; i<nThreads; i++) {
        MSG_DEBUG("Copying netVec[" << i << "] using copyNeededTrainingInfoFast");
        netVec[i]->copyNeededTrainingInfoFast(*mainNet);
        //netVec[i]->operator=(*mainNet);
      }
    };
  
#if !USE_OMP
  int omp_get_num_threads() {return 1;}
  int omp_get_thread_num() {return 0;}
#endif
  
  public:
  
    Training(TuningTool::Backpropagation *n
        , const unsigned bSize
        , const MSG::Level level )
      : IMsgService("Training", MSG::INFO ),
        MsgService( level ),
        mainNet(nullptr),
        netVec(nullptr)
    {
      msg().width(5);
      bestGoal = 10000000000.;
      batchSize = bSize;
      
      int nt = 1;
#if USE_OMP
      #pragma omp parallel shared(nt)
      {
        #pragma omp master
        nt = omp_get_num_threads();
      }
#endif

      nThreads = static_cast<unsigned>(nt);
      chunkSize = static_cast<int>(std::ceil(static_cast<float>(batchSize) 
                                   / static_cast<float>(nThreads)));

      netVec = new TuningTool::Backpropagation* [nThreads];

      MSG_DEBUG("Cloning training neural network " << nThreads 
          << "times (one for each thread).")
      mainNet = netVec[0] = n;
      for (unsigned i=1; i<nThreads; i++)
      {
        netVec[i] = new TuningTool::Backpropagation(*n);
        netVec[i]->setName(netVec[i]->getName() + "_Thread[" + 
            std::to_string(i) + "]" );
      }
    }
  
  
    virtual ~Training()
    {
      MSG_DEBUG("Releasing training algorithm extra threads (" << nThreads - 1
          << ") neural networks ")
      for (unsigned i=1; i<nThreads; i++) {
        delete netVec[i]; netVec[i] = nullptr;
      }
      for ( auto& trainData : trnEvolution ) {
        delete trainData; trainData = nullptr;
      }
      delete netVec;
    };
  
  
    /**
     * @brief Writes the training information of a network in a linked list.
     *
     * This method writes in a linked list in memory the information generated
     * by the network during training, for improved speed. To actually stores
     * this values for posterior use in matlab, you must call, at the end of the
     * training process, the flushErrors method. 
     *
     * @param[in] epoch The epoch number.
     * @param[in] trnError The training error obtained in that epoch.
     * @param[in] valError The validation error obtained in that epoch.
     **/
    virtual void saveTrainInfo(const unsigned epoch, const REAL mse_trn, 
        const REAL mse_val, const REAL sp_val,
        const REAL det_val, const REAL fa_val,
        const REAL mse_tst, const REAL sp_tst, 
        const REAL det_tst, const REAL fa_tst, 
        const ValResult is_best_mse, const ValResult is_best_sp, 
        const ValResult is_best_det, const ValResult is_best_fa,
        const unsigned num_fails_mse, const unsigned num_fails_sp, 
        const unsigned num_fails_det, const unsigned num_fails_fa,
        const bool stop_mse, const bool stop_sp, 
        const bool stop_det, const bool stop_fa)
    {
      TrainData *trainData = new TrainData;    
      trainData->epoch           = epoch;
      trainData->mse_trn         = mse_trn;
      trainData->mse_val         = mse_val;
      trainData->sp_val          = sp_val;
      trainData->det_val         = det_val;
      trainData->fa_val          = fa_val;
      trainData->mse_tst         = mse_tst;
      trainData->sp_tst          = sp_tst;
      trainData->det_tst         = det_tst;
      trainData->fa_tst          = fa_tst;
      trainData->is_best_mse     = is_best_mse;
      trainData->is_best_sp      = is_best_sp;
      trainData->is_best_det     = is_best_det;
      trainData->is_best_fa      = is_best_fa;
      trainData->num_fails_mse   = num_fails_mse;
      trainData->num_fails_sp    = num_fails_sp;
      trainData->num_fails_det   = num_fails_det;
      trainData->num_fails_fa    = num_fails_fa;
      trainData->stop_mse        = stop_mse;
      trainData->stop_sp         = stop_sp;
      trainData->stop_det        = stop_det;
      trainData->stop_fa         = stop_fa;
      trnEvolution.push_back(trainData);
    }

    const std::list<TrainData*>& getTrainInfo() const {
      return trnEvolution;
    }


    virtual void showInfo(const unsigned nEpochs) const = 0;
    
    virtual void isBestNetwork(const REAL currMSEError, 
        const REAL /*currSPError*/, const REAL /*currDetError*/,
        const REAL /*currFaError*/,  ValResult &isBestMSE, ValResult &/*isBestSP*/,
        ValResult &/*isBestDet*/,    ValResult &/*isBestFa*/)
    {
      if (currMSEError < bestGoal)
      {
        bestGoal = currMSEError;
        isBestMSE = BETTER;
      }
      else if (currMSEError > bestGoal) isBestMSE = WORSE;
      else isBestMSE = EQUAL;
    }
   

    virtual void showTrainingStatus(const unsigned epoch, 
        const REAL mseTrn, const REAL mseVal, 
        const REAL /*spVal*/ = 0, 
        const int /*stopsOn*/ = 0)
    {
      MSG_INFO("Epoch " << epoch << ": mse (train) = " 
                << mseTrn << " mse (val) = " << mseVal);
    }
    
    virtual void showTrainingStatus(const unsigned epoch, 
        const REAL mseTrn, const REAL mseVal, const REAL /*spVal*/ = 0, 
        const REAL mseTst = 0, const REAL /*spTst*/ = 0, 
        const int /*stopsOn*/ = 0)

    {
      MSG_INFO("Epoch " << epoch 
          << ": mse (train) = " << mseTrn 
          << " mse (val) = " << mseVal 
          << " mse (tst) = " << mseTst);
    }
  
    virtual void tstNetwork(REAL &mseTst, REAL &spTst, REAL &detTst, REAL &faTst) = 0;
  
    virtual void valNetwork(REAL &mseVal, REAL &spVal, REAL &detVal, REAL &faVal) = 0;
    
    virtual REAL trainNetwork() = 0;  
};

#endif

//// FIXME: In the future, we might want to change to this version
//class DataManager
//{
//  private:
//    std::vector<unsigned> vec;
//    std::vector<unsigned>::const_iterator pos;
//#if USE_OMP
//    std::vector<unsigned> vec2;
//    std::vector<unsigned>::const_iterator pos2;
//#endif
//    unsigned numEvents;
//    unsigned batchSize;
//    unsigned shiftedPos;
//#if !USE_OMP
//    unsigned tmpShift;
//#endif
//    mutable MsgStream m_msg;
//
//    MsgStream& msg() const {
//      return m_msg;
//    }
//
//    bool msgLevel( MSG::Level level ){
//      return m_msg.msgLevel(level);
//    }
//
//    
//  public:
//    DataManager(const unsigned nEvents, const unsigned batchSize)
//      : numEvents(nEvents)
//      , batchSize(batchSize)
//      , shiftedPos(0)
//#if !USE_OMP
//      , tmpShift(0)
//#endif
//      , m_msg("DataManager", MSG::INFO) 
//    {
//      vec.reserve(numEvents);
//      for (unsigned i=0; i<numEvents; i++) {
//        vec.push_back(i);
//      }
//      random_shuffle(vec.begin(), vec.end(), rndidx );
//      pos = vec.begin();
//    }
//    
//    inline unsigned size() const
//    {
//      return numEvents;
//    }
//
//    inline void print() const
//    {
//      msg() << MSG::INFO << "DataManager is shifted (" << shiftedPos << "): [";
//      for ( auto cPos = pos; cPos < pos + 10; ++cPos ) {
//        msg() << *cPos << ",";
//      } msg() << "]" << endreq;
//      msg() << "FullDataManager : [";
//      for ( unsigned cPos = 0; cPos < vec.size(); ++cPos ) {
//        msg() << vec[cPos] << ",";
//      } msg() << "]" << std::endl;
//    }
//    
//    /**
//     * @brief Get random sorted position data at index.
//     *
//     * IMPORTANT: It is assumed that if reading in seriallized manner, that it
//     * will always get index in a increasing way.
//     *
//     **/
//    inline unsigned get(unsigned idx)
//    {
//#if !USE_OMP
//      std::vector<unsigned>::const_iterator currentPos = pos + idx - tmpShift;
//      // Check whether we've finished the current vector
//      if (currentPos == vec.end())
//      {
//        // Re-shufle
//        random_shuffle(vec.begin(), vec.end(), rndidx);
//        // Reset current position, position to start of vector
//        currentPos = pos = vec.begin();
//        // Set that next entries should be temporarly shufled back
//        // until next shift
//        tmpShift = idx;
//      }
//      return *currentPos;
//#else
//      std::vector<unsigned>::const_iterator currentPos = pos + idx;
//      int dist = 0;
//      if ( (dist = (currentPos - vec.end())) >= 0 )
//      {
//        if ( (pos2 + dist) >= vec2.end() ){
//          // FIXME If one day this is needed, implement it by re-sorting vec2
//          // and adding the tmpShift mechanism to subtract from pos2 + dist
//          MSG_FATAL("Used a batch size which is greater than 2 sizes "
//              "of one dataset, this is not suported for now."); 
//        }
//        return *(pos2 + ( dist ));
//      } else {
//        return *currentPos;
//      }
//#endif
//    }
//
//    /**
//     * @brief Inform manager that data should be shifted of nPos for next
//     *        reading
//     *
//     * This will shift the get method to return the results as if it was
//     * started at after batchSize was read, so that it can be used by the
//     * Training algorithms to avoid repetition of the training cicle.
//     **/
//    inline void shift() {
//      shiftedPos += batchSize;
//      // If we have passed the total number of elements,
//      // shift it back the vector size:
//#if !USE_OMP
//      if ( shiftedPos >= numEvents ) {
//        shiftedPos -= numEvents;
//        if ( shiftedPos == 0 ){
//          // Re-shufle, we've got exactly were we wanted to be:
//          random_shuffle(vec.begin(), vec.end(), rndidx);
//          pos = vec.begin();
//        }
//        tmpShift = 0;
//        // Add the remaining shifted positions:
//        pos += shiftedPos;
//      } else {
//        pos += batchSize;
//      }
//#else
//      if ( shiftedPos >= numEvents ) {
//        shiftedPos -= numEvents;
//        if ( shiftedPos == 0 ){
//          // Re-shufle, we've got exactly were we wanted to be:
//          random_shuffle(vec.begin(), vec.end(), rndidx);
//          pos = vec.begin();
//        } else {
//          // It was already shufled before.
//          vec = vec2;
//        }
//        // Add the remaining shifted positions:
//        pos += shiftedPos;
//      } else {
//        // Check if we are reaching a critical edge region
//        if ( shiftedPos + batchSize >= numEvents ) {
//          // So, as we are using parallelism, we need to be sure that
//          // we generate the next random positions before it is needed,
//          // so that we can retrieve positions needed by threads in the 
//          // order they come:
//          vec2 = vec;
//          random_shuffle(vec2.begin(), vec2.end(), rndidx);
//          pos2 = vec2.begin();
//        }
//        pos += batchSize;
//      }
//#endif
//    }
//};
