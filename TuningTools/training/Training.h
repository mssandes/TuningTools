#ifndef TUNINGTOOLS_TRAINING_H
#define TUNINGTOOLS_TRAINING_H

#include "TuningTools/system/defines.h"

#include <list>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdint>

#include "RingerCore/MsgStream.h"
#include "TuningTools/training/TuningUtil.h"
#include "TuningTools/neuralnetwork/Backpropagation.h"
#include "TuningTools/neuralnetwork/RProp.h"
#include "TuningTools/system/ndarray.h"

namespace TuningTool {

//This struct will hold the training info to be ruterned to the user.

struct TrainData
{
  // Receive Operating Point (ROC)
  ROC::setpoint bestsp_point_val;
  ROC::setpoint bestsp_point_tst;
  ROC::setpoint pf_point_val;
  ROC::setpoint pf_point_tst;
  ROC::setpoint pd_point_val;
  ROC::setpoint pd_point_tst;
  // MSE information
  REAL mse_trn;
  REAL mse_val;
  REAL mse_tst;
  // Current stop status
  PerfEval is_best_mse;
  PerfEval is_best_sp;
  PerfEval is_best_pd;
  PerfEval is_best_pf;
  // Current epoch
  unsigned epoch;
  // Number of max fails
  unsigned num_fails_mse;
  unsigned num_fails_sp;
  unsigned num_fails_pd;
  unsigned num_fails_pf;
  // Stop indexs
  unsigned stop_mse_idx;
  unsigned stop_sp_idx;
  unsigned stop_pd_idx;
  unsigned stop_pf_idx;
  // Stop flag
  bool stop_mse;
  bool stop_sp;
  bool stop_pd;
  bool stop_pf;
};


/**
 * @brief Simple DataManager for shuffling data indexes
 **/
class DataManager
{

  private:
    std::vector<unsigned>::const_iterator m_pos;
    std::vector<unsigned> m_vec;
    unsigned m_nEvents;
    
  public:
    DataManager(const unsigned nEvents);

    /**
     * Shuffle data manager indexes
     **/
    void shuffle();
    
    /**
     * Retrieve index from DataManager
     **/
    unsigned get(); 
    
    /**
     * Print DataManager information
     **/
    void print() const;

    /// Getter methods:
    /// @{
    unsigned size() const { return m_nEvents; }
    /// @}

};

/**
 * @brief The root of all Tuning algorithms
 **/
class Training : public MsgService
{
  protected:
    /// Hold training evolution
    std::list<TrainData*>    m_trnEvolution;
    /// Other training neural networks used by the parallel threads
    TuningTool::Backpropagation **m_netVec{nullptr};
    /// Main training neural network
    TuningTool::Backpropagation *m_mainNet{nullptr};
    /// Current training epoch
    uint64_t m_epoch{0};
    /// The maximum number of epochs to train the neural network
    uint64_t m_nEpochs;
    /// Number of threads used
    unsigned m_nThreads;
    /// The number of samples to take meanwhile tuning process
    unsigned m_batchSize;
    /// Number of samples used by each thread:
    int m_chunkSize;
  
    /**
     * Update main neural network gradiants by accumulating all threads gradients
     **/
    void updateGradients();
  
    /**
     * Update main neural network weigths by accumulating all threads weights
     **/
    void updateWeights();
  
#ifndef USE_OMP
  int omp_get_num_threads() {return 1;}
  int omp_get_thread_num() {return 0;}
#endif
  
  public:
  
    /**
     * Ctor
     **/
    Training(TuningTool::Backpropagation *n
            , const unsigned bSize
            , const MSG::Level level = MSG::INFO );

    /**
     * Dtor
     **/
    virtual ~Training();
  
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
    virtual void saveTrainInfo(const unsigned epoch, 
        const REAL mse_trn, 
        const REAL mse_val,           
        const REAL mse_tst,
        const ROC::setpoint _bestsp_point_val,  
        const ROC::setpoint _pd_point_val,
        const ROC::setpoint _pf_point_val,      
        const ROC::setpoint _bestsp_point_tst,
        const ROC::setpoint _pd_point_tst,     
        const ROC::setpoint _pf_point_tst,
        const PerfEval is_best_mse,  const PerfEval is_best_sp, 
        const PerfEval is_best_pd,  const PerfEval is_best_pf,
        const unsigned num_fails_mse, const unsigned num_fails_sp, 
        const unsigned num_fails_pd, const unsigned num_fails_pf,
        const bool stop_mse,          const bool stop_sp, 
        const bool stop_pd,          const bool stop_pf) 
    {
      TrainData *trainData = new TrainData;    
      trainData->epoch               = epoch;
      trainData->mse_trn             = mse_trn;
      trainData->mse_val             = mse_val;
      trainData->mse_tst             = mse_tst;
      trainData->bestsp_point_val.sp = _bestsp_point_val.sp;
      trainData->bestsp_point_val.pd = _bestsp_point_val.pd;
      trainData->bestsp_point_val.pf = _bestsp_point_val.pf;
      trainData->bestsp_point_tst.sp = _bestsp_point_tst.sp;
      trainData->bestsp_point_tst.pd = _bestsp_point_tst.pd;
      trainData->bestsp_point_tst.pf = _bestsp_point_tst.pf;
      trainData->pd_point_val.sp     = _pd_point_val.sp;
      trainData->pd_point_val.pd     = _pd_point_val.pd; //pdection fitted
      trainData->pd_point_val.pf     = _pd_point_val.pf;
      trainData->pd_point_tst.sp     = _pd_point_tst.sp;
      trainData->pd_point_tst.pd     = _pd_point_tst.pd;
      trainData->pd_point_tst.pf     = _pd_point_tst.pf;
      trainData->pf_point_val.sp     = _pf_point_val.sp;
      trainData->pf_point_val.pd     = _pf_point_val.pd;
      trainData->pf_point_val.pf     = _pf_point_val.pf; //pflse alarm fitted
      trainData->pf_point_tst.sp     = _pf_point_tst.sp;
      trainData->pf_point_tst.pd     = _pf_point_tst.pd;
      trainData->pf_point_tst.pf     = _pf_point_tst.pf;
      trainData->is_best_mse         = is_best_mse;
      trainData->is_best_sp          = is_best_sp;
      trainData->is_best_pd          = is_best_pd;
      trainData->is_best_pf          = is_best_pf;
      trainData->num_fails_mse       = num_fails_mse;
      trainData->num_fails_sp        = num_fails_sp;
      trainData->num_fails_pd        = num_fails_pd;
      trainData->num_fails_pf        = num_fails_pf;
      trainData->stop_mse            = stop_mse;
      trainData->stop_sp             = stop_sp;
      trainData->stop_pd             = stop_pd;
      trainData->stop_pf             = stop_pf;
      m_trnEvolution.push_back(trainData);
    }

    const std::list<TrainData*>& getTrainInfo() const { return m_trnEvolution; }

    virtual void showInfo() const = 0;
 
    /**
     * @brief Method dedicated for the full tuning of the machine learning
     **/
    virtual REAL trainNetwork() = 0;  

};

//==============================================================================
inline
void Training::updateGradients(){
  for (unsigned i=1; i<m_nThreads; i++) {
    m_mainNet->addToGradient(*m_netVec[i]);
  }
}
  
//==============================================================================
inline
void Training::updateWeights() {
  m_mainNet->updateWeights(m_batchSize);
  for (unsigned i=1; i<m_nThreads; i++) {
    MSG_DEBUG("Copying m_netVec[" << i << "] using copyNeededTrainingInfoFast");
    m_netVec[i]->copyNeededTrainingInfoFast(*m_mainNet);
    //m_netVec[i]->operator=(*m_mainNet);
  }
}

//// FIXME: In the future, we might want to change to this version
//class DataManager
//{
//  private:
//    std::vector<unsigned> m_vec;
//    std::vector<unsigned>::const_iterator m_pos;
//#ifdef USE_OMP
//    std::vector<unsigned> vec2;
//    std::vector<unsigned>::const_iterator pos2;
//#endif
//    unsigned m_nEvents;
//    unsigned m_batchSize;
//    unsigned shiftedPos;
//#ifndef USE_OMP
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
//    DataManager(const unsigned nEvents, const unsigned m_batchSize)
//      : m_nEvents(nEvents)
//      , m_batchSize(m_batchSize)
//      , shiftedPos(0)
//#ifndef USE_OMP
//      , tmpShift(0)
//#endif
//      , m_msg("DataManager", MSG::INFO) 
//    {
//      m_vec.reserve(m_nEvents);
//      for (unsigned i=0; i<m_nEvents; i++) {
//        m_vec.push_back(i);
//      }
//      random_shuffle(m_vec.begin(), m_vec.end(), rndidx );
//      m_pos = m_vec.begin();
//    }
//    
//    inline unsigned size() const
//    {
//      return m_nEvents;
//    }
//
//    inline void print() const
//    {
//      msg() << MSG::INFO << "DataManager is shifted (" << shiftedPos << "): [";
//      for ( auto cPos = m_pos; cPos < m_pos + 10; ++cPos ) {
//        msg() << *cPos << ",";
//      } msg() << "]" << endreq;
//      msg() << "FullDataManager : [";
//      for ( unsigned cPos = 0; cPos < m_vec.size(); ++cPos ) {
//        msg() << m_vec[cPos] << ",";
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
//#ifndef USE_OMP
//      std::vector<unsigned>::const_iterator currentPos = m_pos + idx - tmpShift;
//      // Check whether we've finished the current vector
//      if (currentPos == m_vec.end())
//      {
//        // Re-shufle
//        random_shuffle(m_vec.begin(), m_vec.end(), rndidx);
//        // Reset current position, position to start of vector
//        currentPos = m_pos = m_vec.begin();
//        // Set that next entries should be temporarly shufled back
//        // until next shift
//        tmpShift = idx;
//      }
//      return *currentPos;
//#else
//      std::vector<unsigned>::const_iterator currentPos = m_pos + idx;
//      int dist = 0;
//      if ( (dist = (currentPos - m_vec.end())) >= 0 )
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
//     * started at after m_batchSize was read, so that it can be used by the
//     * Training algorithms to avoid repetition of the training cicle.
//     **/
//    inline void shift() {
//      shiftedPos += m_batchSize;
//      // If we have passed the total number of elements,
//      // shift it back the vector size:
//#ifndef USE_OMP
//      if ( shiftedPos >= m_nEvents ) {
//        shiftedPos -= m_nEvents;
//        if ( shiftedPos == 0 ){
//          // Re-shufle, we've got exactly were we wanted to be:
//          random_shuffle(m_vec.begin(), m_vec.end(), rndidx);
//          m_pos = m_vec.begin();
//        }
//        tmpShift = 0;
//        // Add the remaining shifted positions:
//        m_pos += shiftedPos;
//      } else {
//        m_pos += m_batchSize;
//      }
//#else
//      if ( shiftedPos >= m_nEvents ) {
//        shiftedPos -= m_nEvents;
//        if ( shiftedPos == 0 ){
//          // Re-shufle, we've got exactly were we wanted to be:
//          random_shuffle(m_vec.begin(), m_vec.end(), rndidx);
//          m_pos = m_vec.begin();
//        } else {
//          // It was already shufled before.
//          m_vec = vec2;
//        }
//        // Add the remaining shifted positions:
//        m_pos += shiftedPos;
//      } else {
//        // Check if we are reaching a critical edge region
//        if ( shiftedPos + m_batchSize >= m_nEvents ) {
//          // So, as we are using parallelism, we need to be sure that
//          // we generate the next random positions before it is needed,
//          // so that we can retrieve positions needed by threads in the 
//          // order they come:
//          vec2 = m_vec;
//          random_shuffle(vec2.begin(), vec2.end(), rndidx);
//          pos2 = vec2.begin();
//        }
//        m_pos += m_batchSize;
//      }
//#endif
//    }
//};

} // namespace TuningTool

#endif // TUNINGTOOLS_TRAINING_H
