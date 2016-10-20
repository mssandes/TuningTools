#ifndef TUNINGTOOLS_TRAINING_PATTERNREC_H
#define TUNINGTOOLS_TRAINING_PATTERNREC_H

// Package include(s):
#include "TuningTools/system/defines.h"
#include "TuningTools/training/TuningUtil.h"
#include "TuningTools/training/Training.h"

// STL include(s):
#include <vector>

namespace TuningTool {

class PatternRecognition : public Training
{
  protected:
    /// The tuning performances colletion
    TuningPerformanceCollection m_perfCol;
    /// Hold ROC parameters and calculate its performance
    ROC m_roc;
    /// Hold current epoch performance for each dataset
    ROC::Performance m_trnRocPerf, m_valRocPerf, m_tstRocPerf;
    /// Hold current epoch MSE for each dataset
    REAL m_trnMSE, m_valMSE, m_tstMSE;
    /// Hold current epoch
    uint64_t m_epoch{0};
    /// Hold max number of epochs
    uint64_t m_nEpochs;
    /// the data manager for training
    std::vector<DataManager*> m_dmTrn;
    /// number of patterns
    unsigned m_nPatterns;
    /// the input train dataset vector
    const REAL **m_inTrnVec{nullptr};
    /// number of train dataset events
    unsigned *m_nTrnEvents{nullptr};
    /// this epoch train dataset outputs
    REAL **m_epochTrnOutputs{nullptr};
    /// the input validation dataset vector
    const REAL **m_inValVec{nullptr};
    /// number of validation dataset events
    unsigned *m_nValEvents{nullptr};
    /// this epoch validation dataset outputs
    REAL **m_epochValOutputs{nullptr};
    /// the input test dataset vector
    const REAL **m_inTstVec{nullptr};
    /// number of test dataset events
    unsigned *m_nTstEvents{nullptr};
    /// this epoch test dataset outputs
    REAL **m_epochTstOutputs{nullptr};
    /// hold the targets for each pattern
    const REAL **m_targetVec{nullptr};
    /// input layer size
    unsigned m_inputLayerSize;
    /// Output layer size
    unsigned m_outputLayerSize;
    /// Whether this training method has test dataset
    bool m_hasTstData;
    /// Whether this training method is using only MSE as benchmark
    bool m_mseOnly;
    
    void allocateDataset( std::vector<Ndarray<REAL,2>*> dataSet
                        , const bool forTrain
                        , const REAL **&inVec, REAL **&out
                        , unsigned *&nEv);

    void deallocateDataset( const bool forTrain
                          , const REAL **&inVec
                          , REAL **&out
                          , unsigned *&nEv);

    /**
     * @brief Propagate the full dataset and retrieve its performance
     **/
    void propagateDataset( const Type::Dataset ds
                         , ROC::Performance& rocPerf
                         , REAL& mseVal 
                         , const bool update = false
                         );

    /**
     * @brief Propagate every dataset and retrieve their performance
     *
     * @param[in] When update is set to true, uses train dataset for updating
     *            main neural network performance
     **/
    void propagateAllDataset( bool update = false );

  public:

    /// Ctors
    /// @{
    PatternRecognition( TuningTool::Backpropagation *net
                      , std::vector< Ndarray<REAL,2>* > inTrn
                      , std::vector< Ndarray<REAL,2>* > inVal
                      , std::vector< Ndarray<REAL,2>* > inTst
                      , const TuningReferenceContainer &refCont
                      , const NetConfHolder &conf
                      );
    /// @}

    virtual ~PatternRecognition();

    /**
     * @brief Realize the full tuning process of the machine learning algorithm
     *        for pattern recognition
     **/
    virtual REAL trainNetwork();

    virtual void showInfo() const;

    /// Message methods
    /// @{
    void setMsgLevel( const MSG::Level lvl );
    void setUseColor( const bool useColor );
    /// @}

};

} // namespace TuningTool

#endif
