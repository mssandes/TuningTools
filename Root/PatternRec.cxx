#include "TuningTools/training/PatternRec.h"

#include "TuningTools/TuningToolPyWrapper.h"

// STL include(s):
#include <algorithm>


namespace TuningTool {

//==============================================================================
PatternRecognition::PatternRecognition(
    TuningTool::Backpropagation *net, std::vector< Ndarray<REAL,2>* > inTrn, 
    std::vector< Ndarray<REAL,2>* > inVal, std::vector< Ndarray<REAL,2>* > inTst,  
    const TuningReferenceContainer &refCont, const NetConfHolder &conf) 
  : IMsgService{ "PatternRecognition" }
  , Training{ net, conf.getBatchSize() }
  , m_nEpochs{ conf.getEpochs() }
  , m_nPatterns{ static_cast<unsigned>(inTrn.size()) }
  , m_outputLayerSize{ (m_nPatterns == 2) ? 1 : m_nPatterns }
  , m_hasTstData{ !inTst.empty() }
  , m_mseOnly{ refCont.mseOnly() }
{
  MSG_DEBUG("Starting a Pattern Recognition Training Object");

  m_nPatterns = inTrn.size();
  MSG_DEBUG("Number of patterns: " << m_nPatterns);
  
  // The last 2 parameters for the training case will not be used by the
  // function, so, there is no problem passing the corresponding validation
  // variables to this first function call.
  MSG_DEBUG("Allocating memory for training data.");
  allocateDataset(inTrn, true, m_inTrnVec, m_epochTrnOutputs, m_nTrnEvents);
  MSG_DEBUG("Allocating memory for validation data.");
  allocateDataset(inVal, false, m_inValVec, m_epochValOutputs, m_nValEvents);
  if (m_hasTstData) {
    MSG_DEBUG("Allocating memory for testing data.");
    allocateDataset(inTst, false, m_inTstVec, m_epochTstOutputs, m_nTstEvents);
  }
  // Creating the targets for each class (maximum sparsed oututs).
  m_targetVec = new const REAL*[m_nPatterns];  
  for (unsigned i=0; i<m_nPatterns; i++)
  {
    REAL *target = new REAL [m_outputLayerSize];
    for (unsigned j=0; j<m_outputLayerSize; j++) target[j] = -1;
    target[i] = 1;
    //Saving the target in the list.
    m_targetVec[i] = target;
  }

  // And the targets
  if ( m_outputLayerSize == 1 ) {
    const REAL sgnTarget = (*m_targetVec)[1]; const REAL bkgTarget = (*m_targetVec)[0];
    m_roc = ROC{ sgnTarget, bkgTarget, conf.getROCResolution(), 
                 conf.getSPSignalWeight(), conf.getSPBackgroundWeight() };
  }

  m_perfCol = TuningPerformanceCollection( refCont, m_epoch
                                         , m_nEpochs
                                         , conf.getMinEpochs()
                                         , conf.getMaxFail()
                                         , conf.getShow()
                                         , m_mseOnly
                                         , m_hasTstData
                                         );

  MSG_DEBUG("Input space dimension: " << m_inputLayerSize);
  MSG_DEBUG("Output space dimension: " << m_outputLayerSize);

  // Get initial performance:
  propagateAllDataset( /* update = */ false );
}

//==============================================================================
void PatternRecognition::allocateDataset(
    std::vector< Ndarray<REAL,2>* > dataSet, const bool forTrain, 
    const REAL **&inVec, REAL **&out, unsigned *&nEv)
{
  inVec = new const REAL* [m_nPatterns];

  nEv = new unsigned [m_nPatterns];
  if ( ! m_mseOnly ) {
    out = new REAL*[m_nPatterns];
  }
  
  for (unsigned i=0; i<m_nPatterns; i++)
  {
    Ndarray<REAL,2>* patData = dataSet[i];
    m_inputLayerSize = patData->getShape(1);
    inVec[i] = patData->getPtr();
    if (forTrain)
    {
      // FIXME When changing to new DM version
      m_dmTrn.push_back(new DataManager(patData->getShape(0)/*, m_batchSize*/));
      MSG_DEBUG("Number of events for pattern " << i 
          << ":" << patData->getShape(0));
    }
    // Important: nEvents will hold the number of batch events used by training data
    nEv[i] = static_cast<unsigned>( 
        ( forTrain && m_batchSize )?(m_batchSize):(patData->getShape(0)) 
      );
    if ( ! m_mseOnly ) out[i] = new REAL [nEv[i] * m_outputLayerSize];
      out = new REAL*[m_batchSize];
    MSG_DEBUG("Number of events used per epoch for pattern " << i << ":" << nEv[i]);
  }
}

//==============================================================================
void PatternRecognition::deallocateDataset( const bool forTrain, 
    const REAL **&inVec, REAL **&out, unsigned *&nEv)
{
  for (unsigned i=0; i<m_nPatterns; i++)
  {
    if (forTrain) delete m_dmTrn[i];
    else if ( ! m_mseOnly ) delete [] out[i];
  }

  delete [] inVec;
  delete [] nEv;
  if ( ! m_mseOnly ) delete [] out;
}

//==============================================================================
PatternRecognition::~PatternRecognition()
{
  // The last 2 parameters for the training case will not be used by the
  // function, so, there is no problem passing the corresponding validation
  // variables to this first function call.
  deallocateDataset(true, m_inTrnVec, m_epochTrnOutputs, m_nTrnEvents);
  deallocateDataset(false, m_inValVec, m_epochValOutputs, m_nValEvents);
  if (m_hasTstData) {
    deallocateDataset(false, m_inTstVec, m_epochTstOutputs, m_nTstEvents);
  }
  for (unsigned i=0; i<m_nPatterns; i++) delete [] m_targetVec[i];
  delete [] m_targetVec;
}


//==============================================================================
void PatternRecognition::propagateDataset( const Type::Dataset ds, 
                                           ROC::Performance& rocPerf, 
                                           REAL &mseVal,
                                           const bool update )
{
  using Type::Dataset;
  // Choose from which input dataset to retrieve information and also which
  // place should we put the outputs:
  const REAL **inVec{nullptr};
  const unsigned *nEvents{nullptr};
  REAL **nnOutputs{nullptr};
  DataManager *dm{nullptr};
  switch(ds){
    case Dataset::Train:
    {
      inVec = m_inTrnVec;
      nEvents = m_nTrnEvents;
      nnOutputs = m_epochTrnOutputs;
      break;
    }
    case Dataset::Validation:
    {
      inVec = m_inValVec;
      nEvents = m_nValEvents;
      nnOutputs = m_epochValOutputs;
      break;
    }
    case Dataset::Test:
    {
      inVec = m_inTstVec;
      nEvents = m_nTstEvents;
      nnOutputs = m_epochTstOutputs;
      break;
    }
    default:
      MSG_FATAL("Unhandled dataset");
  }

  TuningTool::Backpropagation **nv = this->m_netVec;
  TuningTool::Backpropagation *thread_nv{nullptr};
  // Total MSE error:
  REAL gbError = 0;
  // Holds the amount of events presented to the network.
  unsigned totEvents = 0; 
  unsigned inputLayerSize{this->m_inputLayerSize}, outputLayerSize{this->m_outputLayerSize};
#ifdef USE_OMP
  int chunk = m_chunkSize;
#endif

  for(unsigned patIdx=0; patIdx < m_nPatterns; patIdx++)
  {
    // wFactor will allow each pattern to have the same relevance, despite the
    // number of events it contains.
    const REAL *target = m_targetVec[patIdx];
    const REAL *input = m_inTrnVec[patIdx];
    // cOutput hold the output for the given input, whereas nnOutput hold the
    // pointer to the dataset of the PatternRec dataset
    REAL * const nnOutput = ( ! m_mseOnly ) ? nnOutputs[patIdx] : nullptr;
    const REAL *cOutput{nullptr};
    // Some other variables used by looping
    unsigned i, j, thId;
    unsigned pos{0}, cEvents{nEvents[patIdx]};
    if ( update ) dm = m_dmTrn[patIdx];

    // Update counting
    totEvents += cEvents;

#if defined(TUNINGTOOL_DBG_LEVEL) && TUNINGTOOL_DBG_LEVEL > 0
    MSG_DEBUG("Printing Manager BEFORE running for patIdx[" << patIdx << "]");
    if ( msgLevel( MSG::DEBUG ) ){
      dm->print();
    }
#endif

    MSG_DEBUG("Applying training set for pattern " 
        << patIdx << " by randomly selecting " 
        << cEvents << " events (out of " << dm->size() << ").");

#ifdef USE_OMP
    #pragma omp parallel default(none) \
        shared(chunk,nv,input,nnOutput,target,dm,update,cEvents,inputLayerSize,outputLayerSize) \
        private(i,j,thId,cOutput,thread_nv,pos) \
        reduction(+:gbError)
#endif
    { // fork
      thId = omp_get_thread_num();
      thread_nv = nv[thId];

#ifdef USE_OMP
      #pragma omp for schedule(dynamic,chunk) nowait
#endif
      for (i=0; i<cEvents; ++i)
      {
        // FIXME When changing to new DM version
        if ( update ) {
#ifdef USE_OMP
        #pragma omp critical
#endif
          pos = dm->get(/*i*/);

          gbError += thread_nv->applySupervisedInput(
              input + (pos*inputLayerSize), 
              target, 
              cOutput);

          //Calculating the weight and bias update values.
          thread_nv->calculateNewWeights(cOutput, target);
#if defined(TUNINGTOOL_DBG_LEVEL) && TUNINGTOOL_DBG_LEVEL > 0
          if ( i < 10 || i > nEvents - 10 ) {
            MSG_DEBUG( "Thread[" << thId << "] executing index[" 
                << i << "] got random index [" << pos << "] and cOutput was [" 
                << cOutput[0] << "]" );
            if ( msgLevel( MSG::DEBUG ) ) {
              thread_nv->printLayerOutputs();
              thread_nv->printWeigths();
              thread_nv->printDeltas();
            }
          } else {
            MSG_DEBUG( "Thread[" << thId << "] executing index[" 
                << i << "] got random index [" << pos << "]" );
          }
#endif
        } else {
          gbError += thread_nv->applySupervisedInput(input + (i*inputLayerSize), 
                                                     target, cOutput);
        }
        // Copy cOutput to PatternRec outputs:
        for ( j = 0; j < outputLayerSize; ++j ){
          nnOutput[ i*outputLayerSize + j ] = cOutput[j];
        }
      } // no barrier
    } // join

    // FIXME Shift the data manager (when change to new version)
    //dm->shift();
#if defined(TUNINGTOOL_DBG_LEVEL) && TUNINGTOOL_DBG_LEVEL > 0
    if ( update && msgLevel( MSG::DEBUG ) ){
      MSG_DEBUG("Printing Manager AFTER running for patIdx[" << patIdx << "]");
      dm->print();
    }
#endif
  }

  // Update weights and epoch counting:
  if ( update ) {
#if defined(TUNINGTOOL_DBG_LEVEL) && TUNINGTOOL_DBG_LEVEL > 0
    MSG_DEBUG("BEFORE UPDATES:");
    if ( msgLevel( MSG::DEBUG ) ){
      for (unsigned i=0; i<m_nThreads; i++){ 
        MSG_DEBUG("Printing m_netVec[" << i << "] layerOutputs:");
        m_netVec[i]->printLayerOutputs();
        MSG_DEBUG("Printing m_netVec[" << i << "] weigths:");
        m_netVec[i]->printWeigths();
        MSG_DEBUG("Printing m_netVec[" << i << "] deltas:");
        m_netVec[i]->printDeltas();
      }
    }
#endif
    updateGradients();
    updateWeights();
    // Increase epoch counting
    ++m_epoch;
#if defined(TUNINGTOOL_DBG_LEVEL) && TUNINGTOOL_DBG_LEVEL > 0
    MSG_DEBUG("AFTER UPDATES:");
    if ( msgLevel( MSG::DEBUG ) ){
      for (unsigned i=0; i<m_nThreads; i++){ 
        MSG_DEBUG("Printing m_netVec[" << i << "] weigths:");
        m_netVec[i]->printWeigths();
      }
    }
#endif
  }

  // Now retrieve performance: 
  // MSE for this dataset
  mseVal = gbError / static_cast<REAL>(totEvents);

  // Retrieve ROC performance
  if ( ! m_mseOnly ) {
    if ( m_outputLayerSize == 1 ) {
      MSG_WARNING("Retrieving performance for neural networks with more than one output is not implemented.");
      return;
    }
    // Retrieve the neural network outputs (using the continuous pointed memory region):
    std::vector<REAL> sgnOutputs( nnOutputs[1], ( nnOutputs[1] + nEvents[1] ) );
    std::vector<REAL> bkgOutputs( nnOutputs[0], ( nnOutputs[0] + nEvents[0] ) );
    // Retrieve the ROC:
    m_roc.execute( sgnOutputs, bkgOutputs, rocPerf);
  }
}
  
//==============================================================================
void PatternRecognition::propagateAllDataset( const bool update )
{
  if ( update ){
    this->propagateDataset( Type::Dataset::Train
                          , m_trnRocPerf
                          , m_trnMSE
                          , /*update = */ true);
  } else {
    this->propagateDataset( Type::Dataset::Train
                          , m_trnRocPerf
                          , m_trnMSE);
  }
  this->propagateDataset( Type::Dataset::Validation
                        , m_valRocPerf
                        , m_valMSE);
  if( m_hasTstData ) {
    this->propagateDataset( Type::Dataset::Test
                          , m_tstRocPerf
                          , m_tstMSE);
  }
  m_perfCol.update( m_trnMSE , m_valMSE , m_tstMSE
                  , m_trnRocPerf, m_valRocPerf, m_tstRocPerf
                  , *m_mainNet );
  m_perfCol.printEpochInfo();
}

//==============================================================================
REAL PatternRecognition::trainNetwork()
{
  m_trnMSE = m_valMSE = m_tstMSE = std::numeric_limits<REAL>::max();
  try {
    while ( m_epoch++ < m_nEpochs ) {
      propagateAllDataset( /* update = */ true );
    }
    MSG_INFO("Maximum number of epochs (" << m_nEpochs << ") reached."); 
  } catch ( const EarlyStop &) {
    MSG_INFO("Early stop tuning due to failure to performance improvement.");
  }

#if defined(TUNINGTOOL_DBG_LEVEL) && TUNINGTOOL_DBG_LEVEL > 0
  if ( msgLevel( MSG::DEBUG ) ){
    MSG_DEBUG( "Printing last epoch weigths:" );
    m_net->printWeigths();
  }
#endif
  return m_trnMSE;
}

//==============================================================================
void PatternRecognition::showInfo() const
{
  MSG_INFO("TRAINING DATA INFORMATION (Pattern Recognition Optimized Network)");
  MSG_INFO("Training by MSE only      : " << (( m_mseOnly )?"true":"false") );
}

//==============================================================================
void PatternRecognition::setMsgLevel( const MSG::Level lvl ) 
{
  // TODO Create a hook for children that automatically updates children levels
  this->MsgService::setMsgLevel( lvl );
  m_perfCol.setMsgLevel( lvl );
  m_roc.setMsgLevel( lvl );
}

//==============================================================================
void PatternRecognition::setUseColor( const bool useColor ) 
{
  this->MsgService::setUseColor( useColor );
  m_perfCol.setUseColor( useColor );
  m_roc.setUseColor( useColor );
}

} // namespace TuningTool
