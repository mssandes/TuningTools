#include "FastNetTool/FastnetPyWrapper.h"

// STL include(s)
#include <cstdlib>
#include <cstring>

//==============================================================================
FastnetPyWrapper::FastnetPyWrapper()
  : FastnetPyWrapper( MSG::INFO )
{;}

//==============================================================================
FastnetPyWrapper::FastnetPyWrapper( const int msglevel )
  : FastnetPyWrapper( msglevel, std::numeric_limits<unsigned>::max() )
{;}

//==============================================================================
FastnetPyWrapper::FastnetPyWrapper( const int msglevel, 
    const unsigned seed )
  : IMsgService("FastnetPyWrapper"),
    MsgService( msglevel )
{
  // MsgStream Manager object
  m_trainNetwork    = nullptr;
  m_train           = nullptr;
  m_stdTrainingType = true;

  setSeed( seed );
}

//==============================================================================
FastnetPyWrapper::~FastnetPyWrapper()
{

  MSG_DEBUG("Releasing memory...");

  if(m_trainNetwork)  delete m_trainNetwork;
  for(unsigned i = 0; i < m_saveNetworks.size(); ++i) {
    delete m_saveNetworks[i];
  }
  if(!m_trnData.empty()) releaseDataSet( m_trnData );
  if(!m_valData.empty()) releaseDataSet( m_valData );
  if(!m_tstData.empty()) releaseDataSet( m_tstData );
}

//==============================================================================
unsigned FastnetPyWrapper::getSeed() const
{
  return m_seed;
}

//==============================================================================
void FastnetPyWrapper::setSeed( const unsigned seed ) 
{
  unsigned m_seed = ( seed != std::numeric_limits<unsigned int>::max() )?
      ( seed ) : ( time(nullptr) );

  MSG_INFO("Changing pseudo-random number generator seed to (" << 
      m_seed << ")." );

  std::srand( m_seed ); 
}

//==============================================================================
py::list FastnetPyWrapper::train_c()
{
 
  // Output will be: [networks, trainEvolution]
  py::list output;

  TrainGoal trainGoal = m_net.getTrainGoal();
  unsigned nClones = ( trainGoal == MULTI_STOP )?3:1;

  MSG_DEBUG("Cloning initialized network to hold best training epoch...")
  for(unsigned i = 0; i < nClones; ++i) {
    MSG_DEBUG("Cloning for index (" << i << ")" );
    m_saveNetworks.push_back( m_trainNetwork->clone() );
    switch ( i )
    {
      case TRAINNET_DEFAULT_ID:
        if ( trainGoal == MSE_STOP ){
          m_saveNetworks[i]->setName("NN_MSE_STOP");
        } else {
          m_saveNetworks[i]->setName("NN_SP_STOP");
        }
        break;
      case TRAINNET_DET_ID:
        m_saveNetworks[i]->setName("NN_DET_STOP");
        break;
      case TRAINNET_FA_ID:
        m_saveNetworks[i]->setName("NN_FA_STOP");
        break;
      default:
        throw std::runtime_error("Couldn't determine saved network type");
    }
  }
  MSG_DEBUG("Finished cloning...")

  //if(!m_tstData.empty()) m_stdTrainingType = false;
  m_stdTrainingType = false;
  // Check if goolType is mse default training  
  bool useSP = (trainGoal != MSE_STOP)? true : false;

  const unsigned show         = m_net.getShow();
  const unsigned fail_limit   = m_net.getMaxFail();
  const unsigned nEpochs      = m_net.getEpochs();
  const unsigned batchSize    = m_net.getBatchSize();
  const unsigned signalWeight = m_net.getSPSignalWeight();
  const unsigned noiseWeight  = m_net.getSPNoiseWeight();

  MSG_DEBUG("Creating training object...")
  if (m_stdTrainingType)
  {
    //m_train = new StandardTraining(m_network, m_in_trn, m_out_trn, m_in_val, m_out_val, batchSize,  getMsgLevel() );
  } else { // It is a pattern recognition network.
    if(m_tstData.empty())
    {
      m_train = new PatternRecognition(m_trainNetwork, 
          m_trnData, m_valData, m_valData, 
          trainGoal , batchSize, signalWeight, noiseWeight, 
          getMsgLevel() );
    } else {
      // If I don't have tstData , I will use the valData as tstData for training.
      m_train = new PatternRecognition( m_trainNetwork, 
          m_trnData, m_valData, m_tstData, 
          trainGoal , batchSize, signalWeight, noiseWeight, 
          getMsgLevel() );
    }  
  }

#if defined(FASTNET_DBG_LEVEL) && FASTNET_DBG_LEVEL > 0
  MSG_DEBUG("Displaying configuration options...")
  this->showInfo();
  m_trainNetwork->showInfo();
  m_train->showInfo(nEpochs);
#endif

  // Performing the training.
  unsigned num_fails_mse = 0;
  unsigned num_fails_sp  = 0;
  unsigned num_fails_det = 0;
  unsigned num_fails_fa  = 0;
  unsigned dispCounter   = 0;
  REAL mse_val, sp_val, det_val, fa_val, mse_tst, sp_tst, det_tst, fa_tst;
  mse_val = sp_val = det_val = fa_val = mse_tst = sp_tst = det_tst = fa_tst = 0.;
  ValResult is_best_mse, is_best_sp, is_best_det, is_best_fa;
  bool stop_mse, stop_sp, stop_det, stop_fa;

  // Calculating the max_fail limits for each case (MSE and SP, if the case).
  const unsigned fail_limit_mse  = (useSP) ? (fail_limit / 2) : fail_limit; 
  const unsigned fail_limit_sp   = (useSP) ? fail_limit : 0;
  const unsigned fail_limit_det  = (useSP) ? fail_limit : 0;
  const unsigned fail_limit_fa   = (useSP) ? fail_limit : 0;

  REAL best_sp_val, best_det_val, best_fa_val;
  best_sp_val = best_det_val = best_fa_val = 0.;

  bool stop = false;
  int stops_on = 0;
  unsigned epoch(0);

  MSG_DEBUG("Start looping...")

  // Training loop
  for(; epoch < nEpochs; ++epoch){
    MSG_DEBUG("=================== Start of Epoch (" << epoch 
         << ") ===================");

    // Training the network and calculating the new weights.
    const REAL mse_trn = m_train->trainNetwork();

    m_train->valNetwork(mse_val, sp_val, det_val, fa_val);

    // Testing the new network if a testing dataset was passed.
    if (!m_tstData.empty()) m_train->tstNetwork(mse_tst, sp_tst, det_tst, fa_tst);

    // Saving the best weight result.
    m_train->isBestNetwork( mse_val, sp_val, det_val, 1-fa_val, 
        is_best_mse, is_best_sp, is_best_det, is_best_fa);
   
    // Saving best neworks depends on each criteria
    if (is_best_mse == BETTER) {
      num_fails_mse = 0; 
      if (trainGoal == MSE_STOP) {
        m_saveNetworks[TRAINNET_DEFAULT_ID]->copyWeigthsFast(*m_trainNetwork);
      }
    } else if (is_best_mse == WORSE || is_best_mse == EQUAL) {
      ++num_fails_mse;
    }

    if (is_best_sp == BETTER) {
      num_fails_sp = 0; best_sp_val = sp_val;
      if( (trainGoal == SP_STOP) || (trainGoal == MULTI_STOP) ) {
        m_saveNetworks[TRAINNET_DEFAULT_ID]->copyWeigthsFast(*m_trainNetwork);
      }
    } else if (is_best_sp == WORSE || is_best_sp == EQUAL) {
      ++num_fails_sp;
    }
 
    if (is_best_det == BETTER) {
      num_fails_det = 0;  best_det_val = det_val;
      if(trainGoal == MULTI_STOP) {
        m_saveNetworks[TRAINNET_DET_ID]->copyWeigthsFast(*m_trainNetwork);
      }
    } else if (is_best_det == WORSE || is_best_det == EQUAL) {
      ++num_fails_det;
    }
 
    if (is_best_fa == BETTER) {
      num_fails_fa = 0; best_fa_val = fa_val;
      if(trainGoal == MULTI_STOP) {
        m_saveNetworks[TRAINNET_FA_ID]->copyWeigthsFast(*m_trainNetwork);
      }
    } else if (is_best_fa == WORSE || is_best_fa == EQUAL) {
      ++num_fails_fa;
    }

    // Discovering which of the criterias are telling us to stop.
    stop_mse  = num_fails_mse >= fail_limit_mse;
    stop_sp   = num_fails_sp  >= fail_limit_sp;
    stop_det  = num_fails_det >= fail_limit_det;
    stop_fa   = num_fails_fa  >= fail_limit_fa;
    
    // Save train information
    m_train->saveTrainInfo(epoch, mse_trn, mse_val, 
        sp_val, det_val, fa_val, 
        mse_tst, sp_tst, det_tst, fa_tst,
        is_best_mse, is_best_sp, is_best_det, is_best_fa, 
        num_fails_mse, num_fails_sp, num_fails_det, num_fails_fa, 
        stop_mse, stop_sp, stop_det, stop_fa);

    if(epoch > NUMBER_MIN_OF_EPOCHS) {  
      if( (trainGoal == MSE_STOP) && (stop_mse) ) stop = true;
      if( (trainGoal == SP_STOP)  && (stop_mse) && (stop_sp) ) stop = true;
      if( (trainGoal == MULTI_STOP) && (stop_mse) && (stop_sp) && (stop_det) && (stop_fa) ) stop = true;
    }

    // Number of stops flags on
    stops_on = (int)stop_mse + (int)stop_sp + (int)stop_det + (int)stop_fa;

    // Stop loop
    if ( stop ) {
      if ( show ) {
        if ( !m_tstData.empty() ) { 
          m_train->showTrainingStatus( epoch, 
              mse_trn, mse_val, sp_val, mse_tst, sp_tst, 
              stops_on );
        } else {
          m_train->showTrainingStatus( epoch, 
              mse_trn, mse_val, sp_val, 
              stops_on);
        }
        MSG_INFO("Maximum number of failures reached. " 
                        "Finishing training...");
      }
      break;
    }

    // Showing partial results at every "show" epochs (if show != 0).
    if ( show ) {
      if ( !dispCounter || true ) {
        MSG_DEBUG("Epoch " <<  epoch << ": Best values: SP (val) = " << best_sp_val 
            << " DET (val) = " << best_det_val 
            << " FA (det) = " << best_fa_val);
        if ( !m_tstData.empty() ) {
          m_train->showTrainingStatus( epoch, 
              mse_trn, mse_val, sp_val, mse_tst, sp_tst, 
              stops_on );
        } else {
          m_train->showTrainingStatus( epoch, 
              mse_trn, mse_val, sp_val, 
              stops_on );
        }
      }
      dispCounter = (dispCounter + 1) % show;
    }
  } if ( epoch == nEpochs ) {
    MSG_INFO("Maximum number of epochs (" << 
        nEpochs << ") reached. Finishing training...");
  }

#if defined(FASTNET_DBG_LEVEL) && FASTNET_DBG_LEVEL > 0
  if ( msgLevel( MSG::DEBUG ) ){
    MSG_DEBUG( "Printing last epoch weigths:" ){
      m_trainNetwork->printWeigths();
    }
  }
#endif
  // FIXME Delete this:
  m_trainNetwork->printWeigths();

  // Hold the train evolution before remove object
  flushTrainEvolution( m_train->getTrainInfo() );

  // Release memory
  MSG_DEBUG("Releasing train algorithm...");
  delete m_train;

  MSG_DEBUG("Appending neural networks to python list...");
  saveNetworksToPyList(output);

  MSG_DEBUG("Printing list of appended objects...");
  if ( msg().msgLevel( MSG::DEBUG ) ) {
    PyObject_Print(py::object(output[py::len(output)-1]).ptr(), stdout, 0);
  }
  
  MSG_DEBUG("Appending training evolution to python list...");
  output.append( trainEvolutionToPyList() );

  MSG_DEBUG("Exiting train_c...");
  return output;
}


//==============================================================================
py::list FastnetPyWrapper::valid_c( const DiscriminatorPyWrapper &net )
{
  std::vector<REAL> signal, noise;
  py::list output;
  bool useTst = !m_tstData.empty();

  if(useTst){
    signal.reserve( m_tstData[0]->getShape(0)
                  + m_valData[0]->getShape(0)
                  + m_trnData[0]->getShape(0) 
                  );
    noise.reserve( m_tstData[1]->getShape(0)
                 + m_valData[1]->getShape(0)
                 + m_trnData[1]->getShape(0) 
                 );
    MSG_DEBUG("Propagating test dataset signal:");
    sim( net, m_tstData[0], signal);  
    MSG_DEBUG("Propagating test dataset noise:");
    sim( net, m_tstData[1], noise);
    output.append( genRoc(signal, noise, 0.005) );

    MSG_DEBUG("Propagating validation dataset signal:");
    sim( net, m_valData[0], signal);  
    MSG_DEBUG("Propagating validation dataset noise:");
    sim( net, m_valData[1], noise);
    MSG_DEBUG("Propagating train dataset signal:");
    sim( net, m_trnData[0], signal);  
    MSG_DEBUG("Propagating train dataset noise:");
    sim( net, m_trnData[1], noise);
    output.append( genRoc(signal, noise, 0.005) );
  } else {

    signal.reserve( m_valData[0]->getShape(0)
                  + m_trnData[0]->getShape(0) 
                  );
    noise.reserve( m_valData[1]->getShape(0)
                 + m_trnData[1]->getShape(0) 
                 );

    MSG_DEBUG("Propagating validation dataset signal:");
    sim( net, m_valData[0], signal);  
    MSG_DEBUG("Propagating validation dataset noise:");
    sim( net, m_valData[1], noise);
    output.append( genRoc(signal, noise, 0.005) );

    MSG_DEBUG("Propagating train dataset signal:");
    sim( net, m_trnData[0], signal);  
    MSG_DEBUG("Propagating train dataset noise:");
    sim( net, m_trnData[1], noise);
    output.append( genRoc(signal, noise, 0.005) );
  }
  return output;    
}


//==============================================================================
PyObject* FastnetPyWrapper::sim_c( const DiscriminatorPyWrapper &net,
    const py::numeric::array &data )
{

  // Check if our array is on the correct type:
  auto handle = util::get_np_array( data );
  // Create our object holder:
  Ndarray<REAL,2> dataHandler( handle );
  // And extract information from it
  long numOfEvents = dataHandler.getShape(0);
  long inputSize = dataHandler.getShape(1);
  const REAL *inputEvents  = dataHandler.getPtr();

  // Create a PyObject of same length
  PyObject *pyObj = PyArray_ZEROS( 1
      , &numOfEvents
      , type_to_npy_enum<REAL>::enum_val
      , 0 );

  // Obtain an array representation of the python object (we use this
  // for retrieving the raw data from the array):
  PyArrayObject *out(nullptr);
  out->base = pyObj;

  // Retrieve its raw pointer:
  REAL* outputEvents = reinterpret_cast<REAL*>(out->data);

  /* This is commented b/c I think it is not needed */
  // Create a smart pointer handle to it (we need it to be deleted
  // as soon as it is not handled anymore)
  //py::handle<> handle( out );

  // Retrieve output size information
  const std::size_t outputSize = net.getNumNodes( 
      net.getNumLayers() - 1 );

  auto netCopy = net;

  unsigned i;
#if USE_OMP
  int chunk = 1000;
  #pragma omp parallel shared(inputEvents, outputEvents, chunk) \
    private(i) firstprivate(netCopy)
#endif
  {
#if USE_OMP
    #pragma omp for schedule(dynamic,chunk) nowait
#endif
    for ( i=0; i < numOfEvents; ++i )
    {
      std::copy_n( netCopy.propagateInput( inputEvents + (i*inputSize) ), 
          outputSize,
          outputEvents + (i*outputSize));
    }
  }

  // TODO Check if arr(handle) does not allocate more space (it only uses the 
  // handle to refer to the object. 
  // TODO What does happen if I set it to return the PyArray instead?
  //py::numeric::array arr( handle );
  //return arr.copy();
  return pyObj;

}

//==============================================================================
void FastnetPyWrapper::setData( const py::list& data, 
    std::vector< Ndarray<REAL,2>* > FastnetPyWrapper::* const setPtr )
{
  // Retrieve this member property from property pointer and set a reference to
  // it:
  std::vector< Ndarray<REAL,2>* > &set = this->*setPtr;

  // Check if set is empty, where we need to clean its previous memory:
  if ( !set.empty() ) {
    releaseDataSet( set );
  }

  // Loop over list and check for elements in which we can extract:
  for( unsigned pattern = 0; pattern < py::len( data ); pattern++ )
  {
    // Extract our array:
    py::extract<py::numeric::array> extractor( data[pattern] );
    if ( extractor.check() )
    {
      // Extract our array:
      const auto &pyObj = static_cast<py::numeric::array>(extractor());
      // Make sure that the input type is a numpy array and get it:
      auto handle = util::get_np_array( pyObj );
      // Retrieve our dataHandler:
      auto dataHandler = new Ndarray< REAL, 2 >( handle );
      // If we arrived here, it is OK, put it on our data set:
      MSG_DEBUG( "Added dataset of size (" 
                 << dataHandler->getShape(0) << "," 
                 << dataHandler->getShape(1) << ")"
               );
      set.push_back( dataHandler );
    } else {
      // We shouldn't be retrieving this, warn user:
      MSG_WARNING( "Input a list with an object on position " 
          << pattern 
          << " which is not a ctype numpy object (in fact it is of type: " 
          << py::extract<std::string>( 
              data[pattern].attr("__class__").attr("__name__"))()
          << ")." );
    }
  }
}

//==============================================================================
void FastnetPyWrapper::flushTrainEvolution( 
    const std::list<TrainData*> &trnEvolution )
{

  m_trnEvolution.clear();  

  for( const auto& cTrnData : trnEvolution ) 
  {

    TrainDataPyWrapper trainData;

    trainData.setEpoch       ( cTrnData->epoch         );
    trainData.setMseTrn      ( cTrnData->mse_trn       );
    trainData.setMseVal      ( cTrnData->mse_val       );
    trainData.setSPVal       ( cTrnData->sp_val        );
    trainData.setDetVal      ( cTrnData->det_val       );
    trainData.setFaVal       ( cTrnData->fa_val        );
    trainData.setMseTst      ( cTrnData->mse_tst       );
    trainData.setSPTst       ( cTrnData->sp_tst        );
    trainData.setDetTst      ( cTrnData->det_tst       );
    trainData.setFaTst       ( cTrnData->fa_tst        );
    trainData.setIsBestMse   ( cTrnData->is_best_mse   );
    trainData.setIsBestSP    ( cTrnData->is_best_sp    );
    trainData.setIsBestDet   ( cTrnData->is_best_det   );
    trainData.setIsBestFa    ( cTrnData->is_best_fa    );
    trainData.setNumFailsMse ( cTrnData->num_fails_mse );
    trainData.setNumFailsSP  ( cTrnData->num_fails_sp  );
    trainData.setNumFailsDet ( cTrnData->num_fails_det );
    trainData.setNumFailsFa  ( cTrnData->num_fails_fa  );
    trainData.setStopMse     ( cTrnData->stop_mse      );
    trainData.setStopSP      ( cTrnData->stop_sp       );
    trainData.setStopDet     ( cTrnData->stop_det      );
    trainData.setStopFa      ( cTrnData->stop_fa       );

    m_trnEvolution.push_back(trainData);
  }
}

//==============================================================================
void FastnetPyWrapper::showInfo()
{
  MSG_INFO( "FastNetTool::Options param:\n" 
       << "  show          : " << m_net.getShow()        << "\n"
       << "  trainFcn      : " << m_net.getTrainFcn()    << "\n"
       << "  learningRate  :"  << m_net.getLearningRate()<< "\n"
       << "  DecFactor     :"  << m_net.getDecFactor()   << "\n"
       << "  DeltaMax      :"  << m_net.getDeltaMax()    << "\n"
       << "  DeltaMin      :"  << m_net.getDeltaMin()    << "\n"
       << "  IncEta        :"  << m_net.getIncEta()      << "\n"
       << "  DecEta        :"  << m_net.getDecEta()      << "\n"
       << "  InitEta       :"  << m_net.getInitEta()     << "\n"
       << "  Epochs        :"  << m_net.getEpochs() )
}


//==============================================================================
bool FastnetPyWrapper::newff( 
    const py::list &nodes, 
    const py::list &trfFunc, 
    const std::string &trainFcn )
{
  MSG_DEBUG("Allocating FastnetPyWrapper master neural network space...")
  if ( !allocateNetwork(nodes, trfFunc, trainFcn) ) {
    return false;
  }
  MSG_DEBUG("Initialiazing neural network...")
  m_trainNetwork->initWeights();
  return true;
}

//==============================================================================
bool FastnetPyWrapper::loadff( const py::list &nodes, 
    const py::list &trfFunc,  
    const py::list &weights, 
    const py::list &bias, 
    const std::string &trainFcn )
{
  if( !allocateNetwork( nodes, trfFunc, trainFcn) ) {
    return false;
  }

  m_trainNetwork->loadWeights( util::to_std_vector<REAL>(weights), 
      util::to_std_vector<REAL>(bias));
  return true;
}

//==============================================================================
bool FastnetPyWrapper::allocateNetwork( 
    const py::list &nodes, 
    const py::list &trfFunc, 
    const std::string &trainFcn )
{

  // Reset all networks
  if ( m_trainNetwork ){
    delete m_trainNetwork; m_trainNetwork = nullptr;
    for(unsigned i = 0; i < m_saveNetworks.size(); ++i){
      delete m_saveNetworks[i];
    } 
    m_saveNetworks.clear();
  }
 
  std::vector<unsigned> nNodes = util::to_std_vector<unsigned>(nodes);
  m_net.setNodes(nNodes);
  m_net.setTrfFunc( util::to_std_vector<std::string>(trfFunc) );
  m_net.setTrainFcn(trainFcn);

  if ( trainFcn == TRAINRP_ID ) {
    MSG_DEBUG( "Creating RProp object..." );
    m_trainNetwork = new RProp(m_net, getMsgLevel(), "NN_TRAINRP");
  } else if( trainFcn == TRAINGD_ID ) {
    MSG_DEBUG( "Creating Backpropagation object...");
    m_trainNetwork = new Backpropagation(m_net, getMsgLevel(), "NN_TRAINGD");
  } else {
    //MSG_WARNING( "Invalid training algorithm option!" );
    return false;
  }
  return true;
}


//==============================================================================
void FastnetPyWrapper::sim( const DiscriminatorPyWrapper &net,
    const Ndarray<REAL,2> *data,
    std::vector<REAL> &outputVec)
{
  // Retrieve number of input events:
  long numOfEvents = data->getShape(0);
  MSG_DEBUG("numOfEvents: " << numOfEvents);

  // Old end position:
  size_t oldSize = outputVec.size();

  // Increase size to handle data:
  outputVec.resize( oldSize + numOfEvents );

  // Retrieve old end output position:
  std::vector<REAL>::iterator outItr = outputVec.begin() + oldSize;

  // Get the number of outputs from neural network:
  const std::size_t outputSize = net.getNumNodes( net.getNumLayers() - 1 );

  MSG_DEBUG("Creating a copy of neural network: ");
  auto netCopy = net;
  netCopy.setName( netCopy.getName() + "_MultiThread");

  npy_intp i;
  MSG_DEBUG("Initialize loop: ");
#if USE_OMP
  int chunk = 1000;
  #pragma omp parallel shared(data, outItr, chunk) \
      private(i) firstprivate(netCopy)
#endif
  {
#if USE_OMP
    #pragma omp for schedule(dynamic, chunk) nowait
#endif
    for ( i=0; i < numOfEvents; ++i )
    {
      const auto &rings = (*data)[i];
      std::copy_n( netCopy.propagateInput( rings.getPtr() ), 
          outputSize,
          outItr + (i*outputSize));
    }
  }
  MSG_DEBUG("Finished loop.");
}


//==============================================================================
py::list FastnetPyWrapper::genRoc( const std::vector<REAL> &signal, 
    const std::vector<REAL> &noise, 
    REAL resolution )
{

  std::vector<REAL> sp, det, fa, cut;
  util::genRoc( signal, noise, 1, -1, det, fa, sp, cut, resolution);

  py::list output;
  output.append( util::std_vector_to_py_list<REAL>(sp)  );
  output.append( util::std_vector_to_py_list<REAL>(det) );
  output.append( util::std_vector_to_py_list<REAL>(fa)  );
  output.append( util::std_vector_to_py_list<REAL>(cut) );
  return output;
}

//==============================================================================
py::object multiply(const py::numeric::array &m, float f)
{                                                                        
  PyObject* m_obj = PyArray_FROM_OTF(m.ptr(), NPY_FLOAT, NPY_IN_ARRAY);
  if (!m_obj)
    throw WrongTypeError();

  // to avoid memory leaks, let a Boost::Python object manage the array
  py::object temp(py::handle<>(m_obj));

  // check that m is a matrix of doubles
  int k = PyArray_NDIM(m_obj);
  if (k != 2)
    throw WrongSizeError();

  // get direct access to the array data
  const float* data = static_cast<const float*>(PyArray_DATA(m_obj));

  // make the output array, and get access to its data
  PyObject* res = PyArray_SimpleNew(2, PyArray_DIMS(m_obj), NPY_FLOAT);
  float* res_data = static_cast<float*>(PyArray_DATA(res));

  const unsigned size = PyArray_SIZE(m_obj); // number of elements in array
  for (unsigned i = 0; i < size; ++i)
    res_data[i] = f*data[i];

  return py::object(py::handle<>(res)); // go back to using Boost::Python constructs
}

//==============================================================================
py::object multiply(const py::list &list, float f)
{
  py::list output;
  for( unsigned pattern = 0; pattern < py::len( list ); pattern++ )
  {
    py::extract<py::numeric::array> extractor( list[pattern] );
    if ( extractor.check() )
    {
      // Extract our array:
      const auto &pyObj = static_cast<py::numeric::array>(extractor());
      output.append( multiply( pyObj, f ) );
      // Make sure that the input type is a numpy array and get it:
      auto handle = util::get_np_array( pyObj );
      // Retrieve our dataHandler:
      auto dataHandler = new Ndarray< REAL, 2 >( handle );
      std::cout << "Array size is (" << dataHandler->getShape(0) << ","
                << dataHandler->getShape(1) << ")" << std::endl;
      std::cout << "Input array is: [" << std::endl;
      for ( npy_int i = 0; i < dataHandler->getShape(0) && i < 6; ++i){
        std::cout << "[";
        for ( npy_int j = 0; j < dataHandler->getShape(1) && j < 6; ++j){
          std::cout << (*dataHandler)[i][j] << " ";
        } std::cout << "]" << std::endl;
      } std::cout << "]" << std::endl;
      delete dataHandler;
    }
  }
  return output;
}


namespace __expose_FastnetPyWrapper__ 
{

//==============================================================================
void __load_numpy(){
  py::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();
} 


//==============================================================================
void translate_sz(const WrongSizeError& e)                               
{                                                                        
  PyErr_SetString(PyExc_RuntimeError, e.what());                         
}                                                                        

//==============================================================================
void translate_ty(const WrongTypeError& e)
{                                                                        
  PyErr_SetString(PyExc_RuntimeError, e.what());                         
}                                                                        

//==============================================================================
void expose_exceptions()
{
  py::register_exception_translator<WrongSizeError>(&translate_sz);
  py::register_exception_translator<WrongTypeError>(&translate_ty);
}

//==============================================================================
void expose_multiply()
{
  py::object (*arraymultiply)(const py::numeric::array &, float) = &multiply;
  py::object (*listmultiply)(const py::list &, float) = &multiply;

  def("multiply", arraymultiply);
  def("multiply", listmultiply);
}

//==============================================================================
py::object* expose_DiscriminatorPyWrapper()
{
  static py::object _c = py::class_<DiscriminatorPyWrapper>( 
                                    "DiscriminatorPyWrapper", 
                                    py::no_init)
    .def("getNumLayers",            &DiscriminatorPyWrapper::getNumLayers   )
    .def("getNumNodes",             &DiscriminatorPyWrapper::getNumNodes    )
    .def("getBias",                 &DiscriminatorPyWrapper::getBias        )
    .def("getWeight",               &DiscriminatorPyWrapper::getWeight      )
    .def("getTrfFuncName",          &DiscriminatorPyWrapper::getTrfFuncName )
    .def("getName",                 &DiscriminatorPyWrapper::getName        )
  ;
  return &_c;
}

//==============================================================================
py::object* expose_TrainDataPyWrapper()
{
  static py::object _c = py::class_<TrainDataPyWrapper>("TrainDataPyWrapper", 
                                                         py::no_init)
    .add_property("epoch",              &TrainDataPyWrapper::getEpoch       )
    .add_property("mseTrn",             &TrainDataPyWrapper::getMseTrn      )
    .add_property("mseVal",             &TrainDataPyWrapper::getMseVal      )
    .add_property("spVal",              &TrainDataPyWrapper::getSPVal       )
    .add_property("detVal",             &TrainDataPyWrapper::getDetVal      )
    .add_property("faVal",              &TrainDataPyWrapper::getFaVal       )
    .add_property("mseTst",             &TrainDataPyWrapper::getMseTst      )
    .add_property("spTst",              &TrainDataPyWrapper::getSPTst       )
    .add_property("detTst",             &TrainDataPyWrapper::getDetTst      )
    .add_property("faTst",              &TrainDataPyWrapper::getFaTst       )
    .add_property("isBestMse",          &TrainDataPyWrapper::getIsBestMse   )
    .add_property("isBestSP",           &TrainDataPyWrapper::getIsBestSP    )
    .add_property("isBestDet",          &TrainDataPyWrapper::getIsBestDet   )
    .add_property("isBestFa",           &TrainDataPyWrapper::getIsBestFa    )
    .add_property("numFailsMse",        &TrainDataPyWrapper::getNumFailsMse )
    .add_property("numFailsSP",         &TrainDataPyWrapper::getNumFailsSP  )
    .add_property("numFailsDet",        &TrainDataPyWrapper::getNumFailsDet )
    .add_property("numFailsFa",         &TrainDataPyWrapper::getNumFailsFa  )
    .add_property("stopMse",            &TrainDataPyWrapper::getStopMse     )
    .add_property("stopSP",             &TrainDataPyWrapper::getStopSP      )
    .add_property("stopDet",            &TrainDataPyWrapper::getStopDet     )
    .add_property("stopFa",             &TrainDataPyWrapper::getStopFa      )
  ;
  return &_c;
}

//==============================================================================
py::object* expose_FastnetPyWrapper()
{
  static py::object _c = py::class_<FastnetPyWrapper>("FastnetPyWrapper", 
                                                        py::no_init )
    .def( py::init<int>() )
    .def( py::init<int, unsigned>() )
    .def("loadff"                 ,&FastnetPyWrapper::loadff            )
    .def("newff"                  ,&FastnetPyWrapper::newff             )
    .def("train_c"                ,&FastnetPyWrapper::train_c           )
    .def("sim_c"                  ,&FastnetPyWrapper::sim_c             )
    .def("valid_c"                ,&FastnetPyWrapper::valid_c           )
    .def("showInfo"               ,&FastnetPyWrapper::showInfo          )
    .def("useMSE"                 ,&FastnetPyWrapper::useMSE            )
    .def("useSP"                  ,&FastnetPyWrapper::useSP             )
    .def("useAll"                 ,&FastnetPyWrapper::useAll            )
    .def("setFrozenNode"          ,&FastnetPyWrapper::setFrozenNode     )
    .def("setTrainData"           ,&FastnetPyWrapper::setTrainData      )
    .def("setValData"             ,&FastnetPyWrapper::setValData        )
    .def("setTestData"            ,&FastnetPyWrapper::setTestData       )
    .def("setSeed"                ,&FastnetPyWrapper::setSeed           )
    .def("getSeed"                ,&FastnetPyWrapper::getSeed           )
    .add_property("showEvo"       ,&FastnetPyWrapper::getShow
                                  ,&FastnetPyWrapper::setShow           )
    .add_property("maxFail"       ,&FastnetPyWrapper::getMaxFail
                                  ,&FastnetPyWrapper::setMaxFail        )
    .add_property("batchSize"     ,&FastnetPyWrapper::getBatchSize
                                  ,&FastnetPyWrapper::setBatchSize      )
    .add_property("SPNoiseWeight" ,&FastnetPyWrapper::getSPNoiseWeight
                                  ,&FastnetPyWrapper::setSPNoiseWeight  )
    .add_property("SPSignalWeight",&FastnetPyWrapper::getSPSignalWeight
                                  ,&FastnetPyWrapper::setSPSignalWeight )
    .add_property("learningRate"  ,&FastnetPyWrapper::getLearningRate
                                  ,&FastnetPyWrapper::setLearningRate   )
    .add_property("decFactor"     ,&FastnetPyWrapper::getDecFactor
                                  ,&FastnetPyWrapper::setDecFactor      )
    .add_property("deltaMax"      ,&FastnetPyWrapper::getDeltaMax
                                  ,&FastnetPyWrapper::setDeltaMax       )
    .add_property("deltaMin"      ,&FastnetPyWrapper::getDeltaMin
                                  ,&FastnetPyWrapper::setDeltaMin       )
    .add_property("incEta"        ,&FastnetPyWrapper::getIncEta
                                  ,&FastnetPyWrapper::setIncEta         )
    .add_property("decEta"        ,&FastnetPyWrapper::getDecEta
                                  ,&FastnetPyWrapper::setDecEta         )
    .add_property("initEta"       ,&FastnetPyWrapper::getInitEta
                                  ,&FastnetPyWrapper::setInitEta        )
    .add_property("epochs"        ,&FastnetPyWrapper::getEpochs
                                  ,&FastnetPyWrapper::setEpochs         )
  ;
  return &_c;
}

}
