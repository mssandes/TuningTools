#include "TuningTools/TuningToolPyWrapper.h"

// STL include(s)
#include <cstdlib>
#include <cstring>

//==============================================================================
TuningToolPyWrapper::TuningToolPyWrapper( const int msglevel, const bool useColor,
    const unsigned seed )
  : IMsgService("TuningToolPyWrapper"),
    MsgService( msglevel, useColor )
{
  // MsgStream Manager object
  m_net             = nullptr;
  m_trainAlg        = nullptr;
  m_stdTrainingType = true;

  setSeed( seed );
}

//==============================================================================
TuningToolPyWrapper::~TuningToolPyWrapper()
{

  MSG_DEBUG("Releasing memory...");

  if(m_net)  delete m_net;
  for(unsigned i = 0; i < m_saveNetworks.size(); ++i) {
    delete m_saveNetworks[i];
  }
  if(!m_trnData.empty()) releaseDataSet( m_trnData );
  if(!m_valData.empty()) releaseDataSet( m_valData );
  if(!m_tstData.empty()) releaseDataSet( m_tstData );
}

//==============================================================================
unsigned TuningToolPyWrapper::getSeed() const
{
  return m_seed;
}

//==============================================================================
void TuningToolPyWrapper::setSeed( const unsigned seed )
{
  unsigned m_seed = ( seed != std::numeric_limits<unsigned int>::max() )?
      ( seed ) : ( std::chrono::system_clock::now().time_since_epoch().count() );

  MSG_INFO("Changing pseudo-random number generator seed to (" <<
      m_seed << ")." );

  std::srand( m_seed );

  //static std::mt19937 this->generator( seed );
}

//==============================================================================
void TuningToolPyWrapper::setReferences( const py::list& references )
{
  m_references = TuningReferenceContainer( references, this->getMsgLevel(), this->getUseColor() );
}

//==============================================================================
py::list TuningToolPyWrapper::train_c()
{

  // Number of references
  unsigned nRef = m_references.size();

  if ( !nRef ) {
    MSG_FATAL("No reference is available!")
  }

  if ( ! m_net ) {
    MSG_FATAL("Cannot train: no network was initialized!")
  }

  MSG_DEBUG("Cloning initialized network to hold best training epoch...")
  for(unsigned i = 0; i < nRef; ++i) {
    MSG_DEBUG("Cloning for index (" << i << ")" );
    m_saveNetworks.push_back( m_net->clone() );
    m_saveNetworks[i]->setName( m_references.at(i).getName() + "_NN" );
  }
  MSG_DEBUG("Finished cloning...")

  m_stdTrainingType = false;

  MSG_DEBUG("Creating training object...")
  if (m_stdTrainingType)
  {
    MSG_FATAL("Standard training is not yet implemented!");
  } else {
    m_trainAlg = new PatternRecognition( m_net
                                       , m_trnData
                                       , m_valData
                                       , m_tstData
                                       , m_references
                                       , m_netConfHolder
                                       );
    m_trainAlg->setMsgLevel( getMsgLevel() );
    m_trainAlg->setUseColor( getUseColor() );
  }

#if defined(TUNINGTOOL_DBG_LEVEL) && TUNINGTOOL_DBG_LEVEL > 0
  MSG_DEBUG("Displaying configuration options...")
  this->showInfo();
  m_net->showInfo();
  m_trainAlg->showInfo(nEpochs);
#endif

  // Performing the training.
  MSG_DEBUG("Training...");
  m_trainAlg->trainNetwork();
  MSG_DEBUG("Finished training!");

  // Hold the train evolution before remove object
  // TODO Flush train information is invalid
  //flushTrainEvolution( m_trainAlg->getTrainInfo() );

  // TODO Retrieve networks

  // Release memory
  MSG_DEBUG("Releasing tuning algorithm...");
  delete m_trainAlg; m_trainAlg = nullptr;

  // Output will be: [networks, trainEvolution]
  py::list output;

  MSG_DEBUG("Appending neural networks to python list...");
  saveNetworksToPyList(output);

  MSG_DEBUG("Printing list of appended objects...");
#if defined(TUNINGTOOL_DBG_LEVEL) && TUNINGTOOL_DBG_LEVEL > 0
  if ( msg().msgLevel( MSG::DEBUG ) ) {
    PyObject_Print(py::object(output[py::len(output)-1]).ptr(), stdout, 0);
    PyObject_Print("\n", stdout, 0);
  }
#endif

  MSG_DEBUG("Appending training evolution to python list...");
  output.append( trainEvolutionToPyList() );

  MSG_DEBUG("Exiting train_c...");
  return output;
}


//==============================================================================
py::list TuningToolPyWrapper::valid_c( const DiscriminatorPyWrapper &net )
{
  std::vector<REAL> signal, background;
  py::list output;
  bool useTst = !m_tstData.empty();

  if(useTst){
    signal.reserve( m_tstData[0]->getShape(0)
                  + m_valData[0]->getShape(0)
                  + m_trnData[0]->getShape(0)
                  );
    background.reserve( m_tstData[1]->getShape(0)
                 + m_valData[1]->getShape(0)
                 + m_trnData[1]->getShape(0)
                 );
    MSG_DEBUG("Propagating test dataset signal:");
    sim( net, m_tstData[0], signal);
    MSG_DEBUG("Propagating test dataset background:");
    sim( net, m_tstData[1], background);
    output.append( genRoc(signal, background, 0.005) );

    MSG_DEBUG("Propagating validation dataset signal:");
    sim( net, m_valData[0], signal);
    MSG_DEBUG("Propagating validation dataset background:");
    sim( net, m_valData[1], background);
    MSG_DEBUG("Propagating train dataset signal:");
    sim( net, m_trnData[0], signal);
    MSG_DEBUG("Propagating train dataset background:");
    sim( net, m_trnData[1], background);
    output.append( genRoc(signal, background, 0.005) );
  } else {

    signal.reserve( m_valData[0]->getShape(0)
                  + m_trnData[0]->getShape(0)
                  );
    background.reserve( m_valData[1]->getShape(0)
                 + m_trnData[1]->getShape(0)
                 );

    MSG_DEBUG("Propagating validation dataset signal:");
    sim( net, m_valData[0], signal);
    MSG_DEBUG("Propagating validation dataset background:");
    sim( net, m_valData[1], background);
    output.append( genRoc(signal, background, 0.005) );

    MSG_DEBUG("Propagating train dataset signal:");
    sim( net, m_trnData[0], signal);
    MSG_DEBUG("Propagating train dataset background:");
    sim( net, m_trnData[1], background);
    output.append( genRoc(signal, background, 0.005) );
  }
  return output;
}


//==============================================================================
PyObject* TuningToolPyWrapper::sim_c( const DiscriminatorPyWrapper &net,
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
#ifdef USE_OMP
  int chunk = 1000;
  #pragma omp parallel shared(inputEvents, outputEvents, chunk) \
    private(i) firstprivate(netCopy)
#endif
  {
#ifdef USE_OMP
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
void TuningToolPyWrapper::setData( const py::list& data,
    std::vector< Ndarray<REAL,2>* > TuningToolPyWrapper::* const setPtr )
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
void TuningToolPyWrapper::flushTrainEvolution(
    const std::list<TrainData*> &/*trnEvolution*/ )
{

  m_trnEvolution.clear();

  //for( const auto& cTrnData : trnEvolution )
  //{

  //  TrainDataPyWrapper trainData;

  //  trainData.setEpoch       ( cTrnData->epoch         );
  //  trainData.setMseTrn      ( cTrnData->mse_trn       );
  //  trainData.setMseVal      ( cTrnData->mse_val       );
  //  trainData.setMseTst      ( cTrnData->mse_tst       );
  //  trainData.setIsBestMse   ( cTrnData->is_best_mse   );
  //  trainData.setIsBestSP    ( cTrnData->is_best_sp    );
  //  trainData.setIsBestPd   ( cTrnData->is_best_pd   );
  //  trainData.setIsBestPf    ( cTrnData->is_best_pf    );
  //  trainData.setNumFailsMse ( cTrnData->num_fails_mse );
  //  trainData.setNumFailsSP  ( cTrnData->num_fails_sp  );
  //  trainData.setNumFailsPd ( cTrnData->num_fails_pd );
  //  trainData.setNumFailsPf  ( cTrnData->num_fails_pf  );
  //  trainData.setStopMse     ( cTrnData->stop_mse      );
  //  trainData.setStopSP      ( cTrnData->stop_sp       );
  //  trainData.setStopPd     ( cTrnData->stop_pd      );
  //  trainData.setStopPf      ( cTrnData->stop_pf       );

  //  //Expert methods to attach the operating point information into the object
  //  trainData.set_bestsp_point_sp_val ( cTrnData->bestsp_point_val.sp  );
  //  trainData.set_bestsp_point_pd_val( cTrnData->bestsp_point_val.pd );
  //  trainData.set_bestsp_point_pf_val ( cTrnData->bestsp_point_val.pf  );
  //  trainData.set_bestsp_point_sp_tst ( cTrnData->bestsp_point_tst.sp  );
  //  trainData.set_bestsp_point_pd_tst( cTrnData->bestsp_point_tst.pd );
  //  trainData.set_bestsp_point_pf_tst ( cTrnData->bestsp_point_tst.pf  );
  //  trainData.set_pd_point_sp_val    ( cTrnData->pd_point_val.sp     );
  //  trainData.set_pd_point_pd_val   ( cTrnData->pd_point_val.pd    );
  //  trainData.set_pd_point_pf_val    ( cTrnData->pd_point_val.pf     );
  //  trainData.set_pd_point_sp_tst    ( cTrnData->pd_point_tst.sp     );
  //  trainData.set_pd_point_pd_tst   ( cTrnData->pd_point_tst.pd    );
  //  trainData.set_pd_point_pf_tst    ( cTrnData->pd_point_tst.pf     );
  //  trainData.set_pf_point_sp_val     ( cTrnData->pf_point_val.sp      );
  //  trainData.set_pf_point_pd_val    ( cTrnData->pf_point_val.pd     );
  //  trainData.set_pf_point_pf_val     ( cTrnData->pf_point_val.pf      );
  //  trainData.set_pf_point_sp_tst     ( cTrnData->pf_point_tst.sp      );
  //  trainData.set_pf_point_pd_tst    ( cTrnData->pf_point_tst.pd     );
  //  trainData.set_pf_point_pf_tst     ( cTrnData->pf_point_tst.pf      );

  //  m_trnEvolution.push_back(trainData);
  //}
}

//==============================================================================
void TuningToolPyWrapper::showInfo()
{
  MSG_INFO( "TuningTools::Options param:\n"
       << "  show          : " << m_netConfHolder.getShow()        << "\n"
       << "  trainFcn      : " << m_netConfHolder.getTrainFcn()    << "\n"
       << "  learningRate  :"  << m_netConfHolder.getLearningRate()<< "\n"
       << "  DecFactor     :"  << m_netConfHolder.getDecFactor()   << "\n"
       << "  DeltaMax      :"  << m_netConfHolder.getDeltaMax()    << "\n"
       << "  DeltaMin      :"  << m_netConfHolder.getDeltaMin()    << "\n"
       << "  IncEta        :"  << m_netConfHolder.getIncEta()      << "\n"
       << "  DecEta        :"  << m_netConfHolder.getDecEta()      << "\n"
       << "  InitEta       :"  << m_netConfHolder.getInitEta()     << "\n"
       << "  Epochs        :"  << m_netConfHolder.getEpochs() )
}


//==============================================================================
bool TuningToolPyWrapper::newff(
    const py::list &nodes,
    const py::list &trfFunc,
    const std::string &trainFcn )
{
  MSG_DEBUG("Allocating TuningToolPyWrapper master neural network space...")
  if ( !allocateNetwork(nodes, trfFunc, trainFcn) ) {
    return false;
  }
  MSG_DEBUG("Initialiazing neural network...")
  m_net->initWeights();
  return true;
}

//==============================================================================
bool TuningToolPyWrapper::loadff( const py::list &nodes,
    const py::list &trfFunc,
    const py::list &weights,
    const py::list &bias,
    const std::string &trainFcn )
{
  if( !allocateNetwork( nodes, trfFunc, trainFcn) ) {
    return false;
  }

  m_net->loadWeights( util::to_std_vector<REAL>(weights),
      util::to_std_vector<REAL>(bias));
  return true;
}

//==============================================================================
bool TuningToolPyWrapper::allocateNetwork(
    const py::list &nodes,
    const py::list &trfFunc,
    const std::string &trainFcn )
{

  // Reset all networks
  if ( m_net ){
    delete m_net; m_net = nullptr;
    for(unsigned i = 0; i < m_saveNetworks.size(); ++i){
      delete m_saveNetworks[i];
    }
    m_saveNetworks.clear();
  }

  std::vector<unsigned> nNodes = util::to_std_vector<unsigned>(nodes);
  m_netConfHolder.setNodes(nNodes);
  m_netConfHolder.setTrfFunc( util::to_std_vector<std::string>(trfFunc) );
  m_netConfHolder.setTrainFcn(trainFcn);

  if ( trainFcn == TRAINRP_ID ) {
    MSG_DEBUG( "Creating RProp object..." );
    m_net = new RProp(m_netConfHolder, getMsgLevel(), "NN_TRAINRP");
  } else if( trainFcn == TRAINGD_ID ) {
    MSG_DEBUG( "Creating Backpropagation object...");
    m_net = new Backpropagation(m_netConfHolder, getMsgLevel(), "NN_TRAINGD");
  } else {
    MSG_WARNING( "Invalid training algorithm option(" << trainFcn << ")!" );
    return false;
  }
  m_net->setUseColor( getUseColor() );
  return true;
}


//==============================================================================
void TuningToolPyWrapper::sim( const DiscriminatorPyWrapper &net,
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
#ifdef USE_OMP
  int chunk = 1000;
  #pragma omp parallel shared(data, outItr, chunk) \
      private(i) firstprivate(netCopy)
#endif
  {
#ifdef USE_OMP
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
py::list TuningToolPyWrapper::genRoc( const std::vector<REAL> &/*signal*/,
    const std::vector<REAL> &/*background*/,
    REAL /*resolution*/ )
{

  std::vector<REAL> sp, pd, pf, cut;
  // TODO Use ROC method or expose ROC method to outside?
  //util::genRoc( signal, background, 1, -1, pd, pf, sp, cut, resolution);

  py::list output;
  output.append( util::std_vector_to_py_list<REAL>(sp)  );
  output.append( util::std_vector_to_py_list<REAL>(pd) );
  output.append( util::std_vector_to_py_list<REAL>(pf)  );
  output.append( util::std_vector_to_py_list<REAL>(cut) );
  return output;
}

namespace __expose_TuningToolPyWrapper__
{

//==============================================================================
void __load_numpy(){
  py::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();
}

//==============================================================================
void expose_exceptions()
{
  py::register_exception_translator<WrongDictError>(&translate_de);
}

//==============================================================================
//void expose_multiply()
//{
//  py::object (*arraymultiply)(const py::numeric::array &, float) = &multiply;
//  py::object (*listmultiply)(const py::list &, float) = &multiply;
//
//  def("multiply", arraymultiply);
//  def("multiply", listmultiply);
//}

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
    //.add_property("epoch",               &TrainDataPyWrapper::getEpoch                )
    //.add_property("mseTrn",              &TrainDataPyWrapper::getMseTrn               )
    //.add_property("mseVal",              &TrainDataPyWrapper::getMseVal               )
    //.add_property("mseTst",              &TrainDataPyWrapper::getMseTst               )
    //.add_property("bestsp_point_sp_val", &TrainDataPyWrapper::get_bestsp_point_sp_val )
    //.add_property("bestsp_point_pd_val", &TrainDataPyWrapper::get_bestsp_point_pd_val )
    //.add_property("bestsp_point_pf_val", &TrainDataPyWrapper::get_bestsp_point_pf_val )
    //.add_property("bestsp_point_sp_tst", &TrainDataPyWrapper::get_bestsp_point_sp_tst )
    //.add_property("bestsp_point_pd_tst", &TrainDataPyWrapper::get_bestsp_point_pd_tst )
    //.add_property("bestsp_point_pf_tst", &TrainDataPyWrapper::get_bestsp_point_pf_tst )
    //.add_property("pd_point_sp_val",     &TrainDataPyWrapper::get_pd_point_sp_val     )
    //.add_property("pd_point_pd_val",     &TrainDataPyWrapper::get_pd_point_pd_val     )
    //.add_property("pd_point_pf_val",     &TrainDataPyWrapper::get_pd_point_pf_val     )
    //.add_property("pd_point_sp_tst",     &TrainDataPyWrapper::get_pd_point_sp_tst     )
    //.add_property("pd_point_pd_tst",     &TrainDataPyWrapper::get_pd_point_pd_tst     )
    //.add_property("pd_point_pf_tst",     &TrainDataPyWrapper::get_pd_point_pf_tst     )
    //.add_property("pf_point_sp_val",     &TrainDataPyWrapper::get_pf_point_sp_val     )
    //.add_property("pf_point_pd_val",     &TrainDataPyWrapper::get_pf_point_pd_val     )
    //.add_property("pf_point_pf_val",     &TrainDataPyWrapper::get_pf_point_pf_val     )
    //.add_property("pf_point_sp_tst",     &TrainDataPyWrapper::get_pf_point_sp_tst     )
    //.add_property("pf_point_pd_tst",     &TrainDataPyWrapper::get_pf_point_pd_tst     )
    //.add_property("pf_point_pf_tst",     &TrainDataPyWrapper::get_pf_point_pf_tst     )
    //.add_property("isBestMse",           &TrainDataPyWrapper::getIsBestMse            )
    //.add_property("isBestSP",            &TrainDataPyWrapper::getIsBestSP             )
    //.add_property("isBestPd",            &TrainDataPyWrapper::getIsBestPd             )
    //.add_property("isBestPf",            &TrainDataPyWrapper::getIsBestPf             )
    //.add_property("numFailsMse",         &TrainDataPyWrapper::getNumFailsMse          )
    //.add_property("numFailsSP",          &TrainDataPyWrapper::getNumFailsSP           )
    //.add_property("numFailsPd",          &TrainDataPyWrapper::getNumFailsPd           )
    //.add_property("numFailsPf",          &TrainDataPyWrapper::getNumFailsPf           )
    //.add_property("stopMse",             &TrainDataPyWrapper::getStopMse              )
    //.add_property("stopSP",              &TrainDataPyWrapper::getStopSP               )
    //.add_property("stopPd",              &TrainDataPyWrapper::getStopPd               )
    //.add_property("stopPf",              &TrainDataPyWrapper::getStopPf               )
  ;
  return &_c;
}

//==============================================================================
py::object* expose_TuningToolPyWrapper()
{
  static py::object _c = py::class_<TuningToolPyWrapper>("TuningToolPyWrapper",
                                                        py::no_init )
    .def( py::init<int>() )
    .def( py::init<int, unsigned>() )
    .def("loadff"                 ,&TuningToolPyWrapper::loadff            )
    .def("newff"                  ,&TuningToolPyWrapper::newff             )
    .def("train_c"                ,&TuningToolPyWrapper::train_c           )
    .def("sim_c"                  ,&TuningToolPyWrapper::sim_c             )
    .def("valid_c"                ,&TuningToolPyWrapper::valid_c           )
    .def("showInfo"               ,&TuningToolPyWrapper::showInfo          )
    .def("setFrozenNode"          ,&TuningToolPyWrapper::setFrozenNode     )
    .def("setTrainData"           ,&TuningToolPyWrapper::setTrainData      )
    .def("setValData"             ,&TuningToolPyWrapper::setValData        )
    .def("setTestData"            ,&TuningToolPyWrapper::setTestData       )
    .def("setSeed"                ,&TuningToolPyWrapper::setSeed           )
    .def("getSeed"                ,&TuningToolPyWrapper::getSeed           )
    .add_property("showEvo"       ,&TuningToolPyWrapper::getShow
                                  ,&TuningToolPyWrapper::setShow           )
    .add_property("maxFail"       ,&TuningToolPyWrapper::getMaxFail
                                  ,&TuningToolPyWrapper::setMaxFail        )
    .add_property("batchSize"     ,&TuningToolPyWrapper::getBatchSize
                                  ,&TuningToolPyWrapper::setBatchSize      )
    .add_property("SPBackgroundWeight" ,&TuningToolPyWrapper::getSPBackgroundWeight
                                  ,&TuningToolPyWrapper::setSPBackgroundWeight  )
    .add_property("SPSignalWeight",&TuningToolPyWrapper::getSPSignalWeight
                                  ,&TuningToolPyWrapper::setSPSignalWeight )
    .add_property("learningRate"  ,&TuningToolPyWrapper::getLearningRate
                                  ,&TuningToolPyWrapper::setLearningRate   )
    .add_property("decFactor"     ,&TuningToolPyWrapper::getDecFactor
                                  ,&TuningToolPyWrapper::setDecFactor      )
    .add_property("deltaMax"      ,&TuningToolPyWrapper::getDeltaMax
                                  ,&TuningToolPyWrapper::setDeltaMax       )
    .add_property("deltaMin"      ,&TuningToolPyWrapper::getDeltaMin
                                  ,&TuningToolPyWrapper::setDeltaMin       )
    .add_property("incEta"        ,&TuningToolPyWrapper::getIncEta
                                  ,&TuningToolPyWrapper::setIncEta         )
    .add_property("decEta"        ,&TuningToolPyWrapper::getDecEta
                                  ,&TuningToolPyWrapper::setDecEta         )
    .add_property("initEta"       ,&TuningToolPyWrapper::getInitEta
                                  ,&TuningToolPyWrapper::setInitEta        )
    .add_property("epochs"        ,&TuningToolPyWrapper::getEpochs
                                  ,&TuningToolPyWrapper::setEpochs         )
  ;
  return &_c;
}

}
