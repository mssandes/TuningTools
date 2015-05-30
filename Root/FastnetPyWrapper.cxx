

#include "FastNetTool/FastnetPyWrapper.h"

///Constructor
FastnetPyWrapper::FastnetPyWrapper(unsigned msglevel){

  m_appName         = "FastnetPyWrapper";
  setMsgLevel(msglevel);
  m_trainNetwork    = NULL;
  m_train           = NULL;
  m_stdTrainingType = true;

  ///MsgStream Manager object
  m_log = new MsgStream(m_appName, m_msgLevel);
  m_net = new INeuralNetwork();

  MSG_INFO(m_log, "python/c++ interface was created.");
}

FastnetPyWrapper::~FastnetPyWrapper(){

  MSG_INFO(m_log, "Release memory!");

  if(m_trainNetwork)  delete m_trainNetwork;
  for(unsigned i = 0; i < m_saveNetworks.size(); ++i) delete m_saveNetworks[i];
  if(!m_trnData.empty()) releaseDataSet( m_trnData );
  if(!m_valData.empty()) releaseDataSet( m_valData );
  if(!m_tstData.empty()) releaseDataSet( m_tstData );
  
  delete m_net;
  delete m_log;
 
}


bool FastnetPyWrapper::newff( py::list nodes, py::list trfFunc, string trainFcn ){

  ///Reset all networks
  if(m_trainNetwork){
    delete m_trainNetwork;
    for(unsigned i = 0; i < m_saveNetworks.size(); ++i)
      delete m_saveNetworks[i];
    m_trainNetwork = NULL;
    m_saveNetworks.clear();
  }
 
  m_net->setNodes(util::to_std_vector<unsigned>(nodes));
  m_net->setTrfFunc(util::to_std_vector<string>(trfFunc));
  m_net->setTrainFcn(trainFcn);

  if(trainFcn == TRAINRP_ID){
    m_trainNetwork = new RProp(m_net, m_msgLevel);
    MSG_DEBUG(m_log, "RProp object was created into the python interface!");
  }else if(trainFcn == TRAINGD_ID){
    m_trainNetwork = new Backpropagation(m_net, m_msgLevel);
    MSG_DEBUG(m_log, "Backpropagation object was created into the python interface!");
  }else{
    MSG_WARNING(m_log, "Invalid training algorithm option!");
    return false;
  }

  TrainGoal trainGoal = m_net->getTrainGoal();
  unsigned nClones = ( trainGoal == MULTI_STOP )? 3:1;
  for(unsigned i = 0; i < nClones; ++i) m_saveNetworks.push_back(m_trainNetwork->clone());
  return true;
}

///Main trainig loop
py::list  FastnetPyWrapper::train(){
 
  ///Output will be: [networks, trainEvolution]
  py::list output;


  //if(!m_tstData.empty()) m_stdTrainingType = false;
  m_stdTrainingType = false;
  TrainGoal trainGoal          = m_net->getTrainGoal();
  
  ///Check if goolType is mse default training  
  bool useSP = (trainGoal != MSE_STOP)? true : false;

  const unsigned show         = m_net->getShow();
  const unsigned fail_limit   = m_net->getMaxFail();
  const unsigned nEpochs      = m_net->getEpochs();
  const unsigned batchSize    = m_net->getBatchSize();
  const unsigned signalWeight = m_net->getSPSignalWeight();
  const unsigned noiseWeight  = m_net->getSPNoiseWeight();

  if (m_stdTrainingType)
  {
    //m_train = new StandardTraining(m_network, m_in_trn, m_out_trn, m_in_val, m_out_val, batchSize,  m_msgLevel );
  }
  else // It is a pattern recognition network.
  {
    if(m_tstData.empty())
      m_train = new PatternRecognition(m_trainNetwork, m_trnData, m_valData, m_valData, trainGoal , batchSize, signalWeight, noiseWeight, m_msgLevel);
    else{
      ///If I don't have tstData , I will uses the valData as tstData for training.
      m_train = new PatternRecognition(m_trainNetwork, m_trnData, m_valData, m_tstData, trainGoal , batchSize, signalWeight, noiseWeight, m_msgLevel);
    }  
  }

  if(m_msgLevel <= DEBUG){
    m_trainNetwork->showInfo();
    m_train->showInfo(nEpochs);
  }

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

  //Calculating the max_fail limits for each case (MSE and SP, if the case).
  const unsigned fail_limit_mse  = (useSP) ? (fail_limit / 2) : fail_limit; 
  const unsigned fail_limit_sp   = (useSP) ? fail_limit : 0;
  const unsigned fail_limit_det  = (useSP) ? fail_limit : 0;
  const unsigned fail_limit_fa   = (useSP) ? fail_limit : 0;

  REAL best_sp_val, best_det_val, best_fa_val;
  best_sp_val = best_det_val = best_fa_val = 0.;

  bool stop = false;
  ///For monitoring
  int stops_on = 0;
  ///Training loop
  for(unsigned epoch=0; epoch < nEpochs; epoch++){

    //Training the network and calculating the new weights.
    const REAL mse_trn = m_train->trainNetwork();

    m_train->valNetwork(mse_val, sp_val, det_val, fa_val);

    //Testing the new network if a testing dataset was passed.
    if (!m_tstData.empty()) m_train->tstNetwork(mse_tst, sp_tst, det_tst, fa_tst);

    // Saving the best weight result.
    m_train->isBestNetwork(mse_val, sp_val, det_val, 1-fa_val, is_best_mse, is_best_sp, is_best_det, is_best_fa);
   
    /// Saving best neworks depends on each criteria
    if (is_best_mse == BETTER){
      num_fails_mse = 0; 
      if(trainGoal == MSE_STOP) (*m_saveNetworks[TRAINNET_DEFAULT_ID]) = (*m_trainNetwork);
    }else if (is_best_mse == WORSE || is_best_mse == EQUAL) num_fails_mse++;

    if (is_best_sp == BETTER){
      num_fails_sp = 0; best_sp_val = sp_val;
      if( (trainGoal == SP_STOP) || (trainGoal == MULTI_STOP) ) (*m_saveNetworks[TRAINNET_DEFAULT_ID]) = (*m_trainNetwork);
    }else if (is_best_sp == WORSE || is_best_sp == EQUAL) num_fails_sp++;
 
    if (is_best_det == BETTER){
      num_fails_det = 0;  best_det_val = det_val;
      if(trainGoal == MULTI_STOP) (*m_saveNetworks[TRAINNET_DET_ID]) = (*m_trainNetwork);
    }else if (is_best_det == WORSE || is_best_det == EQUAL) num_fails_det++;
 
    if (is_best_fa == BETTER){
      num_fails_fa = 0; best_fa_val = fa_val;
      if(trainGoal == MULTI_STOP) (*m_saveNetworks[TRAINNET_FA_ID]) = (*m_trainNetwork);
    }else if (is_best_fa == WORSE || is_best_fa == EQUAL) num_fails_fa++;


    //Knowing whether the criterias are telling us to stop.
    stop_mse  = num_fails_mse >= fail_limit_mse;
    stop_sp   = num_fails_sp  >= fail_limit_sp;
    stop_det  = num_fails_det >= fail_limit_det;
    stop_fa   = num_fails_fa  >= fail_limit_fa;
    
    //Save train information
    m_train->saveTrainInfo(epoch, mse_trn, mse_val, sp_val, det_val, fa_val, mse_tst, sp_tst, det_tst, fa_tst,
                           is_best_mse, is_best_sp, is_best_det, is_best_fa, num_fails_mse, num_fails_sp, 
                           num_fails_det, num_fails_fa, stop_mse, stop_sp, stop_det, stop_fa);

    
    if( (trainGoal == MSE_STOP) && (stop_mse) ) stop = true;
    if( (trainGoal == SP_STOP)  && (stop_mse) && (stop_sp) ) stop = true;
    if( (trainGoal == MULTI_STOP) && (stop_mse) && (stop_sp) && (stop_det) && (stop_fa) ) stop = true;

    ///Number of stops flags on
    stops_on = (int)stop_sp + (int)stop_det + (int)stop_fa;

    //Showing partial results at every "show" epochs (if show != 0).
    if (show)
    {
      if (!dispCounter)
      {
        MSG_DEBUG(m_log, "best values: SP (val) = " << best_sp_val << " DET (val) = " << best_det_val << " FA (det) = " << best_fa_val);
        if (!m_tstData.empty()) m_train->showTrainingStatus(epoch, mse_trn, mse_val, sp_val, mse_tst, sp_tst, stops_on );
        m_train->showTrainingStatus(epoch, mse_trn, mse_val, sp_val, stops_on);
      }
      dispCounter = (dispCounter + 1) % show;
    }

    ///Stop loop
    if ( stop )
    {
      if (show){
        MSG_INFO(m_log, "Maximum number of failures reached. Finishing training...");
      }  
      break;
    }

  }///Loop

  ///Hold the train evolution before destroyer the object
  flushTrainEvolution( m_train->getTrainInfo() );

  ///Release memory
  delete m_train;

  output.append( saveNetworksToPyList() );
  output.append( trainEvolutionToPyList() );
  return output;
}


py::list FastnetPyWrapper::sim(  DiscriminatorPyWrapper net, py::list data )
{
  MSG_DEBUG(m_log, "Applying input propagation for simulation step..." );

  py::list output;
  DataHandler<REAL> *dataHandler = new DataHandler<REAL>( data );
  const unsigned numEvents = dataHandler->getNumRows();
  const unsigned outputSize = net.getNumNodes(net.getNumLayers()-1);
  const unsigned inputSize = dataHandler->getNumCols();
  const unsigned numBytes2Copy = outputSize * sizeof(REAL);
  REAL *inputEvents = dataHandler->getPtr();
  REAL *outputEvents = new REAL[outputSize*numEvents];

  unsigned i;
  int chunk = 1000;
  #pragma omp parallel shared(inputEvents,outputEvents,chunk) private(i) firstprivate(net)
  {
    #pragma omp for schedule(dynamic,chunk) nowait
    for (i=0; i<numEvents; i++)
    {
      memcpy( &outputEvents[i*outputSize], net.propagateInput( &inputEvents[i*inputSize]), numBytes2Copy);
    }
  }

  ///Parse between c++ and python list using boost
  for(unsigned i = 0; i < numEvents; ++i){
    if(outputSize == 1){
      output.append(outputEvents[i*outputSize]);     
    }else{
      py::list out;
      for(unsigned j = 0; j < outputSize; ++j)  out.append(outputEvents[j+outputSize*i]);
      output.append(out);
    }///Output parse
  }

  delete dataHandler;
  return output; ///Return boost python list
}


void FastnetPyWrapper::setTrainData( py::list data ){

  if(!m_trnData.empty()) releaseDataSet( m_trnData );
  for(unsigned pattern=0; pattern < py::len( data ); pattern++ ){
    DataHandler<REAL> *dataHandler = new DataHandler<REAL>( py::extract<py::list>(data[pattern]) );
    m_trnData.push_back( dataHandler );
  }
}

void FastnetPyWrapper::setValData( py::list data ){

  if(!m_valData.empty()) releaseDataSet( m_valData );
  for(unsigned pattern=0; pattern < py::len( data ); pattern++ ){
    DataHandler<REAL> *dataHandler = new DataHandler<REAL>( py::extract<py::list>(data[pattern]) );
    m_valData.push_back( dataHandler );
  }
}

void FastnetPyWrapper::setTestData( py::list data ){

  if(!m_tstData.empty()) releaseDataSet( m_tstData );
  for(unsigned pattern=0; pattern < py::len( data ); pattern++ ){
    DataHandler<REAL> *dataHandler = new DataHandler<REAL>( py::extract<py::list>(data[pattern]) );
    m_tstData.push_back( dataHandler );
  }
}

void FastnetPyWrapper::flushTrainEvolution( std::list<TrainData> trnEvolution )
{
  m_trnEvolution.clear();  
  for(std::list<TrainData>::iterator at = trnEvolution.begin(); at != trnEvolution.end(); at++) 
  {
    TrainDataPyWrapper trainData;
    trainData.setEpoch((*at).epoch);
    trainData.setMseTrn((*at).mse_trn);
    trainData.setMseVal((*at).mse_val);
    trainData.setSPVal((*at).sp_val);
    trainData.setDetVal((*at).det_val);
    trainData.setFaVal((*at).fa_val);
    trainData.setMseTst((*at).mse_tst);
    trainData.setSPTst((*at).sp_tst);
    trainData.setDetTst((*at).det_tst);
    trainData.setFaTst((*at).fa_tst);
    trainData.setIsBestMse((*at).is_best_mse);
    trainData.setIsBestSP((*at).is_best_sp);
    trainData.setIsBestDet((*at).is_best_det);
    trainData.setIsBestFa((*at).is_best_fa);
    trainData.setNumFailsMse((*at).num_fails_mse);
    trainData.setNumFailsSP((*at).num_fails_sp);
    trainData.setNumFailsDet((*at).num_fails_det);
    trainData.setNumFailsFa((*at).num_fails_fa);
    trainData.setStopMse((*at).stop_mse);
    trainData.setStopSP((*at).stop_sp);
    trainData.setStopDet((*at).stop_det);
    trainData.setStopFa((*at).stop_fa);
    m_trnEvolution.push_back(trainData);
  }
}


void FastnetPyWrapper::showInfo(){
      
  cout << "FastNetTool::python::train param:\n" 
       << "  trainFcn      : " << m_net->getTrainFcn()    << "\n"
       << "  learningRate  :"  << m_net->getLearningRate()<< "\n"
       << "  DecFactor     :"  << m_net->getDecFactor()   << "\n"
       << "  DeltaMax      :"  << m_net->getDeltaMax()    << "\n"
       << "  DeltaMin      :"  << m_net->getDeltaMin()    << "\n"
       << "  IncEta        :"  << m_net->getIncEta()      << "\n"
       << "  DecEta        :"  << m_net->getDecEta()      << "\n"
       << "  InitEta       :"  << m_net->getInitEta()     << "\n"
       << "  Epochs        :"  << m_net->getEpochs()      << endl;
}










