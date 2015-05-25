

#include "FastNetTool/FastnetPyWrapper.h"

///Constructor
FastnetPyWrapper::FastnetPyWrapper(unsigned msglevel){

  m_appName  = "FastnetPyWrapper";
  setMsgLevel(msglevel);
  m_network  = NULL;
  m_train    = NULL;
  //m_in_trn = m_out_trn = m_in_val = m_out_val = m_in_tst = NULL;
  m_stdTrainingType = true;

  ///MsgStream Manager object
  m_log = new MsgStream(m_appName, m_msgLevel);
  m_net = new INeuralNetwork();

  MSG_INFO(m_log, "python/c++ interface was created.");
}

FastnetPyWrapper::~FastnetPyWrapper(){
  MSG_INFO(m_log, "Release memory!");
  if(m_network)  delete m_network;
  //if(m_in_trn)   delete m_in_trn;
  //if(m_out_trn)  delete m_out_trn;
  //if(m_in_val)   delete m_in_val;
  //if(m_out_val)  delete m_out_val;
  //if(m_in_tst)   delete m_in_tst;

  if(!m_trnData.empty()) releaseDataSet( m_trnData );
  if(!m_valData.empty()) releaseDataSet( m_valData );
  if(!m_tstData.empty()) releaseDataSet( m_tstData );
  
  delete m_net;
  delete m_log;
 
}


bool FastnetPyWrapper::newff( py::list nodes, py::list trfFunc, string trainFcn ){

  if(m_network){
    delete m_network;
    m_network = NULL;
  }
 
  m_net->setNodes(util::to_std_vector<unsigned>(nodes));
  m_net->setTrfFunc(util::to_std_vector<string>(trfFunc));
  m_net->setTrainFcn(trainFcn);

  if(trainFcn == TRAINRP_ID){
    m_network = new RProp(m_net, m_msgLevel);
    MSG_DEBUG(m_log, "RProp object was created into the python interface!");
  }else if(trainFcn == TRAINGD_ID){
    m_network = new Backpropagation(m_net, m_msgLevel);
    MSG_DEBUG(m_log, "Backpropagation object was created into the python interface!");
  }else{
    MSG_WARNING(m_log, "Invalid training algorithm option!");
    return false;
  }

  return true;
}

///Main trainig loop
bool FastnetPyWrapper::train(){
 
  //if(!m_tstData.empty()) m_stdTrainingType = false;
  m_stdTrainingType = false;

  //if(!m_in_trn || !m_in_val || !m_out_trn || !m_out_val){
  //  MSG_WARNING(m_log, "You must set the datasets [in_trn, out_trn, in_val, out_val] before start the train.");
  //  //return false;
  //}

  ///Train param 
  bool useSP                = m_net->getUseSP();
  const unsigned show       = m_net->getShow();
  const unsigned fail_limit = m_net->getMaxFail();
  const unsigned nEpochs    = m_net->getEpochs();
  const unsigned batchSize  = m_net->getBatchSize();
  const unsigned signalWeight = m_net->getSPSignalWeight();
  const unsigned noiseWeight  = m_net->getSPNoiseWeight();

  if (m_stdTrainingType)
  {
    //m_train = new StandardTraining(m_network, m_in_trn, m_out_trn, m_in_val, m_out_val, batchSize,  m_msgLevel );
  }
  else // It is a pattern recognition network.
  {
    if(m_tstData.empty())
      m_train = new PatternRecognition(m_network, m_trnData, m_valData, m_valData, useSP, batchSize, signalWeight, noiseWeight, m_msgLevel);
    else{
      ///If I don't have tstData , I will uses the valData as tstData for training.
      m_train = new PatternRecognition(m_network, m_trnData, m_valData, m_tstData, useSP, batchSize, signalWeight, noiseWeight, m_msgLevel);
    }  
  }

  if(m_msgLevel <= DEBUG){
    m_network->showInfo();
    m_train->showInfo(nEpochs);
  }

  // Performing the training.
  unsigned num_fails_mse = 0;
  unsigned num_fails_sp = 0;
  unsigned dispCounter = 0;
  REAL mse_val, sp_val, mse_tst, sp_tst;
  mse_val = sp_val = mse_tst = sp_tst = 0.;
  ValResult is_best_mse, is_best_sp;
  bool stop_mse, stop_sp;

  //Calculating the max_fail limits for each case (MSE and SP, if the case).
  const unsigned fail_limit_mse = (useSP) ? (fail_limit / 2) : fail_limit;
  const unsigned fail_limit_sp = (useSP) ? fail_limit : 0;
  ValResult &is_best = (useSP) ? is_best_sp :  is_best_mse;
  REAL &val_data = (useSP) ? sp_val : mse_val;
  REAL &tst_data = (useSP) ? sp_tst : mse_tst;

  ///Training loop
  for(unsigned epoch=0; epoch < nEpochs; epoch++){

    //Training the network and calculating the new weights.
    const REAL mse_trn = m_train->trainNetwork();

    m_train->valNetwork(mse_val, sp_val);

    //Testing the new network if a testing dataset was passed.
    if (!m_tstData.empty()) m_train->tstNetwork(mse_tst, sp_tst);

    // Saving the best weight result.
    m_train->isBestNetwork(mse_val, sp_val, is_best_mse, is_best_sp);

    if (is_best_mse == BETTER) num_fails_mse = 0;
    else if (is_best_mse == WORSE || is_best_mse == EQUAL) num_fails_mse++;

    if (is_best_sp == BETTER) num_fails_sp = 0;
    else if (is_best_sp == WORSE || is_best_sp == EQUAL) num_fails_sp++;
    
    if (is_best == BETTER) m_network->saveBestTrain();

    //Showing partial results at every "show" epochs (if show != 0).
    if (show)
    {
      if (!dispCounter)
      {
        if (!m_tstData.empty()) m_train->showTrainingStatus(epoch, mse_trn, val_data, tst_data);
        else m_train->showTrainingStatus(epoch, mse_trn, val_data);
      }
      dispCounter = (dispCounter + 1) % show;
    }

    //Knowing whether the criterias are telling us to stop.
    stop_mse = num_fails_mse >= fail_limit_mse;
    stop_sp = num_fails_sp >= fail_limit_sp;
    m_train->saveTrainInfo(epoch, mse_trn, mse_val, sp_val, mse_tst, sp_tst, is_best_mse, 
                          is_best_sp, num_fails_mse, num_fails_sp, stop_mse, stop_sp);

    //MSG_INFO(m_log, "num_fails_sp = " << num_fails_sp << " num_fails_mse = " << num_fails_mse);
    if ( (stop_mse) && (stop_sp) )
    {
      if (show) MSG_INFO(m_log, "Maximum number of failures reached. Finishing training...");
      break;
    }


  }

  ///Hold the train evolution before destroyer the object
  flushTrainEvolution( m_train->getTrainInfo() );

  ///Release memory
  delete m_train;
}


py::list FastnetPyWrapper::sim( py::list data )
{
  MSG_DEBUG(m_log, "Applying input propagation for simulation step...");
  py::list outputList;

  if(m_network)
  {
    // Creating the neural network to use.
    //FeedForward net(m_network);
    Backpropagation *net = m_network;

    DataHandler<REAL> *dataHandler = new DataHandler<REAL>( data );
    const unsigned numEvents = dataHandler->getNumRows();
    const unsigned outputSize = net->getNumNodes(net->getNumLayers()-1);
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
        memcpy( &outputEvents[i*outputSize], net->propagateInput( &inputEvents[i*inputSize]), numBytes2Copy);
      }
    }

    ///Parse between c++ and python list using boost
    for(unsigned i = 0; i < numEvents; ++i){
      if(outputSize == 1){
        outputList.append(outputEvents[i*outputSize]);     
      }else{
        py::list out;
        for(unsigned j = 0; j < outputSize; ++j)  out.append(outputEvents[j+outputSize*i]);
        outputList.append(out);
      }///Output parse
    }

    delete dataHandler;
  }else{
    MSG_WARNING(m_log, "There is no Network object trained into the memory. Please train your network using train().");
  }
  
  return outputList; ///Return boost python list
}


void FastnetPyWrapper::setTrainData( py::list data ){

  if(!m_trnData.empty()) releaseDataSet( m_trnData );
  for(unsigned pattern=0; pattern < py::len( data ); pattern++ ){
    DataHandler<REAL> *dataHandler = new DataHandler<REAL>( py::extract<py::list>(data[pattern]) );
    m_trnData.push_back( dataHandler );
    //m_trnData[pattern]->showInfo();
  }

}

void FastnetPyWrapper::setValData( py::list data ){

  if(!m_valData.empty()) releaseDataSet( m_valData );
  for(unsigned pattern=0; pattern < py::len( data ); pattern++ ){
    DataHandler<REAL> *dataHandler = new DataHandler<REAL>( py::extract<py::list>(data[pattern]) );
    m_valData.push_back( dataHandler );
    //m_valData[pattern]->showInfo();
  }

}

void FastnetPyWrapper::setTestData( py::list data ){

  if(!m_tstData.empty()) releaseDataSet( m_tstData );
  for(unsigned pattern=0; pattern < py::len( data ); pattern++ ){
    DataHandler<REAL> *dataHandler = new DataHandler<REAL>( py::extract<py::list>(data[pattern]) );
    m_tstData.push_back( dataHandler );
    //m_trnData[pattern]->showInfo();
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
    trainData.setMseTst((*at).mse_tst);
    trainData.setSPTst((*at).sp_tst);
    trainData.setIsBestMse((*at).is_best_mse);
    trainData.setIsBestSP((*at).is_best_sp);
    trainData.setNumFailsMse((*at).num_fails_mse);
    trainData.setNumFailsSP((*at).num_fails_sp);
    trainData.setStopMse((*at).stop_mse);
    trainData.setStopSP((*at).stop_sp);
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










