

#include "FastNetTool/FastnetPyWrapper.h"

///Constructor
FastnetPyWrapper::FastnetPyWrapper(){

  m_appName  = "FastnetPyWrapper";
  m_msgLevel = DEBUG;
  m_net      = NULL;
  m_network  = NULL;
  m_train    = NULL;
  m_stdTrainingType = true;

  ///MsgStream Manager object
  m_log = new MsgStream(m_appName, m_msgLevel); 
  MSG_INFO(m_log, "python/c++ interface was created.");
}

FastnetPyWrapper::~FastnetPyWrapper(){

}


bool FastnetPyWrapper::newff( py::list nodes, py::list trfFunc, string trainFcn, REAL batchSize, bool usingBias ){

  ///release memory
  if(m_net) delete m_net;
  if(m_network){
    delete m_network;
    m_network = NULL;
  }

  ///apply the net struct
  m_net = new INeuralNetwork(util::to_std_vector<unsigned>(nodes), util::to_std_vector<string>(trfFunc), usingBias );
  m_net->setTrainFcn(trainFcn);
  m_net->setBatchSize(batchSize);

  if(m_net->getTrainFcn() == "rprop"){
    m_network = new RProp(m_net, m_msgLevel);
    MSG_DEBUG(m_log, "RProp object was created into the python interface!");
  }else if(m_net->getTrainFcn() == "bfg"){
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

  if(!m_in_tst) m_stdTrainingType = false;
  if(!m_in_trn || !m_in_val || !m_out_trn || !m_out_val){
    MSG_WARNING(m_log, "You must set the datasets [in_trn, out_trn, in_val, out_val] before start the train.");
    return false;
  }
  
   
  if (m_stdTrainingType)
  {
    m_train = new StandardTraining(m_network, m_in_trn, m_out_trn, m_in_val, m_out_val, m_net->getBatchSize(), m_msgLevel );
  }
  else // It is a pattern recognition network.
  {
    //m_train = new PatternRecognition(net, args[IN_TRN_IDX], args[IN_VAL_IDX], tstData, useSP, batchSize, signalWeight, noiseWeight);
  }  


  return true;
}

///Release memory
bool FastnetPyWrapper::finalize(){ 
  delete m_log;
  if(m_net)      delete m_net;
  if(m_network)  delete m_network;
  if(m_in_trn)   delete m_in_trn;
  if(m_out_trn)  delete m_out_trn;
  if(m_in_val)   delete m_in_val;
  if(m_out_val)  delete m_out_val;
  if(m_in_tst)   delete m_in_tst;
  return true;
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










