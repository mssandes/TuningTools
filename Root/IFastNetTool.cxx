

#include "FastNetTool/IFastNetTool.h"

///Constructor
IFastNetTool::IFastNetTool(){

  m_appName  = "IFastNetTool_python";
  m_msgLevel = DEBUG;
  m_net      = NULL;
  m_network  = NULL;
  m_train    = NULL;
  m_stdTrainingType = true;

  ///MsgStream Manager object
  m_log = new MsgStream(m_appName, m_msgLevel); 
  MSG_INFO(m_log, "python/c++ interface was created.");
}

IFastNetTool::~IFastNetTool(){

}


///Initialize step is necessary to set all objects and configuration that i
//need to traininig my neural network. After initialize all these objects, you
//need apply the train() method to start the event loop.
bool IFastNetTool::initialize(){

  if(!m_net){
    MSG_WARNING(m_log, "Before initialize this class, you must apply setParam to alloc the INeuralNetwork into the memory.");
    return false;
  }

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
bool IFastNetTool::train(){
  return true;
}

///Release memory
bool IFastNetTool::finalize(){ 
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



void IFastNetTool::showInfo(){
      
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










