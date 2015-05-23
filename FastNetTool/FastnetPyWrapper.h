
#ifndef FASTNETTOOL_IFASTNETTOOL_H
#define FASTNETTOOL_IFASTNETTOOL_H

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <boost/python.hpp>
#include "FastNetTool/system/util.h"
#include "FastNetTool/system/defines.h"
#include "FastNetTool/system/macros.h"
#include "FastNetTool/system/DataHandler.h"
#include "FastNetTool/system/MsgStream.h"
#include "FastNetTool/neuralnetwork/INeuralNetwork.h"
#include "FastNetTool/neuralnetwork/Backpropagation.h"
#include "FastNetTool/neuralnetwork/RProp.h"
#include "FastNetTool/neuralnetwork/FeedForward.h"
#include "FastNetTool/training/Standard.h"
#include "FastNetTool/training/PatternRec.h"

namespace py = boost::python;
using namespace std;
using namespace msg;
using namespace FastNet;


///Helper class
class TrainDataPyWrapper{

  private:

    unsigned m_epoch;
    REAL m_mse_trn;
    REAL m_mse_val;
    REAL m_sp_val;
    REAL m_mse_tst;
    REAL m_sp_tst;
    ValResult m_is_best_mse;
    ValResult m_is_best_sp;
    unsigned m_num_fails_mse;
    unsigned m_num_fails_sp;
    bool m_stop_mse;
    bool m_stop_sp;


  public:
    PRIMITIVE_SETTER_AND_GETTER(unsigned  , setEpoch, getEpoch, m_epoch);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setMseTrn, getMseTrn, m_mse_trn);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setMseVal, getMseVal, m_mse_val);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setSPVal, getSPVal, m_sp_val);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setMseTst, getMseTst, m_mse_tst);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setSPTst, getSPTst, m_sp_tst);
    PRIMITIVE_SETTER_AND_GETTER(unsigned  , setNumFailsMse, getNumFailsMse, m_num_fails_mse);
    PRIMITIVE_SETTER_AND_GETTER(unsigned  , setNumFailsSP, getNumFailsSP, m_num_fails_sp);
    PRIMITIVE_SETTER_AND_GETTER(bool      , setStopMse, getStopMse, m_stop_mse);
    PRIMITIVE_SETTER_AND_GETTER(bool      , setStopSP, getStopSP, m_stop_sp);

    PRIMITIVE_SETTER(ValResult , setIsBestMse, m_is_best_mse);
    PRIMITIVE_SETTER(ValResult , setIsBestSP,  m_is_best_sp);

    bool getIsBestMse(){return (m_is_best_mse == BETTER) ? true:false;}
    bool getIsBestSP(){ return (m_is_best_sp  == BETTER) ? true:false;}


};///Helper class








///Interface class between the python and c++ fastnet core
class FastnetPyWrapper{

  private:
    ///MsgStream manager
    MsgStream *m_log;
    Level      m_msgLevel;
    string     m_appName;

    vector<DataHandler<REAL>*> m_trnData;
    vector<DataHandler<REAL>*> m_valData;
    vector<DataHandler<REAL>*> m_tstData;
    vector<DataHandler<REAL>*> m_simData;
    

    ///FastNet Core
    INeuralNetwork     *m_net;///Configuration object 
    Backpropagation    *m_network;
    Training           *m_train; 

    bool m_stdTrainingType;

    ///Hold a list of TrainDataPyWrapper
    vector<TrainDataPyWrapper> m_trnEvolution;

    void flushTrainEvolution( std::list<TrainData> trnEvolution );

    void releaseDataSet( vector<DataHandler<REAL>*> vec )
    {
      for(unsigned pattern=0; pattern < vec.size(); ++pattern)
      delete vec[pattern];
      vec.clear();
    }


  public:
    
    ///Default constructor
    FastnetPyWrapper(unsigned msglevel);
    ///Destructor
    ~FastnetPyWrapper();

    ///initialize all fastNet classes
    bool newff( py::list nodes, py::list trfFunc, string trainFcn = TRAINRP_ID );
    ///Train function
    bool train();
    ///Simulatio function
    py::list sim( py::list data );
    ///Return a list of TrainDataPyWrapper
    py::list getTrainEvolution(){
      py::list pylist;
      for(vector<TrainDataPyWrapper>::iterator at = m_trnEvolution.begin(); at!=m_trnEvolution.end(); ++at)
        pylist.append((*at));
      return pylist;
    };


    void showInfo();

    void setTrainData( py::list data );
    void setValData(   py::list data );
    void setTestData(  py::list data );

    ///Frozen node for training.
    bool setFrozenNode(unsigned layer, unsigned node, bool status=true){
      if(m_net)  return m_net->setFrozenNode(layer, node, status);
      return false;
    };
    
    void setMsgLevel(unsigned level){
      if(level == 0)       m_msgLevel = VERBOSE;
      else if(level == 1)  m_msgLevel = DEBUG;
      else if(level == 2)  m_msgLevel = INFO;
      else if(level == 3)  m_msgLevel = WARNING;
      else if(level == 4)  m_msgLevel = FATAL;
      else{
        cout << "option not found." << endl;
      }
    };
    

    ///Macros for helper
    OBJECT_SETTER_AND_GETTER(m_net, string,   setTrainFcn       , getTrainFcn       );      
    OBJECT_SETTER_AND_GETTER(m_net, bool,     setUseSP          , getUseSP          );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL,     setSPSignalWeight , getSPSignalWeight );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL,     setSPNoiseWeight  , getSPNoiseWeight  );      
    OBJECT_SETTER_AND_GETTER(m_net, unsigned, setMaxFail        , getMaxFail        );      
    OBJECT_SETTER_AND_GETTER(m_net, unsigned, setBatchSize      , getBatchSize      );      
    OBJECT_SETTER_AND_GETTER(m_net, unsigned, setEpochs         , getEpochs         );      
    OBJECT_SETTER_AND_GETTER(m_net, unsigned, setShow           , getShow           );      

    OBJECT_SETTER_AND_GETTER(m_net, REAL, setLearningRate, getLearningRate);      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setDecFactor   , getDecFactor   );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setDeltaMax    , getDeltaMax    );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setDeltaMin    , getDeltaMin    );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setIncEta      , getIncEta      );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setDecEta      , getDecEta      );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setInitEta     , getInitEta     );      

    /*
    DATAHANDLER_SETTER_AND_GETTER(m_in_trn  , REAL, set_in_trn  , get_in_trn  );
    DATAHANDLER_SETTER_AND_GETTER(m_out_trn , REAL, set_out_trn , get_out_trn );
    DATAHANDLER_SETTER_AND_GETTER(m_in_val  , REAL, set_in_val  , get_in_val  );
    DATAHANDLER_SETTER_AND_GETTER(m_out_val , REAL, set_out_val , get_out_val );
    DATAHANDLER_SETTER_AND_GETTER(m_in_tst  , REAL, set_in_tst  , get_in_tst  );
    */

 
};





///BOOST module
BOOST_PYTHON_MODULE(libFastNetTool){
  using namespace boost::python;

  class_<TrainDataPyWrapper>("TrainDataPyWrapper")
    .def("epoch",              &TrainDataPyWrapper::getEpoch)
    .def("mse_trn",            &TrainDataPyWrapper::getMseTrn)
    .def("mse_val",            &TrainDataPyWrapper::getMseVal)
    .def("sp_val",             &TrainDataPyWrapper::getSPVal)
    .def("mse_tst",            &TrainDataPyWrapper::getMseTst)
    .def("sp_tst",             &TrainDataPyWrapper::getSPTst)
    .def("is_best_mse",        &TrainDataPyWrapper::getIsBestMse)
    .def("is_best_sp",         &TrainDataPyWrapper::getIsBestSP)
    .def("num_fails_mse",      &TrainDataPyWrapper::getNumFailsMse)
    .def("num_fails_sp",       &TrainDataPyWrapper::getNumFailsSP)
    .def("stop_mse",           &TrainDataPyWrapper::getStopMse)
    .def("stop_sp",            &TrainDataPyWrapper::getStopSP)
    ;

  class_<FastnetPyWrapper>("FastnetPyWrapper",init<unsigned>())

    .def("newff"              ,&FastnetPyWrapper::newff)
    .def("train"              ,&FastnetPyWrapper::train)
    .def("sim"                ,&FastnetPyWrapper::sim)
    .def("showInfo"           ,&FastnetPyWrapper::showInfo)
 
    .def("setFrozenNode"      ,&FastnetPyWrapper::setFrozenNode)
    .def("setTrainData"       ,&FastnetPyWrapper::setTrainData )
    .def("setValData"         ,&FastnetPyWrapper::setValData )
    .def("setTestData"        ,&FastnetPyWrapper::setTestData )
    .def("setShow"            ,&FastnetPyWrapper::setShow )
    

    .def("setUseSp"       ,&FastnetPyWrapper::setUseSP)
    .def("setMaxFail"     ,&FastnetPyWrapper::setMaxFail)
    .def("setBatchSize"   ,&FastnetPyWrapper::setBatchSize)
    .def("setSPNoiseWeight"    ,&FastnetPyWrapper::setSPNoiseWeight)
    .def("setSPSignalWeight"   ,&FastnetPyWrapper::setSPSignalWeight)
    
    .def("setLearningRate",&FastnetPyWrapper::setLearningRate)
    .def("setDecFactor"   ,&FastnetPyWrapper::setDecFactor)
    .def("setDeltaMax"    ,&FastnetPyWrapper::setDeltaMax)
    .def("setDeltaMin"    ,&FastnetPyWrapper::setDeltaMin)
    .def("setIncEta"      ,&FastnetPyWrapper::setIncEta)
    .def("setDecEta"      ,&FastnetPyWrapper::setDecEta)
    .def("setInitEta"     ,&FastnetPyWrapper::setInitEta)
    .def("setEpochs"      ,&FastnetPyWrapper::setEpochs)

    .def("getTrainFcn"    ,&FastnetPyWrapper::getTrainFcn)
    .def("getUseSp"       ,&FastnetPyWrapper::getUseSP)
    .def("getMaxFail"     ,&FastnetPyWrapper::getMaxFail)
    .def("getBatchSize"   ,&FastnetPyWrapper::getBatchSize)
    .def("getSPNoiseWeight"    ,&FastnetPyWrapper::getSPNoiseWeight)
    .def("getSPSignalWeight"   ,&FastnetPyWrapper::getSPSignalWeight)
 
    
    .def("getLearningRate",&FastnetPyWrapper::getLearningRate)
    .def("getDecFactor"   ,&FastnetPyWrapper::getDecFactor)
    .def("getDeltaMax"    ,&FastnetPyWrapper::getDeltaMax)
    .def("getDeltaMin"    ,&FastnetPyWrapper::getDeltaMin)
    .def("getIncEta"      ,&FastnetPyWrapper::getIncEta)
    .def("getDecEta"      ,&FastnetPyWrapper::getDecEta)
    .def("getInitEta"     ,&FastnetPyWrapper::getInitEta)
    .def("getEpochs"      ,&FastnetPyWrapper::getEpochs)
    
    .def("getTrainEvolution"      ,&FastnetPyWrapper::getTrainEvolution)

 ;
}

#endif
