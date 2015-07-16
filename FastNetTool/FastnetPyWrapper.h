/*
  Author (FastNet c++ core): Rodrigo Coura Torres
  email: torres.rc@gmail.com
  Authol (FastNet python interface): Joao Victor da Fonseca Pinto
  email: jodafons@cern.ch

  Introduction:
    The FastNet c++ core It was implemented base on this link:
    https://github.com/rctorres/fastnet
    The old version used the matlab interface. In this new version,
    we have the python configuration with out matlab. Here, I add
    more on stop criteria. The multi stop criteria. This criteria
    uses the detection probability, the index SP and the false 
    alarm to save three networks and stop the training.

  comments:
    You need the boost , RootCore and gcc to compile this!
*/

#ifndef FASTNETTOOL_FASTNETTOOLPYWRAPPER_H
#define FASTNETTOOL_FASTNETTOOLPYWRAPPER_H

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
    REAL m_det_val;
    REAL m_fa_val;
    REAL m_mse_tst;
    REAL m_sp_tst;
    REAL m_det_tst;
    REAL m_fa_tst;
    ValResult m_is_best_mse;
    ValResult m_is_best_sp;
    ValResult m_is_best_det;
    ValResult m_is_best_fa;
    unsigned m_num_fails_mse;
    unsigned m_num_fails_sp;
    unsigned m_num_fails_det;
    unsigned m_num_fails_fa;
    bool m_stop_mse;
    bool m_stop_sp;
    bool m_stop_det;
    bool m_stop_fa;


  public:
    PRIMITIVE_SETTER_AND_GETTER(unsigned  , setEpoch, getEpoch, m_epoch);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setMseTrn, getMseTrn, m_mse_trn);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setMseVal, getMseVal, m_mse_val);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setSPVal, getSPVal, m_sp_val);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setDetVal, getDetVal, m_det_val);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setFaVal, getFaVal, m_fa_val);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setMseTst, getMseTst, m_mse_tst);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setSPTst, getSPTst, m_sp_tst);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setDetTst, getDetTst, m_det_tst);
    PRIMITIVE_SETTER_AND_GETTER(REAL      , setFaTst, getFaTst, m_fa_tst);
    PRIMITIVE_SETTER_AND_GETTER(unsigned  , setNumFailsMse, getNumFailsMse, m_num_fails_mse);
    PRIMITIVE_SETTER_AND_GETTER(unsigned  , setNumFailsSP, getNumFailsSP, m_num_fails_sp);
    PRIMITIVE_SETTER_AND_GETTER(unsigned  , setNumFailsDet, getNumFailsDet, m_num_fails_det);
    PRIMITIVE_SETTER_AND_GETTER(unsigned  , setNumFailsFa, getNumFailsFa, m_num_fails_fa);
    PRIMITIVE_SETTER_AND_GETTER(bool      , setStopMse, getStopMse, m_stop_mse);
    PRIMITIVE_SETTER_AND_GETTER(bool      , setStopSP, getStopSP, m_stop_sp);
    PRIMITIVE_SETTER_AND_GETTER(bool      , setStopDet, getStopDet, m_stop_det);
    PRIMITIVE_SETTER_AND_GETTER(bool      , setStopFa, getStopFa, m_stop_fa);

    PRIMITIVE_SETTER(ValResult , setIsBestMse, m_is_best_mse);
    PRIMITIVE_SETTER(ValResult , setIsBestSP,  m_is_best_sp);
    PRIMITIVE_SETTER(ValResult , setIsBestDet, m_is_best_det);
    PRIMITIVE_SETTER(ValResult , setIsBestFa,  m_is_best_fa);

    bool getIsBestMse(){ return (m_is_best_mse == BETTER)  ? true:false;}
    bool getIsBestSP(){  return (m_is_best_sp  == BETTER)  ? true:false;}
    bool getIsBestDet(){ return (m_is_best_det  == BETTER) ? true:false;}
    bool getIsBestFa(){  return (m_is_best_fa  == BETTER)  ? true:false;}


};///Helper class

//==========================================================================================
//==========================================================================================
//==========================================================================================
//==========================================================================================

class DiscriminatorPyWrapper : public NeuralNetwork{

  public:
    DiscriminatorPyWrapper( const NeuralNetwork &net):NeuralNetwork(net){};
    ~DiscriminatorPyWrapper(){}; 
};///Helper class

//==========================================================================================
//==========================================================================================
//==========================================================================================
//==========================================================================================

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
    INeuralNetwork        *m_net;
    Backpropagation       *m_trainNetwork;
    Training              *m_train; 
    vector<NeuralNetwork*> m_saveNetworks;

    bool m_stdTrainingType;

    ///Hold a list of TrainDataPyWrapper
    vector<TrainDataPyWrapper> m_trnEvolution;

  private:

    void flushTrainEvolution( std::list<TrainData> trnEvolution );
    ///Allocate network space into the memory
    bool allocateNetwork( py::list nodes, py::list trfFunc, string trainFcn );

    void releaseDataSet( vector<DataHandler<REAL>*> vec )
    {
      for(unsigned pattern=0; pattern < vec.size(); ++pattern){
        if(vec[pattern])  delete vec[pattern];
      }
      vec.clear();
    }

    ///Return a list of TrainDataPyWrapper
    py::list trainEvolutionToPyList(){
      py::list trainList;
      for(vector<TrainDataPyWrapper>::iterator at = m_trnEvolution.begin(); at!=m_trnEvolution.end(); ++at) trainList.append((*at));
      return trainList;
    };

    ///Return a list of DiscriminatorPyWrapper::NeuralNetwork to python 
    py::list saveNetworksToPyList(){
      py::list netList;
      for(unsigned i=0; i < m_saveNetworks.size(); ++i) netList.append( DiscriminatorPyWrapper((*m_saveNetworks[i])) );
      return netList;
    };

    ///Simulatio function retrn a list of outputs
    DataHandler<REAL> sim( DiscriminatorPyWrapper net, DataHandler<REAL> *data);
    py::list genRoc( vector<REAL> signalVec, vector<REAL> noiseVec, REAL resolution );

 public:
    
    ///Default constructor
    FastnetPyWrapper(unsigned msglevel);
    ///Destructor
    ~FastnetPyWrapper();

    ///initialize all fastNet classes
    bool newff( py::list nodes, py::list trfFunc, string trainFcn = TRAINRP_ID );
    bool loadff( py::list nodes, py::list trfFunc, py::list weight, py::list bias ,string trainFcn = TRAINRP_ID);

    /*
      This function return a list of networks and a list of TrainData evolution. 
      If MSE_STOP or SP_STOP was enable, this will return a list o one element. 
      But in the other case, MULTI_STOP will return a list where: 
          [network_stop_by_sp, network_stop_by_det, network_stop_by_fa]
      The train data evolution is a list of TrainDataPyWrapper and networks
      is a list of DiscriminatorPyWrapper. Basically, the outputs are:
          [list_of_DeiscriminatorPyWrapper, list_of_TrainDataPyWrapper]
    */
    py::list train_c();
    py::list sim_c( DiscriminatorPyWrapper net, py::list input );
    py::list valid_c( DiscriminatorPyWrapper net );
    
   
    void showInfo();
    void setTrainData( py::list data , const unsigned inputSize);
    void setValData(   py::list data , const unsigned inputSize);
    void setTestData(  py::list data , const unsigned inputSize);

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

    ///Goal train selection 
    void useMSE(){  m_net->setTrainGoal( MSE_STOP );    };
    void useSP(){   m_net->setTrainGoal( SP_STOP );     };
    void useAll(){  m_net->setTrainGoal( MULTI_STOP );  };

    ///Macros for helper
    OBJECT_SETTER_AND_GETTER(m_net, string,   setTrainFcn       , getTrainFcn       );      
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
};


///BOOST module
BOOST_PYTHON_MODULE(libFastNetTool){
  using namespace boost::python;

  class_<DiscriminatorPyWrapper>("DiscriminatorPyWrapper",no_init)
    
    .def("getNumLayers",            &DiscriminatorPyWrapper::getNumLayers   )
    .def("getNumNodes",             &DiscriminatorPyWrapper::getNumNodes    )
    .def("getBias",                 &DiscriminatorPyWrapper::getBias        )
    .def("getWeight",               &DiscriminatorPyWrapper::getWeight      )
    .def("getTrfFuncName",          &DiscriminatorPyWrapper::getTrfFuncName )
    ;
  ///=================================================================================
  class_<TrainDataPyWrapper>("TrainDataPyWrapper", no_init)
    
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
  ///=================================================================================
  class_<FastnetPyWrapper>("FastnetPyWrapper",init<unsigned>())

    .def("loadff"             ,&FastnetPyWrapper::loadff        )
    .def("newff"              ,&FastnetPyWrapper::newff         )
    .def("train_c"            ,&FastnetPyWrapper::train_c       )
    .def("sim_c"              ,&FastnetPyWrapper::sim_c         )
    .def("valid_c"            ,&FastnetPyWrapper::valid_c       )
    .def("showInfo"           ,&FastnetPyWrapper::showInfo      )    
    .def("useMSE"             ,&FastnetPyWrapper::useMSE        )
    .def("useSP"              ,&FastnetPyWrapper::useSP         )
    .def("useAll"             ,&FastnetPyWrapper::useAll        )
    .def("setFrozenNode"      ,&FastnetPyWrapper::setFrozenNode )
    .def("setTrainData"       ,&FastnetPyWrapper::setTrainData  )
    .def("setValData"         ,&FastnetPyWrapper::setValData    )
    .def("setTestData"        ,&FastnetPyWrapper::setTestData   )
    
    .add_property("show"          ,&FastnetPyWrapper::getShow           ,&FastnetPyWrapper::setShow           )
    .add_property("maxFail"       ,&FastnetPyWrapper::getMaxFail        ,&FastnetPyWrapper::setMaxFail        )
    .add_property("batchSize"     ,&FastnetPyWrapper::getBatchSize      ,&FastnetPyWrapper::setBatchSize      )
    .add_property("SPNoiseWeight" ,&FastnetPyWrapper::getSPNoiseWeight  ,&FastnetPyWrapper::setSPNoiseWeight  )
    .add_property("SPSignalWeight",&FastnetPyWrapper::getSPSignalWeight ,&FastnetPyWrapper::setSPSignalWeight )
    .add_property("learningRate"  ,&FastnetPyWrapper::getLearningRate   ,&FastnetPyWrapper::setLearningRate   )
    .add_property("decFactor"     ,&FastnetPyWrapper::getDecFactor      ,&FastnetPyWrapper::setDecFactor      )
    .add_property("deltaMax"      ,&FastnetPyWrapper::getDeltaMax       ,&FastnetPyWrapper::setDeltaMax       )
    .add_property("deltaMin"      ,&FastnetPyWrapper::getDeltaMin       ,&FastnetPyWrapper::setDeltaMin       )
    .add_property("incEta"        ,&FastnetPyWrapper::getIncEta         ,&FastnetPyWrapper::setIncEta         )
    .add_property("decEta"        ,&FastnetPyWrapper::getDecEta         ,&FastnetPyWrapper::setDecEta         )
    .add_property("initEta"       ,&FastnetPyWrapper::getInitEta        ,&FastnetPyWrapper::setInitEta        )
    .add_property("epochs"        ,&FastnetPyWrapper::getEpochs         ,&FastnetPyWrapper::setEpochs         )
    ;
}
#endif
