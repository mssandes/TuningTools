
#ifndef FASTNETTOOL_IFASTNETTOOL_H
#define FASTNETTOOL_IFASTNETTOOL_H

#include <iostream>
#include <string>
#include <vector>
#include <boost/python.hpp>
#include "FastNetTool/system/util.h"
#include "FastNetTool/system/defines.h"
#include "FastNetTool/system/DataHandler.h"
#include "FastNetTool/system/MsgStream.h"
#include "FastNetTool/neuralnetwork/INeuralNetwork.h"
#include "FastNetTool/neuralnetwork/Backpropagation.h"
#include "FastNetTool/neuralnetwork/RProp.h"
#include "FastNetTool/training/Standard.h"
//#include "FastNetTool/training/PatternRec.h"

#define OBJECT_SETTER_AND_GETTER(OBJ, TYPE, SETTER, GETTER)\
                                                            \
  TYPE GETTER(){                                            \
    return OBJ->GETTER();                                   \
  }                                                         \
  void SETTER(TYPE value){                                  \
    OBJ->SETTER(value);                                     \
    return;                                                 \
  }                                                         \
                                                            \

#define DATAHANDLER_SETTER_AND_GETTER(OBJ, TYPE, SETTER, GETTER)\
  DataHandler<TYPE>* GETTER(){                                          \
    return OBJ;                                                         \
  }                                                                     \
                                                                        \
  void SETTER##_2D(boost::python::list data, unsigned row, unsigned col){ \
      OBJ = new DataHandler<TYPE>(data, row,col);                       \
  }                                                                     \
  void SETTER##_1D(boost::python::list data, unsigned col){             \
      OBJ = new DataHandler<TYPE>(data, col);                           \
  }                                                                     \




namespace py = boost::python;
using namespace std;
using namespace msg;
using namespace FastNet;

class FastnetPyWrapper{

  private:
    ///MsgStream manager
    MsgStream *m_log;
    Level      m_msgLevel;
    string     m_appName;

    ///Matrixs objects for training and validation
    DataHandler<REAL> *m_in_trn;
    DataHandler<REAL> *m_out_trn;
    DataHandler<REAL> *m_in_val;
    DataHandler<REAL> *m_out_val;
    DataHandler<REAL> *m_in_tst;

    ///FastNet Core
    INeuralNetwork     *m_net;///Configuration object 
    Backpropagation    *m_network;
    Training           *m_train; 
    
    bool m_stdTrainingType;


  public:
    
    ///Default constructor
    FastnetPyWrapper();
    ///Destructor
    ~FastnetPyWrapper();

    ///initialize all fastNet classes
    bool newff( py::list nodes, py::list trfFunc, string trainFcn, REAL batchSize, bool usingBias );
    ///Train function
    bool train();
    ///Release memory
    bool finalize();

    void showInfo();

   
    ///Frozen node for training.
    bool setFrozenNode(unsigned layer, unsigned node, bool status=true){
      if(m_net)  return m_net->setFrozenNode(layer, node, status);
      return false;
    };
    


    ///Macros for helper
    OBJECT_SETTER_AND_GETTER(m_net, string, setTrainFcn  , getTrainFcn    );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setLearningRate, getLearningRate);      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setDecFactor   , getDecFactor   );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setDeltaMax    , getDeltaMax    );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setDeltaMin    , getDeltaMin    );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setIncEta      , getIncEta      );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setDecEta      , getDecEta      );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setInitEta     , getInitEta     );      
    OBJECT_SETTER_AND_GETTER(m_net, REAL, setEpochs      , getEpochs      );      

    DATAHANDLER_SETTER_AND_GETTER(m_in_trn  , REAL, set_in_trn  , get_in_trn  );
    DATAHANDLER_SETTER_AND_GETTER(m_out_trn , REAL, set_out_trn , get_out_trn );
    DATAHANDLER_SETTER_AND_GETTER(m_in_val  , REAL, set_in_val  , get_in_val  );
    DATAHANDLER_SETTER_AND_GETTER(m_out_val , REAL, set_out_val , get_out_val );
    DATAHANDLER_SETTER_AND_GETTER(m_in_tst  , REAL, set_in_tst  , get_in_tst  );


 
};





///BOOST module
BOOST_PYTHON_MODULE(libFastNetTool){
  using namespace boost::python;
  class_<FastnetPyWrapper>("FastnetPyWrapper")

    .def("newff"              ,&FastnetPyWrapper::newff)
    .def("finalize"           ,&FastnetPyWrapper::finalize)
    .def("train"              ,&FastnetPyWrapper::train)
    .def("showInfo"           ,&FastnetPyWrapper::showInfo)
 
    .def("setFrozenNode"      ,&FastnetPyWrapper::setFrozenNode)

    .def("set_in_trn_1D"     ,&FastnetPyWrapper::set_in_trn_1D)
    .def("set_in_trn_2D"     ,&FastnetPyWrapper::set_in_trn_2D)
    //.def("get_in_trn"      ,&FastnetPyWrapper::get_in_trn)
    .def("set_out_trn_1D"    ,&FastnetPyWrapper::set_out_trn_1D)
    .def("set_out_trn_2D"    ,&FastnetPyWrapper::set_out_trn_2D)
    //.def("get_out_trn"     ,&FastnetPyWrapper::get_out_trn)
    .def("set_in_val_1D"     ,&FastnetPyWrapper::set_in_val_1D)
    .def("set_in_val_2D"     ,&FastnetPyWrapper::set_in_val_2D)
    //.def("get_in_trn"      ,&FastnetPyWrapper::get_in_trn)
    .def("set_out_val_1D"    ,&FastnetPyWrapper::set_out_val_1D)
    .def("set_out_val_2D"    ,&FastnetPyWrapper::set_out_val_2D)
    //.def("get_in_trn"      ,&FastnetPyWrapper::get_in_trn)
    .def("set_in_tst_1D"     ,&FastnetPyWrapper::set_in_tst_1D)
    .def("set_in_tst_2D"     ,&FastnetPyWrapper::set_in_tst_2D)
    //.def("get_in_trn"      ,&FastnetPyWrapper::get_in_trn)
 
    .def("setTrainFcn"    ,&FastnetPyWrapper::setTrainFcn)
    .def("setLearningRate",&FastnetPyWrapper::setLearningRate)
    .def("setDecFactor"   ,&FastnetPyWrapper::setDecFactor)
    .def("setDeltaMax"    ,&FastnetPyWrapper::setDeltaMax)
    .def("setDeltaMin"    ,&FastnetPyWrapper::setDeltaMin)
    .def("setIncEta"      ,&FastnetPyWrapper::setIncEta)
    .def("setDecEta"      ,&FastnetPyWrapper::setDecEta)
    .def("setInitEta"     ,&FastnetPyWrapper::setInitEta)
    .def("setEpochs"      ,&FastnetPyWrapper::setEpochs)

    .def("getTrainFcn"    ,&FastnetPyWrapper::getTrainFcn)
    .def("getLearningRate",&FastnetPyWrapper::getLearningRate)
    .def("getDecFactor"   ,&FastnetPyWrapper::getDecFactor)
    .def("getDeltaMax"    ,&FastnetPyWrapper::getDeltaMax)
    .def("getDeltaMin"    ,&FastnetPyWrapper::getDeltaMin)
    .def("getIncEta"      ,&FastnetPyWrapper::getIncEta)
    .def("getDecEta"      ,&FastnetPyWrapper::getDecEta)
    .def("getInitEta"     ,&FastnetPyWrapper::getInitEta)
    .def("getEpochs"      ,&FastnetPyWrapper::getEpochs)

 ;
}

#endif
