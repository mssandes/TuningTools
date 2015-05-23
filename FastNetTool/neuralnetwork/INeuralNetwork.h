#ifndef FASTNETTOOL_SYSTEM_INEURALNETWORK_H
#define FASTNETTOOL_SYSTEM_INEURALNETWORK_H


#include <vector>
#include <string>
#include <iostream> 
#include "FastNetTool/system/defines.h"
#include "FastNetTool/system/MsgStream.h"
#include "FastNetTool/system/macros.h"

// namespaces declarations
using namespace std;
using namespace msg;

namespace FastNet{

  //Interface Classes
  class INeuralNetwork{

    private:
      ///MsgLevel for future
      Level m_msgLevel = INFO;

      ///Network struct parameters
      vector<unsigned>     m_nNodes;
      vector<string>       m_trfFuncStr;
      vector<bool>         m_usingBias;
      vector<vector<bool>> m_frozenNodes;

      ///Train parameters

      string m_trainFcn       = TRAINRP_ID;
      REAL m_learningRate     = 0.05;
      REAL m_decFactor        = 1;
      REAL m_deltaMax         = 50;
      REAL m_deltaMin         = 1E-6;
      REAL m_incEta           = 1.10;
      REAL m_decEta           = 0.5;
      REAL m_initEta          = 0.1;
      bool m_useSP            = true;
      unsigned m_maxFail      = 50;
      unsigned m_nEpochs      = 1000;
      unsigned m_batchSize    = 10;
      unsigned m_show         = 2;
      REAL m_sp_signal_weight = 1;  
      REAL m_sp_noise_weight  = 1;

    public:
      /// Defaul constructor
      //This inteface will be used to hold all single information about the
      //network struct. This single constructor are inplemented to build a
      //single network with three layers.
      INeuralNetwork(){};
      ~INeuralNetwork(){};

      
      void setNodes( vector<unsigned> nodes ){
        
        m_nNodes = nodes;
        m_usingBias.clear();
        m_frozenNodes.clear();

        for(unsigned i=0; i < m_nNodes.size()-1; ++i){
          m_usingBias.push_back(true);
        }

        //Initialize frozen nodes status with false
        for(unsigned layer=0; layer<m_nNodes.size()-1; ++layer){
          vector<bool> nodes;
          for(unsigned node=0; node<m_nNodes[layer+1]; ++node){
            nodes.push_back(false);
          }  
          m_frozenNodes.push_back(nodes);
        }

      };
      
      void setUsingBias(unsigned layer, bool status = true){
        m_usingBias[layer] = status;
      }

      ///Frozen node for training.
      bool setFrozenNode(unsigned layer, unsigned node, bool status=true){
        //Add some protections
        if(layer < 1 || layer > m_nNodes.size()){
          cout << "Invalide layer for frozen status." << endl;
          return false;
        }

        if(node > m_nNodes[layer]){
          cout << "Invalide node for frozen status." << endl;
          return false;
        }
        m_frozenNodes[layer][node] = status;
        return true;
      };

      ///Return frozen status
      bool isFrozenNode(unsigned layer, unsigned node){return m_frozenNodes[layer][node];};
      ///Return the layer bias status. The dafault is true
      bool isUsingBias(unsigned layer){return m_usingBias[layer];};
      ///Return the nNodes vector
      vector<unsigned> getNodes(){return m_nNodes;};
      ///Return the number of nodes into the layer.
      unsigned getNumberOfNodes(unsigned layer=0){return m_nNodes[layer];};
      ///Return the max number of layer.
      unsigned getNumberOfLayers(){return m_nNodes.size();};
      ///Return the tranfer function.
      string getTrfFuncStr(unsigned layer){return m_trfFuncStr[layer];};
      
      ///Macros to help
      PRIMITIVE_SETTER_AND_GETTER(string, setTrainFcn  , getTrainFcn    , m_trainFcn    );      
      PRIMITIVE_SETTER_AND_GETTER(bool, setUseSP       , getUseSP       , m_useSP       );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setMaxFail     , getMaxFail     , m_maxFail     );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setSPSignalWeight    , getSPSignalWeight    , m_sp_signal_weight    );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setSPNoiseWeight     , getSPNoiseWeight     , m_sp_noise_weight     );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setLearningRate, getLearningRate, m_learningRate);      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setDecFactor   , getDecFactor   , m_decFactor   );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setDeltaMax    , getDeltaMax    , m_deltaMax    );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setDeltaMin    , getDeltaMin    , m_deltaMin    );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setIncEta      , getIncEta      , m_incEta      );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setDecEta      , getDecEta      , m_decEta      );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setInitEta     , getInitEta     , m_initEta     );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setEpochs      , getEpochs      , m_nEpochs     );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setBatchSize   , getBatchSize   , m_batchSize   );      
      PRIMITIVE_SETTER_AND_GETTER(REAL, setShow        , getShow        , m_show        );      
      PRIMITIVE_SETTER_AND_GETTER(vector<string>, setTrfFunc   , getTrfFunc    , m_trfFuncStr    );      
      PRIMITIVE_SETTER_AND_GETTER(Level, setMsglevel   , getMsgLevel    , m_msgLevel    );      
  };


}
#endif
