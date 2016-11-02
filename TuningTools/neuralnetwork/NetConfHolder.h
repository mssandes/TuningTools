#ifndef TUNINGTOOLS_SYSTEM_INEURALNETWORK_H
#define TUNINGTOOLS_SYSTEM_INEURALNETWORK_H

// Package include(s)
#include "TuningTools/system/defines.h"

#include "RingerCore/MsgStream.h"
#include "TuningTools/system/macros.h"

// STL include(s)
#include <vector>
#include <string>
#include <iostream> 
#include <cstdint> 

namespace TuningTool {

//Interface Classes
class NetConfHolder : public MsgService {

  private:
    ///Network struct parameters
    std::vector<unsigned>       m_nNodes;
    std::vector<std::string>    m_trfFuncStr;
    std::vector<bool>           m_usingBias;
    std::vector< std::vector<bool> > m_frozenNodes;

    /// NN Training Algorithm Parameters
    /// @{
    std::string m_trainAlgFcn = TRAINRP_ID;
    /// Steepest decendent algorithm
    /// @{
    /// Learning rate factor
    REAL m_learningRate       = 0.05;
    /// The decreasing factor to the learning rate after each epoch (not currently being used!)
    REAL m_decFactor          = 1;
    /// @}
    /// Resilient back-propagation algorithm
    /// @{
    /// Maximum delta accepted by algorithm
    REAL m_deltaMax           = 50;
    /// Mimumum delta accepted by algorithm
    REAL m_deltaMin           = 1E-6;
    /// Increasing factor apply to eta
    REAL m_incEta             = 1.10; // FIXME increase delta is different from the default (1.2)
    /// Decreasing factor applied to eta
    REAL m_decEta             = 0.5;
    /// Initial eta (delta0)
    REAL m_initEta            = 0.1; // delta0
    /// @}
    /// Standard tuning parameters:
    /// @{
    /// Maximum number of epochs
    uint64_t m_nEpochs        = 1000;
    /// Training batch size
    unsigned m_batchSize      = 10;
    /// Interval to show performance improvement
    unsigned m_show           = 5;
    /// Maximum allowed failures to improve performance
    unsigned m_maxFail        = 50;
    /// Minimum number of epochs to start evaluating performance
    unsigned m_minEpochs      = 5;
    /// Parameters used to ROC calculation
    /// @{
    /// ROC resolution
    REAL m_rocResolution      = 0.01;
    /// ROC Signal weight used for SP calculation
    REAL m_spSignalWeight     = 1.;
    /// ROC Background weight used for SP calculation
    REAL m_spBackgroundWeight = 1.;
    /// @}

  public:
    /**
     * @brief Defaul constructor
     *
     * This configuration holder has every information which will be used to
     * set the network struct. This single constructor are inplemented to build
     * a single network with three layers.
     **/
    NetConfHolder( MSG::Level level = MSG::INFO ) 
      : IMsgService("NetConfHolder"),
        MsgService( level ){;}
    
    
    /**
     * Set neural network configuration nodes
     **/
    void setNodes( std::vector<unsigned> nodes ){
      m_nNodes = nodes;
      m_usingBias.clear();
      m_frozenNodes.clear();

      for(unsigned i=0; i < m_nNodes.size()-1; ++i){
        m_usingBias.push_back(true);
      }

      //Initialize frozen nodes status with false
      for(unsigned layer=0; layer<m_nNodes.size()-1; ++layer){
        std::vector<bool> nodes;
        for(unsigned node=0; node<m_nNodes[layer+1]; ++node){
          nodes.push_back(false);
        }  
        m_frozenNodes.push_back(nodes);
      }
    };
    
    /**
     * Set a layer number to use bias
     **/
    void setUsingBias(unsigned layer, bool status = true){
      m_usingBias[layer] = status;
    }

    /**
     * @brief Set neural network frozen nodes.
     **/
    bool setFrozenNode(unsigned layer, unsigned node, bool status=true){
      //Add some protections
      if(layer < 1 || layer > m_nNodes.size()){
        MSG_ERROR("Invalid layer for frozen status.");
        return false;
      }
      if(node > m_nNodes[layer]){
        MSG_ERROR("Invalid node for frozen status.");
        return false;
      }
      m_frozenNodes[layer][node] = status;
      return true;
    };

    /// Return frozen status
    bool isFrozenNode(unsigned layer, unsigned node) const 
    {
      return m_frozenNodes[layer][node];
    }

    /// Return the layer bias status. The dafault is true
    bool isUsingBias(unsigned layer) const 
    {
      return m_usingBias[layer];
    }

    /// Return the nNodes std::vector
    std::vector<unsigned> getNodes() const 
    {
      return m_nNodes;
    }

    /// Return the number of nodes into the layer.
    unsigned getNumberOfNodes(unsigned layer=0) const 
    {
      return m_nNodes[layer];
    }

    /// Return the max number of layer.
    unsigned getNumberOfLayers() const {
      return m_nNodes.size();
    }

    /// Return the tranfer function.
    std::string getTrfFuncStr(unsigned layer) const 
    {
      return m_trfFuncStr[layer];
    }
    
    /// Define get and setter for the properties
    OBJECT_SETTER_AND_GETTER(std::string,                 setTrainFcn,           getTrainFcn,           m_trainAlgFcn         );
    PRIMITIVE_SETTER_AND_GETTER(std::vector<std::string>, setTrfFunc,            getTrfFunc,            m_trfFuncStr          );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setLearningRate,       getLearningRate,       m_learningRate        );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setDecFactor,          getDecFactor,          m_decFactor           );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setDeltaMax,           getDeltaMax,           m_deltaMax            );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setDeltaMin,           getDeltaMin,           m_deltaMin            );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setIncEta,             getIncEta,             m_incEta              );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setDecEta,             getDecEta,             m_decEta              );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setInitEta,            getInitEta,            m_initEta             );
    PRIMITIVE_SETTER_AND_GETTER(unsigned,                 setBatchSize,          getBatchSize,          m_batchSize           );
    PRIMITIVE_SETTER_AND_GETTER(std::uint64_t,            setEpochs,             getEpochs,             m_nEpochs             );
    PRIMITIVE_SETTER_AND_GETTER(unsigned,                 setShow,               getShow,               m_show                );
    PRIMITIVE_SETTER_AND_GETTER(unsigned,                 setMaxFail,            getMaxFail,            m_maxFail             );
    PRIMITIVE_SETTER_AND_GETTER(unsigned,                 setMinEpochs,          getMinEpochs,          m_minEpochs           );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setROCResolution,      getROCResolution,      m_rocResolution       );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setSPSignalWeight,     getSPSignalWeight,     m_spSignalWeight      );
    PRIMITIVE_SETTER_AND_GETTER(REAL,                     setSPBackgroundWeight, getSPBackgroundWeight, m_spBackgroundWeight  );
};


}
#endif
