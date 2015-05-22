
#include "FastNetTool/neuralnetwork/NeuralNetwork.h"
#include "FastNetTool/system/util.h"

#include <iostream>
#include <new>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>

using namespace std;
using namespace msg;

namespace FastNet
{
  NeuralNetwork::NeuralNetwork( INeuralNetwork *net, Level msglevel ):m_msgLevel(msglevel)
  {
    ///Application name is set by default to MsgStream monitoring
    m_appName  = "NeuralNetwork";

    ///Get nodes configuration from interface
    nNodes     = net->getNodes();

    // alloc MsgStream manager
    m_log        = new MsgStream(m_appName, m_msgLevel);
    
    MSG_DEBUG(m_log, "Number of nodes in layer 0 " << nNodes[0] );
    for (size_t i=0; i< nNodes.size()-1; i++)
    {
      MSG_DEBUG(m_log, "Number of nodes in layer " << (i+1) << ": " << nNodes[(i+1)]);
      const string transFunction = net->getTrfFuncStr(i);
      this->usingBias.push_back(net->isUsingBias(i));
      MSG_DEBUG(m_log, "Layer " << (i+1) << " is using bias? " << this->usingBias[i]);

      if (transFunction == TGH_ID)
      {
        this->trfFunc.push_back(&NeuralNetwork::hyperbolicTangent);
        MSG_DEBUG(m_log, "Transfer function in layer " << (i+1) << ": tanh");
      }
      else if (transFunction == LIN_ID)
      {
        this->trfFunc.push_back(&NeuralNetwork::linear);
        MSG_DEBUG(m_log, "Transfer function in layer " << (i+1) << ": purelin");
      }
      else throw "Transfer function not specified!";
    }

    //Allocating the memory for the values.
    try {allocateSpace(nNodes);}
    catch (bad_alloc xa) {throw;}
    
    // This will be a pointer to the input event.
    layerOutputs[0] = NULL;
    
    //Init weights and bias vector
    initWeights();
    
    MSG_DEBUG(m_log, "NeuralNetwork class was created.");
  }
  
  NeuralNetwork::NeuralNetwork(const NeuralNetwork &net)
  {
    m_appName = "NeuralNetwork";
    m_log = new MsgStream(m_appName , net.getMsgLevel() );
    //Allocating the memory for the values.
    try {allocateSpace(net.nNodes);}
    catch (bad_alloc xa) {throw;}
    (*this) = net;
  }


  void NeuralNetwork::operator=(const NeuralNetwork &net)
  {
    nNodes.clear();
    usingBias.clear();
    trfFunc.clear();
    nNodes.assign(net.nNodes.begin(), net.nNodes.end());
    usingBias.assign(net.usingBias.begin(), net.usingBias.end());
    trfFunc.assign(net.trfFunc.begin(), net.trfFunc.end());
      
    layerOutputs[0] = net.layerOutputs[0]; // This will be a pointer to the input event.
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      memcpy(bias[i], net.bias[i], nNodes[i+1]*sizeof(REAL));
      memcpy(layerOutputs[i+1], net.layerOutputs[i+1], nNodes[i+1]*sizeof(REAL));

      for (unsigned j=0; j<nNodes[i+1]; j++) memcpy(weights[i][j], net.weights[i][j], nNodes[i]*sizeof(REAL));
    } 
  }


  void NeuralNetwork::initWeights()
  {
    //Processing layers and init weights
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      for (unsigned j=0; j<nNodes[(i+1)]; j++)
      {
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          weights[i][j][k] = static_cast<REAL>( util::rand_float_range(-1, 1)  );
        }
        bias[i][j] = (usingBias[i]) ? static_cast<REAL>( util::rand_float_range(-1,1) ) : 0.;
      }
    }

    //Apply Nguyen-Widrow weight initialization algorithm
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      float beta = 0.7*pow((float) nNodes[i], (float) 1/nNodes[0]);
      for (unsigned j=0; j<nNodes[(i+1)]; j++)
      {
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          weights[i][j][k] = beta*weights[i][j][k]*util::get_norm_of_weight(weights[i][j], nNodes[i]);
        }
        bias[i][j] = beta * bias[i][j] * bias[i][j];
      }
    }

  }


  void NeuralNetwork::allocateSpace(const vector<unsigned> &nNodes)
  {
    MSG_DEBUG(m_log, "Allocating all the space that the NeuralNetwork class will need.");
    try
    {
      layerOutputs = new REAL* [nNodes.size()];
      layerOutputs[0] = NULL; // This will be a pointer to the input event.
    
      const unsigned size = nNodes.size()-1;
      
      bias = new REAL* [size];
      weights = new REAL** [size];

      for (unsigned i=0; i<size; i++)
      {
        bias[i] = new REAL [nNodes[i+1]];
        layerOutputs[i+1] = new REAL [nNodes[i+1]];
        weights[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++) weights[i][j] = new REAL [nNodes[i]];
      }
    }
    catch (bad_alloc xa)
    {
      MSG_FATAL(m_log, "bad alloc, abort!");
      throw;
    }   

    MSG_DEBUG(m_log, "good alloc space memory.");
  }
  

  NeuralNetwork::~NeuralNetwork()
  {
    MSG_DEBUG(m_log, "Releasing all memory allocated by NeuralNetwork.");
    const unsigned size = nNodes.size() - 1;

    // Deallocating the bias and weight matrices.
    releaseMatrix(bias);
    releaseMatrix(weights);
    
    // Deallocating the hidden outputs matrix.
    if (layerOutputs)
    {
      for (unsigned i=1; i<size; i++)
      {
        if (layerOutputs[i]) delete [] layerOutputs[i];
      }

      delete [] layerOutputs;
    }

    delete m_log;
  }
  
  
  void NeuralNetwork::showInfo() const
  {
    MSG_INFO(m_log, "NEURAL NETWORK CONFIGURATION INFO");
    MSG_INFO(m_log, "Number of Layers (including the input): " << nNodes.size());
    
    for (unsigned i=0; i<nNodes.size(); i++)
    {
      MSG_INFO(m_log, "\nLayer " << i << " Configuration:");
      MSG_INFO(m_log, "Number of Nodes   : " << nNodes[i]);
      
      if (i)
      {
        std::ostringstream aux;
        aux << "Transfer function : ";
        if (trfFunc[(i-1)] == (&NeuralNetwork::hyperbolicTangent)) aux << "tanh";
        else if (trfFunc[(i-1)] == (&NeuralNetwork::linear)) aux << "purelin";
        else aux << "UNKNOWN!";

        aux << "\nUsing bias        : ";
        if (usingBias[(i-1)]) aux << "true";
        else  aux << "false";
        MSG_INFO(m_log, aux.str());
      }      
    }
  }


  inline const REAL* NeuralNetwork::propagateInput(const REAL *input)
  {
    const unsigned size = (nNodes.size() - 1);

    //Placing the input. though we are removing the const' ness no changes are perfomed.
    layerOutputs[0] = const_cast<REAL*>(input);

    //Propagating the input through the network.
    for (unsigned i=0; i<size; i++)
    {
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        layerOutputs[i+1][j] = bias[i][j];
        
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          layerOutputs[i+1][j] += layerOutputs[i][k] * weights[i][j][k];
        }

        layerOutputs[i+1][j] = CALL_TRF_FUNC(trfFunc[i])(layerOutputs[i+1][j], false);
      }
    }
    
    //Returning the network's output.
    return layerOutputs[size];
  }
  

  void NeuralNetwork::releaseMatrix(REAL **b)
  {
    if (b)
    {
      for (unsigned i=0; i<(nNodes.size()-1); i++)
      {
        if (b[i]) delete [] b[i];
      }
      delete [] b;
      b = NULL;
    }
  }


  void NeuralNetwork::releaseMatrix(REAL ***w)
  {
    if (w)
    {
      for (unsigned i=0; i<(nNodes.size()-1); i++)
      {
        if (w[i])
        {
          for (unsigned j=0; j<nNodes[i+1]; j++)
          {
            if (w[i][j]) delete [] w[i][j];
          }
          delete [] w[i];
        }
      }
      delete [] w;
      w = NULL;
    }
  }


  void NeuralNetwork::setUsingBias(const unsigned layer, const bool val)
  {
    usingBias[layer] = val;
    
    //If not using layers, we assign the biases values
    //in the layer to 0.
    if(!usingBias[layer])
    {
      for (unsigned i=0; i<nNodes[(layer+1)]; i++)
      {
        bias[layer][i] = 0;
      }
    }
  }
}
