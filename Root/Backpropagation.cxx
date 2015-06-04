

#include <vector>
#include <string>
#include <cstdlib>
#include <typeinfo>
#include <sstream>

#include "FastNetTool/neuralnetwork/Backpropagation.h"

namespace FastNet
{
 
  Backpropagation::Backpropagation(INeuralNetwork *net, Level msglevel) : NeuralNetwork(net, msglevel), m_msgLevel(msglevel)
  {
    ///Application name is set by default to MsgStream monitoring
    m_appName  = "BackPropagation";

    // alloc MsgStream manager
    m_log        = new MsgStream(m_appName, m_msgLevel);
 
    //We first test whether the values exists, otherwise, we use default ones.
    this->learningRate = net->getLearningRate();
    this->decFactor = net->getDecFactor();

    try {allocateSpace(nNodes);}
    catch (bad_alloc xa) {throw;}


    //Verifying if there are frozen nodes and seting them.
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      // For the frozen nodes, we first initialize them all as unfrozen.
      setFrozen(i, false);
      
      /* 
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        MSG_DEBUG(m_log, "aki2");
        if (nNodes[i] < nNodes[(i+1)]) setFrozen(i, j, net.isFrozenNode(i,j) );
        else throw "Node to be frozen is invalid!";
      }*/

      //Initializing dw and db.
      for (unsigned j=0; j<nNodes[i+1]; j++) 
      {
        this->db[i][j] = 0.;
        this->sigma[i][j] = 0.;
        for (unsigned k=0; k<nNodes[i]; k++) this->dw[i][j][k] = 0.;
      }
    }

    MSG_DEBUG(m_log, "BackPropagation class was created.");
  }

  Backpropagation::Backpropagation(const Backpropagation &net) : NeuralNetwork(net)
  {
    m_appName   = "Backpropagation";
    m_msgLevel  = net.getMsgLevel();
    m_log = new MsgStream(m_appName , m_msgLevel );
    try {allocateSpace(net.nNodes);}
    catch (bad_alloc xa) {throw;}
    (*this) = net; 
    MSG_DEBUG(m_log, "Attributing all values using assignment operator for Backpropagation class");
  }


  Backpropagation& Backpropagation::operator=(const Backpropagation &net)
  { 

    NeuralNetwork::operator=(net);
   
    learningRate = net.learningRate;
    decFactor = net.decFactor;

    for (unsigned i=0; i<(nNodes.size() - 1); i++)
    {
      memcpy(frozenNode[i], net.frozenNode[i], nNodes[i+1]*sizeof(bool));
      memcpy(db[i], net.db[i], nNodes[i+1]*sizeof(REAL));
      memcpy(sigma[i], net.sigma[i], nNodes[i+1]*sizeof(REAL));
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        memcpy(dw[i][j], net.dw[i][j], nNodes[i]*sizeof(REAL));
      }
    }
    return *this;
  }
  

  void Backpropagation::allocateSpace(const vector<unsigned> &nNodes)
  {
    MSG_DEBUG(m_log, "Allocating all the space that the Backpropagation class will need.");
    const unsigned size = nNodes.size() - 1;
    try
    {
      frozenNode = new bool* [size];
      db = new REAL* [size];
      sigma = new REAL* [size];
      dw = new REAL** [size];
      for (unsigned i=0; i<size; i++)
      {
        frozenNode[i] = new bool [nNodes[i+1]];
        db[i] = new REAL [nNodes[i+1]];
        sigma[i] = new REAL [nNodes[i+1]];
        dw[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++)
        {
          dw[i][j] = new REAL [nNodes[i]];
        }
      }
    }
    catch (bad_alloc xa)
    {
      throw;
    }

    MSG_DEBUG(m_log, "good Alloc space memory.");
  }

  Backpropagation::~Backpropagation()
  {
    MSG_DEBUG(m_log, "Releasing all memory allocated by Backpropagation.");
    releaseMatrix(db);
    releaseMatrix(dw);
    releaseMatrix(sigma);

    // Deallocating the frozenNode matrix.
    if (frozenNode)
    {
      for (unsigned i=0; i<(nNodes.size()-1); i++) if (frozenNode[i]) delete [] frozenNode[i];
      delete [] frozenNode;
    }

    delete m_log;
  }

  void Backpropagation::retropropagateError(const REAL *output, const REAL *target)
  {
    const unsigned size = nNodes.size() - 1;

    for (unsigned i=0; i<nNodes[size]; i++) sigma[size-1][i] = (target[i] - output[i]) * CALL_TRF_FUNC(trfFunc[size-1])(output[i], true);

    //Retropropagating the error.
    for (int i=(size-2); i>=0; i--)
    {
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        sigma[i][j] = 0;

        for (unsigned k=0; k<nNodes[i+2]; k++)
        {
          sigma[i][j] += sigma[i+1][k] * weights[(i+1)][k][j];
        }

        sigma[i][j] *= CALL_TRF_FUNC(trfFunc[i])(layerOutputs[i+1][j], true);
      }
    }
  }
  

  void Backpropagation::calculateNewWeights(const REAL *output, const REAL *target)
  {
    const unsigned size = nNodes.size() - 1;

    retropropagateError(output, target);

    //Accumulating the deltas.
    for (unsigned i=0; i<size; i++)
    {
      for (unsigned j=0; j<nNodes[(i+1)]; j++)
      {
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          dw[i][j][k] += (sigma[i][j] * layerOutputs[i][k]);
        }

        db[i][j] += (sigma[i][j]);
      }
    }
  }


  void Backpropagation::addToGradient(const Backpropagation &net)
  {
    //Accumulating the deltas.
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      for (unsigned j=0; j<nNodes[(i+1)]; j++)
      {
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          dw[i][j][k] += net.dw[i][j][k];
        }
        db[i][j] += net.db[i][j];
      }
    }
  }

  void Backpropagation::updateWeights(const unsigned numEvents)
  {
    const REAL val = 1. / static_cast<REAL>(numEvents);
    
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      for (unsigned j=0; j<nNodes[(i+1)]; j++)
      {
        //If the node is frozen, we just reset the accumulators,
        //otherwise, we actually train the weights connected to it.
        if (frozenNode[i][j])
        {
          MSG_DEBUG(m_log, "Skipping updating node " << j << " from hidden layer " << i << ", since it is frozen!");
          for (unsigned k=0; k<nNodes[i]; k++) dw[i][j][k] = 0;
          if (usingBias[i]) db[i][j] = 0;
          else bias[i][j] = 0;
        }
        else
        {
          for (unsigned k=0; k<nNodes[i]; k++)
          {
            weights[i][j][k] += (learningRate * val * dw[i][j][k]);
            dw[i][j][k] = 0;
          }

          if (usingBias[i])
          {
            bias[i][j] += (learningRate * val * db[i][j]);
            db[i][j] = 0;
          }
          else
          {
            bias[i][j] = 0;
          }
        }
      }
    }
  }


  void Backpropagation::showInfo() const
  {
    NeuralNetwork::showInfo();
    MSG_INFO(m_log, "TRAINING ALGORITHM INFORMATION:");
    MSG_INFO(m_log, "Training algorithm : Gradient Descent");
    MSG_INFO(m_log, "Learning rate      : " << learningRate);
    MSG_INFO(m_log, "Decreasing factor  : " << decFactor);
        
    for (unsigned i=0; i<nNodes.size()-1; i++) 
    {
      std::ostringstream aux;
      aux << "Frozen Nodes in hidden layer " << i << ":";
      bool frozen = false;
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        if (frozenNode[i][j])
        {
          aux << " " << j;
          frozen = true;
        }
      }
      if (!frozen) aux << " NONE";
      MSG_INFO(m_log, aux.str());

    }
  }


  bool Backpropagation::isFrozen(unsigned layer) const
  {
    for (unsigned i=0; i<nNodes[layer+1]; i++)
    {
      if (!frozenNode[layer][i]) return false;
    }

    return true;
  }


  inline REAL Backpropagation::applySupervisedInput(const REAL *input, const REAL *target, const REAL* &output)
  {
    int size = (nNodes.size()-1);
    REAL error = 0;

    //Propagating the input.
    output = propagateInput(input);
      
    //Calculating the error.
    for (unsigned i=0; i<nNodes[size]; i++){
      error += SQR(target[i] - output[i]);
    }
    //Returning the MSE
    return (error / nNodes[size]);
  }


}
