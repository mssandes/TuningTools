#include "FastNetTool/neuralnetwork/FeedForward.h"

namespace FastNet
{

//==============================================================================
FeedForward::FeedForward(const FeedForward &net) 
  : IMsgService("FeedForward"),
    NeuralNetwork(net){;}

//==============================================================================
FeedForward::FeedForward(const NetConfHolder &net, 
                         const MSG::Level msglevel, 
                         const std::string &name) 
  : IMsgService("FeedForward"),
    NeuralNetwork(net, msglevel, name){;}

//==============================================================================
NeuralNetwork *FeedForward::clone()
{
  return new FeedForward(*this);
}

//==============================================================================
FeedForward::~FeedForward() {;}

}
