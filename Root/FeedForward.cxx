#include "FastNetTool/neuralnetwork/FeedForward.h"

using namespace std;

namespace FastNet
{
    FeedForward::FeedForward(const FeedForward &net) : NeuralNetwork(net){}
    FeedForward::FeedForward(INeuralNetwork &net, Level msglevel) : NeuralNetwork(net, msglevel){}
    NeuralNetwork *FeedForward::clone(){return new FeedForward(*this);}      
    FeedForward::~FeedForward() {}
}
