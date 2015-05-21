#ifndef FASTNETTOOL_FEEDFORWARD_H
#define FASTNETTOOL_FEEDFORWARD_H

#include <vector>
#include <cstring>

#include "FastNetTool/system/defines.h"
#include "FastNetTool/system/MsgStream.h"
#include "FastNetTool/neuralnetwork/NeuralNetwork.h"
#include "FastNetTool/neuralnetwork/INeuralNetwork.h"

using namespace std;
using namespace msg;

namespace FastNet
{
  /** 
  This class should be used for network production, when no training is necessary,
  just feedforward the incoming events, fot output collection.
  */
  class FeedForward : public NeuralNetwork 
  {

    private:

      ///Name of the aplication
      string        m_appName;
      ///Hold the output level that can be: verbose, debug, info, warning or
      //fatal. This will be administrated by the MsgStream Class manager.
      Level         m_msgLevel;
      /// MsgStream for monitoring
      MsgStream     *m_log;
    
    public:

      ///Copy constructor
      /**This constructor should be used to create a new network which is an exactly copy 
        of another network.
        @param[in] netÊThe network that we will copy the parameters from.
      */
      FeedForward(const FeedForward &net);

      /// Constructor taking the parameters for a matlab net structure.
      /**
      This constructor should be called when the network parameters are stored in a matlab
      network structure.
      @param[in] netStr The Matlab network structure as returned by newff.
      */
      FeedForward(INeuralNetwork *net, Level msglevel);

      /// Returns a clone of the object.
      /**
      Returns a clone of the calling object. The clone is dynamically allocated,
      so it must be released with delete at the end of its use.
      @return A dynamically allocated clone of the calling object.
      */
      virtual NeuralNetwork *clone();

      /// Class destructor.
      /**
       Releases all the dynamically allocated memory used by the class.
      */
      virtual ~FeedForward();

      /// Get name from MsgStream Manager
      string getAppName() const {return m_appName;};

      /// Get Level of output from MsgStream Manager
      Level getMsgLevel() const {return m_msgLevel;};
 
  
  };
}

#endif
