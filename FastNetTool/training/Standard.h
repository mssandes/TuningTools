#ifndef FASTNETTOOL_TRAINING_STANDARD_H
#define FASTNETTOOL_TRAINING_STANDARD_H

#include "FastNetTool/training/Training.h"
#include "FastNetTool/system/DataHandler.h"
#include "FastNetTool/system/MsgStream.h"
#include "FastNetTool/system/defines.h"

using namespace msg;

class StandardTraining : public Training
{
  private:
    ///Name of the aplication
    string        m_appName;
    ///Hold the output level that can be: verbose, debug, info, warning or
    //fatal. This will be administrated by the MsgStream Class manager.
    Level         m_msgLevel;
    /// MsgStream for monitoring
    MsgStream     *m_log;

  protected:
    const REAL *inTrnData;
    const REAL *outTrnData;
    const REAL *inValData;
    const REAL *outValData;
    unsigned inputSize;
    unsigned outputSize;
    unsigned numValEvents;
    DataManager *dmTrn;
  
  public:
    StandardTraining(FastNet::Backpropagation *net, const DataHandler<REAL> *inTrn,  const DataHandler<REAL> *outTrn, 
                          const DataHandler<REAL> *inVal, const DataHandler<REAL> *outVal, const unsigned bSize, Level msglevel);
  
    virtual ~StandardTraining();
    
    virtual void tstNetwork(REAL &mseTst, REAL &spTst){mseTst = spTst = 0.;};
  
  
    /// Applies the validating set for the network's validation.
    /**
    This method takes the one or more validating events (input and targets) and presents them
    to the network. At the end, the mean training error is returned. Since it is a validating function,
    the network is not modified, and no updating weights values are calculated. This method only
    presents the validating sets and calculates the mean validating error obtained.
    of this class are not modified inside this method, since it is only a network validating process.
    @return The mean validating error obtained after the entire training set is presented to the network.
    */
    virtual void valNetwork(REAL &mseVal, REAL &spVal);
  
  
    /// Applies the training set for the network's training.
    /**
    This method takes the one or more training events (input and targets) and presents them
    to the network, calculating the new mean (if batch training is being used) update values 
    after each input-output pair is presented. At the end, the mean training error is returned.
    @return The mean training error obtained after the entire training set is presented to the network.
    */
    virtual REAL trainNetwork();
    
    virtual void showInfo(const unsigned nEpochs) const;
};

#endif
