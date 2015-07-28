#include "FastNetTool/training/PatternRec.h"

//==============================================================================
PatternRecognition::PatternRecognition(
    FastNet::Backpropagation *net, vector<DataHandler<REAL>*> inTrn, 
    vector<DataHandler<REAL>*> inVal, vector<DataHandler<REAL>*>inTst,  
    const TrainGoal mode, const unsigned bSize,
    const REAL signalWeight, const REAL noiseWeight, Level msglevel) 
  : Training(net, bSize, msglevel), 
    trainGoal(mode)
{
  m_appName = "PatternRecognition";
  m_msgLevel = msglevel;
  m_log = new MsgStream(m_appName, m_msgLevel);

  MSG_DEBUG(m_log, "Starting a Pattern Recognition Training Object");
  
  // Initialize weights for SP calculation
  this->signalWeight = signalWeight;
  this->noiseWeight = noiseWeight;

  bool hasTstData = !inTst.empty();
  useSP = (trainGoal != MSE_STOP) ? true: false; 

  if (useSP) {
    bestGoalSP  = 0.;
    bestGoalDet = 0.;
    bestGoalFa  = 0.;
    if(trainGoal == SP_STOP) {
      MSG_DEBUG(m_log, "I'll use SP validating criterium.");
    }
    if(trainGoal == MULTI_STOP) {
      MSG_DEBUG(m_log, "I'll use SP, DET and FA validating criterium.");
    }
  } else {
    MSG_DEBUG(m_log, "I'll not use SP/DET/FA  validating criterium.");
  }
  
  numPatterns = inTrn.size();
  MSG_DEBUG(m_log, "Number of patterns: " << numPatterns);
  outputSize = (numPatterns == 2) ? 1 : numPatterns;
  
  // The last 2 parameters for the training case will not be used by the
  // function, so, there is no problem passing the corresponding validation
  // variables to this first function call.
  MSG_DEBUG(m_log, "Allocating memory for the training data.");
  allocateDataset(inTrn, true, inTrnList, epochValOutputs, numValEvents);
  MSG_DEBUG(m_log, "Allocating memory for the validation data.");
  allocateDataset(inVal, false, inValList, epochValOutputs, numValEvents);
  if (hasTstData) {
    MSG_DEBUG(m_log, "Allocating memory for the testing data.");
    allocateDataset(inTst, false, inTstList, epochTstOutputs, numTstEvents);
  }
  //Creating the targets for each class (maximum sparsed oututs).
  targList = new const REAL* [numPatterns];  
  for (unsigned i=0; i<numPatterns; i++)
  {
    REAL *target = new REAL [outputSize];
    for (unsigned j=0; j<outputSize; j++) target[j] = -1;
    target[i] = 1;
    //Saving the target in the list.
    targList[i] = target;    
  }
  
  MSG_DEBUG(m_log, "Input events dimension: " << inputSize);
  MSG_DEBUG(m_log, "Output events dimension: " << outputSize);
}

//==============================================================================
void PatternRecognition::allocateDataset(
    vector<DataHandler<REAL>*> dataSet, const bool forTrain, 
    const REAL **&inList, REAL **&out, unsigned *&nEv)
{
  inList = new const REAL* [numPatterns];

  if (!forTrain)
  {
    nEv = new unsigned [numPatterns];
    if (useSP) out = new REAL* [numPatterns];
  }
  
  for (unsigned i=0; i<numPatterns; i++)
  {
    DataHandler<REAL> *patData = dataSet[i];
    inputSize = patData->getNumCols();
    inList[i] = patData->getPtr();

    if (forTrain)
    {
      dmTrn.push_back(new DataManager(patData->getNumRows()));
      MSG_DEBUG(m_log, "Number of events for pattern " << i << ":" << patData->getNumRows());
    } else {
      nEv[i] = static_cast<unsigned>(patData->getNumRows());
      if (useSP) out[i] = new REAL [nEv[i]];
      MSG_DEBUG(m_log, "Number of events for pattern " << i << ":" << nEv[i]);
    }
  }


}

//==============================================================================
void PatternRecognition::deallocateDataset( const bool forTrain, 
    const REAL **&inList, 
    REAL **&out, 
    unsigned *&nEv)
{
  for (unsigned i=0; i<numPatterns; i++)
  {
    if (forTrain) delete dmTrn[i];
    else if (useSP) delete [] out[i];
  }

  delete [] inList;
  if (!forTrain)
  {
    delete [] nEv;
    if (useSP) delete [] out;
  }
}

//==============================================================================
PatternRecognition::~PatternRecognition()
{
  // The last 2 parameters for the training case will not be used by the
  // function, so, there is no problem passing the corresponding validation
  // variables to this first function call.
  deallocateDataset(true, inTrnList, epochValOutputs, numValEvents);
  deallocateDataset(false, inValList, epochValOutputs, numValEvents);
  if (hasTstData) {
    deallocateDataset(false, inTstList, epochTstOutputs, numTstEvents);
  }
  for (unsigned i=0; i<numPatterns; i++) delete [] targList[i];
  delete [] targList;
}


//==============================================================================
REAL PatternRecognition::sp(const unsigned *nEvents, 
    REAL **epochOutputs, 
    REAL &det,  
    REAL &fa )
{
  unsigned TARG_SIGNAL, TARG_NOISE;
  
  // We consider that our signal has target output +1 and the noise, -1. So, the
  // if below help us figure out which class is the signal.
  if (targList[0][0] > targList[1][0]) // target[0] is our signal.
  {
    TARG_NOISE = 1;
    TARG_SIGNAL = 0;    
  }
  else //target[1] is the signal.
  {
    TARG_NOISE = 0;
    TARG_SIGNAL = 1;
  }

  const REAL *signal = epochOutputs[TARG_SIGNAL];
  const REAL *noise = epochOutputs[TARG_NOISE];
  const REAL signalTarget = targList[TARG_SIGNAL][0];
  const REAL noiseTarget = targList[TARG_NOISE][0];
  const int numSignalEvents = static_cast<int>(nEvents[TARG_SIGNAL]);
  const int numNoiseEvents = static_cast<int>(nEvents[TARG_NOISE]);
  const REAL RESOLUTION = 0.01;
  REAL maxSP = -1.;
  int i;
  int chunk = chunkSize;


  for (REAL pos = noiseTarget; pos < signalTarget; pos += RESOLUTION)
  {
    REAL sigEffic = 0.;
    REAL noiseEffic = 0.;
    unsigned se, ne;
    
    #pragma omp parallel shared(signal, noise, sigEffic, noiseEffic) \
      private(i,se,ne)
    {
      se = ne = 0;
      
      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numSignalEvents; i++) if (signal[i] >= pos) se++;
      
      #pragma omp critical
      sigEffic += static_cast<REAL>(se);

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numNoiseEvents; i++) if (noise[i] < pos) ne++;
      
      #pragma omp critical
      noiseEffic += static_cast<REAL>(ne);
    }
    
    sigEffic /= static_cast<REAL>(numSignalEvents);
    noiseEffic /= static_cast<REAL>(numNoiseEvents);

    // Use weights for signal and noise efficiencies
    sigEffic *= signalWeight;
    noiseEffic *= noiseWeight;

    //Using normalized SP calculation.
    const REAL sp = ((sigEffic + noiseEffic) / 2) * sqrt(sigEffic * noiseEffic);
    if (sp > maxSP){
      maxSP = sp;
      det = sigEffic;
      fa  = 1 - noiseEffic;
    }
  }
  
  return sqrt(maxSP); // This sqrt is so that the SP value is in percent.
}


//==============================================================================
void PatternRecognition::getNetworkErrors(
    const REAL **inList, 
    const unsigned *nEvents,
    REAL **epochOutputs, 
    REAL &mseRet, 
    REAL &spRet, 
    REAL &detRet, 
    REAL &faRet)
{
  REAL gbError = 0.;
  FastNet::Backpropagation **nv = netVec;
  int totEvents = 0;
  
  for (unsigned pat=0; pat<numPatterns; pat++)
  {
    totEvents += nEvents[pat];
 
    const REAL *target = targList[pat];
    const REAL *input = inList[pat];
    const REAL *output;
    const int numEvents = nEvents[pat];
    REAL error = 0.;
    int i, thId;
    int chunk = chunkSize;

    REAL *outList = (useSP) ? epochOutputs[pat] : NULL;
    
    MSG_DEBUG(m_log, "Applying performance calculation for pattern " 
        << pat << " (" << numEvents << " events).");
    
    #pragma omp parallel shared(input,target,chunk,nv,gbError,pat) \
      private(i,thId,output,error)
    {
      thId = omp_get_thread_num();
      error = 0.;

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numEvents; i++)
      {
        error += nv[thId]->applySupervisedInput(&input[i*inputSize], 
            target, 
            output);
        if (useSP) outList[i] = output[0];
      }

      #pragma omp critical
      gbError += error;
    }
  }

  mseRet = gbError / static_cast<REAL>(totEvents);
  if (useSP)  spRet = sp(nEvents, epochOutputs, detRet, faRet);
}


//==============================================================================
REAL PatternRecognition::trainNetwork()
{
  MSG_DEBUG(m_log, "Starting training process for an epoch.");
  REAL gbError = 0;
  FastNet::Backpropagation **nv = netVec;
  int totEvents = 0; // Holds the amount of events presented to the network.

  for(unsigned pat=0; pat<numPatterns; pat++)
  {
    // wFactor will allow each pattern to have the same relevance, despite the
    // number of events it contains.
    const REAL *target = targList[pat];
    const REAL *input = inTrnList[pat];
    const REAL *output;
    REAL error = 0.;
    int i, thId;
    int chunk = chunkSize;
    unsigned pos = 0;
    DataManager *dm = dmTrn[pat];

    const int nEvents = (batchSize) ? batchSize : dm->size();
    totEvents += nEvents;
    MSG_DEBUG(m_log, "Applying training set for pattern " 
        << pat << " by randomly selecting " 
        << nEvents << " events (out of " << dm->size() << ").");
   
    #pragma omp parallel shared(input,target,chunk,nv,gbError,pat,dm) \
      private(i,thId,output,error,pos)
    {
      thId = omp_get_thread_num();
      error = 0.;

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<nEvents; i++)
      {
        #pragma omp critical
        pos = dm->get();

        error += nv[thId]->applySupervisedInput(
            &input[pos*inputSize], 
            target, 
            output);

        //Calculating the weight and bias update values.
        nv[thId]->calculateNewWeights(output, target);
      }

      #pragma omp critical
      gbError += error;
    }
  }

  updateGradients();
  updateWeights();
  return (gbError / static_cast<REAL>(totEvents));
}
  

//==============================================================================
void PatternRecognition::showInfo(const unsigned nEpochs) const
{
  MSG_INFO(m_log, "TRAINING DATA INFORMATION "
      "(Pattern Recognition Optimized Network)");
  MSG_INFO(m_log, "Number of Epochs          : " 
      << nEpochs);
  MSG_INFO(m_log, "Using SP  Stopping Criteria      : " 
      << (((trainGoal == SP_STOP) || (trainGoal == MULTI_STOP))  
        ? "true" : "false"));
  MSG_INFO(m_log, "Using DET Stopping Criteria      : " 
      << ((trainGoal == MULTI_STOP)  ? "true" : "false"));
  MSG_INFO(m_log, "Using FA  Stopping Criteria      : " 
      << ((trainGoal == MULTI_STOP)  ? "true" : "false"));
}

//==============================================================================
void PatternRecognition::isBestNetwork(
    const REAL currMSEError, const REAL currSPError, const REAL currDetError, 
    const REAL currFaError, ValResult &isBestMSE, ValResult &isBestSP,
    ValResult &isBestDet,   ValResult &isBestFa)
{
  // Knowing whether we have a better network, according to the MSE validation
  // criterium.
  Training::isBestNetwork(
      currMSEError, currSPError, currDetError, currFaError,  
      isBestMSE, isBestSP, isBestDet, isBestFa);

  // Knowing whether we have a better network, accorting to: SP, Detection or
  // false alarm
  isBestGoal( currSPError,   bestGoalSP  , isBestSP);
  isBestGoal( currDetError,  bestGoalDet , isBestDet);
  isBestGoal( currFaError,   bestGoalFa  , isBestFa);
}
  

//==============================================================================
void PatternRecognition::showTrainingStatus(
    const unsigned epoch, const REAL mseTrn, const REAL mseVal, const REAL spVal, 
    const REAL mseTst, const REAL spTst, const int stopsOn)
{
  if(trainGoal == SP_STOP) {
    MSG_INFO(m_log, "Epoch " << setw(5) << epoch 
        << ": mse (train) = " << mseTrn 
        << " SP (val) = " << spVal 
        << " SP (tst) = " << spTst);
  } else if (trainGoal == MULTI_STOP) {
    MSG_INFO(m_log, "Epoch " << setw(5) << epoch 
        << ": mse (train) = " << mseTrn 
        << " SP (val) = " << spVal 
        << " SP (tst) = " << spTst 
        << " stops = "<< stopsOn );
  } else {
    Training::showTrainingStatus(epoch, 
        mseTrn, mseVal, spVal, mseTst, spTst, 
        stopsOn);
  }
}

//==============================================================================
void PatternRecognition::showTrainingStatus(
    const unsigned epoch, 
    const REAL mseTrn,  const REAL mseVal, const REAL spVal, 
    const int stopsOn)
{
  if (trainGoal == SP_STOP) {
    MSG_INFO(m_log, "Epoch " << setw(5) << epoch 
        << ": mse (train) = " << mseTrn << " SP (val) = " << spVal );
  } else if (trainGoal == MULTI_STOP) {
    MSG_INFO(m_log, "Epoch " << setw(5) << epoch 
        << ": mse (train) = " << mseTrn << " SP (val) = " << spVal 
        << " stops = "<< stopsOn );
  } else {
    Training::showTrainingStatus( epoch, mseTrn, mseVal, spVal, stopsOn );
  }
}

