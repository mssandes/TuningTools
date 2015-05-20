
#include "FastNetTool/training/Standard.h"

StandardTraining::StandardTraining(FastNet::Backpropagation *net, const DataHandler<REAL> *inTrn, const DataHandler<REAL> *outTrn, 
                                   const DataHandler<REAL> *inVal, const DataHandler<REAL> *outVal, 
                                   const unsigned bSize, Level msglevel) : Training(net, bSize, msglevel), m_msgLevel(msglevel)
{
  m_appName = "StandartTraining";
  m_log = new MsgStream(m_appName, m_msgLevel);

  MSG_DEBUG(m_log, "Creating StandardTraining object.");
 
  if ( inTrn->getNumRows() != inVal->getNumRows() ) throw "Input training and validating events dimension does not match!";
  if ( outTrn->getNumRows() != outVal->getNumRows() ) throw "Output training and validating events dimension does not match!";
  if ( inTrn->getNumCols() != outTrn->getNumCols() ) throw "Number of input and target training events does not match!";
  if ( inVal->getNumCols() != outVal->getNumCols() ) throw "Number of input and target validating events does not match!";

  inTrnData  = static_cast<const REAL*>(inTrn->getPtr());
  outTrnData = static_cast<const REAL*>(outTrn->getPtr());
  inValData  = static_cast<const REAL*>(inVal->getPtr());
  outValData = static_cast<const REAL*>(outVal->getPtr());
  inputSize  = inTrn->getNumRows();
  outputSize = outTrn->getNumRows();
 
  dmTrn = new DataManager( inTrn->getNumCols() );
  numValEvents = inVal->getNumCols();
}

StandardTraining::~StandardTraining()
{
  delete m_log;
  delete dmTrn;
}

void StandardTraining::valNetwork(REAL &mseVal, REAL &spVal)
{
  REAL gbError = 0.;
  REAL error = 0.;
  const REAL *output;

  const REAL *input = inValData;
  const REAL *target = outValData;
  const int numEvents = static_cast<int>(numValEvents);
  
  int chunk = chunkSize;
  int i, thId;
  FastNet::Backpropagation **nv = netVec;

  #pragma omp parallel shared(input,target,chunk,nv,gbError) private(i,thId,output,error)
  {
    thId = omp_get_thread_num();
    error = 0.;

    #pragma omp for schedule(dynamic,chunk) nowait
    for (i=0; i<numEvents; i++)
    {
      error += nv[thId]->applySupervisedInput(&input[i*inputSize], &target[i*outputSize], output);
    }

    #pragma omp critical
    gbError += error;
  }
  
  mseVal = gbError / static_cast<REAL>(numEvents);
}


REAL StandardTraining::trainNetwork()
{
  unsigned pos;
  REAL gbError = 0.;
  REAL error = 0.;
  const REAL *output;

  const REAL *input = inTrnData;
  const REAL *target = outTrnData;

  int chunk = chunkSize;
  int i, thId;
  FastNet::Backpropagation **nv = netVec;
  DataManager *dm = dmTrn;
  const int nEvents = (batchSize) ? batchSize : dm->size();

  #pragma omp parallel shared(input,target,chunk,nv,gbError,dm) private(i,thId,output,error,pos)
  {
    thId = omp_get_thread_num(); 
    error = 0.;

    #pragma omp for schedule(dynamic,chunk) nowait
    for (i=0; i<nEvents; i++)
    {
        #pragma omp critical
        pos = dm->get();
        
        error += nv[thId]->applySupervisedInput(&input[pos*inputSize], &target[pos*outputSize], output);
        nv[thId]->calculateNewWeights(output, &target[pos*outputSize]);
    }

    #pragma omp critical
    gbError += error;    
  }

  updateGradients();
  updateWeights();
  return (gbError / static_cast<REAL>(nEvents));
}

  
void StandardTraining::showInfo(const unsigned nEpochs) const
{
  MSG_INFO(m_log, "TRAINING DATA INFORMATION (Standard Network)");
  MSG_INFO(m_log, "Number of Epochs          : " << nEpochs);
}
