#include <TuningTools/training/Training.h>

namespace TuningTool {

namespace {
  // FIXME: Change this to be a better distribution generator
  int rndidx (int i) { return std::rand()%i; }
}
//==============================================================================
DataManager::DataManager(const unsigned nEvents)
  : m_nEvents(nEvents)
{
  m_vec.reserve(m_nEvents);
  for (unsigned i=0; i<m_nEvents; i++) {
    m_vec.push_back(i);
  }
  random_shuffle(m_vec.begin(), m_vec.end(), rndidx );
  m_pos = m_vec.begin();
}

//==============================================================================
unsigned DataManager::get()
{
  if (m_pos == m_vec.end())
    shuffle();
  return *m_pos++;
}

//==============================================================================
void DataManager::shuffle()
{
  random_shuffle(m_vec.begin(), m_vec.end(), rndidx);
  m_pos = m_vec.begin();
}

//==============================================================================
void DataManager::print() const
{
  std::cout << "DataManager is : [";
  for ( unsigned cPos = 0; cPos < 10; ++cPos ) {
    std::cout << m_vec[cPos] << ",";
  } std::cout << "]" << std::endl;
}

//==============================================================================
Training::Training(TuningTool::Backpropagation *n
                    , const unsigned bSize
                    , const MSG::Level level )
  : IMsgService("Training", MSG::INFO ),
    MsgService( level )
{
  msg().width(5);
  m_batchSize = bSize;
  
  int nt = 1;
#ifdef USE_OMP
  #pragma omp parallel shared(nt)
  {
    #pragma omp master
    nt = omp_get_num_threads();
  }
#endif

  m_nThreads = static_cast<unsigned>(nt);
  m_chunkSize = static_cast<int>(std::ceil(static_cast<float>(m_batchSize) 
                               / static_cast<float>(m_nThreads)));

  m_netVec = new TuningTool::Backpropagation* [m_nThreads];

  MSG_DEBUG("Cloning training neural network " << m_nThreads 
      << " times (one for each thread).")
  m_mainNet = m_netVec[0] = n;
  for (unsigned i=1; i<m_nThreads; i++)
  {
    m_netVec[i] = new TuningTool::Backpropagation(*n);
    m_netVec[i]->setName(m_netVec[i]->getName() + "_Thread[" + 
        std::to_string(i) + "]" );
  }
}
    
//==============================================================================
Training::~Training()
{
  MSG_DEBUG("Releasing training algorithm extra threads (" << m_nThreads - 1
      << ") neural networks ")
  for (unsigned i=1; i<m_nThreads; i++) {
    delete m_netVec[i]; m_netVec[i] = nullptr;
  }
  for ( auto& trainData : m_trnEvolution ) {
    delete trainData; trainData = nullptr;
  }
  delete m_netVec;
};

} // namespace TuningTool

