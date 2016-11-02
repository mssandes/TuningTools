#include "TuningTools/training/TuningUtil.h"

// Package include(s)
#include <TuningTools/system/util.h>
#include <TuningTools/neuralnetwork/NeuralNetwork.h>

// STL include(s)
#include <algorithm>

namespace TuningTool {

using Type::Dataset;
using Type::benchmark;

//==============================================================================
void ROC::execute(const std::vector<REAL> &signal, 
                  const std::vector<REAL> &background,
                  ROC::Performance& rocPerf)
{
  const unsigned nSignal = signal.size();
  const unsigned nBackground  = background.size();
  const REAL* sPtr = signal.data();
  const REAL* nPtr = background.data();
  auto &sp = rocPerf.sp; auto &pd = rocPerf.pd;
  auto &pf = rocPerf.sp; auto &cut = rocPerf.cut;
  sp.reserve( m_nPoints ); pd.reserve( m_nPoints );
  pf.reserve( m_nPoints ); cut.reserve( m_nPoints );

#ifdef USE_OMP
  int chunk = 1000;
#endif

  for (REAL pos = m_backgroundTarget; pos < m_signalTarget; pos += m_resolution)
  {
    REAL sigEffic(0.), backgroundEffic (0.);
    unsigned se, ne, i(0);
#ifdef USE_OMP
    #pragma omp parallel shared(sPtr, nPtr, sigEffic, backgroundEffic) private(i,se,ne)
#endif
    {
      se = ne = 0;
#ifdef USE_OMP
      #pragma omp for schedule(dynamic,chunk) nowait
#endif
      for (i=0; i<nSignal; i++) if (sPtr[i] >= pos) ++se;
      
#ifdef USE_OMP
      #pragma omp critical
#endif
      sigEffic += static_cast<REAL>(se);

#ifdef USE_OMP
      #pragma omp for schedule(dynamic,chunk) nowait
#endif
      for (i=0; i<nBackground; i++) if (nPtr[i] < pos) ++ne;
      
#ifdef USE_OMP
      #pragma omp critical
#endif
      backgroundEffic += static_cast<REAL>(ne);
    }
    
    sigEffic /= static_cast<REAL>(nSignal);
    backgroundEffic /= static_cast<REAL>(nBackground);

    // Use weights for signal and background efficiencies
    sigEffic *= m_signalWeight;
    backgroundEffic *= m_backgroundWeight;

    //Using normalized SP calculation.
    sp.push_back( util::calcSP( sigEffic, backgroundEffic ) );
    pd.push_back( sigEffic );
    pf.push_back( 1-backgroundEffic );
    cut.push_back( pos );
  }
}

//==============================================================================
ROC &ROC::operator=(const ROC &roc)
{
  if (this!=&roc)
  {
    this->~ROC();
    new (this) ROC(roc);
  }
  return *this;
}

//==============================================================================
void DatasetPerformanceEvolution::addPerf( const ROC::setpoint &perf ) 
{
  if ( m_refType != benchmark::MSE ){
    m_spEvo.push_back(perf.sp); m_pdEvo.push_back(perf.pd);
    m_pfEvo.push_back(perf.pf); m_cutEvo.push_back(perf.cut);
  }
}

//==============================================================================
void DatasetPerformanceEvolution::addPerf( const REAL mseVal ) 
{
  if ( m_refType != benchmark::MSE ){
    m_mseEvo.push_back(mseVal);
  }
}

//==============================================================================
REAL DatasetPerformanceEvolution::getSP() const
{
  return m_spEvo.at(m_epoch);
}

//==============================================================================
REAL DatasetPerformanceEvolution::getPd() const
{
  return m_pdEvo.at(m_epoch);
}

//==============================================================================
REAL DatasetPerformanceEvolution::getPf() const
{
  return m_pfEvo.at(m_epoch);
}

//==============================================================================
REAL DatasetPerformanceEvolution::getCut() const
{
  return m_cutEvo.at(m_epoch);
}

//==============================================================================
REAL DatasetPerformanceEvolution::getMSE() const
{
  return m_mseEvo.at(m_epoch);
}


//==============================================================================
PerformanceEvolution::PerformanceEvolution( const TuningReferenceContainer& refs
    , const uint64_t nEpochs
    , const uint64_t epoch
    , const bool hasTstData )
  : m_savedEpoch{ refs.size(), 0 }
  , m_epoch{ epoch }
  , m_hasTstData{ hasTstData }
{
  m_perfEvoCol.reserve( refs.size() );
  Type::Dataset lastDS = Type::getLastDS(hasTstData);
  for ( const auto ref : refs ) {
    PerformanceEvolutionRawContainer rawCont; rawCont.reserve( static_cast<unsigned>(lastDS) + 1 );
    for ( unsigned ds = static_cast<unsigned>(Type::Dataset::Train)
        ; ds <= static_cast<unsigned>(lastDS)
        ; ++ds )
    {
      rawCont.push_back( 
                         DatasetPerformanceEvolution{ nEpochs
                                                    , epoch
                                                    , static_cast<Type::Dataset>(ds)
                                                    , ref.refType() 
                                                    } 
                       );
    } m_perfEvoCol.push_back( rawCont );
  }
}

//==============================================================================
TuningReference::TuningReference( const std::string& name 
                                , const Type::benchmark refType
                                , const REAL refVal
                                , const REAL maxRefDelta )
  : m_refType{ refType }
  , m_refVal{ refVal }
  , m_maxRefDelta{ maxRefDelta }
{
  setName( name );
}

//==============================================================================
void TuningReference::setName( const std::string &name )
{
  this->MsgService::setLogName( name + "_Reference" );
  m_name = name;
}

//==============================================================================
// TODO This method should receive instead an std::vector of references
// and the conversion let to the TuningWrapper
TuningReferenceContainer::TuningReferenceContainer( const py::list& references, 
                          MSG::Level msglevel, 
                          bool useColor )
  : IMsgService( "TuningReferenceContainer" ),
    MsgService( msglevel, useColor )
{
  // Before releasing, make sure that all references are ok!
  bool error{false};
  for( unsigned idx = 0; idx < py::len( references ); ++idx )
  {
    const auto &item = references[idx];
    // Extract our array:
    py::extract< py::dict > extractor( item );
    if ( extractor.check() ) {
      const auto &dictObj = static_cast<py::dict>(extractor());
      if( !dictObj.has_key("name") ) { error = true; MSG_ERROR("Dict has no 'name' key!") };
      if( !dictObj.has_key("reference") ) { error = true; MSG_ERROR("Dict has no 'reference' key!") };
      if( !dictObj.has_key("refVal") ) { error = true; MSG_ERROR("Dict has no 'refVal' key!") };
      if( !dictObj.has_key("maxRefDelta") ) { error = true; MSG_ERROR("Dict has no 'maxRefDelta' key!") };
      if( error ) throw WrongDictError();
    } else {
      // We shouldn't be retrieving this, warn user:
      MSG_WARNING( "Input a list with an object on position "
          << idx
          << " which is not a dict object (in fact it is of type: "
          << py::extract<std::string>(
              item.attr("__class__").attr("__name__"))()
          << ")." );
      return;
    }
    ++idx;
  }

  // Release old references
  m_vec.reserve( py::len(references) );

  // Loop over list and check for elements in which we can extract:
  for( unsigned idx = 0; idx < py::len( references ); ++idx )
  {
    const auto &item = references[idx];
    // Extract our python reference values and put it into a c++ object:
    py::extract< py::dict > extractor( item );
    const auto &dictObj = static_cast<py::dict>(extractor());
    // retrieve the information
    std::string name = py::extract<std::string>( dictObj.attr("name") );
    Type::benchmark refType = static_cast<Type::benchmark>( py::extract<unsigned int>( dictObj.attr("reference") )() );
    REAL refVal = py::extract<float>( dictObj.attr("refVal") );
    REAL maxRefDelta = py::extract<float>( dictObj.attr("maxRefDelta") );
    switch ( refType ){
      case Type::benchmark::MSE :
      {
        if ( m_usesMSE ) {
          MSG_FATAL( "Attempted to set MSE stop twice!" );
        } else {
          m_usesMSE = true;
        }
        break;
      }
      case Type::benchmark::SP :
      {
        if ( m_usesSP ) {
          MSG_FATAL( "Attempted to set SP stop twice!" );
        } else {
          m_usesSP = true;
        }
        break;
      }
      default:
        break;
    }
    // and add it to our references:
    m_vec.push_back( 
                     TuningReference{name
                                    , refType
                                    , refVal
                                    , maxRefDelta} 
                   );
    // log what happened:
    msg() << MSG::INFO << "Retrieved TuningReference ( " << name << ")";
    if ( refType == Type::benchmark::Pd || refType == Type::benchmark::Pf ) {
      msg() << " Its parameters are, value: " << refVal << " | maxDelta: " << maxRefDelta;
    }
    msg() << endreq;
  }
  if ( ! m_usesMSE ) {
    m_vec.push_back( TuningReference( "TuningByMSE", benchmark::MSE ) );
    m_usesMSE = true;
  }
}

//==============================================================================
std::size_t TuningReferenceContainer::spRefIdx() const 
{ 
  std::size_t ret{0};
  for ( const auto &ref : *this ) {
    if ( ref.refType() == Type::benchmark::SP ) {
      return ret; 
    }
    ++ret;
  }
  return std::numeric_limits<std::size_t>::max();
}

//==============================================================================
std::size_t TuningReferenceContainer::mseRefIdx() const 
{ 
  std::size_t ret{0};
  for ( const auto &ref : *this ) {
    if ( ref.refType() == benchmark::MSE ) {
      return ret; 
    }
    ++ret;
  }
  return std::numeric_limits<std::size_t>::max();
}

/**
 * @brief helper class for subtracting in by value
 **/
class DistanceToValFctor
{
  REAL m_val;
  public:
    DistanceToValFctor( REAL val ):m_val(val){;}
    void operator()(REAL &in){
      in -= m_val;
      in = std::abs( in );
    }
};

//==============================================================================
PerfEval TuningPerformance::getBest( const REAL mseVal
    , const ROC::Performance& rocPerf 
    , const NeuralNetwork& newEpochNN
    , REAL TuningPerformance::* const mseMember
    , ROC::setpoint TuningPerformance::* const rocPerfMember
    , DatasetPerformanceEvolution &dsPerfEvo 
    , const bool evalEarlyStop
    , const bool update )
{
  // We will not request to early stop by default
  m_requestedEarlyStop = false;

  // Initialize the returning variable:
  auto perfEval = PerfEval::WORSE;

  // Allocate references to members:
  auto &m_mse = this->*mseMember;
  auto &m_perf = this->*rocPerfMember;
  benchmark benchType = m_reference.refType();

  // shortcuts to the roc values:
  const auto &sp = rocPerf.sp; const auto &pd = rocPerf.pd;
  const auto &pf = rocPerf.sp; const auto &cut = rocPerf.cut;

  // The index of best found performance in this epoch
  uint64_t idx{0};

  // Evaluate performance:
  switch ( benchType ) {
    case benchmark::SP:
      // Get index of highest sp-index:
      idx = std::distance(sp.begin(), std::max_element(sp.begin(), sp.end()));
      break;
    case benchmark::MSE:
      if ( mseVal < m_mse )
        perfEval = PerfEval::BETTER;
      break;
    case benchmark::Pd:
    case benchmark::Pf:
    {
      // Work on copy:
      std::vector<REAL> refVec = (benchType == benchmark::Pd)?pd:pf;
      // No need to copy benchmark, no operation done on it:
      const std::vector<REAL> &benchmarkVec = (benchType == benchmark::Pd)?pf:pd;
      // The algorithm reference operation:
      const REAL refVal = m_reference.refVal();
      // Remove the reference
      basic_paralel::for_each( refVec.begin(), refVec.end(), DistanceToValFctor{refVal} );
      // Retrieve nearest index
      idx = std::distance(refVec.begin(), std::min_element(refVec.begin(), refVec.end()));
      // Check if delta is greater than it is allowed
      if ( refVec[idx] > m_reference.maxRefDelta() ) {
        // When greater than the limit, we get the first point that we
        // are better than the reference:
        if( benchType == benchmark::Pd ){
          if ( idx > 0 ) --idx;
          // and then get the best sp-index, evaluating only in the
          // region that we are performing better than the reference
          idx = std::distance(sp.begin(), std::max_element(sp.begin() + idx, sp.end() ));
        } else {
          unsigned nPoints = cut.size();
          if ( idx < nPoints ) ++idx;
          // and then get the best sp-index, evaluating only in the
          // region that we are performing better than the reference
          idx = std::distance(sp.begin(), std::max_element(sp.begin(), sp.begin() + idx ));
        }
        // Now check we are better than previous performance
        if ( sp[idx] > m_perf.sp )
          perfEval = PerfEval::BETTER;
      } else {
        // Retrieve previous benchmark best performance:
        const REAL prevPerf = (benchType == benchmark::Pd)?(-m_perf.pf):m_perf.pd;
        // Retrieve current benchmark performance:
        const REAL cPerf    = (benchType == benchmark::Pd)?(-benchmarkVec[idx]):benchmarkVec[idx];
        if ( cPerf > prevPerf ) {
          perfEval = PerfEval::BETTER;
        }
      }
      break;
    }
  }

  // Add performance to history:
  ROC::setpoint newPerf;
  dsPerfEvo.addPerf( mseVal );
  if ( ! m_mseOnly && benchType != benchmark::MSE ) {
    newPerf = rocPerf.at(idx);
    dsPerfEvo.addPerf( newPerf );
  }

  // Check if there was an improvement:
  if ( m_epoch > m_minEpochs ){
    // Update our performance if requested
    if ( update ){
      switch ( perfEval ){
        case PerfEval::BETTER:
          // Set that our last improvement was in this epoch:
          m_lastImprove = m_epoch;
          // Update our performance:
          m_mse = mseVal; 
          if ( ! m_mseOnly && benchType != benchmark::MSE ) {
            m_perf = newPerf;
          }
          // and the discriminator:
          m_nn.copyWeigthsFast( newEpochNN );
          break;
        default:
          break;
      }
    }
    if ( evalEarlyStop ) {
      switch ( perfEval ){
        case PerfEval::EQUAL:
        case PerfEval::WORSE:
          if ( ( m_epoch - m_lastImprove ) >= m_maxFail ) {
            m_requestedEarlyStop = true;
            throw EarlyStop();
          }
          break;
        default:
          break;
      }
    }
  }
  return perfEval;
}

//==============================================================================
bool TuningPerformance::update( const REAL trnMSE , const REAL valMSE , const REAL tstMSE
    , const ROC::Performance& trnRocPerf
    , const ROC::Performance& valRocPerf
    , const ROC::Performance& tstRocPerf
    , const NeuralNetwork& newEpochNN)
{
  // Get best performance for training dataset
  this->getBest( trnMSE, trnRocPerf, newEpochNN
               , &TuningPerformance::m_trnMSE
               , &TuningPerformance::m_trnPerf
               , m_perfEvo.at( static_cast<unsigned>(Type::Dataset::Train) )
               , /*evalEarlyStop = */ false 
               , /*update = */ false );
  // Get best performance for validation dataset
  // We evaluate early stopping in this dataset, but also update
  // performance in case there is no test data available
  PerfEval perfEval = this->getBest( valMSE, valRocPerf, newEpochNN
                                   , &TuningPerformance::m_valMSE
                                   , &TuningPerformance::m_valPerf
                                   , m_perfEvo.at( static_cast<unsigned>(Type::Dataset::Validation) )
                                   , /*evalEarlyStop = */ true
                                   , /*update = */ !m_hasTstData );
  // Get best performance for test dataset
  if ( m_hasTstData ){
    this->getBest( tstMSE, tstRocPerf, newEpochNN
                 , &TuningPerformance::m_tstMSE
                 , &TuningPerformance::m_tstPerf
                 , m_perfEvo.at( static_cast<unsigned>(Type::Dataset::Test) )
                 , /*evalEarlyStop = */ false
                 , /*update = */ true );
  }
  return perfEval == PerfEval::BETTER;
}

//==============================================================================
void TuningPerformance::printEpochInfo() const
{
  if ( m_reference.refType() == benchmark::MSE ) {
    msg() << MSG::INFO << "Epoch " << m_epoch << " : MSE " 
          << m_perfEvo.at(static_cast<unsigned>(Type::Dataset::Train)).getMSE() << " (train) " 
          << m_perfEvo.at(static_cast<unsigned>(Type::Dataset::Validation)).getMSE() << " (val) ";
    if ( m_hasTstData ){
      msg() << m_perfEvo.at(static_cast<unsigned>(Type::Dataset::Test)).getMSE() << " (test) ";
    } 
    msg() << " : LastImprove " << this->lastImprove()
          << " : EarlyStop " << util::boolStr(this->requestedEarlyStop())
          << endreq;
  } else if ( ! m_mseOnly ) {
    Type::Dataset lastDS = Type::getLastDS(m_hasTstData);
    for ( unsigned ds = static_cast<unsigned>(Type::Dataset::Train)
        ; ds <= static_cast<unsigned>(lastDS)
        ; ++ds )
    {
      MSG_INFO( "Epoch " << m_epoch << to_str(static_cast<Type::Dataset>(ds)) 
                         << " : SP "          << m_perfEvo.at(ds).getSP()
                         << " : Pd "          << m_perfEvo.at(ds).getPd()
                         << " : Pf "          << m_perfEvo.at(ds).getPf()
                         << " : cut "         << m_perfEvo.at(ds).getCut()
                         << " : LastImprove " << this->lastImprove()
                         << " : EarlyStop "   << util::boolStr(this->requestedEarlyStop())
             );
    }
  }
}

//==============================================================================
void TuningPerformance::setName( const std::string &name )
{
  this->MsgService::setLogName( name + "_Performance" );
}

//==============================================================================
TuningPerformanceCollection::TuningPerformanceCollection( 
    const TuningReferenceContainer& refs
  , const uint64_t &epoch
  , const uint64_t nEpochs
  , const unsigned minEpochs
  , const unsigned maxFail
  , const unsigned show
  , const bool mseOnly
  , const bool hasTstData) 
  : m_perfEvo{ refs, epoch, nEpochs, hasTstData }
  , m_epoch{ epoch }
  , m_show{show}
{
  // We will hold one performance for each reference:
  m_perfs.reserve( refs.size() );
  // And allocate them accordingly:
  unsigned idx{0};
  for ( const auto& ref : refs ) {
    m_perfs.push_back( TuningPerformance{ ref, m_perfEvo.at(idx)
                                        , epoch, minEpochs, maxFail
                                        , mseOnly, hasTstData } );
    ++idx;
  }
  m_mseRefIdx = refs.mseRefIdx();
  m_spRefIdx = refs.spRefIdx();
  m_nPerfs = m_perfs.size();
}

//==============================================================================
void TuningPerformanceCollection::update( const REAL trnMSE , const REAL valMSE , const REAL tstMSE
    , const ROC::Performance& trnRocPerf
    , const ROC::Performance& valRocPerf
    , const ROC::Performance& tstRocPerf
    , const NeuralNetwork& newEpochNN )
{
  // Reset number of stop requests:
  m_earlyStopCount = 0;
  unsigned perfIdx{0};
  for ( auto& perf : *this ) {
    try {
      if ( perf.update( trnMSE, valMSE, tstMSE
                      , trnRocPerf 
                      , valRocPerf
                      , tstRocPerf
                      , newEpochNN )  )
      {
        // Flag that this performance improved
        m_perfEvo.improved( perfIdx );
      }
    } catch ( const EarlyStop & ){
      // An performance has requested to stop tuning, but we continue
      // training until all performances have requested
      ++m_earlyStopCount;
    }
    ++perfIdx;
  }
  // Check if all performance evaluations have requested to stop
  // tuning
  if ( m_earlyStopCount == m_nPerfs ) {
    throw EarlyStop();
  }
}

//==============================================================================
void TuningPerformanceCollection::printEpochInfo() const
{
  if ( m_epoch % m_show ) {
    for ( const auto& perf : *this ) {
      perf.printEpochInfo();
    }
  }
}

//==============================================================================
TuningPerformanceCollection &TuningPerformanceCollection::operator=(const TuningPerformanceCollection &tpc)
{
  if (this!=&tpc)
  {
    this->~TuningPerformanceCollection();
    new (this) TuningPerformanceCollection(tpc);
  }
  return *this;
}

} // namespace TuningTool
