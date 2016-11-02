#ifndef TUNINGTOOLS_TRAINING_TUNINGUTIL_H
#define TUNINGTOOLS_TRAINING_TUNINGUTIL_H

// First include must be the defines, always!
#include "TuningTools/system/defines.h"

// Package include(s):
#include "RingerCore/MsgStream.h"
#include "TuningTools/neuralnetwork/NeuralNetwork.h"

// STL include(s):
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <memory>

// Boost include(s):
#include <boost/python.hpp>
namespace py = boost::python;

// FIXME Should implement some container template at util to reduce this coding copy and paste

// FIXME Should implement ROC from 100% detection to 0% if only using Pd benchmarks and stop
//       if retrieved all benchmarks?

namespace Type {
enum class benchmark : std::uint8_t {
  SP = 0,
  Pd = 1,
  Pf = 2,
  MSE = 3
};

//==============================================================================
inline
std::string to_str(const benchmark b){
  switch (b){
    case benchmark::SP: return "SP";
    case benchmark::Pd: return "Pd";
    case benchmark::Pf: return "Pf";
    case benchmark::MSE: return "MSE";
  }
}

enum class Dataset : std::int8_t {
  Train = 0,
  Validation = 1,
  Test = 2,
  Operation = 3,
};

//==============================================================================
inline
std::string to_str(const Dataset d){
  switch (d){
    case Dataset::Train:      return "Train";
    case Dataset::Validation: return "Validation";
    case Dataset::Test:       return "Test";
    case Dataset::Operation:  return "Operation";
  }
}


//==============================================================================
inline
Dataset getLastDS( bool hasTstData ) {
  return (hasTstData)?(Dataset::Test):(Dataset::Validation);
}
}

namespace TuningTool {

struct WrongDictError : public std::exception {                          
  const char* what() const throw() { return "Missing key on dict."; } 
};

enum class PerfEval : std::int8_t {
  WORSE = -1, 
  EQUAL = 0, 
  BETTER = 1
};

/**
 * @brief Calculate ROC performance
 **/
class ROC : public MsgService {
  public:
    /**
     * @brief One ROC performance point
     *
     * Current standard does not use percentage values.
     **/
    struct setpoint{
      REAL sp, pd, pf, cut;

      setpoint( const REAL sp = 0., const REAL pd = 0.
              , const REAL pf = 100., const REAL cut = 0.)
        : sp{sp} , pd{pd}
        , pf{pf} , cut{cut} {;}
    };

    /**
     * @brief The full ROC calculated performance
     **/
    struct Performance {
      /// The sum-product index for each cut
      std::vector<REAL> sp;
      /// The ROC P_D value for each cut
      std::vector<REAL> pd;
      /// The ROC P_F value for each cut
      std::vector<REAL> pf;
      /// The cuts grid where the ROC was valuated
      std::vector<REAL> cut;

      /**
       * Return performance at index
       **/
      setpoint at(std::size_t idx) const { return setpoint{sp.at(idx), pd.at(idx), pf.at(idx), cut.at(idx)}; }
    };

  private:

    /// Signal target
    REAL m_signalTarget;
    /// Background target
    REAL m_backgroundTarget;

    /// The resolution to create the cuts;
    const REAL m_resolution{0.01};
    /// Signal weight for calculating the sp-index
    const REAL m_signalWeight{1.};
    /// Background weight for calculating the sp-index
    const REAL m_backgroundWeight{1.};

    /// Number of points allocated in the ROC
    unsigned m_nPoints;

  public:

    /// Ctors
    /// @{
    ROC(){;}
    ROC( REAL signalTarget, REAL backgroundTarget, 
         const REAL resolution = 0.01, 
         REAL signalWeight = 1,
         REAL backgroundWeight = 1)
      : m_resolution( resolution ),
        m_signalWeight( signalWeight ),
        m_backgroundWeight( backgroundWeight )
    {
      if ( m_signalTarget < m_backgroundTarget ) {
        std::runtime_error("Cannot allocate ROC with signal target lesser than background target!");
      }
      m_nPoints = std::floor( (signalTarget - backgroundTarget) / resolution ) + 1;
    }
    /// @}

    /**
     * @brief Fast solution for enabling copying ROC objects
     *
     * XXX This is only used on PatternRec ctor, so shouldn't be an issue. 
     **/
    ROC &operator=(const ROC &rhs);

    /**
     * @brief Run ROC generation
     **/
    void execute(const std::vector<REAL> &signal, 
                 const std::vector<REAL> &background,
                 Performance& rocPerf);
};

/**
 * @brief Hold one dataset performance evolution
 **/
class DatasetPerformanceEvolution : public MsgService {
  private:

    /// The performance vectors
    std::vector<REAL> m_spEvo, m_pdEvo, m_pfEvo, m_cutEvo, m_mseEvo;
    /// Keep weak reference to Pattern Recognition epoch couting
    const uint64_t &m_epoch;
    /// Which dataset information that is being held
    const Type::Dataset m_ds;
    /// Which reference type is being held
    const Type::benchmark m_refType;
    /// Improved

  public:

    /// @{
    DatasetPerformanceEvolution()
      : m_epoch{*static_cast<std::uint64_t*>(nullptr)}
      , m_ds{Type::Dataset::Train}
      , m_refType{Type::benchmark::MSE}
      {;}
    DatasetPerformanceEvolution( const uint64_t nEpochs
                               , const uint64_t epoch
                               , const Type::Dataset ds
                               , const Type::benchmark refType )
      : m_epoch{ epoch }
      , m_ds{ ds }
      , m_refType{ refType }
    {
      if ( refType == Type::benchmark::MSE ) {
        m_mseEvo.reserve( nEpochs );
      } else {
        m_spEvo.reserve( nEpochs ); m_pdEvo.reserve( nEpochs ); 
        m_pfEvo.reserve( nEpochs ); m_cutEvo.reserve( nEpochs );
      }
    }
    /// @}

    /**
     * @brief Adds performance to evolution history
     **/
    void addPerf( const ROC::setpoint &perf );
    /**
     * @brief Adds mse value to evolution history
     **/
    void addPerf( const REAL mseVal );

    /// Getter methods
    /// @{
    REAL getSP() const;
    REAL getPd() const;
    REAL getPf() const;
    REAL getCut() const;
    REAL getMSE() const;
    /// @}

};

typedef std::vector< DatasetPerformanceEvolution > PerformanceEvolutionRawContainer;
typedef std::vector< PerformanceEvolutionRawContainer > PerformanceRawCollection;

class TuningReferenceContainer;

/**
 * @brief DatasetPerformanceEvolution std::vector specialization
 * TODO Not implemented as there seems to be no need for this right
 * now
 **/
//class ReferencePerformanceEvolutionVec : public std::vector< ReferencePerformanceEvolution >
//{
//  public:
//
//    /**
//     * Copies current epoch performance
//     **/
//    setPerf( const ReferencePerformanceEvolutionVec& refPerf );
//    for ( Type::Dataset ds = Train; ds <= Type::Dataset::Test; ++ds ){
//      perfEvo.setPerf( ds, m_perfs[ m_spRefIdx ].getPerf( ds ) );
//    }
//}

/**
 * @brief Holds performance evolution for every tuning reference
 **/
class PerformanceEvolution : public MsgService {
  private:
    /// Collection of performance evolution
    PerformanceRawCollection m_perfEvoCol;
    /// Hold each performance best epoch
    std::vector<std::uint64_t> m_savedEpoch;
    /// Hold current epoch weak reference
    const std::uint64_t &m_epoch;
    /// Whether the performance evoluation has test dataset
    const bool m_hasTstData;

  public:

    /// Ctors
    /// @{
    PerformanceEvolution()
      : m_epoch{ *static_cast<std::uint64_t*>(nullptr) }
      , m_hasTstData{ false }
      {;}
    PerformanceEvolution( const TuningReferenceContainer& refs
                        , const uint64_t nEpochs
                        , const uint64_t epoch
                        , const bool hasTstData );
    /// @}

    typedef PerformanceRawCollection::size_type size_type;
    typedef PerformanceRawCollection::value_type value_type;
    typedef PerformanceRawCollection::reference reference;
    typedef PerformanceRawCollection::const_reference const_reference;
    typedef PerformanceRawCollection::iterator iterator;
    typedef PerformanceRawCollection::const_iterator const_iterator;

    /// Container methods
    /// @{
    reference       at( size_type pos )               { return m_perfEvoCol.at(pos); }
    const_reference at( size_type pos ) const         { return m_perfEvoCol.at(pos); }
    reference       operator[]( size_type pos )       { return m_perfEvoCol[pos];    }
    const_reference operator[]( size_type pos ) const { return m_perfEvoCol[pos];    }
    iterator        begin()                           { return m_perfEvoCol.begin(); }
    const_iterator  begin() const                     { return m_perfEvoCol.begin(); }
    iterator        end()                             { return m_perfEvoCol.end();   }
    const_iterator  end() const                       { return m_perfEvoCol.end();   }
    /// @}
    
    /**
     * @brief Flag that performance improved for keeping in history
     **/
    void improved( unsigned perfIdx ) { m_savedEpoch.at( perfIdx ) = m_epoch; }

    /// Message methods
    /// @{
    void setMsgLevel( const MSG::Level lvl );
    void setUseColor( const bool color );
    /// @}

    /// Get methods
    /// @{
    uint64_t savedEpoch( unsigned pos ){ return m_savedEpoch.at(pos); }
    /// @}
};

/**
 * @brief Holds the Tuning Reference performance
 **/
class TuningReference : public MsgService {

  private:
    /// The reference name
    std::string m_name;
    /// The reference benchmark type
    Type::benchmark m_refType;
    /// The reference value
    REAL m_refVal;
    /// The reference maximum allowed delta to consider same value as
    /// the reference itself
    REAL m_maxRefDelta;

  public:

    /// ctors
    /// @{
    TuningReference(){;} 
    TuningReference( const std::string& name 
                   , const Type::benchmark m_refType
                   , const REAL refVal = 999.
                   , const REAL maxRefDelta = 0.2 );
    /// @}

    /// Getter methods
    /// @{
    std::string getName() const { return m_name; }
    Type::benchmark refType() const { return m_refType; }
    REAL refVal() const { return m_refVal; }
    REAL maxRefDelta() const { return m_maxRefDelta; }
    /// @}

    /// Setter methods
    /// @{
    void setName( const std::string &name );
    /// @}
};

typedef std::vector<TuningReference> TuningReferenceVec;

/**
 * @brief Holds all Tuning Reference performances to be used during
 *        the training
 **/
class TuningReferenceContainer : public MsgService
{
  private:
    /// The reference vector
    TuningReferenceVec m_vec;
    /// Whether current container uses MSE performance:
    bool m_usesMSE{false};
    /// Whether current container uses SP performance:
    bool m_usesSP{false};

  public:

    typedef TuningReferenceVec::size_type size_type;
    typedef TuningReferenceVec::value_type value_type;
    typedef TuningReferenceVec::reference reference;
    typedef TuningReferenceVec::const_reference const_reference;
    typedef TuningReferenceVec::iterator iterator;
    typedef TuningReferenceVec::const_iterator const_iterator;

    /// Ctors
    /// @{
    TuningReferenceContainer(){;} 
    TuningReferenceContainer( const py::list& references
                            , MSG::Level msglevel = MSG::INFO
                            , bool useColor = false );
    /// @}

    bool mseOnly() const { return m_usesMSE && this->size() == 1; }

    /// Container methods
    /// @{
    reference       at( size_type pos )               { return m_vec.at(pos); }
    const_reference at( size_type pos ) const         { return m_vec.at(pos); }
    reference       operator[]( size_type pos )       { return m_vec[pos];    }
    const_reference operator[]( size_type pos ) const { return m_vec[pos];    }
    iterator        begin()                           { return m_vec.begin(); }
    const_iterator  begin() const                     { return m_vec.begin(); }
    iterator        end()                             { return m_vec.end();   }
    const_iterator  end() const                       { return m_vec.end();   }
    /// @}

    /// Getter methods
    /// @{
    bool usesMSE() const { return m_usesMSE; }
    bool usesSP() const { return m_usesSP; }
    /// Returns SP reference index
    std::size_t spRefIdx() const;
    /// Returns MSE reference index
    std::size_t mseRefIdx() const;
    size_t size() const { return m_vec.size(); }
    const TuningReferenceVec& vec() const { return this->m_vec; }
    /// @}
};

/**
 * Raised when there is no reason to continue tuning
 **/
class EarlyStop : public std::exception {
  public:
    const char* what() const noexcept override {
      return "EarlyStop exception raised and not catched.";
    }
};

/**
 * @brief Checks if performance was better than previous and hold best
 *        paramaters for the tuned ML for the given reference
 * 
 * It uses the TuningReference as the requested reference operation.
 **/
class TuningPerformance : public MsgService {
  private:
    /// Neural network with best performance
    NeuralNetwork m_nn;
    /// The reference we are using for the performance evaluation
    const TuningReference &m_reference;
    /// @brief Performance evoluation raw container in which we will fill our
    ///        history
    PerformanceEvolutionRawContainer &m_perfEvo;
    ///  Each dataset MSE for the best performance
    REAL m_trnMSE, m_valMSE, m_tstMSE;
    /// Each dataset chosen ROC point with best performance
    ROC::setpoint m_trnPerf, m_valPerf, m_tstPerf;
    /// Current epoch reference (linked to the PatternRec epoch)
    const uint64_t& m_epoch;
    /// Last previous epoch which we had an performance improvement
    uint64_t m_lastImprove{0};
    /// Whether we have requested to early stop or not
    bool m_requestedEarlyStop{false};
    /// Minimum number of epochs in order to evaluate performance
    const unsigned m_minEpochs;
    /// Maximum number of fails in order to throw an EarlyStop request
    const unsigned m_maxFail;
    /// Whether we don't need to access ROC information
    const bool m_mseOnly;
    /// Whether we have access to test data
    const bool m_hasTstData;

    /**
     * @brief Retrieve best performance using reference and possibly
     *        update ML technique and evaluate early stop
     **/
    PerfEval getBest( const REAL mseVal
        , const ROC::Performance& rocPerf 
        , const NeuralNetwork& newEpochNN
        , REAL TuningPerformance::* const mseMember
        , ROC::setpoint TuningPerformance::* const rocPerfMember
        , DatasetPerformanceEvolution &dsPerfEvo 
        , const bool evalEarlyStop
        , const bool update );

  public:

    /// Ctors
    ///@{
    TuningPerformance()
      : m_reference{ *static_cast<TuningReference*>(nullptr) }
      , m_perfEvo{ *static_cast<PerformanceEvolutionRawContainer*>(nullptr) }
      , m_epoch{ *static_cast<std::uint64_t*>(nullptr) }
      , m_minEpochs{ 0 }
      , m_maxFail{ 0 }
      , m_mseOnly{ false }
      , m_hasTstData{ false }
    {;} 
    TuningPerformance( const TuningReference &ref
                     , PerformanceEvolutionRawContainer &perfEvo
                     , const uint64_t& epoch
                     , const unsigned minEpochs
                     , const unsigned maxFail
                     , const bool mseOnly
                     , const bool hasTstData
                     )
      : m_reference{ ref }
      , m_perfEvo{ perfEvo }
      , m_epoch{ epoch }
      , m_minEpochs{ minEpochs }
      , m_maxFail{ maxFail }
      , m_mseOnly{ mseOnly }
      , m_hasTstData{ hasTstData }
    { 
      setName( m_reference.getName() );
    }
    ///@}

    /**
     * Update performance
     **/
    bool update( const REAL trnMSE , const REAL valMSE , const REAL tstMSE
               , const ROC::Performance& trnRocPerf
               , const ROC::Performance& valRocPerf
               , const ROC::Performance& tstRocPerf
               , const NeuralNetwork& newEpochNN);

    /**
     * Print performance at current epoch
     **/
    void printEpochInfo() const;

    /// Getter methods
    /// @{
    void getNN( NeuralNetwork& nn ) const { nn = m_nn; }
    uint64_t lastImprove() const { return m_lastImprove; }
    bool requestedEarlyStop() const { return m_requestedEarlyStop; }
    /// @}
    
    /// Setter methods
    /// @{
    void setName( const std::string &name );
    /// @}
};

typedef std::vector<TuningPerformance> TuningPerformanceVec;

/**
 * @brief Retrieve best performance at all requested tuning reference
 *        points
 *
 * Also feed the PerformanceEvolution during its update
 **/
class TuningPerformanceCollection : public MsgService {
  private:

    /// The tuning performance to update
    PerformanceEvolution m_perfEvo;
    /// Hold performances
    TuningPerformanceVec m_perfs;
    /// Number of early stop counts:
    unsigned m_earlyStopCount{0};
    /// The number of performances we hold
    unsigned m_nPerfs;
    /// Weak reference to the curent epoch
    const std::uint64_t &m_epoch;
    /// Number of epochs to show performance
    unsigned m_show;
    /// The MSE reference index 
    unsigned m_mseRefIdx;
    /// The MSE reference index 
    unsigned m_spRefIdx;

  public:

    /// Ctors
    /// @{
    TuningPerformanceCollection()
      : m_epoch{ *static_cast<std::uint64_t*>(nullptr) }
      {;}
    TuningPerformanceCollection( const TuningReferenceContainer& refs
                               , const uint64_t &epoch
                               , const uint64_t nEpoch
                               , const unsigned minEpochs
                               , const unsigned maxFail
                               , const unsigned show
                               , const bool mseOnly
                               , const bool hasTstData);
    /// @}

    /**
     * @brief Main method for updating tuning performances
     *
     * @param[in] The train dataset MSE value;
     * @param[in] The validation dataset MSE value;
     * @param[in] The test dataset MSE value;
     * @param[in] The train dataset ROC performance;
     * @param[in] The validation dataset ROC performance;
     * @param[in] The test dataset ROC performance;
     * @param[in] The Neural Network weights to keep in case
     *            there is an performance improvement;
     **/
    void update( const REAL trnMSE , const REAL valMSE , const REAL tstMSE
        , const ROC::Performance& trnRocPerf
        , const ROC::Performance& valRocPerf
        , const ROC::Performance& tstRocPerf
        , const NeuralNetwork& newEpochNN );

    void printEpochInfo() const;

    typedef TuningPerformanceVec::size_type size_type;
    typedef TuningPerformanceVec::value_type value_type;
    typedef TuningPerformanceVec::reference reference;
    typedef TuningPerformanceVec::const_reference const_reference;
    typedef TuningPerformanceVec::iterator iterator;
    typedef TuningPerformanceVec::const_iterator const_iterator;

    /// Container methods
    /// @{
    reference       at( size_type pos )               { return m_perfs.at(pos); }
    const_reference at( size_type pos ) const         { return m_perfs.at(pos); }
    reference       operator[]( size_type pos )       { return m_perfs[pos];    }
    const_reference operator[]( size_type pos ) const { return m_perfs[pos];    }
    iterator        begin()                           { return m_perfs.begin(); }
    const_iterator  begin() const                     { return m_perfs.begin(); }
    iterator        end()                             { return m_perfs.end();   }
    const_iterator  end() const                       { return m_perfs.end();   }
    /// @}
    
    /// Msg methods:
    /// @{
    void setMsgLevel( const MSG::Level lvl );
    void setUseColor( const bool useColor );
    /// @}

    /**
     * @brief Fast solution for enabling copying ROC objects
     *
     * XXX This is only used on PatternRec ctor, so shouldn't be an issue. 
     **/
    TuningPerformanceCollection &operator=(const TuningPerformanceCollection &rhs);

    /// Getter Methods
    /// @{
    const TuningPerformanceVec& perfs() const { return m_perfs; }
    unsigned earlyStopCount() const { return m_earlyStopCount; }
    /// @}
};

//==============================================================================
inline
void PerformanceEvolution::setMsgLevel( const MSG::Level lvl )
{
  this->MsgService::setMsgLevel( lvl );
  for ( auto &perf : *this ) {
    for ( auto &dsPerf : perf ){
      dsPerf.setMsgLevel( lvl );
    }
  }
}

//==============================================================================
inline
void PerformanceEvolution::setUseColor( const bool useColor )
{
  this->MsgService::setUseColor( useColor );
  for ( auto &perf : *this ) {
    for ( auto &dsPerf : perf ){
      dsPerf.setUseColor( useColor );
    }
  }
}

//==============================================================================
inline
void TuningPerformanceCollection::setMsgLevel( const MSG::Level lvl )
{
  this->MsgService::setMsgLevel( lvl );
  for ( auto &perf : *this ) {
    perf.setMsgLevel( lvl );
  }
  m_perfEvo.setMsgLevel( lvl );
}

//==============================================================================
inline
void TuningPerformanceCollection::setUseColor( const bool useColor )
{
  this->MsgService::setUseColor( useColor );
  for ( auto &perf : *this ) {
    perf.setUseColor( useColor );
  }
  m_perfEvo.setUseColor( useColor );
}

} // namespace TuningTool

#endif // TUNINGTOOLS_TRAINING_TUNINGUTIL_H
