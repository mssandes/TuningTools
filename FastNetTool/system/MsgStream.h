#ifndef FASTNETTOOL_MSGSTREAM_H
#define FASTNETTOOL_MSGSTREAM_H

#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "FastNetTool/system/defines.h"

/**
 * Macro for using within MsgService inherited classes.
 *
 * It will check if message is above level before requesting to enter it into
 * MsgStream to check it.
 **/
#define MSG_LVL_CHK(xmsg, lvl)  do { \
  if ( msgLevel( lvl ) ) { \
    msg() << lvl << xmsg << endreq; \
  } \
} while (0);

/**
 * Macro for check and displaying DEBUG messages.
 *
 * We only display DEBUG messages if building with debug set
 **/
#if defined(FASTNET_DBG_LEVEL) && FASTNET_DBG_LEVEL > 0
# define MSG_DEBUG(xmsg) MSG_LVL_CHK( xmsg, ::MSG::DEBUG )
#else
# define MSG_DEBUG(xmsg) while(0) { std::ostringstream s; s << xmsg; }
#endif // DEBUG > 0

/**
 * Macro for displaying INFO messages
 **/
#define MSG_INFO(xmsg)    MSG_LVL_CHK( xmsg, ::MSG::INFO    )

/**
 * Macro for displaying WARNING messages
 **/
#define MSG_WARNING(xmsg) MSG_LVL_CHK( xmsg, ::MSG::WARNING )

/**
 * Macro for displaying VERBOSE messages
 **/
#define MSG_ERROR(xmsg)   MSG_LVL_CHK( xmsg, ::MSG::ERROR   )

/**
 * Macro for displaying FATAL messages
 *
 * It will also raise a std::runtime_error with same message.
 **/
#define MSG_FATAL(xmsg)                                 \
  msg() << MSG::FATAL << xmsg;                          \
  auto e = std::runtime_error( msg().stream().str() );  \
  msg().doOutput();                                     \
  throw e;

namespace MSG
{

/**
 * The multiple message levels available
 **/
enum Level {
  VERBOSE = 0,
  DEBUG   = 1,
  INFO    = 2,
  WARNING = 3,
  ERROR   = 4,
  FATAL   = 5
};

} // namespace MSG

/**
 * @brief Get C string from enum
 **/
//==============================================================================
inline
const char* to_str(const MSG::Level lvl)
{
  switch (lvl){
    case MSG::VERBOSE:
      return "VERBOSE";
    case MSG::DEBUG:
      return "DEBUG";
    case MSG::INFO:
      return "INFO";
    case MSG::WARNING:
      return "WARNING";
    case MSG::ERROR:
      return "ERROR";
    case MSG::FATAL:
      return "FATAL";
    default:
      throw std::runtime_error("Couldn't determine LEVEL enumeration.");
  }
}

/**
 * @brief Overload ostringstream to accept MSG::Level
 *
 * In fact, we only need this prototype for the compile to accept the macros
 * declared above. It is implemented only for convinience, as it should be the
 * wanted behavior in case anyone decides to enter the MSG::Level accidently to
 * an ostringstream.
 **/
inline
std::ostringstream& operator<<(std::ostringstream &s, 
                               MSG::Level /*level*/)
{
  return s;
}

/**
 * @brief Overload operator << to accept display vectors
 **/
template <typename T>                                                      
std::ostream& operator<< ( std::ostream& out, const std::vector< T >& vec )
{                                                                          
  // A little prefix:                                                      
  out << "[";                                                              
  // Print the contents:                                                   
  for( size_t i = 0; i < vec.size(); ++i ) {                               
     out << vec[ i ];                                                      
     if( i < vec.size() - 1 ) {                                            
        out << ", ";                                                       
     }                                                                     
  }                                                                        
  // A little postfix:                                                     
  out << "]";                                                              
  // Return the stream:                                                    
  return out;                                                              
}                                                                          

/**
 * @brief Adds logging capability to inherited classes
 *
 * Based on Athena framework's MsgStream
 **/
class MsgStream
{

  public:

    /**
     * @brief Helper class for displaying MsgStream messages
     **/
    class Message {
      public: 
        /// Number of charatecters before displaying message
        static constexpr unsigned space_between_log_and_msg = 45;

        /// Ctor
        explicit Message( const std::string &logName, 
                 const MSG::Level level,
                 const std::string &message ) 
          : m_formatted_msg("")
        {
          const char* lvl_str = to_str(level);
          const size_t end_log_name = space_between_log_and_msg 
                                      - (std::strlen(lvl_str) + 2);
          m_formatted_msg += logName.substr(0, end_log_name );
          m_formatted_msg.append( end_log_name 
                                  - m_formatted_msg.length() + 1
                                  , 0x20);
          m_formatted_msg += lvl_str;
          m_formatted_msg += 0x20;
          m_formatted_msg += message;
        }

        /// Overloads std::cout printing capabilities
        friend std::ostream &operator<<( 
            std::ostream &stream, 
            Message &msg );

      private:
        /// The formated message
        std::string m_formatted_msg;
    };

    /// Ctor using standard configuration
    /// @{
    MsgStream() 
      : m_streamName("FastNetTool_Log"),
        m_level(MSG::INFO),
        m_currentLevel(MSG::INFO),
        m_active(true){;}

    /// Ctor using integer for setting logname and msgLevel
    explicit MsgStream(const std::string &logname, const int msgLevel)
      : m_streamName(logname),
        m_level( ( msgLevel > MSG::FATAL ) ? ( MSG::FATAL )
                                           : ( ( msgLevel < MSG::VERBOSE ) ? ( MSG::VERBOSE )
                                                                           : ( static_cast<MSG::Level>( msgLevel ) ) 
                                             )
               ),
        m_currentLevel(MSG::INFO),
        m_active( m_level <= m_currentLevel )
        {;}

    /// Ctor setting logname and msgLevel
    explicit MsgStream(const std::string &logname, const MSG::Level msgLevel)
      : m_streamName(logname),
        m_level(msgLevel),
        m_currentLevel(MSG::INFO),
        m_active(m_level <= m_currentLevel)
        {;}

    /// Copy constructor
    MsgStream(const MsgStream& msg)
      : m_streamName(msg.m_streamName),
        m_level(msg.m_level),
        m_currentLevel(msg.m_currentLevel),
        m_active(msg.m_active)
    {
      try { // ignore exception if we cannot copy the string
        m_stream.flags( msg.m_stream.flags() );
      } catch (...) {;}
    }
    /// @}

    /// Inline methods
    /// @{
    /// Retrieve current message level
    MSG::Level level() const
    { 
      return m_level;
    }

    /// Retrieve a reference to current logname
    const std::string& logName() const 
    {
      return m_streamName;
    }
    
    /// Change logger level
    void setLevel(MSG::Level msgLevel){
      if ( msgLevel != m_level ){
        m_level = msgLevel;
        // Report change of level:
        report(m_currentLevel);
      }
    }

    /// Change this logger name
    void setLogName(const std::string &logName)
    {
      m_streamName = logName;
    }

    /// oMsgStream flush emulation
    MsgStream& flush()    {
      if ( isActive() ) m_stream.flush();
      return *this;
    }

    /// Operators overload:
    /// @{
    /// Accept MsgStream modifiers (this will call endreq/endmsg)
    MsgStream& operator<<(MsgStream& (*_f)(MsgStream&)) {
      if ( isActive() ) _f(*this);
      return *this;
    }
    /// Accept oMsgStream modifiers
    MsgStream& operator<<(std::ostream& (*_f)(std::ostream&)) {
      if ( isActive() ) _f(m_stream);
      return *this;
    }

    /// Accept ios modifiers
    MsgStream& operator<<(std::ios& (*_f)(std::ios&)) {
      if ( isActive() ) _f(m_stream);
      return *this;
    }

    /// Accept MsgStream activation using MsgStreamer operator
    MsgStream& operator<< (MSG::Level level) {
      return report(level);
    }

    /// Accept MsgStream activation using MsgStreamer operator
    MsgStream& operator<<(long long arg) {
      try {
        // this may throw, and we cannot afford it if the stream is used in a catch block
        m_stream << arg;
      } catch (...) {}
      return *this;
    }
    /// @}

    /// Initialize report of new message: activate if print level is sufficient.
    MsgStream& report(MSG::Level lvl) {
      if ( ( m_currentLevel = MSG::Level(lvl) ) >= level() ) {
        activate();
      } else {
        deactivate();
      }
      return *this;
    }

    /// Activate MsgStream
    void activate() {
      m_active = true;
    }
    /// Deactivate MsgStream
    void deactivate() {
      m_active = false;
    }

    /// Accessor: is MsgStream active
    bool isActive() const {
      return m_active;
    }

    /// Check if logger is active at message level
    bool msgLevel(const MSG::Level lvl) const {
      return this->level() <= lvl;
    }

    /// IOS emulation
    /// @{
    long flags() const {
      return isActive() ? m_stream.flags()    : 0;
    }
    long flags(std::ios_base::fmtflags v) {
      return isActive() ? m_stream.flags(v)  :  0;
    }
    long setf(std::ios_base::fmtflags v) {
      return isActive() ? m_stream.setf(v)  :  0;
    }
    int width() const {
      return isActive() ? m_stream.width()    : 0;
    }
    int width(int v) {
      return isActive() ? m_stream.width(v)    : 0;
    }
    char fill() const {
      return isActive() ? m_stream.fill()     : (char)-1;
    }
    char fill(char v) {
      return isActive() ? m_stream.fill(v)     : (char)-1;
    }
    int precision() const  {
      return isActive() ? m_stream.precision(): 0;
    }
    int precision(int v) {
      return isActive() ? m_stream.precision(v): 0;
    }
    int rdstate() const  {
      return isActive() ? m_stream.rdstate () : std::ios_base::failbit;
    }
    int good() const  {
      return isActive() ? m_stream.good ()    : 0;
    }
    int eof() const  {
      return isActive() ? m_stream.eof ()     : 0;
    }
    int bad() const  {
      return isActive() ? m_stream.bad()      : 0;
    }
    long setf(std::ios_base::fmtflags _f, std::ios_base::fmtflags _m) {
      return isActive() ? m_stream.setf(_f, _m)   : 0;
    }
    void unsetf(std::ios_base::fmtflags _l)    {
      if ( isActive() ) m_stream.unsetf(_l);
    }
    void clear(std::ios_base::iostate _i = std::ios_base::failbit)  {
      if ( isActive() ) m_stream.clear(_i);
    }
    /// @}
    /// @}

    /// Print information about the log
    void print( const MSG::Level lvl );

    /// Output method
    MsgStream& doOutput();

    /// Access string MsgStream
    std::ostringstream& stream(){
      return m_stream;
    }

  private:

    /// Stream name used for identifying the origin from the message display
    std::string m_streamName;
    /// The message level where the stream should accept the input
    MSG::Level m_level;
    /// Message level for the current input string
    MSG::Level m_currentLevel;

    /// The auxiliary caching string:
    std::ostringstream m_stream;

    /// Whether we are active
    bool m_active;
};

/// MsgStream Modifier: endmsg. Calls the output method of the MsgStream
inline
MsgStream& endmsg(MsgStream& s) 
{
  return s.doOutput();
}
#define endreq endmsg


#if defined (__GNUC__)
inline MsgStream& operator << (MsgStream& s,
                               const std::_Setiosflags &manip) {
  try {
    // this may throw, and we cannot afford it if the stream is used in a catch block
    if ( s.isActive() ) s.stream() << manip;
  } catch(...) {}
  return s;
}
inline MsgStream& operator << (MsgStream& s,
                               const std::_Resetiosflags &manip)      {
  try {
    // this may throw, and we cannot afford it if the stream is used in a catch block
    if ( s.isActive() ) s.stream() << manip;
  } catch (...) {}
  return s;
}
inline MsgStream& operator << (MsgStream& s,
                               const std::_Setbase &manip)    {
  try {
    // this may throw, and we cannot afford it if the stream is used in a catch block
    if ( s.isActive() ) s.stream() << manip;
  } catch (...) {}
  return s;
}
inline MsgStream& operator << (MsgStream& s,
                               const std::_Setprecision &manip)       {
  try {
    // this may throw, and we cannot afford it if the stream is used in a catch block
    if ( s.isActive() ) s.stream() << manip;
  } catch (...) {}
  return s;
}
inline MsgStream& operator << (MsgStream& s,
                               const std::_Setw &manip)       {
  try {
    // this may throw, and we cannot afford it if the stream is used in a catch block
    if ( s.isActive() ) s.stream() << manip;
  } catch (...) {}
  return s;
}

namespace MSG {
  inline
  MsgStream& dec(MsgStream& log) {
    log.setf(std::ios_base::dec, std::ios_base::basefield);
    return log;
  }
  inline
  MsgStream& hex(MsgStream& log) {
    log.setf(std::ios_base::hex, std::ios_base::basefield);
    return log;
  }
}

#else // GCC, version << 3
/// I/O Manipulator for setfill
template<class _Tm> inline
MsgStream& operator << (MsgStream& s, const std::smanip<_Tm>& manip)  {
  try {
    // this may throw, and we cannot afford it if the stream is used in a catch block
    if ( s.isActive() ) s.stream() << manip;
  } catch (...) {}
  return s;
}
#endif    // WIN32 or (__GNUC__)

/// Specialization to avoid the generation of implementations for char[].
/// \see {<a href="https://savannah.cern.ch/bugs/?87340">bug #87340</a>}
inline MsgStream& operator<< (MsgStream& s, const char *arg){
  try {
    // this may throw, and we cannot afford it if the stream is used in a catch block
    if ( s.isActive() ) s.stream() << arg;
  } catch (...) {}
  return s;
}

/// General templated stream operator
template <typename T>
MsgStream& operator<< (MsgStream& lhs, const T& arg)  {
  if(lhs.isActive())
    try {
      // this may throw, and we cannot afford it if the stream is used in a catch block
      lhs.stream() << arg;
    }
    catch (...) {}
  return lhs;
}

#ifdef __GNUC__
/// compiler is stupid. Must specialize
template<typename T>
MsgStream& operator << (MsgStream& lhs, const std::_Setfill<T> &manip) {
  if ( lhs.isActive() )
    try {
      // this may throw, and we cannot afford it if the stream is used in a catch block
      lhs.stream() << manip;
    } catch(...) {}
  return lhs;
}
#endif

//==============================================================================
inline
std::ostream &operator<<( 
    std::ostream &stream, 
    MsgStream::Message &msg )
{
  return stream << msg.m_formatted_msg;
}

/**
 * @brief Adds messaging capability to interface
 **/
class IMsgService 
{

  protected:

    /// Stores the default stream name for this interface
    std::string m_defName;

    /// Default level for this interface
    MSG::Level m_defLevel;

  public:
    /**
     * @brief Builds default interface
     **/
    IMsgService()
      : m_defName("MsgStream"),
        m_defLevel( MSG::INFO ){;}

    /**
     * @brief Defines default log name and default level for Messaging Service
     **/
    IMsgService( const std::string& defName, 
                 const MSG::Level defLevel = MSG::INFO )
      : m_defName(defName),
        m_defLevel(defLevel){;}

    virtual ~IMsgService(){;}

    /// Check if stream will display at message level
    bool msgLevel(const MSG::Level lvl) const
    {
      return msg().msgLevel(lvl);
    }

    /// Change stream message level display
    void setMsgLevel(const MSG::Level lvl)
    {
      msg().setLevel(lvl);
    }

    /// Get Level of output from MsgStream Manager
    MSG::Level getMsgLevel() const 
    {
      return msg().level();
    }

    /// Change stream display name
    void setLogName(const std::string &name)
    {
      msg().setLogName(name);
    }

    /// Get stream name
    const std::string& getLogName() const 
    {
      return msg().logName();
    }

    /// Get streamer
    virtual MsgStream& msg() = 0;
    /// cv get streamer
    virtual MsgStream& msg() const = 0;
};


/**
 * @brief Consolidate message service capabilitiy
 **/
class MsgService : virtual public IMsgService
{
  public:

    /**
     * @brief Builds Message service using default configuration set by
     *        interface
     **/
    MsgService() :
      m_log( m_defName,  m_defLevel ){;}

    explicit MsgService(const int lvl)
      : m_log( m_defName, lvl){;}

    /**
     * @brief Builds Message service changing level
     **/
    explicit MsgService(const MSG::Level lvl)
      : m_log( m_defName, lvl){;}

    /// Retrieve log
    MsgStream& msg() override final {
      return m_log;
    }

    /// Const Retrieve log
    MsgStream& msg() const override final {
      return m_log;
    }

  private:
    /// Message Stream:
    mutable MsgStream m_log;
};

#endif
