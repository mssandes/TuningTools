#ifndef FASTNETTOOL_MSGSTREAM_H
#define FASTNETTOOL_MSGSTREAM_H

#include <string>
#include <iostream>
#include <sstream>

#define SPACE_BETWEEN_LOG_AND_MSG	45// 32
#define MSG_RED				            "\033[1;31m"
#define MSG_REDBOX		          	"\033[1;41;1m"
#define MSG_NONE		            	"\033[0m"

#define MSG_DEBUG(CLASS,MSG) do { \
  if ( CLASS->level() <= ::msg::DEBUG ) { \
    std::ostringstream s; s << MSG; CLASS->debug(s.str()); \
  } \
} while(0) 
#define MSG_INFO(CLASS,MSG)  do { \
  if ( CLASS->level() <= ::msg::INFO ) { \
    std::ostringstream s; s << MSG; CLASS->info(s.str()); \
  } \
} while(0)
#define MSG_WARNING(CLASS,MSG) do { \
  if ( CLASS->level() <= ::msg::WARNING ) { \
    std::ostringstream s; s << MSG; CLASS->warning(s.str()); \
  } \
} while(0)
#define MSG_ERROR(CLASS,MSG) do { \
  if ( CLASS->level() <= ::msg::ERROR ) { \
    std::ostringstream s; s << MSG; CLASS->error(s.str()); \
  } \
}  while (0)
#define MSG_FATAL(CLASS,MSG) do { \
  if ( CLASS->level() <= ::msg::FATAL ) { \
    std::ostringstream s; s << MSG; CLASS->fatal(s.str()); \
  } \
} while (0)

namespace msg
{

enum Level {
  VERBOSE = 0,
  DEBUG   = 1,
  WARNING = 2,
  INFO    = 3,
  ERROR   = 4,
  FATAL   = 5
};

class MsgStream
{

  public:
    MsgStream();
    MsgStream(std::string logname, Level msglevel);
    ~MsgStream();
    
    void setLevel(Level msglevel);
    void setLogName(std::string logname);
    void print();

    void debug(  std::string msg);
    void warning(std::string msg);
    void info(   std::string msg);
    void error(  std::string msg);
    void fatal(  std::string msg);

    Level level() const
    { 
      return m_msgLevel;
    }

  private:
    std::string m_logName;
    Level m_msgLevel;
};

} // namespace msg

#endif
