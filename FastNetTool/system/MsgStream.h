
#ifndef FASTNETTOOL_MSGSTREAM_H
#define FASTNETTOOL_MSGSTREAM_H

#include <string>
#include <iostream>
#include <sstream>

#define SPACE_BETWEEN_LOG_AND_MSG	45// 32
#define MSG_RED				            "\033[1;31m"
#define MSG_REDBOX		          	"\033[1;41;1m"
#define MSG_NONE		            	"\033[0m"

#define MSG_DEBUG(CLASS,MSG){ std::ostringstream s; s << MSG; CLASS->debug(s.str());} 
#define MSG_INFO(CLASS,MSG){ std::ostringstream s; s << MSG; CLASS->info(s.str());} 
#define MSG_WARNING(CLASS,MSG){ std::ostringstream s; s << MSG; CLASS->warning(s.str());} 
#define MSG_ERROR(CLASS,MSG){ std::ostringstream s; s << MSG; CLASS->error(s.str());} 
#define MSG_FATAL(CLASS,MSG){ std::ostringstream s; s << MSG; CLASS->fatal(s.str());} 
//typedef std::ostringstream os;

namespace msg{
//VERBOSE < DEBUG < INFO < WARNING < ERROR < FATAL
  enum Level{
    VERBOSE,
    DEBUG,
    WARNING,
    INFO,
    ERROR,
    FATAL
  };

  class MsgStream{

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

    private:
      std::string m_logName;
      Level m_msgLevel;
  };
}// namespace
#endif


