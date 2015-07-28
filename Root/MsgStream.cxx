
#include "FastNetTool/system/MsgStream.h"

//VERBOSE < DEBUG < INFO < WARNING < ERROR < FATAL
using namespace msg;

//==============================================================================
MsgStream::MsgStream()
{
  m_msgLevel = VERBOSE;
  m_logName  = "none";
}

//==============================================================================
MsgStream::MsgStream(std::string logname, Level msglevel)
{
  m_msgLevel = msglevel;
  m_logName  = logname;
}

//==============================================================================
MsgStream::~MsgStream(){ }

//==============================================================================
void MsgStream::setLevel(Level msglevel)
{
  m_msgLevel = msglevel;
}

//==============================================================================
void MsgStream::setLogName(std::string logname)
{
  m_logName = logname;
}

//==============================================================================
void MsgStream::print()
{
  
  std::cout.fill(' ');
  std::cout << m_logName;
  std::cout.width(SPACE_BETWEEN_LOG_AND_MSG-m_logName.length());
  std::cout << "INFO " << "logfile with level messenge = " << m_msgLevel << " and logname = " << m_logName  << std::endl;

}

//==============================================================================
void MsgStream::debug(std::string msg)
{
  if(m_msgLevel <= DEBUG){
    std::cout.fill(' ');
    std::cout << m_logName;
    std::cout.width(SPACE_BETWEEN_LOG_AND_MSG-m_logName.length());
    std::cout << "DEBUG " << msg << std::endl;    
  }
}

//==============================================================================
void MsgStream::info(std::string msg)
{
  if(m_msgLevel <= INFO){
    std::cout.fill(' ');
    std::cout << m_logName;
    std::cout.width(SPACE_BETWEEN_LOG_AND_MSG-m_logName.length());
    std::cout << "INFO " << msg << std::endl;
  }
}

//==============================================================================
void MsgStream::warning(std::string msg)
{
  if(m_msgLevel <= WARNING){
    std::cout.fill(' ');
    std::cout << m_logName;
    std::cout.width(SPACE_BETWEEN_LOG_AND_MSG-m_logName.length());
    std::cout << MSG_RED <<"WARNING " << msg << MSG_NONE << std::endl;
  }
}

//==============================================================================
void MsgStream::error(std::string msg)
{
  if(m_msgLevel <= ERROR){
    std::cout.fill(' ');
    std::cout << m_logName;
    std::cout.width(SPACE_BETWEEN_LOG_AND_MSG-m_logName.length());
    std::cout << MSG_RED <<"ERROR " << msg << MSG_NONE << std::endl;
  }
}

//==============================================================================
void MsgStream::fatal(std::string msg)
{
  if(m_msgLevel <= FATAL){
    std::cout.fill(' ');
    std::cout << m_logName;
    std::cout.width(SPACE_BETWEEN_LOG_AND_MSG-m_logName.length());
    std::cout << MSG_REDBOX <<"FATAL " << msg << MSG_NONE << std::endl;
  }
}
