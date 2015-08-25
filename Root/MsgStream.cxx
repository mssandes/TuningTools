#include "FastNetTool/system/MsgStream.h"

//VERBOSE < DEBUG < INFO < WARNING < ERROR < FATAL

constexpr unsigned MsgStream::Message::space_between_log_and_msg;
//==============================================================================
MsgStream& MsgStream::doOutput() 
{
  try {
    // This piece of code may throw and we cannot afford it when we print a
    // message in the middle of a catch block.
    if ( isActive() )   {
      Message msg(m_streamName, m_currentLevel, m_stream.str());
#if USE_OMP
      #pragma omp critical
#endif
      std::cout << msg << std::endl;
    }
    // Reset our stream
    m_stream.str("");
  } catch(...) {}
  return *this;
}

//==============================================================================
void MsgStream::print( const MSG::Level lvl )
{
  // FIXME Check if stream is empty and print its output before printing
  MSG::Level prev_lvl = m_currentLevel;
  m_stream << lvl << " stream with message level "
    << to_str(prev_lvl) << " and logname \"" 
    << m_streamName << "\"";
  doOutput();
}

