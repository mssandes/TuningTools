

#include <iostream>
#include <string>
#include <vector>
#include <boost/python.hpp>
#include "FastNetTool/system/util.h"
#include "FastNetTool/system/defines.h"


namespace py = boost::python;
using namespace std;

class FastNetTool{

  private:
    
    vector<REAL> m_nNodes;


  public:
    
    FastNetTool()
    {
    };


    void setNodes(py::list nodes){
      m_nNodes = util::to_std_vector<REAL>(nodes);
    };

}



BOOST_PYTHON_MODULE(FastNetTool){

  class_<FastNetTool>("FastNetTool")
    .def("setNodes", &FastNetTool::setNodes)
  ;
}


