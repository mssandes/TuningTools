#ifndef BOOST_TEST
#define BOOST_TEST
#include <boost/python.hpp>
namespace py = boost::python;
int main() { py::list l; return 0; }
BOOST_PYTHON_MODULE(mymodule)
{}
#endif // BOOST_TEST
