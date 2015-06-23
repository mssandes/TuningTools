#
# Installs boost_1_58_0. It will take a while, only source this if you are sure
# that your system does not contains the boost needed version already installed.
#

test "x$1" = "x" -o "x$2" = "x" && echo "$0: Wrong number of arguments" && exit 1

NEW_ENV_FILE=$1
PYTHON_INCLUDE=$2

CXX=`root-config --cxx`


BOOST_LOCAL_PATH=$PWD
boost_include=$BOOST_LOCAL_PATH/include
boost_lib=$BOOST_LOCAL_PATH/lib
if test \! -f $boost_include/boost/python.hpp -o \! -d $boost_lib/
then
  if ! $CXX $PYTHON_INCLUDE -P boost_test.h
  then
    echo "It is needed to install boost python library." 
    test \! -f boost_1_58_0.tar.gz && wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz
    test \! -e boost_1_58_0 && echo -n "Extracting files..." && tar xfz boost_1_58_0.tar.gz && echo " done!"
    echo "Installing boost..."
    cd boost_1_58_0
    ./bootstrap.sh --prefix=$BOOST_LOCAL_PATH --with-libraries=python
    ./b2 install --prefix=$BOOST_LOCAL_PATH --with-python -j$ROOTCORE_NCPUS
    cd -
  else
    echo "Boost installed at file system" && exit 0
  fi
else
  echo "Boost needed libraries already installed."
fi

echo "test \"\${CPATH#*$boost_include}\" = \"\${CPATH}\" && export CPATH=$boost_include:\$CPATH || true" >> $NEW_ENV_FILE
echo "test \"\${LD_LIBRARY_PATH#*$boost_lib}\" = \"\${LD_LIBRARY_PATH}\" && export LD_LIBRARY_PATH=$boost_lib:\$LD_LIBRARY_PATH || true" >> $NEW_ENV_FILE
source $NEW_ENV_FILE || { echo "Couldn't set environment" && exit 1; }
`$CXX $PYTHON_INCLUDE -P boost_test.h` || { echo "Couldn't install boost" && exit 1; }
