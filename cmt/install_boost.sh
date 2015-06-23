#
# Installs boost_1_58_0. It will take a while, only source this if you are sure
# that your system does not contains the boost needed version already installed.
#

test "x$1" = "x" -o "x$2" = "x" && echo "$0: Wrong number of arguments" && exit 1

CXX=`root-config --cxx`

if ! $CXX $2 -P boost_test.h > /dev/null 2> /dev/null
then
  NEW_ENV_FILE=$1
  echo "It is needed to install boost python library." 
  BOOST_LOCAL_PATH=$PWD
  test ! -f boost_1_58_0.tar.gz && wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz
  tar xfz boost_1_58_0.tar.gz
  cd boost_1_58_0
  ./bootstrap.sh --prefix=$BOOST_LOCAL_PATH --with-libraries=python
  ./b2 install --prefix=$BOOST_LOCAL_PATH --with-python -j$ROOTCORE_NCPUS
  cd -
  boost_include=$BOOST_LOCAL_PATH/include
  boost_lib=$BOOST_LOCAL_PATH/lib
  echo "test \"\${CPATH#*$boost_include}\" = \"\${CPATH}\" && export CPATH=$boost_include:\$CPATH" >> $NEW_ENV_FILE
  echo "test \"\${LD_LIBRARY_PATH#*$boost_lib}\" = \"\${LD_LIBRARY_PATH}\" && export LD_LIBRARY_PATH=$boost_lib:\$LD_LIBRARY_PATH" >> $NEW_ENV_FILE
  source $NEW_ENV_FILE && echo "Couldn't set environment" && exit 1
  `$CXX $2 -P boost_test.h` && echo "Couldn't install boost" && exit 1
else
  echo "Boost needed libraries already installed."
fi
