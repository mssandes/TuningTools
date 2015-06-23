#
# Installs boost_1_58_0. It will take a while, only source this if you are sure
# that your system does not contains the boost needed version already installed.
#

test "x$1" = "x" -o "x$2" = "x" -o "x$3" = "x" && echo "$0: Wrong number of arguments" && exit 1

MAKEFILE=$1
NEW_ENV_FILE=$2
PYTHON_INCLUDE=$3

CXX=`root-config --cxx`


BOOST_LOCAL_PATH=$PWD
boost_include=$BOOST_LOCAL_PATH/include
boost_lib=$BOOST_LOCAL_PATH/lib
if test \! -f $boost_include/boost/python.hpp -o \! -d $boost_lib/
then
  if ! $CXX $PYTHON_INCLUDE -P boost_test.h > /dev/null 2> /dev/null
  then
    echo "It is needed to install boost python library." 
    test \! -f boost_1_58_0.tar.gz && wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz
    test \! -e boost_1_58_0 && echo -n "Extracting files..." && tar xfz boost_1_58_0.tar.gz && echo " done!"
    echo "Installing boost..."
    cd boost_1_58_0
    ./bootstrap.sh --prefix=$BOOST_LOCAL_PATH --with-libraries=python
    ./b2 install --prefix=$BOOST_LOCAL_PATH --with-python -j$ROOTCORE_NCPUS
    cd -
    sleep 3
  else
    echo "Boost installed at file system" && exit 0
  fi
else
  echo "Boost needed libraries already installed."
fi

echo "OLD ENV"
cat $MAKEFILE
env
old_field=`$ROOTCOREDIR/scripts/get_field.sh $MAKEFILE PACKAGE_LDFLAGS`
if test "${old_field#*-L$boost_lib}" = "$old_field"
then
  $ROOTCOREDIR/scripts/set_field.sh $MAKEFILE PACKAGE_LDFLAGS "$old_field -L$boost_lib"  
fi

echo "test \"\${CPATH#*$boost_include}\" = \"\${CPATH}\" && export CPATH=$boost_include:\$CPATH || true" >> $NEW_ENV_FILE
echo "test \"\${LD_LIBRARY_PATH#*$boost_lib}\" = \"\${LD_LIBRARY_PATH}\" && export LD_LIBRARY_PATH=$boost_lib:\$LD_LIBRARY_PATH || true" >> $NEW_ENV_FILE
if test "`root-config --arch`" = "macosx64"
then
  echo "test \"\${DYLD_LIBRARY_PATH#*$boost_lib}\" = \"\${DYLD_LIBRARY_PATH}\" && export DYLD_LIBRARY_PATH=$boost_lib:\$DYLD_LIBRARY_PATH || true" >>  $NEW_ENV_FILE
fi
source $NEW_ENV_FILE || { echo "Couldn't set environment" && exit 1; }
`$CXX $PYTHON_INCLUDE -P boost_test.h > /dev/null 2> /dev/null` || { echo "Couldn't install boost" && exit 1; }

echo "NEW ENV"
cat $MAKEFILE
env
