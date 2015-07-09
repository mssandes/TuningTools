#
# Installs boost_1_58_0. It will take a while, only source this if you are sure
# that your system does not contains the boost needed version already installed.
#

test "x$1" = "x" -o "x$2" = "x" -o "x$3" = "x" && echo "$0: Wrong number of arguments" && exit 1

MAKEFILE=$1
NEW_ENV_FILE=$2
PYTHON_INCLUDE=$3

CXX=`root-config --cxx`


BOOST_LOCAL_PATH=\$ROOTCOREBIN/../FastNetTool/cmt
boost_include=$BOOST_LOCAL_PATH/include
boost_lib=$BOOST_LOCAL_PATH/lib
if test \! -f `eval echo "$boost_include/boost/python.hpp"` -o \! -d `eval echo "$boost_lib/"`
then
  if ! $CXX $PYTHON_INCLUDE -P boost_test.h > /dev/null 2> /dev/null
  then
    echo "It is needed to install boost python library." 
    test \! -f boost_1_58_0.tar.gz && wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz
    test \! -e boost_1_58_0 && echo -n "Extracting files..." && tar xfz boost_1_58_0.tar.gz && echo " done!"
    echo "Installing boost..."
    cd boost_1_58_0
    if ./bootstrap.sh --prefix=`eval echo "$BOOST_LOCAL_PATH"` --with-libraries=python > /dev/null
    then
      echo "Finished setting bootstrap successfully."
    else
      echo "Couldn't source bootstrap.sh." && exit 1
    fi
    if ./b2 install --prefix=`eval echo "$BOOST_LOCAL_PATH"` --with-python -j$ROOTCORE_NCPUS > /dev/null
    then
      echo "Sucessfully compiled boost."
    else
      echo "Couldn't install boost." && exit 1
    fi
    cd - > /dev/null
    sleep 3
  else
    echo "Boost installed at file system" && exit 0
  fi
else
  echo "Boost needed libraries already installed."
fi

old_field=`$ROOTCOREDIR/scripts/get_field.sh $MAKEFILE PACKAGE_LDFLAGS`
if test "${old_field#*-L$boost_lib}" = "$old_field"
then
  $ROOTCOREDIR/scripts/set_field.sh $MAKEFILE PACKAGE_LDFLAGS "$old_field -L$boost_lib"  
fi
arch=`root-config --arch`
if test "$arch" = "macosx64"
then
  include_marker="-isystem "
else
  include_marker=-I
fi
old_field=`$ROOTCOREDIR/scripts/get_field.sh $MAKEFILE PACKAGE_CXXFLAGS`
if test "${old_field#*$include_marker$boost_lib}" = "$old_field"
then
  $ROOTCOREDIR/scripts/set_field.sh $MAKEFILE PACKAGE_CXXFLAGS "$old_field $include_marker$boost_include"  
fi

echo "test \"\${CPATH#*$boost_include}\" = \"\${CPATH}\" && export CPATH=$boost_include:\$CPATH || true" >> $NEW_ENV_FILE
echo "test \"\${LD_LIBRARY_PATH#*$boost_lib}\" = \"\${LD_LIBRARY_PATH}\" && export LD_LIBRARY_PATH=$boost_lib:\$LD_LIBRARY_PATH || true" >> $NEW_ENV_FILE
if test "$arch" = "macosx64"
then
  echo "test \"\${DYLD_LIBRARY_PATH#*$boost_lib}\" = \"\${DYLD_LIBRARY_PATH}\" && export DYLD_LIBRARY_PATH=$boost_lib:\$DYLD_LIBRARY_PATH || true" >>  $NEW_ENV_FILE
fi
source $NEW_ENV_FILE || { echo "Couldn't set environment" && exit 1; }
# Final test:
echo -n "Checking boost installation..." && { `$CXX $PYTHON_INCLUDE -P boost_test.h > /dev/null 2> /dev/null` || { echo "Couldn't install boost!" && exit 1; } && echo " sucessfully installed!"; }
