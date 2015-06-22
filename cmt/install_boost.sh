#
# Installs boost_1_58_0. It will take a while, only source this if you are sure
# that your system does not contains the boost needed version already installed.
#

# FIXME:
# - We may want this to check if the boost is already installed in
# standard places and contains the python library. 
# - We would also want this to install only the boost python library. 

if ! $ROOTCOREDIR/scripts/test_cc.sh compile boost_test.h
then
  echo "It is needed to install boost python library." 
  BOOST_LOCAL_PATH=$PWD
  if test ! -f boost_1_58_0.tar.gz
  then
    wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz
  fi
  tar xfvz boost_1_58_0.tar.gz
  cd boost_1_58_0
  ./bootstrap.sh --prefix=$BOOST_LOCAL_PATH --with-libraries=python
  ./b2 install --prefix=$BOOST_LOCAL_PATH --with-python
  cd -
  boost_include=$BOOST_LOCAL_PATH/include
  boost_lib=$BOOST_LOCAL_PATH/lib
  echo "test \"\${PATH#*$boost_include}\" = \"\${PATH}\" && export PATH=$boost_include:\$PATH" >> $NEW_ENV_FILE
  echo "test \"\${LD_LIBRARY_PATH#*$boost_lib}\" = \"\${LD_LIBRARY_PATH}\" && export LD_LIBRARY_PATH=$boost_lib:\$LD_LIBRARY_PATH" >> $NEW_ENV_FILE
  source $NEW_ENV_FILE
  if ! $ROOTCOREDIR/scripts/test_cc.sh compile boost_test.h
  then
    echo "Couldn't install boost"
    exit 1
  fi
else
  echo "Boost needed libraries already installed."
fi


source boostsetup.sh
