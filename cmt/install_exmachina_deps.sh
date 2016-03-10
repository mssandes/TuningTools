# Retrieve python information from RingerCore script
source "$ROOTCOREBIN/../RootCoreMacros/retrieve_python_info.sh" \
    || { echo "Couldn't load python information." && exit 1;}
# We also source the RingerCore to garantee that we get the numpy path if it
# was locally installed
source "$ROOTCOREBIN/../RingerCore/cmt/$BASE_NEW_ENV_FILE" 

# TODO Check if python import errors are due to no package or due to an import error.
# TODO Check if binaries or libraries are already on the install_path
# TODO Put source and temp binaries to be on tmp dir

# Files version when afs not available:
cython_version=Cython-0.23.4.tar.gz
sklearn_version=0.17.tar.gz
cmake_version=cmake-3.4.1.tar.gz
armadillo_version=armadillo-6.400.3.tar.gz

## pip
pip_install_file=$DEP_AREA/get-pip.py
pip_install_path=$INSTALL_AREA/pip; pip_install_path_bslash=$INSTALL_AREA_BSLASH/pip
if test \! -f $pip_install_file; then
  echo "Downloading ${pip_install_file}..."
  curl -s -o $pip_install_file https://bootstrap.pypa.io/get-pip.py || { echo "Couldn't download pip!" && return 1; }
fi
pip_version=$(pip --version | grep -o "python [[:digit:]].[[:digit:]]" | sed "s/python /python/")
if ! type pip > /dev/null 2>&1 || test "$pip_version" != "$PYTHON_LIB_VERSION"; then
  echo "Installing pip..."
  if test -e $pip_install_path; then
    rm -r $pip_install_path || { echo "Couldn't remove old installed pip. Please remove it manually on path \"$pip_install_path\" and try again." && return 1; }
  fi
  python $pip_install_file --root $pip_install_path --ignore-installed > /dev/null || { echo "Couldn't install pip." && return 1; }
  mv $(dirname $(find $pip_install_path -name "pip" -type f)) $pip_install_path/bin
  mv $(find $pip_install_path -name "site-packages" -type d) $pip_install_path
  rm -r $(find $pip_install_path  -maxdepth 1 -mindepth 1 -not -name "site-packages" -a -not -name "bin")
  sed -i.bak "s_#!.*_#!/usr/bin/env python_" $pip_install_path/bin/pip
else
  echo "No need to install pip."
fi
test -d "$pip_install_path" && export pip_install_path_bslash
test -d "$pip_install_path/bin" && add_to_env_file PATH "$pip_install_path_bslash/bin"
test -d "$pip_install_path/site-packages" && add_to_env_file PYTHONPATH "$pip_install_path_bslash/site-packages"
source $NEW_ENV_FILE

# Cython
cython_tgz_file=$DEP_AREA/cython.tgz
cython_install_path=$INSTALL_AREA/cython; cython_install_path_bslash=$INSTALL_AREA_BSLASH/cython
if test \! -f $cython_tgz_file; then
  echo "Downloading ${cython_tgz_file}..."
  cython_afs_path=/afs/cern.ch/user/w/wsfreund/public/misc/cython.tgz
  if test -f $cython_afs_path; then
    cp $cython_afs_path $cython_tgz_file
  else
    curl -s -o $cython_tgz_file http://cython.org/release/${cython_version} || { echo "Couldn't download Cython!" && return 1; }
  fi
fi
if ! python -c "import Cython" > /dev/null 2>&1; then
  echo "Installing Cython..."
  cython_folder=$(tar xfzv $cython_tgz_file --skip-old-files -C $DEP_AREA 2> /dev/null)
  test -z "$cython_folder" && { echo "Couldn't extract Cython!" && return 1;}
  cython_folder=$(echo $cython_folder | cut -f1 -d ' ' )
  cython_folder=$DEP_AREA/${cython_folder%%\/*};
  if test -e $cython_install_path; then
    rm -r $cython_install_path || { echo "Couldn't remove old installed cython. Please remove it manually on path \"$cython_install_path\" and try again." && return 1; }
  fi
  mkdir -p $cython_install_path
  cd $cython_folder; tmp_cython_install_folder="$cython_install_path/lib/$PYTHON_LIB_VERSION/site-packages/"
  mkdir -p $tmp_cython_install_folder
  export PYTHONPATH="$tmp_cython_install_folder:$PYTHONPATH"
  python setup.py install --prefix $cython_install_path > /dev/null || { echo "Couldn't install cython." && return 1;}
  cd - > /dev/null
  mv $(find $cython_install_path -name "site-packages" -type d) $cython_install_path
  rm -r $(find $cython_install_path  -maxdepth 1 -mindepth 1 -not -name "site-packages" -a -not -name "bin")
else
  echo "No need to install cython."
fi
test -d "$cython_install_path" && export cython_install_path_bslash
test -d "$cython_install_path/bin" && add_to_env_file PATH "$cython_install_path_bslash/bin"
test -d "$cython_install_path/site-packages" && add_to_env_file PYTHONPATH "$cython_install_path_bslash/site-packages"
source $NEW_ENV_FILE

# TODO ATLAS, blas

# scipy
scipy_tgz_file=$DEP_AREA/scipy.tgz
scipy_install_path=$INSTALL_AREA/scipy; scipy_install_path_bslash=$INSTALL_AREA_BSLASH/scipy
if test \! -f $scipy_tgz_file; then
  scipy_afs_path=/afs/cern.ch/user/w/wsfreund/public/misc/scipy.tgz
  if test -f $scipy_afs_path; then
    cp $scipy_afs_path $scipy_tgz_file
  else
    wget -q -O $scipy_tgz_file "http://sourceforge.net/projects/scipy/files/latest/download\?source\=files"
  fi
fi
if ! python -c "import scipy.linalg" > /dev/null 2>&1; then
  echo "Installing scipy..."
  scipy_folder=$(tar xfzv $scipy_tgz_file --skip-old-files -C $DEP_AREA 2> /dev/null)
  test -z "$scipy_folder" && { echo "Couldn't extract scipy!" && return 1;}
  scipy_folder=$DEP_AREA/$(echo $scipy_folder | cut -f1 -d ' ' )
  if test -e $scipy_install_path; then
    rm -r $scipy_install_path || { echo "Couldn't remove old installed scipy. Please remove it manually on path \"$scipy_install_path\" and try again." && return 1; }
  fi
  mkdir -p $scipy_install_path
  cd $scipy_folder
  python setup.py install --prefix $scipy_install_path > /dev/null || { echo "Couldn't install scipy." && return 1;}
  cd - > /dev/null
  mv $(find $scipy_install_path -name "site-packages" -type d) $scipy_install_path
  rm -r $(find $scipy_install_path  -maxdepth 1 -mindepth 1 -not -name "site-packages" -a -not -name "bin")
else
  echo "No need to install scipy."
fi
test -d "$scipy_install_path" && export scipy_install_path_bslash
test -d "$scipy_install_path/site-packages" && add_to_env_file PYTHONPATH "$scipy_install_path_bslash/site-packages"
source $NEW_ENV_FILE

# Sklearn
sklearn_tgz_file=$DEP_AREA/sklearn.tgz
sklearn_install_path=$INSTALL_AREA/sklearn; sklearn_install_path_bslash=$INSTALL_AREA_BSLASH/sklearn
if test \! -f $sklearn_tgz_file; then
  sklearn_afs_path=/afs/cern.ch/user/w/wsfreund/public/misc/sklearn.tgz
  if test -f $sklearn_afs_path; then
    cp $sklearn_afs_path $sklearn_tgz_file
  else
    wget -q -O $sklearn_tgz_file https://github.com/scikit-learn/scikit-learn/archive/${sklearn_version}
  fi
fi
if ! { python -c "import sklearn" > /dev/null 2>&1 || \
    { find "$sklearn_install_path/site-packages" -maxdepth 0 -empty 2>/dev/null | read v; } }
then
  echo "Installing sklearn..."
  sklearn_folder=$(tar xfzv $sklearn_tgz_file --skip-old-files -C $DEP_AREA 2> /dev/null)
  test -z "$sklearn_folder" && { echo "Couldn't extract sklearn!" && return 1;}
  sklearn_folder=$DEP_AREA/$(echo $sklearn_folder | cut -f1 -d ' ' )
  if test -e $sklearn_install_path; then
    rm -r $sklearn_install_path || { echo "Couldn't remove old installed sklearn. Please remove it manually on path \"$sklearn_install_path\" and try again." && return 1; }
  fi
  mkdir -p $sklearn_install_path
  cd $sklearn_folder
  python setup.py install --prefix $sklearn_install_path > /dev/null || { echo "Couldn't install sklearn." && return 1;}
  cd - > /dev/null
  mv $(find $sklearn_install_path -name "site-packages" -type d) $sklearn_install_path
  rm -r $(find $sklearn_install_path  -maxdepth 1 -mindepth 1 -not -name "site-packages" -a -not -name "bin")
else
  echo "No need to install sklearn."
fi
test -d "$sklearn_install_path" && export sklearn_install_path_bslash
test -d "$sklearn_install_path/site-packages" && add_to_env_file PYTHONPATH "$sklearn_install_path_bslash/site-packages"
source $NEW_ENV_FILE

# Cmake
cmake_tgz_file=$DEP_AREA/cmake.tgz
cmake_install_path=$INSTALL_AREA/cmake; cmake_install_path_bslash=$INSTALL_AREA_BSLASH/cmake
if test \! -f $cmake_tgz_file; then
  cmake_afs_path=/afs/cern.ch/user/w/wsfreund/public/misc/cmake.tgz
  if test -f $cmake_afs_path; then
    cp $cmake_afs_path $cmake_tgz_file
  else
    wget -q -O $cmake_tgz_file https://cmake.org/files/v3.4/${cmake_version}
  fi
fi
if ! { type cmake > /dev/null 2>&1 || test -f "$cmake_install_path/bin/cmake"; }; then
  echo "Installing cmake..."
  cmake_folder=$(tar xfzv $cmake_tgz_file --skip-old-files -C $DEP_AREA 2> /dev/null)
  test -z "$cmake_folder" && { echo "Couldn't extract cmake!" && return 1;}
  cmake_folder=$(echo $cmake_folder | cut -f1 -d ' ' )
  cmake_folder=$DEP_AREA/${cmake_folder%%\/*}
  if test -e $cmake_install_path; then
    rm -r $cmake_install_path || { echo "Couldn't remove old installed cmake. Please remove it manually on path \"$cmake_install_path\" and try again." && return 1; }
  fi
  mkdir -p $cmake_install_path
  cd $cmake_folder
  ./bootstrap --prefix=$cmake_install_path --parallel=$ROOTCORE_NCPUS > /dev/null || { echo "Couldn't bootstrap cmake." && return 1;}
  make install > /dev/null || { echo "Couldn't make cmake." && return 1; }
  cd - > /dev/null
else
  echo "No need to install cmake."
fi
test -d "$cmake_install_path" && export cmake_install_path_bslash
test -d "$cmake_install_path/bin" && add_to_env_file PATH "$cmake_install_path_bslash/bin"
source $NEW_ENV_FILE

# Armadillo
armadillo_tgz_file=$DEP_AREA/armadillo.tgz
armadillo_install_path=$INSTALL_AREA/armadillo; armadillo_install_path_bslash=$INSTALL_AREA_BSLASH/armadillo
if test \! -f $armadillo_tgz_file; then
  armadillo_afs_path=/afs/cern.ch/user/w/wsfreund/public/misc/armadillo.tgz
  if test -f $armadillo_afs_path; then
    cp $armadillo_afs_path $armadillo_tgz_file
  else
    wget -q -O $armadillo_tgz_file http://sourceforge.net/projects/arma/files/${armadillo_version}
  fi
fi
if ! { ldconfig -p | grep "libarmadillo" || find_lib "libarmadillo" > /dev/null 2>&1; }; then
  echo "Installing armadillo..."
  armadillo_folder=$(tar xfzv $armadillo_tgz_file --skip-old-files -C $DEP_AREA 2> /dev/null)
  test -z "$armadillo_folder" && { echo "Couldn't extract armadillo!" && return 1;}
  armadillo_folder=$DEP_AREA/$(echo $armadillo_folder | cut -f1 -d ' ' )
  if test -e $armadillo_install_path; then
    rm -r $armadillo_install_path || { echo "Couldn't remove old installed armadillo. Please remove it manually on path \"$armadillo_install_path\" and try again." && return 1; }
  fi
  mkdir -p $armadillo_install_path
  cd $armadillo_folder
  cmake -DCMAKE_INSTALL_PREFIX=$armadillo_install_path . > /dev/null || { echo "Couldn't prepare armadillo installation." && return 1;}
  make install > /dev/null || { echo "Couldn't install armadillo." && return 1; }
  cd - > /dev/null
else
  echo "No need to install armadillo."
fi
test -d "$armadillo_install_path" && export armadillo_install_path_bslash
test -d "$armadillo_install_path/lib" && add_to_env_file LD_LIBRARY_PATH "$armadillo_install_path_bslash/lib"
export ARMADILLO_INCLUDE_PATH=$armadillo_install_path/include
export ARMADILLO_LIB_PATH=$armadillo_install_path/lib
source $NEW_ENV_FILE


# Exmachina
exmachina_folder=$DEP_AREA/ExMachina; exmachina_folder_bslash=$DEP_AREA_BSLASH/ExMachina
exmachina_install_path=$INSTALL_AREA/ExMachina; exmachina_install_path_bslash=$INSTALL_AREA_BSLASH/ExMachina
if test \! -d $exmachina_folder; then
  git clone https://github.com/Tiamaty/ExMachina.git $exmachina_folder
fi
if ! python -c "import exmachina" > /dev/null 2>&1; then
  echo "Installing ExMachina..."
  if test -e $exmachina_install_path; then
    rm -r $exmachina_install_path || { echo "Couldn't remove old installed exmachina. Please remove it manually on path \"$exmachina_install_path\" and try again." && return 1; }
  fi
  mkdir -p $exmachina_install_path
  cd $exmachina_folder
  oldCXX=$CXX; test -z $CXX && export CXX=g++
  oldCC=$CC; test -z $CC && export CC=gcc
  python setup.py build_ext --build-lib="$exmachina_install_path" > /dev/null || { echo "Couldn't install ExMachina." && return 1;}
  export CXX=$oldCXX;
  export CC=$oldCC;
  cd - > /dev/null
else
  echo "No need to install ExMachina."
fi
test -d "$exmachina_install_path" && export exmachina_install_path_bslash
test -d "$exmachina_install_path" && add_to_env_file LD_LIBRARY_PATH "$exmachina_install_path_bslash"
test -d "$exmachina_install_path" && add_to_env_file PYTHONPATH "$exmachina_install_path_bslash"
test -d "$exmachina_install_path" && add_to_env_file PYTHONPATH "$exmachina_folder_bslash"
source $NEW_ENV_FILE
