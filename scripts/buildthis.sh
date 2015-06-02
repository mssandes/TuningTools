
# Setup a fresh directory for the trigger tutorial.
# Config for asetup... have to do this funny
#since setupATLAS isn't passed in, but this is what setupATLASf
# does (when not run via a source command in bash).

#export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
#source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh

source ~/public/setupDQ2.zsh
rcSetup Base,2.3.X,rel_4
# Now build everything
rc find_packages
#rc compile
PYTHONFASTNET_UTIL_PATH="$ROOTCOREBIN/../FastNetTool/python/"
PYTHONFASTNET_PATH="$ROOTCOREBIN/lib/x86_64-slc6-gcc48-opt/"
#fix for numpy
export PYTHONPATH="/afs/cern.ch/sw/lcg/external/pyanalysis/1.0_python2.6/x86_64-slc5-gcc43-opt/lib/python2.6/site-packages:$LD_LIBRARY_PATH:$PYTHONFASTNET_PATH:$PYTHONFASTNET_UTIL_PATH"


