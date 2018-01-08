################################################################################
# Package: TuningTools
################################################################################

# Declare the package name:
atlas_subdir( TuningTools )

if( XAOD_STANDALONE )
  # We don't install TuningTools on GRID if we are on XAOD_STANDALONE mode
  if( NOT DEFINED $ENV{PanDA_TaskID} )
    # Declare the package's dependencies:
    #atlas_depends_on_subdirs( PUBLIC
    #  ElectronPhotonSelectorTools
    #  RingerSelectorTools
    #  )
    atlas_depends_on_subdirs( PUBLIC
                              TuningTools
                              RingerCore
                              PRIVATE
                              ${extra_dep}
                              )

    # External dependencies:
		find_package( Boost COMPONENTS python )
		#find_package( ROOT COMPONENTS Core Hist Gpad Tree MathCore Hist MathMore RIO )
		find_package( ROOT COMPONENTS Core Hist Tree RIO Hist )
    set( Python_ADDITIONAL_VERSIONS 2.7.13 2.7.5)
    find_package( PythonInterp REQUIRED)
    find_package( PythonLibs REQUIRED)

    ## Find numpy (taken from caffe):
    ## Install numpy instead:
    ## https://github.com/UCATLAS/xAODAnaHelpers/blob/master/CMakeLists.txt
    if(PYTHONINTERP_FOUND)
      execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
        "import numpy as n; print(n.__version__); print(n.get_include());"
        RESULT_VARIABLE __result
        OUTPUT_VARIABLE __output
        OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(__result MATCHES 0)
        string(REGEX REPLACE ";" "\\\\;" __values ${__output})
        string(REGEX REPLACE "\r?\n" ";"    __values ${__values})
        list(GET __values 0 NUMPY_VERSION)
        list(GET __values 1 NUMPY_INCLUDE_DIR)

        string(REGEX MATCH "^([0-9])+\\.([0-9])+\\.([0-9])+" __ver_check "${NUMPY_VERSION}")
        if(NOT "${__ver_check}" STREQUAL "")
          set(NUMPY_VERSION_MAJOR ${CMAKE_MATCH_1})
          set(NUMPY_VERSION_MINOR ${CMAKE_MATCH_2})
          set(NUMPY_VERSION_PATCH ${CMAKE_MATCH_3})
          math(EXPR NUMPY_VERSION_DECIMAL
            "(${NUMPY_VERSION_MAJOR} * 10000) + (${NUMPY_VERSION_MINOR} * 100) + ${NUMPY_VERSION_PATCH}")
          string(REGEX REPLACE "\\\\" "/"  NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR})
        else()
         unset(NUMPY_VERSION)
         unset(NUMPY_INCLUDE_DIR)
         message(STATUS "Requested NumPy version and include path, but got instead:\n${__output}\n")
        endif()
      endif()
    else()
      message(STATUS "To find NumPy Python interpretator is required to be found.")
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NumPy REQUIRED_VARS NUMPY_INCLUDE_DIR NUMPY_VERSION
                                            VERSION_VAR   NUMPY_VERSION)

    if(NUMPY_FOUND)
      message(STATUS "NumPy ver. ${NUMPY_VERSION} found (include: ${NUMPY_INCLUDE_DIR})")
    endif()

    #get_cmake_property(_variableNames VARIABLES)
    #foreach (_variableName ${_variableNames})
    #    message(STATUS "${_variableName}=${${_variableName}}")
    #endforeach()

    if (NOT DEFINED _USING_MULT_THREAD)
        set(_USING_MULT_THREAD 1)
    endif()
    add_definitions(-DUSING_MULTI_THREAD=${_USING_MULT_THREAD} -D__TUNINGTOOLS_LIBRARY_NAME__=libTuningToolsLib)

    # New style root dictionary:
		atlas_add_root_dictionary( TuningToolsLib TuningToolsDictSource
															ROOT_HEADERS TuningTools/*.h Root/LinkDef.h
															EXTERNAL_PACKAGES ROOT
		)

    # Component(s) in the package:
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()

    if( NUMPY_FOUND AND Boost_PYTHON_FOUND AND PYTHONLIBS_FOUND )
      # Then we can proceed to a full installation:
      atlas_add_library( TuningToolsLib
                         Root/*.cxx ${TuningToolsDictSource}
                         PUBLIC_HEADERS TuningTools
                         INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${ROOT_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} # ${PYTHON_INCLUDE_DIRS}
                         LINK_LIBRARIES ${Boost_PYTHON_LIBRARIES} ${ROOT_LIBRARIES} ${PYTHON_LIBRARIES} RingerCoreLib
                       )
    else()
      message( STATUS "TuningTools will be built without FastNet core!" )
    endif()

    #atlas_add_dictionary( TuningToolsDict
    #                      TuningTools/TuningToolsDict.h
    #                      TuningTools/selection.xml
    #                      INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${ROOT_INCLUDE_DIRS}
    #                      LINK_LIBRARIES ${Boost_LIBRARIES} ${ROOT_LIBRARIES} AsgTools AthenaKernel CxxUtils xAODCaloRings xAODEgamma xAODTracking PATCoreLib ElectronPhotonSelectorToolsLib AthContainers xAODBase GaudiKernel PathResolver TrkTrackSummary TuningToolsLib )

    # Old style root dicionary:
    #atlas_add_dictionary( TuningToolsDict
    #  TuningTools/TuningToolsDict.h
    #  TuningTools/selection.xml
    #  LINK_LIBRARIES TuningToolsLib )

    #atlas_add_root_dictionary( TuningToolsLib
		#													ROOT_HEADERS TuningTools/*.h Root/LinkDef.h
		#													EXTERNAL_PACKAGES ROOT
		#)

    #atlas_add_dictionary( TuningToolsPythonDict
    #  TuningTools/TuningToolsDict.h
    #  TuningTools/selection.xml
    #  LINK_LIBRARIES TuningToolsLib )

  endif()
endif()

# Install files from the package:
atlas_install_python_modules( python/*.py python/dataframe python/keras_util python/monitoring python/parsers )
atlas_install_scripts( scripts/standalone/* scripts/grid_scripts/* )


