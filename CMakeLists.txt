CMAKE_MINIMUM_REQUIRED(VERSION 3.13 FATAL_ERROR)
PROJECT(HPDBSCAN)

# include additional modules
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)

# set c++ properties
ENABLE_LANGUAGE(CXX)
SET(CMAKE_CXX_STANDARD 17)

# set build options
#SET(CMAKE_CXX_FLAGS "-Ofast -Wall -Wextra")
#SET(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
set(CMAKE_C_FLAGS_RELEASE "-Ofast")

# Always output diagnostic colors
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "-fdiagnostics-color ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "-fcolor-diagnostics ${CMAKE_CXX_FLAGS}")
endif()

# add the include and source directories
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR
   CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    option(USE_AVX512 "USE x86 AVX512 optimizations" OFF)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR
       CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    option(USE_SVE "Use Arm SVE optimizations" OFF)
endif()

option(USE_ND_OPTIMIZATIONS "Optimize special cases for some number of dimensions in the data" ON)

set(additional_sources)
if(USE_AVX512)
    set(additional_sources src/spatial_index_avx512.cpp)
    set_source_files_properties(src/spatial_index_avx512.cpp 
        PROPERTIES COMPILE_FLAGS -mavx512f)
endif()

if(USE_SVE)
    set(additional_sources src/spatial_index_sve.cpp)
endif()

# set up the build targets
## set up the shared library compilation
ADD_LIBRARY(hpdbscan SHARED src/hpdbscan.cpp ${additional_sources})
# set up the cli executable
ADD_EXECUTABLE(hpdbscan-bin src/cli.cpp ${additional_sources})
ADD_EXECUTABLE(hpdbscan64-bin src/cli.cpp ${additional_sources})

target_compile_definitions(hpdbscan64-bin PRIVATE USE_INT64_INDEX)

if(USE_SVE)
    target_compile_definitions(hpdbscan PRIVATE USE_SVE)
    target_compile_definitions(hpdbscan-bin PRIVATE USE_SVE)
    target_compile_definitions(hpdbscan64-bin PRIVATE USE_SVE)
endif()
if(USE_AVX512)
    target_compile_definitions(hpdbscan PRIVATE USE_AVX512)
    target_compile_definitions(hpdbscan-bin PRIVATE USE_AVX512)
    target_compile_definitions(hpdbscan64-bin PRIVATE USE_AVX512)
endif()
if(USE_ND_OPTIMIZATIONS)
    target_compile_definitions(hpdbscan PRIVATE USE_ND_OPTIMIZATIONS)
    target_compile_definitions(hpdbscan-bin PRIVATE USE_ND_OPTIMIZATIONS)
    target_compile_definitions(hpdbscan64-bin PRIVATE USE_ND_OPTIMIZATIONS)
endif()

# NOTE: JUST FOR compile_commands.json/autocompletion, remove for release
# TARGET_COMPILE_DEFINITIONS(hpdbscan PRIVATE WITH_OUTPUT)

# define options
TARGET_COMPILE_DEFINITIONS(hpdbscan-bin PRIVATE WITH_OUTPUT)
TARGET_COMPILE_DEFINITIONS(hpdbscan64-bin PRIVATE WITH_OUTPUT)

# check dependencies
## mpi, optional for distributed computing
FIND_PACKAGE(MPI)
IF(MPI_FOUND)
    SET(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
    TARGET_COMPILE_DEFINITIONS(hpdbscan PRIVATE WITH_MPI)
    TARGET_COMPILE_DEFINITIONS(hpdbscan-bin PRIVATE WITH_MPI)
    TARGET_COMPILE_DEFINITIONS(hpdbscan64-bin PRIVATE WITH_MPI)
ELSE()
    MESSAGE("BUILDING WITHOUT MPI SUPPORT")
ENDIF()

## openmp
FIND_PACKAGE(OpenMP 4.0 REQUIRED)
TARGET_LINK_LIBRARIES(hpdbscan PRIVATE ${OpenMP_CXX_FLAGS})
TARGET_COMPILE_OPTIONS(hpdbscan PRIVATE ${OpenMP_CXX_FLAGS})
TARGET_LINK_LIBRARIES(hpdbscan-bin PRIVATE ${OpenMP_CXX_FLAGS})
TARGET_LINK_LIBRARIES(hpdbscan64-bin PRIVATE ${OpenMP_CXX_FLAGS})
TARGET_COMPILE_OPTIONS(hpdbscan-bin PRIVATE ${OpenMP_CXX_FLAGS})
TARGET_COMPILE_OPTIONS(hpdbscan64-bin PRIVATE ${OpenMP_CXX_FLAGS})

## hdf5
FIND_PACKAGE(HDF5 1.8.10 REQUIRED)
INCLUDE_DIRECTORIES("${HDF5_INCLUDE_DIRS}")
TARGET_LINK_LIBRARIES(hpdbscan-bin PRIVATE "${HDF5_LIBRARIES}")
TARGET_LINK_LIBRARIES(hpdbscan64-bin PRIVATE "${HDF5_LIBRARIES}")

## swig and python detection for optional bindings
FIND_PACKAGE(SWIG)
IF(SWIG_FOUND)
    include(UseSWIG)
    FIND_PACKAGE(PythonInterp)
    IF(PYTHONINTERP_FOUND)
        FIND_PACKAGE(PythonLibs)
        IF("${PYTHON_INCLUDE_DIRS}" STREQUAL "")
            MESSAGE("PYTHON HEADERS NOT FOUND, BUILDING WITHOUT BINDINGS")
            MESSAGE("TRY INSTALLING THE python-dev OR python-devel PACKAGE")
        ELSE()
            INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
            FIND_PACKAGE(NumPy)
            IF(NUMPY_FOUND)

                #EXECUTE_PROCESS(COMMAND swig -c++ -python -I"${PYTHON_INCLUDE_DIRS}" 
                #    -I"${NUMPY_INCLUDE_DIR}" 
                #    -o "${CMAKE_CURRENT_BINARY_DIR}/hpdbscan_wrap.cpp" 
                #    "${CMAKE_CURRENT_SOURCE_DIR}/swig/hpdbscan.i")
                #ADD_LIBRARY(hpdbscan-binding SHARED ${CMAKE_CURRENT_BINARY_DIR}/hpdbscan_wrap.cpp)
                set_property(SOURCE swig/hpdbscan.i PROPERTY CPLUSPLUS ON)
                swig_add_library(hpdbscan-binding 
                    TYPE SHARED 
                    LANGUAGE python 
                    SOURCES swig/hpdbscan.i)
                SET_TARGET_PROPERTIES(hpdbscan-binding PROPERTIES 
                    PREFIX "" 
                    OUTPUT_NAME _hpdbscan
                    USE_TARGET_INCLUDE_DIRECTORIES TRUE)
                IF(MPI_FOUND)
                    FIND_PACKAGE(MPI4PY)
                    IF(MPI4PY_FOUND)
                        TARGET_COMPILE_DEFINITIONS(hpdbscan-binding PRIVATE WITH_MPI)
                        FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/hpdbscan.py "from mpi4py import MPI\n")
                    ELSE()
                        MESSAGE("MPI FOUND, BUT MPI4PY MISSING, BINDING WILL BE BUILT WITHOUT MPI SUPPORT")
                    ENDIF()
                ENDIF()
                TARGET_COMPILE_OPTIONS(hpdbscan-binding PRIVATE -Wno-unused-parameter)
                TARGET_INCLUDE_DIRECTORIES(hpdbscan-binding PRIVATE ${NUMPY_INCLUDE_DIR})
                TARGET_LINK_LIBRARIES(hpdbscan-binding PRIVATE ${OpenMP_CXX_FLAGS})
                TARGET_COMPILE_OPTIONS(hpdbscan-binding PRIVATE ${OpenMP_CXX_FLAGS})
                TARGET_LINK_LIBRARIES(hpdbscan-binding PRIVATE "${HDF5_LIBRARIES}")
            ELSE()
                MESSAGE("NUMPY NOT FOUND, BUILDINGS WITHOUT BINDINGS")
            ENDIF()
        ENDIF()
    ELSE()
        MESSAGE("PYTHON INTERPRETER NOT FOUND, BUILDING WITHOUT BINDINGS")
    ENDIF()
ELSE()
    MESSAGE("SWIG NOT FOUND, BUILDING WITHOUT BINDINGS")
ENDIF()

# rename the binary
SET_TARGET_PROPERTIES(hpdbscan-bin PROPERTIES OUTPUT_NAME hpdbscan)
SET_TARGET_PROPERTIES(hpdbscan64-bin PROPERTIES OUTPUT_NAME hpdbscan64)
