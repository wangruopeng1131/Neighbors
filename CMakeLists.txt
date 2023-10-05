cmake_minimum_required(VERSION 3.25)
project(Neighbors)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}"/Eigen3/share/eigen3/cmake")
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_definitions(
        -DEIGEN_NO_DEBUG
        -DEIGEN_DEFAULT_TO_ROW_MAJOR
)

if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

add_executable(Neighbors BinaryTree.h BallTree.h KDTree.h partitionNodes.h main.cpp
        utlis.h)
