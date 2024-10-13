cmake_minimum_required(VERSION 3.22)

project(BiCG)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

# source file
file(GLOB SOURCES "src/*.cpp")

# executable file
add_executable(main ${SOURCES})

include_directories(./include ./CXXBLAS/include)
if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(${BLAS_INCLUDE_DIRS})
    target_link_libraries(main ${BLAS_LIBRARIES})
endif(BLAS_FOUND)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
message("OpenMP found.")
# target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(main Eigen3::Eigen OpenMP::OpenMP_CXX)# ${CMAKE_SOURCE_DIR}/CXXBLAS/lib/libcxxblas.a)

# to change oprimization, run this command in shell
# cmake .. -DCMAKE_BUILD_TYPE=Release