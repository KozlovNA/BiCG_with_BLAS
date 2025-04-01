cmake_minimum_required(VERSION 3.22)

project(BiCG)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(main "./src/main.cpp")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# set(BLAS_LIBRARIES "/home/starman/openblas/lib/libopenblas.so")
set(OPEN_BLAS_PATH "/home/starman/openblas")
target_include_directories(main PUBLIC "./include" "${OPEN_BLAS_PATH}/bin/include" "./CXXBLAS/include")
# target_link_directories(main PUBLIC "${OPEN_BLAS_PATH}/lib")
find_package(OpenMP REQUIRED)
add_subdirectory(CXXBLAS)
target_link_libraries(main "${OPEN_BLAS_PATH}/lib/libopenblas.so" Eigen3::Eigen BLAS::BLAS_CXX OpenMP::OpenMP_CXX)

# to change oprimization, run this command in shell
# cmake .. -DCMAKE_BUILD_TYPE=Release
