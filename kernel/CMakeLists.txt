cmake_minimum_required(VERSION 3.10)
# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CUDA_FLAGS "-std=c++17 -g -G")
project(pytorch_cmake_example LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
set(CMAKE_PREFIX_PATH /home/ldh0033@auburn.edu/.local/lib/python3.10/site-packages/torch/share/cmake)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Modify if you need a different default value
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()
# set(ENABLE_PRECOMPILED_HEADERS off)

# Shared is another word for dynamic.
add_library(library SHARED gemm_lowbit_kernel.h gemm_lowbit_kernel.cu) #gemm_lowbit.cpp
#public so that the dependent can see and use header.
target_link_libraries(library PUBLIC ${TORCH_LIBRARIES} Python::Python)
add_executable(running gemm_lowbit.cpp)
target_link_libraries(running library)
# set_source_files_properties(gemm_lowbit.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

# add_executable(running gemm_lowbit.cpp gemm_lowbit_kernel.h gemm_lowbit_kernel.cu)
# target_link_libraries(running PUBLIC ${TORCH_LIBRARIES} Python::Python)