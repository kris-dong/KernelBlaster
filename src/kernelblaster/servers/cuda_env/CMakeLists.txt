# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda)

enable_language(CUDA)

# Unused library that is sometimes missing when running outside of docker.
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt IMPORTED INTERFACE)
endif()

# Suppress warnings before finding Torch
set(CMAKE_MESSAGE_LOG_LEVEL "ERROR")

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Clear all the architecture flags that PyTorch injected
string(REGEX REPLACE "-gencode[= ]arch=compute_[0-9]+,code=sm_[0-9]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REGEX REPLACE "-gencode[= ]arch=compute_[0-9]+,code=compute_[0-9]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# Now set only the architectures we want from the command line
set(GPU_ARCH_VERSION "" CACHE STRING "GPU architecture version to compile for")
if(NOT GPU_ARCH_VERSION)
    message(FATAL_ERROR "Please specify GPU_ARCH_VERSION: -DGPU_ARCH_VERSION=89")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${GPU_ARCH_VERSION},code=sm_${GPU_ARCH_VERSION}")

# Reset log level after finding Torch if needed
set(CMAKE_MESSAGE_LOG_LEVEL "STATUS") 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

# add a method which adds executable with all the flags
function(add_cuda_executable TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
    # This line is for ncu to collect source. It will add runtime overhead, so it should not be used for benchmarking
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_include_directories(${TARGET_NAME} PRIVATE
        "${Python3_INCLUDE_DIRS}"
    )
    # Link base libraries
    target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}" ${Python3_LIBRARIES})
endfunction()

# Use the function for the executables
add_cuda_executable(main main.cpp cuda_model.cu)
