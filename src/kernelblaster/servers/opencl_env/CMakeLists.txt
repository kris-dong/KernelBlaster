cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(opencl_kernel C)

# Find OpenCL
find_package(OpenCL QUIET)

if(NOT OpenCL_FOUND)
    set(OPENCL_INCLUDE_DIR "/usr/include" CACHE PATH "OpenCL include directory")
    add_library(OpenCL::OpenCL SHARED IMPORTED)
    find_library(OPENCL_LIB OpenCL
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib
    )
    if(OPENCL_LIB)
        set_target_properties(OpenCL::OpenCL PROPERTIES
            IMPORTED_LOCATION "${OPENCL_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIR}"
        )
    else()
        message(FATAL_ERROR "OpenCL library not found")
    endif()
endif()

set(OPENCL_VERSION "200" CACHE STRING "OpenCL target version (e.g. 200 for CL2.0)")

add_executable(main main.c)
target_link_libraries(main OpenCL::OpenCL m)
target_compile_definitions(main PRIVATE CL_TARGET_OPENCL_VERSION=${OPENCL_VERSION})
