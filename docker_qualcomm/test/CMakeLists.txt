cmake_minimum_required(VERSION 3.10)
project(adreno_opencl_test C)

# Find OpenCL
find_package(OpenCL QUIET)

if(NOT OpenCL_FOUND)
    # Fallback: use system headers + find the ICD loader lib
    set(OPENCL_INCLUDE_DIR "/usr/include" CACHE PATH "OpenCL include directory")
    set(ADRENO_SDK_ROOT "/opt/qualcomm/adreno-gpu-sdk/opencl-sdk" CACHE PATH "Adreno SDK root")

    add_library(OpenCL::OpenCL SHARED IMPORTED)

    # On the RB5 board the ICD loader is at /usr/lib/aarch64-linux-gnu/libOpenCL.so
    find_library(OPENCL_LIB OpenCL
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib
        NO_DEFAULT_PATH
    )
    if(NOT OPENCL_LIB)
        find_library(OPENCL_LIB OpenCL)
    endif()

    if(OPENCL_LIB)
        set_target_properties(OpenCL::OpenCL PROPERTIES
            IMPORTED_LOCATION "${OPENCL_LIB}"
        )
    else()
        message(WARNING "No libOpenCL.so found - will attempt link anyway")
        set_target_properties(OpenCL::OpenCL PROPERTIES
            IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libOpenCL.so"
        )
    endif()

    set_target_properties(OpenCL::OpenCL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIR}"
    )
endif()

# Host test binary (native arm64 on the RB5 board)
add_executable(test_opencl_compile test_host.c)
target_link_libraries(test_opencl_compile OpenCL::OpenCL)
target_compile_definitions(test_opencl_compile PRIVATE CL_TARGET_OPENCL_VERSION=200)
