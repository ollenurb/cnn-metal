cmake_minimum_required(VERSION 3.16)
project(CNN CXX)

# fundamental to export lsp commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# misc C++ flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER clang++)

# set output dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)

# set sources
set(
    SOURCES
    src/metal_cpp_impl.cc
    src/context.cc
    src/main_metal.cc
    src/layers.cc
)

set(
    HEADERS
    src/types.hpp
    src/context.hpp
    src/layers.hpp
    src/model.hpp
)

# define target
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# include sources to the target
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# include metal-cpp sources to the target
target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_SOURCE_DIR}/metal-cpp"
)

# link libraries
target_link_libraries(${PROJECT_NAME}
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    "-framework Foundation"
    "-framework QuartzCore"
)

# set compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -O2
)

# path to the metal source in the source tree
set(METAL_SRC "${CMAKE_SOURCE_DIR}/src/ops.metal")

# ensure xcrun exists to invoke metal and metallib
find_program(XCRUN_EXECUTABLE xcrun)
if(NOT XCRUN_EXECUTABLE)
    message(FATAL_ERROR "xcrun not found. Xcode command line tools are required to compile Metal shaders.")
endif()

# setup metallib libraries
set(METAL_LIB "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ops.metallib")
set(METAL_AIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ops.air")

# compiling metal kernels require multiple steps using the xcode sdk
add_custom_command(
    OUTPUT "${METAL_LIB}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c "${METAL_SRC}" -o "${METAL_AIR}"
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib "${METAL_AIR}" -o "${METAL_LIB}"
    DEPENDS "${METAL_SRC}"
    BYPRODUCTS "${METAL_AIR}" "${METAL_LIB}"
    COMMENT "Compiling Metal kernel ${METAL_SRC} into ${METAL_LIB}"
    VERBATIM
)

# add the custom target (to the Makefile)
add_custom_target(metal_kernels ALL DEPENDS "${METAL_LIB}")

# executable must be built after the metal library has been created
add_dependencies(${PROJECT_NAME} metal_kernels)

# set a macro with the metal library path
target_compile_definitions(${PROJECT_NAME} PRIVATE
    METAL_LIB=\"${METAL_LIB}\"
)
