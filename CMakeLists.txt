cmake_minimum_required(VERSION 3.10)
project(agenttest CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
##打开gdb调试
set(CMAKE_BUILD_TYPE Debug)
enable_language(CXX)
enable_language(C)

# Set C++ as default for linking
set_source_files_properties(test.c PROPERTIES LANGUAGE CXX)
file(GLOB AGENT_SRCS ${CMAKE_SOURCE_DIR}/Src/*.c)
file(GLOB EXAMPLE_SRCS ${CMAKE_SOURCE_DIR}/example/*.c)
add_executable(agenttest ${AGENT_SRCS} ${EXAMPLE_SRCS})

set(LLAMA_DIR ${CMAKE_SOURCE_DIR}/../llama.cpp)
set(LLAMA_BUILD_DIR ${LLAMA_DIR}/build)
# Enable OpenMP
find_package(OpenMP REQUIRED)

target_include_directories(agenttest PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_BUILD_DIR}/src
    ${CMAKE_SOURCE_DIR}/Inc
)

target_link_libraries(agenttest PRIVATE
    ${LLAMA_BUILD_DIR}/src/libllama.a
    ${LLAMA_BUILD_DIR}/common/libllama-common.a
    ${LLAMA_BUILD_DIR}/common/libllama-common-base.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-base.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-cpu.a
    OpenMP::OpenMP_CXX
    m
    pthread
    stdc++
    stdc++fs
)