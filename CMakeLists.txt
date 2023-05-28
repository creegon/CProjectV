cmake_minimum_required(VERSION 3.12)
project(YourProjectName)

# 设置源文件和可执行文件名
set(SOURCES finalGEMM.cpp)
set(EXECUTABLE_NAME exe)

# 设置OpenBLAS的库和头文件路径
set(OPENBLAS_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(OPENBLAS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加可执行文件
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# 添加编译选项以启用SSE3
target_compile_options(${EXECUTABLE_NAME} PRIVATE -msse3 -O3 -march=native -funroll-loops -ffast-math)
# target_compile_options(${EXECUTABLE_NAME} PRIVATE -msse3)

# 指定头文件路径
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${OPENBLAS_INCLUDE_DIR})

# 指定库文件路径和链接库
find_library(OPENBLAS_LIBRARY libopenblas.a PATHS ${CMAKE_CURRENT_SOURCE_DIR})
if (OPENBLAS_LIBRARY)
    target_link_libraries(${EXECUTABLE_NAME} ${OPENBLAS_LIBRARY})
else()
    message(FATAL_ERROR "OpenBLAS library not found")
endif()
