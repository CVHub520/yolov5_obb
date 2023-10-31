cmake_minimum_required(VERSION 3.5)

project(ylov5_obb VERSION 1.0 LANGUAGES CXX)

add_compile_options(-std=c++11)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
set(CMAKE_BUILD_TYPE Debug)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# opencv-4.7.0
set(OpenCV_DIR "/usr/local" CACHE PATH "Location of opencv libraries" FORCE)

# onnxruntime
# set(ONNXRUNTIME_ROOT_PATH "/path/to/onnxruntime/")
# set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
#                              ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
#                              ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/)
# set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)


if (NOT DEFINED OpenCV_DIR OR ${OpenCV_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OpenCV_DIR with -DOpenCV_DIR=/path/to/your/custom/opencv")
endif()

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")

# sudo apt-get install libeigen3-dev
include_directories(include)
include_directories(
    # ${ONNXRUNTIME_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/obb/include
)

find_package(OpenCV REQUIRED core imgcodecs imgproc dnn highgui PATHS ${OpenCV_DIR})

file(GLOB SEGMENT ${CMAKE_SOURCE_DIR}/obb/src/*.cpp)

# 编译成可执行文件
add_executable(main
            ${SEGMENT}
            main.cpp
        )

target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
    # ${ONNXRUNTIME_LIB}
    Eigen3::Eigen
)


