cmake_minimum_required(VERSION 2.8)

project(cluster_point)
set(CMAKE_BUILD_TYPE DEBUG)
set(OpenCV_DIR /home/parallels/opencv/release/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SRC ./src/*.cpp ./src/DBow/*.cpp)
add_executable(cluster_point ${SRC})

target_link_libraries(cluster_point ${OpenCV_LIBS})

