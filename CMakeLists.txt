cmake_minimum_required(VERSION 3.5)
project(caffe_mint)

set(CMAKE_CXX_STANDARD 11)


include_directories(include)
link_directories("/home/yd/.local/caffe/lib")

include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

set(SOURCE_FILES main.cpp)
add_executable(caffe_mint ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
target_link_libraries(caffe_mint ${OpenCV_LIBS} )
target_link_libraries(caffe_mint caffe cublas boost_system glog)
