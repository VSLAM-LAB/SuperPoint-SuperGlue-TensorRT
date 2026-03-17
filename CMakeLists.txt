cmake_minimum_required(VERSION 3.5)
project(superpointglue)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "release")
add_definitions(-w)

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer)

find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME}_lib SHARED
  src/super_point.cpp
  src/super_glue.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}_lib
  nvinfer
  nvonnxparser
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  yaml-cpp
  tensorrtbuffer
)

option(SUPERPOINTGLUE_BUILD_EXECUTABLES "Build inference executables" ON)
if(SUPERPOINTGLUE_BUILD_EXECUTABLES)
  add_executable(${PROJECT_NAME}_image inference_image.cpp)
  add_executable(${PROJECT_NAME}_sequence inference_sequence.cpp)
  add_executable(${PROJECT_NAME}_inference_image_profile inference_image_profile.cpp)

  target_link_libraries(${PROJECT_NAME}_image ${PROJECT_NAME}_lib)
  target_link_libraries(${PROJECT_NAME}_sequence ${PROJECT_NAME}_lib)
  target_link_libraries(${PROJECT_NAME}_inference_image_profile ${PROJECT_NAME}_lib)
endif()