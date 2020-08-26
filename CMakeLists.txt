cmake_minimum_required(VERSION 2.8.3)
project(gas_detector)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
)

find_package(OpenCV REQUIRED)


catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS roscpp std_msgs cv_bridge sensor_msgs image_transport                
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(gas_detector src/gas_detector.cpp)
target_link_libraries(gas_detector ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(gas_detector lept tesseract)
add_dependencies(gas_detector gas_detector)

