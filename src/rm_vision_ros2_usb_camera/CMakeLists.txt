cmake_minimum_required(VERSION 3.8)
project(usb_camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(usb_camera_node src/usb_camera_node.cpp)
ament_target_dependencies(usb_camera_node
  rclcpp sensor_msgs image_transport cv_bridge camera_info_manager
)
target_link_libraries(usb_camera_node ${OpenCV_LIBS})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

install(TARGETS usb_camera_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
