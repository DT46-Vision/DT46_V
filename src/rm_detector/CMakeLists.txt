cmake_minimum_required(VERSION 3.8)
project(rm_detector)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖项
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rm_interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)

# 添加可执行文件
add_executable(rm_detector_node 
  src/node.cpp
  src/detector.cpp
  src/classifier.cpp
  src/pnp.cpp
)

# 指定头文件路径
target_include_directories(rm_detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 链接依赖项
ament_target_dependencies(rm_detector_node
  rclcpp
  sensor_msgs
  std_msgs
  OpenCV
  cv_bridge
  rm_interfaces
  ament_index_cpp
)

# 安装目标
install(TARGETS rm_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装 launch 和 config 文件
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# 安装头文件
install(DIRECTORY include/
  DESTINATION include/
)

# 安装模型文件
install(DIRECTORY model/
  DESTINATION share/${PROJECT_NAME}/model
)

# 测试相关配置
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()