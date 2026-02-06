cmake_minimum_required(VERSION 3.8)
project(rm_vision_bringup)
# 仅仅引用该变量以消除警告
set(dummy_var ${CMAKE_EXPORT_COMPILE_COMMANDS})
find_package(ament_cmake_auto REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
)
