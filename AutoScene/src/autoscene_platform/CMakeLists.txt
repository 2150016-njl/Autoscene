cmake_minimum_required(VERSION 3.0.2)
project(autoscene_platform)

find_package(catkin REQUIRED COMPONENTS
  autoscene_msgs
  rospy
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/platform_comm_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
