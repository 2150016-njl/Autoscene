cmake_minimum_required(VERSION 3.0.2)
project(autoscene_ego)

find_package(catkin REQUIRED COMPONENTS
  autoscene_msgs
  rospy
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/ego_adapter_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
