cmake_minimum_required(VERSION 3.0.2)
project(autoscene_runtime)

find_package(catkin REQUIRED COMPONENTS
  autoscene_msgs
  rospy
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/scenario_runtime_node.py
  scripts/calibrate_origins.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
