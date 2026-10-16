cmake_minimum_required(VERSION 3.0.2)
project(autoscene_experiment)

find_package(catkin REQUIRED COMPONENTS
  autoscene_msgs
  rospy
  std_srvs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/experiment_manager_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
