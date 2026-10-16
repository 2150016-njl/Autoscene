cmake_minimum_required(VERSION 3.0.2)
project(autoscene_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  std_msgs
)

add_message_files(
  FILES
  EvaluationMetrics.msg
  ExperimentStatus.msg
  PlatformCommStatus.msg
  PlatformState.msg
  PlatformStateArray.msg
  ScenarioStatus.msg
  SimulationStatus.msg
  SystemControl.msg
  Trajectory.msg
  TrajectoryArray.msg
  TrajectoryPoint.msg
  VehicleState.msg
  VehicleStateArray.msg
  WorldState.msg
)

add_service_files(
  FILES
  ExportEvaluation.srv
  LoadExperiment.srv
  ResetRuntime.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime std_msgs
)
