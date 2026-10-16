# 单个轨迹点。坐标系由外层 Trajectory.frame_id/header.frame_id 指定。
# AutoScene 内部通常使用 world/map 坐标；platform_comm_node 下发前会转换为平台大地投影坐标。

# 相对轨迹起点的时间，单位：秒。
float64 time_from_start
# 轨迹点位姿；position 使用米，orientation 与 heading 保持一致。
geometry_msgs/Pose pose
# 轨迹点速度，单位：m/s。
geometry_msgs/Twist velocity
# 轨迹点加速度，单位：m/s^2。
geometry_msgs/Accel acceleration
# 平面航向角 yaw，单位：弧度；yaw=0 指向 world +x。
float64 heading
# 曲率，单位：1/m；当前 MVP 可为 0，预留给后续控制/评价。
float64 curvature
