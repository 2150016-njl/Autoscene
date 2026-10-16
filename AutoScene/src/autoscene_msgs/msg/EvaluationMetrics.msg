std_msgs/Header header
string experiment_id
# 是否发生碰撞；MVP 中基于中心点距离阈值判断。
bool collision
# 自车到最近 agent 的距离，单位：米；无 agent 时为 -1。
float64 min_distance
# 简化 TTC，单位：秒；不可计算时为 -1。
float64 time_to_collision
# 场景是否成功完成。
bool scenario_success
# 当前/平均自车速度，单位：m/s；MVP 中为当前帧速度。
float64 ego_speed_mean
# 最大自车速度，单位：m/s；MVP 中为当前帧速度。
float64 ego_speed_max
# 自车当前状态到规划轨迹首点的偏差，单位：米。
float64 trajectory_deviation
# 规划轨迹最大加速度模长，单位：m/s^2。
float64 comfort_acceleration
# 规划轨迹最大 jerk，单位：m/s^3。
float64 comfort_jerk
# 平台跟踪误差均值，单位：米。
float64 platform_tracking_error
# JSON 形式的附加诊断信息。
string detail_json
