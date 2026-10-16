std_msgs/Header header
# 当前场景 ID。
string scenario_id
# 仿真时间，单位：秒。
float64 sim_time
# 自车状态，使用 world/map 坐标。
autoscene_msgs/VehicleState ego
# 周车/行人/其他实体状态列表，使用 world/map 坐标。
autoscene_msgs/VehicleState[] agents
