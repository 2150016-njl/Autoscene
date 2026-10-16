std_msgs/Header header
# AutoScene 内部车辆 ID，例如 ego_001、agent_001。
string vehicle_id
# 真实运动平台 ID；纯虚拟车辆可为空。
string platform_id
# 轨迹来源，例如 external_ads、unitraj_pluto_agent、unitraj_pluto_ego_fallback。
string source
# 坐标系名称，通常与 header.frame_id 一致，例如 map。
string frame_id
# 按时间递增的轨迹点列表。
autoscene_msgs/TrajectoryPoint[] points
# false 表示该轨迹不可用于控制/下发。
bool valid
# 轨迹置信度，0~1；用于后续融合或诊断。
float64 confidence
