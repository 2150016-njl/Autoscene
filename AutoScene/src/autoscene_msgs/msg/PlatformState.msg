std_msgs/Header header
# 真实运动平台 ID。
string platform_id
# 该平台绑定的 AutoScene agent/vehicle ID。
string vehicle_id
# 状态来源或状态机文本，例如 ads_udp_feedback。
string state
# 已转换到 AutoScene world/map 坐标的位姿。
geometry_msgs/Pose pose
# 已转换到 AutoScene world/map 坐标的速度。
geometry_msgs/Twist twist
# 平面航向角 yaw，单位：弧度。
float64 heading
# 平台反馈的跟踪误差，单位：米；通常来自上游协议 lat_dev_m。
float64 tracking_error
# 距离最近一次平台反馈的时间，单位：秒。
float64 feedback_age
# false 表示反馈不可用或协议校验严重失败。
bool valid
