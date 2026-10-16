std_msgs/Header header
# AutoScene 内部车辆 ID。
string vehicle_id
# virtual / physical_platform / ego 等实体类型。
string entity_type
# world/map 坐标下的车辆位姿。
geometry_msgs/Pose pose
# world/map 坐标下的速度。
geometry_msgs/Twist twist
# world/map 坐标下的加速度。
geometry_msgs/Accel accel
# 平面航向角 yaw，单位：弧度。
float64 heading
# 车辆长度，单位：米。
float64 length
# 车辆宽度，单位：米。
float64 width
# false 表示该对象当前不可用或未成功匹配。
bool valid
