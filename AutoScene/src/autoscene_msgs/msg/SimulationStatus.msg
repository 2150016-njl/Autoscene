std_msgs/Header header
# 当前仿真后端，例如 unitraj_pluto_metadrive / synthetic_metadrive_adapter。
string backend
# 仿真是否运行中。
bool running
# 仿真时间，单位：秒。
float64 sim_time
# 实时率；MVP 中默认 1.0。
float64 real_time_factor
# 后端状态或错误说明。
string message
