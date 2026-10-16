std_msgs/Header header
string platform_id
string vehicle_id
# 配置中该平台是否启用。
bool enabled
# 最近 1 秒内是否收到平台反馈。
bool connected
# 已发送轨迹 UDP 包数量。
uint32 tx_count
# 已接收上游反馈 UDP 包数量。
uint32 rx_count
# 距离最近一次发送的时间，单位：秒；无发送时为 -1。
float64 last_tx_age
# 距离最近一次接收的时间，单位：秒；无接收时为 -1。
float64 last_rx_age
# 通信诊断文本，例如 waiting_target / tx_ok / rx_error。
string detail
