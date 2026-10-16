# AutoScene 上位机（PyQt5）

当前源码版本：`1.3.1`。已有 deb 包仍是旧版本；本次请从 `frontend/` 源码启动。

## 1.3.1 启动错误与地图内算法选择

- 修复 rosbridge 服务失败返回字符串时的 `TypeError: 'str' object does not support item assignment`。
  保留服务的原始错误，释放界面忙碌状态，不把服务失败误判为 WebSocket 断线。
- 地图下方实车、virtual BV 表格的车名旁直接显示“算法”下拉框：默认 / Pluto / VBD
  （同时保留 Pluto Hybrid）。地图上方选择 EGO 或已配置车辆后，也能直接选择“该车算法”。
- 各入口同步同一份配置；“默认”删除该车的 planner_method 覆盖，继承后端默认设置。

注意：rosbridge 只是通信桥。它连接成功，并不表示 AutoScene 实验后端已运行。
在 ROS 主机的 AutoScene 工作空间中，已完成编译及算法环境配置的情况下执行：

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch autoscene_bringup autoscene_hybrid_ui.launch
```

UI 使用 `_ui.launch` 非交互入口；不是终端菜单使用的 `autoscene_hybrid.launch`。
另开终端（同样 source ROS 和工作空间），用 `rosservice list` 确认存在
`/autoscene/experiment/start`。如果报找不到包或导入失败，需先完成后端部署，
仅启动 rosbridge 或重启前端不能代替这一步。远程 ROS 也必须拿到更新后的 AutoScene
后端、模型和数据集；前端选择的远程数据目录必须是 ROS 主机实际存在的路径。

## 1.3.0 字号与逐车算法

- 顶部独立工具栏直接显示“界面字号 / A− 缩小 / A+ 放大”，无需寻找底部入口。
- 点击顶部“逐车选择算法…”：主车、实车候选、绑定实车和已放置周车均有单独算法下拉框。
- 选择写入 `ego.planner_method` / `vehicles.agents[].planner_method`，后端按 track 分组执行
  Pluto、VBD、Pluto Hybrid，平台反馈执行权不变。改动后需重新准备和启动。
- VBD 频次控件已移除；后端默认自适应降频、单模型共享和后台预取。缓存按仿真时间截取，
  预测过期则报错，不补造轨迹。首次预测/重置仍需要等待模型。

前后端需同时更新。详见 [场景编辑与算法操作说明](SCENE_EDITOR_zh-CN.md)。

## 1.2.0 场景编辑

- 主车、实车候选和已绑定实车支持拖拽/旋转及 x、y、朝向数值编辑。
- virtual BV 使用显式源车辆选择，车道吸附可选，默认不吸附。
- nuPlan 过滤显示帧无效对象；地图按折线合并绘制，减少图元数量。
- 界面字号 100%–200%，保存偏好；左右侧栏可用 F8/F9 隐藏。
- 场景构建模式可编辑背景车，另存完整轨迹为新 PKL 数据集，保留源文件。

操作方法和导出范围见 [场景编辑说明](SCENE_EDITOR_zh-CN.md)。

## 1.1.2 交付与操作改进

- 默认按“前端与 ROS 在同一台 Linux 主机”运行，只需选择一次场景数据目录。
- 只有取消“ROS 与前端在同一台主机”后，才显示远程 ROS 数据目录。
- rosbridge 默认地址改为通用的 `ws://127.0.0.1:9090`，不再绑定开发机 IP。
- virtual BV 提供明确的“删除所选 BV”和逐行删除按钮。
- `initial` 模式自动把参与实验车辆的超限速度修正到 `4.000 m/s`；未绑定候选车不参与实验。

## 1.1.1 界面修复

- 左侧配置栏设置可读最小宽度，旧的过窄布局记录会自动恢复。
- 本机与 ROS 数据目录改为上下排版，完整路径可通过鼠标悬停查看。
- platform 通信状态改为短中文分行，小屏幕下不再横向截断。

## 1.1.0 可交付性改进

- ROS 断线时自动锁定“应用 / 准备 / 启动 / 停止 / Reset”，重连后恢复。
- rosbridge 服务和 rosparam 请求具有 10 秒超时，不再永久卡在“准备中”。
- 左侧同时显示“本机数据目录”和“ROS 数据目录”，适配 Mac + VirtualBox / 远程 ROS。
- 实验运行期间锁定场景和配置编辑，避免正在执行的参数与界面内容分叉。
- Reset 增加确认，并且只有 ROS Reset 成功后才清除本地绑定与 virtual BV。
- “初始状态录入”支持一键把超限速度修正到算法允许上限。
- 加载数据后默认聚焦车辆工作区；按 `F` 仍可查看完整地图。
- 保存窗口布局；关闭仍在运行的实验界面时给出明确警告。

Linux 安装包：

```bash
sudo apt install ./autoscene-ui_1.1.2_all.deb
autoscene-ui
```

软件包只安装前端，不安装或修改 ROS、AutoScene 算法和模型。

放到 `UI/` 下即可，例如 `UI/autoscene_ui/`。

## 推荐：独立 Conda 前端环境

这个 Conda 环境只供 PyQt 前端使用，不安装、替换或修改 ROS。ROS1 Noetic 和
AutoScene 后端继续使用 Ubuntu 主机原来的系统环境。

```bash
chmod +x create_conda_env.sh run_conda_frontend.sh check_conda_ros.sh
./create_conda_env.sh
./check_conda_ros.sh
./run_conda_frontend.sh
```

环境固定名为 `autoscene-ui`，并设置 `PYTHONNOUSERSITE=1`，因此不会偷偷读取
`~/.local` 中的 Python 包。完整交付包中 `frontend/` 与 `AutoScene/` 是同级目录，
启动脚本会自动找到源码、配置和随包数据。

## 手动跑起来

```bash
conda activate autoscene-ui

# 1) 离线：不需要 ROS，先确认界面与交互
python autoscene_ui.py --config ../AutoScene/src/autoscene_bringup/config/hybrid.yaml

# 2) 前端与 ROS 在同一台 Ubuntu 主机
python autoscene_ui.py --backend rosbridge --bridge ws://127.0.0.1:9090 \
    --config ../AutoScene/src/autoscene_bringup/config/hybrid.yaml \
    --dataset ../AutoScene/data/tjwest260725intersection
```

Mac 不需要安装 ROS1。`--backend rosbridge` 下，本机只装
`websocket-client`，所有话题、服务和 rosparam 都经 WebSocket 进入虚拟机。
前端与 ROS 同机时只需一个目录。只有本机数据集路径和远程 ROS 路径不同时，才传
`--ros-dataset-path`，界面也只会在取消“ROS 与前端在同一台主机”后显示第二条路径。
使用一键脚本时分别设置 `AUTOSCENE_DATASET`（前端本机）和
`AUTOSCENE_ROS_DATASET`（ROS 主机）。

## 文件

| 文件 | 作用 |
| --- | --- |
| `autoscene_ui.py` | 入口，解析参数 |
| `autoscene_gui/config_model.py` | YAML 读写 + 启动前校验（hybrid→tjwest/initial、速度上限、端点数量、端点重复绑定） |
| `autoscene_gui/scenario_loader.py` | 数据集加载：`sd_*.pkl` → 车辆位姿/轨迹/`map_features`；未加载时用示意路口 |
| `autoscene_gui/scene_view.py` | 2D 俯视图；virtual BV 可选中、拖拽移动、拖手柄旋转 |
| `autoscene_gui/theme.py` | 全局桌面主题、按钮层级和表格/表单视觉样式 |
| `autoscene_gui/ros_client.py` | 三种后端：mock / rosbridge / rospy，接口一致 |
| `autoscene_gui/main_window.py` | 三栏主窗口，串起全部交互 |

## 与 manager 的对应关系

UI 替代的是 `experiment_manager_node.py` 的交互式终端菜单：

| 终端菜单 | UI 位置 |
| --- | --- |
| 选模式 / 数据集 / history_source / scenario_index | 左侧配置树 |
| 设置 platform / virtual BV 数量 | 不再手输——由地图上的绑定与放置自动计数 |
| 打印候选 CBV 并逐辆输入 track_id | 地图上直接点实物车位，在表里选 `platform_001/002` |
| history=initial 时逐车输入 x/y/heading/speed | virtual BV 用地图点选放置 + 拖拽/旋转；速度在表里改 |
| 准备 / 启动 / 停止 / reset | 准备写 rosparam；其余调用 `experiment/start,stop,reset` 服务 |

## CAD 风格地图操作

- 鼠标滚轮：以光标为中心连续缩放
- Mac 触控板双指捏合：以光标为中心连续缩放（兼容原生与 Qt pinch 手势）
- 中键拖拽，或按住 `Space` 后左键拖拽：平移画布
- 左键拖过空白区域：框选车辆
- `F` / `0` / 双击空白：适应全部地图要素
- `A`：聚焦车辆工作区，忽略远处零散地图要素
- `+` / `-`：分级放大、缩小；工具栏同步显示当前倍率

地图按数据集 `metadata.current_time_index` 显示实验起点（tjwest/Pluto 默认第 20 帧），
而不是误用整段轨迹的最后一帧。实验启动后，车辆图元由
`/autoscene/world/state` 增量更新，不会重建地图或重置当前缩放位置。

tjwest 数据集地图在 UI 中保留原始全局坐标；发送实验配置时，前端会自动减去
第 0 帧 ego 原点，换算成 MetaDrive 使用的局部坐标。接收 `world/state` 时再加回
同一原点，因此车辆不会因两套坐标系混用而被判定在道路外或跳出地图。

放置 virtual BV 时，点击点会吸附到最近车道中心线，默认 heading 使用车道切线，
并根据绑定车辆的原始方向选择正向/反向。距车道超过 8 m 的点击会被拒绝；
吸附完成后仍可拖动车身和旋转手柄微调。

准备按钮把 YAML 各顶层块写入 `/project`、`/runtime`、`/scenario`、
`/vehicles` 等真实参数路径，全部成功后 UI 显示本地 prepared；不调用不存在的
`experiment/prepare`。点击启动后，非交互 manager 才校验数据集、scenario index、
track 绑定、初始状态和平台通信配置，并创建实际运行节点。

## 实车通信现状（重要）

代码里 `platform_comm_node.py` 已经实现了完整的 UDP 链路：
高斯投影 ↔ AutoScene 坐标换算、下发 MsgID 12（50 点 / 0.1 s）、接收 MsgID 11 反馈、
发布 `platform/comm_status`。**但当前 `hybrid.yaml` 是 dry-run 配置**：

```yaml
platform:
  receive_udp: false
  send_udp: false
  emulate_controlled_feedback: true   # 反馈由节点自己模拟
```

也就是说：节点通信能力有，但现在一个真实报文都不收发。要接实车需要改三处，
UI 的 platform 面板会把这三项和 LIVE/DRY-RUN 标识一起显示出来：

1. `receive_udp: true` / `send_udp: true`
2. `emulate_controlled_feedback: false`
3. 配好全局 `platform.earth_to_autoscene.{tx,ty,yaw}`，并为每个
   `platform.platforms[]` 填写 `vehicle_id`、`send_port` 和反馈源地址

UI 侧刻意只读展示这三项——按你的要求，硬件开关不放进界面里改，避免误触发实车动作。

## VirtualBox 与实车网络

- 网卡 1 使用 NAT；把宿主 `127.0.0.1:9090` 转发到虚拟机 `9090`，供 UI 连接 rosbridge。
- 网卡 2 使用桥接，桥到连接实车交换机的有线/USB 网卡；虚拟机在该网卡上配置
  `192.168.88.3/24`。不要把实车 UDP 走 NAT。
- 虚拟机内启动：

```bash
source /opt/ros/noetic/setup.bash
source ~/AutoScene/devel/setup.bash
roslaunch autoscene_bringup autoscene_hybrid.launch
# 另一个终端
roslaunch rosbridge_server rosbridge_websocket.launch
```

首次联调保持 dry-run，确认 UI 状态、轨迹和端点绑定正确后，再切换三项硬件开关。
