# AutoScene 逻辑精简与闭环修复包

本包以本轮上传的最新代码为基线，保持现有工程路径和启动命令，不拆分 hybrid 配置。

## 直接覆盖

在工程根目录执行：

```bash
cd /home/nie/场景生成运动平台_0721/AutoScene
unzip -o AutoScene_clean_logic_patch.zip -d /tmp/AutoScene_clean_logic_patch
cp -a /tmp/AutoScene_clean_logic_patch/AutoScene_clean_logic_patch/src/. ./src/
chmod +x \
  src/autoscene_ego/scripts/ego_adapter_node.py \
  src/autoscene_evaluation/scripts/evaluator_node.py \
  src/autoscene_experiment/scripts/experiment_manager_node_0802备份.py \
  src/autoscene_platform/scripts/platform_comm_node.py \
  src/autoscene_runtime/scripts/calibrate_origins.py \
  src/autoscene_runtime/scripts/scenario_runtime_node.py
catkin_make
source devel/setup.bash
```

也可从工程根目录应用补丁：

```bash
patch -p1 < /tmp/AutoScene_clean_logic_patch/AutoScene_clean_logic_patch/AutoScene_clean_logic.patch
```

## 启动命令

纯仿真：

```bash
roslaunch autoscene_bringup autoscene_simulation.launch \
  config:=$(rospack find autoscene_bringup)/config/simulation.yaml
```

混合模式：

```bash
roslaunch autoscene_bringup autoscene_hybrid.launch \
  config:=$(rospack find autoscene_bringup)/config/hybrid.yaml
```

`hybrid.yaml` 是唯一混合配置：

```yaml
# 无硬件 dry-run
receive_udp: false
send_udp: false
emulate_controlled_feedback: true

# 真实硬件
receive_udp: true
send_udp: true
emulate_controlled_feedback: false
```

真实硬件前还必须填写各平台 `origin`，并确认 `pose_source` 与 `include_link_header` 符合现场协议。

## 运行工作流

### simulation

```text
scenario_runtime_node -> MetaDrive/Pluto -> world/state -> evaluator_node
```

不再启动无意义的 ego simulation echo 和 platform simulation echo 节点。

### hybrid dry-run

```text
scenario_runtime_node -> MetaDrive/Pluto
platform_comm_node    -> 显式模拟平台反馈与 UDP payload 审核
world/state           -> runtime 直接使用 MetaDrive 状态
```

### hybrid hardware

```text
UDP actual state -> platform_comm_node -> agents/state_feedback
                 -> scenario_runtime_node 覆盖 mirror/写实际历史
                 -> Pluto 规划
                 -> 单次轨迹对齐缓存
                 -> platform_comm_node 局部坐标转换/UDP 下发
```

### external ego

```text
external/ego/state -> ego_adapter_node 坐标转换 -> ego/state_feedback
```

有效外部状态立即发布，不再等待 runtime 的 simulation ego 消息。

## 主要修复

- 恢复原始 dataset warmup 和 Pluto 时间轴；删除 initial 的错误伪实现。
- `initial` 仍保留配置接口，但 manager/runtime 会明确报“未实现”，不会静默运行错误逻辑。
- 修复 ego 推理引擎 `self.Sim` 未初始化问题。
- adversarial/max 角色改为显式 `planner_mode`，不再依赖 BV 列表序号或环境变量。
- Pluto 轨迹只在 policy 中对齐一次；MetaDrive 与 ROS 平台轨迹读取同一缓存。
- 真实平台历史由 runtime 单独维护；policy 不再重复覆盖该历史。
- 实物状态缺失时不回退 MetaDrive 动态状态；超时会冻结并最终发布失败状态。
- simulation 直接使用仿真状态，减少 topic 回环、延迟和超时故障点。
- 平台使用一个共享 `local_ip:local_port` socket；删除旧的逐 endpoint bind/recv 逻辑。
- UDP 反馈增加重复/乱序样本过滤。
- 每包坐标转换固定使用一次平台锚点，避免同一包中途切换坐标系。
- 删除把速度/加速度数值 0 当作“缺失”的错误兜底。
- runtime 根据规划速度明确生成加速度，再由平台节点投影到车辆纵向。
- `closeloop_sim.py` 改为 import-safe 工厂，不再在导入时解析 argv 或修改 cwd。
- worker 正常结束自动关闭终端；错误结束保留终端日志。
- rosbag 使用 SIGINT 等待索引收尾，并在启动/停止时检查真实进程状态。
- launch 与 YAML mode 不一致时立即拒绝启动。

## 未改动项

`pluto_inference.py` 与上传的 `pluto_inference_原始.py` 字节完全一致。没有证据表明模型推理实现本身需要改动，因此本轮只修复它外部的时间轴、角色映射、历史所有权和轨迹导出链路。

## 审核范围

详见 `严格审核报告.txt`。当前环境无法接入你的 ROS 工作区、MetaDrive 数据集、GPU 模型和真实运动平台，因此完成的是静态审核、协议级 stub 测试与配置/launch 检查；真实硬件联调仍需现场抓包确认协议选项。
