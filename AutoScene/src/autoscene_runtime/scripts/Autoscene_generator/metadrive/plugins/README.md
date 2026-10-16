# MetaDrive OSM Plugins 使用文档

本文档说明 `metadrive/plugins/` 中 OSM 地图插件的常用流程，覆盖地图提取、地图处理和场景生成。默认从仓库根目录 `metadrive-main/` 执行命令。

## 目录结构

```text
metadrive/plugins/
├── manager/osm_map_manager.py                 # OSMMapIO 地图解析/导出，OSMMapManager 环境地图管理
├── envs/osm_env.py                            # OSMEnv 环境与默认配置
├── examples/drive_in_real_osm.py              # 批量 rollout、保存 GIF、导出 ScenarioDescription
├── utils/add_osm_name_tags.py                 # 给 node/way 补 name 标签
├── utils/add_lanelet_relations_from_geometry.py # 从几何关系推断 lanelet 拓扑
└── map_files/                                 # 推荐存放 .osm 地图文件
```

## 运行准备

先进入 MetaDrive 子仓库，并确保当前 Python 环境已经安装 MetaDrive 依赖：

```bash
cd /data_set/UniTraj_benchmark/metadrive-main
export ROOT_PATH="$PWD"
```

`OSMMapIO` 的路径规则：

- 传入绝对路径，或传入当前目录下已存在的路径时，直接使用该路径。
- 只传文件名时，会按 `$ROOT_PATH/metadrive/plugins/map_files/<file>` 查找。
- 未设置 `ROOT_PATH` 时，会从 `osm_map_manager.py` 所在位置回推到 `metadrive-main/`。

## 一、地图提取

地图提取入口在 `manager/osm_map_manager.py` 的 `OSMMapIO`。它支持：

- `.osm -> map_features`
- `.pkl -> map_features`
- `.pkl/map_features -> 自定义 .osm`

### 1. 从 pkl 导出 OSM

适用于已经从 nuPlan、Waymo 或其他管线转换出的 pickle 文件。输入 pkl 需要包含以下任一结构：

- `map_features`：MetaDrive/ScenarioDescription 风格地图特征。
- `road_points`：车道中心线点集；脚本会按 `lane_width` 自动生成左右边界。
- 可选 `road_connection_types`：用于补 `entry_lanes`、`exit_lanes`、`left_neighbor`、`right_neighbor`。
- 可选 `route`、`ego_route` 或 `global_route`：会导出为 `md_kind=ego_route` 的 route way。

示例：

```bash
python -c "from metadrive.plugins.manager.osm_map_manager import OSMMapIO; OSMMapIO.convert_pickle_to_osm('/path/to/input.pkl', 'metadrive/plugins/map_files/my_map_raw.osm', overwrite=True, lane_width=3.5)"
```

导出的 OSM 会给每个 node 写入 `local_x/local_y/local_z`，因此后续解析会优先使用局部米制坐标，不依赖经纬度投影。

### 2. 直接使用已有 OSM

将 `.osm` 放到 `metadrive/plugins/map_files/` 下，或在配置里使用绝对路径。`OSMMapIO` 会解析：

- way 的 `feature_id`，没有时使用 way id。
- way 的 `type` / `Type` / `LaneType` 等标签，用于判断车道、边界、停止线、红绿灯等类型。
- lanelet relation：`<tag k="type" v="lanelet">`，并读取 `center`、`left`、`right` 成员。
- `entry_lanes`、`exit_lanes`、`left_neighbor`、`right_neighbor` 标签或 lanelet 几何关系。

## 二、地图处理

推荐处理顺序：

```text
raw.osm
  -> add_osm_name_tags.py
  -> add_lanelet_relations_from_geometry.py
  -> final_topology_fixed.osm
```

### 1. 补 name 标签

部分工具或人工检查流程更依赖 `name` 标签。默认把元素自身 `id` 写入 `name`：

```bash
python metadrive/plugins/utils/add_osm_name_tags.py \
  metadrive/plugins/map_files/my_map_raw.osm \
  metadrive/plugins/map_files/my_map_name.osm \
  --overwrite
```

如果原始 OSM 有自定义 `Id` 标签，并希望用它作为 `name`：

```bash
python metadrive/plugins/utils/add_osm_name_tags.py \
  metadrive/plugins/map_files/my_map_raw.osm \
  metadrive/plugins/map_files/my_map_name.osm \
  --name-source Id \
  --overwrite
```

### 2. 推断 lanelet 拓扑

当 OSM 只有中心线和边界线、缺少 lanelet relation 时，可以用几何匹配生成 relation：

```bash
python metadrive/plugins/utils/add_lanelet_relations_from_geometry.py \
  metadrive/plugins/map_files/my_map_name.osm \
  metadrive/plugins/map_files/my_map_name_topology_fixed.osm \
  --overwrite \
  --replace-existing \
  --allow-partial \
  --add-metadrive-type-tags \
  --report-path metadrive/plugins/map_files/my_map_name_topology_report.csv
```

脚本识别规则：

- 中心线 way：`Type=Lane` 或 `subtype=centerlane`。
- 边界 way：`Type=LaneBoundary`，或 `subtype=solid/dashed/curbstone`。
- 输出 relation：包含 `center`，尽量匹配 `left` 和 `right`，并写入 `type=lanelet`。

常用参数：

- `--dry-run`：只计算匹配结果，不写输出文件。
- `--report-path`：输出每条中心线的左右边界匹配 CSV。
- `--replace-existing`：删除已有 lanelet relation 后重新生成。
- `--allow-partial`：只有一侧边界匹配成功时也生成 relation。
- `--max-distance`：边界搜索距离，默认 6 米；地图车道宽度异常时可调整。
- `--sample-count`：每条中心线采样点数，默认 17。

### 3. 检查可用 lane id

`drive_in_real_osm.py` 默认配置里没有暴露 `osm_file_path` 命令行参数，先把脚本尾部 `cfg["osm_file_path"]` 改成目标 OSM，再运行：

```bash
python metadrive/plugins/examples/drive_in_real_osm.py --list_lanes
```

输出的 lane id 需要用于后续 ego / adversary 初始位置配置。

## 三、场景生成

场景生成入口是 `examples/drive_in_real_osm.py`。它会：

1. 使用 `OSMEnv` 加载 OSM 地图。
2. 按模板生成 ego 和 adversary 的初始位置。
3. rollout 指定步数。
4. 可选保存 top-down GIF。
5. 可选导出 MetaDrive `ScenarioDescription` pkl。

### 1. 修改地图与车辆模板

在 `drive_in_real_osm.py` 底部的 `cfg` 中设置地图：

```python
cfg = {
    "osm_file_path": "my_map_name_topology_fixed.osm",
    "num_scenarios": 100,
    ...
}
```

在 `build_custom_init_cfg()` 中设置初始位置模板：

```python
relative_template = {
    "ego": {
        "lane_id": ["-101886"],
        "s": [20.0, 30.0],
        "base_speed": 12.0,
    },
    "adversaries": [
        {
            "vehicle_id": "1",
            "name": "adv_1",
            "lane_offset": -1,
            "relative_s": [10.0, 20.0],
            "base_speed": 10.0,
        },
    ],
}
```

相对位置语义：

- `s` / `base_s`：沿 lane 中心线的纵向距离，单位米。
- `lane_offset=0`：与参考车同车道。
- `lane_offset=-1`：参考车右侧一条车道。
- `lane_offset=+1`：参考车左侧一条车道。
- `relative_s > 0`：在参考车前方。
- `relative_s < 0`：在参考车后方。
- adversary 如果显式设置 `lane_id`，则使用绝对车道初始化，忽略 `relative_to/lane_offset/relative_s`。

当前脚本把三个采样变量映射为：

- `c`：ego 的 `s`。
- `a`：`adv_1` 相对 ego 的 `relative_s`。
- `b`：`adv_2` 使用绝对初始化时的 `base_s`。

### 2. 快速生成一批场景

```bash
python metadrive/plugins/examples/drive_in_real_osm.py \
  --use_custom_init true \
  --variant_seed 0 \
  --num_variants 100 \
  --max_steps 100 \
  --top_down true \
  --save_scenarios true \
  --scenario_dir metadrive/plugins/examples/scenario_record/my_run \
  --save_gif true \
  --gif_dir metadrive/plugins/examples/gif_record/my_run
```

### 3. 网格采样

用于生成热力图或系统扫描：

```bash
python metadrive/plugins/examples/drive_in_real_osm.py \
  --sample_mode grid \
  --use_timing_filter false \
  --ego_s_start 20 --ego_s_stop 40 --ego_s_step 5 \
  --adv1_a_start -5 --adv1_a_stop 15 --adv1_a_step 5 \
  --adv2_b_start 0 --adv2_b_stop 30 --adv2_b_step 5 \
  --seeds_per_grid 1 \
  --num_variants 100 \
  --max_steps 100 \
  --save_scenarios true \
  --scenario_dir metadrive/plugins/examples/scenario_record/grid_run
```

### 4. 时间约束采样

默认 `--use_timing_filter true`，脚本会按到达时间过滤 `c/a/b`：

- `red_before_black = t_black - t_red`
- `ego_after_black = t_ego - t_black`

可通过参数放宽或收紧：

```bash
python metadrive/plugins/examples/drive_in_real_osm.py \
  --sample_mode random \
  --use_timing_filter true \
  --red_black_arrival_min -1.0 \
  --red_black_arrival_max 2.0 \
  --ego_black_arrival_min 0.3 \
  --ego_black_arrival_max 2.0 \
  --timing_sample_attempts 20000 \
  --num_variants 100 \
  --save_scenarios true \
  --scenario_dir metadrive/plugins/examples/scenario_record/timing_run
```

如果最终 `accepted < requested`，增大 `--timing_sample_attempts`，扩大 `c/a/b` 范围，或放宽到达时间约束。

## 输出说明

启用 `--save_scenarios true` 后，`scenario_dir` 下会生成：

```text
scenario_dir/
├── dataset_mapping.pkl
├── dataset_summary.pkl
└── sd_osm_v1_<filename_token>.pkl
```

每个 `sd_osm_v1_*.pkl` 是 MetaDrive `ScenarioDescription` 字典，脚本会补充：

- `metadata.variant_seed`
- `metadata.ego_vehicle_parameters`
- `metadata.track_length`
- `metadata.filename_token`

默认 `--export_planned_global_paths true`，导出时会用车辆规划路径覆盖/补齐轨迹，并把 horizon 对齐到 `--max_steps`。若需要保留原始记录轨迹：

```bash
python metadrive/plugins/examples/drive_in_real_osm.py \
  --export_planned_global_paths false \
  --save_scenarios true
```

启用 `--save_gif true` 后，`gif_dir` 下会生成：

```text
osm_episode_<filename_token>.gif
```

## 常见问题

### 找不到 OSM 文件

使用绝对路径，或确认：

```bash
export ROOT_PATH="/data_set/UniTraj_benchmark/metadrive-main"
```

并把 OSM 放在：

```text
metadrive/plugins/map_files/
```

### lane id 不存在

先运行：

```bash
python metadrive/plugins/examples/drive_in_real_osm.py --list_lanes
```

然后把 `cfg["agent_configs"]["default_agent"]` 和 `build_custom_init_cfg()` 中的 `lane_id` / `destination` 更新为当前地图存在的 id。

### 相邻车道解析失败

常见原因是 OSM 缺少 `left_neighbor/right_neighbor` 或 lanelet relation 不完整。优先运行 `add_lanelet_relations_from_geometry.py`，并查看 `--report-path` 生成的 CSV。确实需要跨断裂拓扑投影时，可在 adversary 模板里显式设置：

```python
"allow_geometric_projection_fallback": True
```

该选项可能把车辆投到几何上接近但语义不正确的 lane，建议只在确认地图拓扑断裂时使用。

### 没有生成 pkl

确认命令里包含：

```bash
--save_scenarios true
```

同时注意 `scenario_dir` 是相对当前工作目录解析的路径，推荐使用完整相对路径或绝对路径。

### 只想生成场景，不保存 GIF

```bash
python metadrive/plugins/examples/drive_in_real_osm.py \
  --save_gif false \
  --save_scenarios true
```

## 推荐完整流程

```bash
cd /data_set/UniTraj_benchmark/metadrive-main
export ROOT_PATH="$PWD"

python -c "from metadrive.plugins.manager.osm_map_manager import OSMMapIO; OSMMapIO.convert_pickle_to_osm('/path/to/input.pkl', 'metadrive/plugins/map_files/my_map_raw.osm', overwrite=True, lane_width=3.5)"

python metadrive/plugins/utils/add_osm_name_tags.py \
  metadrive/plugins/map_files/my_map_raw.osm \
  metadrive/plugins/map_files/my_map_name.osm \
  --overwrite

python metadrive/plugins/utils/add_lanelet_relations_from_geometry.py \
  metadrive/plugins/map_files/my_map_name.osm \
  metadrive/plugins/map_files/my_map_name_topology_fixed.osm \
  --overwrite \
  --replace-existing \
  --allow-partial \
  --add-metadrive-type-tags \
  --report-path metadrive/plugins/map_files/my_map_name_topology_report.csv

# 修改 drive_in_real_osm.py 中的 cfg["osm_file_path"] 和车辆 lane 模板后：
python metadrive/plugins/examples/drive_in_real_osm.py --list_lanes

python metadrive/plugins/examples/drive_in_real_osm.py \
  --use_custom_init true \
  --variant_seed 0 \
  --num_variants 100 \
  --max_steps 100 \
  --top_down true \
  --save_scenarios true \
  --scenario_dir metadrive/plugins/examples/scenario_record/my_run \
  --save_gif true \
  --gif_dir metadrive/plugins/examples/gif_record/my_run
```
