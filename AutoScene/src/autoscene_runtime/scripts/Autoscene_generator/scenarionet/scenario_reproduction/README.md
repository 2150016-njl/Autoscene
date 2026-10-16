# 数据构建 Pipeline

将 ROS bag 数据和 OSM 地图数据转换为 [ScenarioNet](https://github.com/opendilab/ScenarioNet) 标准格式的数据集，用于自动驾驶场景的渲染、训练和测试。

## 数据流总览

```
bag数据 ──导出──▶ Obj_info_batch_X.csv     TesT_Field_Tongji_West.osm
                    │                               │
                    ▼                               ▼
         ObjectInfoProcessor              OSMMapProcessor (dongfeng.py)
         (dongfeng.py:63-161)             (dongfeng.py:164-312)
                    │                               │
                    ▼                               ▼
              obj_info_X.pkl               map_features_polygon.pkl
              (人车轨迹)                      (地图特征 + 车道多边形)
                    │                               │
                    └──────────┬────────────────────┘
                               ▼
                    ScenarioDatasetBuilder
                    (dongfeng.py:315-525)
                               │
                               ▼
              sd_<scenario_id>.pkl          ← ScenarioNet 标准格式
              dataset_summary.pkl
```

## 一键运行

```bash
python scenario_reproduction/dongfeng.py
```

将 CSV 和 OSM 直接转换成 ScenarioNet 数据集。

**默认输入：** `scenario_reproduction/` 下
- `Obj_info_batch_1.csv` — 目标物体轨迹（人/车位置、速度、航向角）
- `TesT_Field_Tongji_West_with_relations.osm` — 同济西校区高精地图（含 lanelet 关系）

**默认中间产物：** `scenario_reproduction/` 下
- `obj_info_1.pkl` — 序列化的人车轨迹
- `map_features_polygon.pkl` — 序列化的地图特征（含车道多边形）

**默认输出目录：** `scenario_reproduction/tjwest260528obj_info_1/`
```
├── sd_tjwest260528obj_info_1.pkl   ← 主场景文件
└── dataset_summary.pkl               ← 数据集摘要
```

### 通过命令行参数自定义

```bash
python scenario_reproduction/dongfeng.py \
    --csv <CSV路径> \
    --osm <OSM路径> \
    --scenario-id <场景ID> \
    --dataset-name <数据集名称> \
    --output-dir <输出目录> \
    --obj-info-pkl <obj_info缓存路径> \
    --map-features-pkl <地图特征缓存路径> \
    --no-polygons  # 不带车道多边形（体积更小，但无法做车道关联）
```

### 分步运行

```bash
# Step 1: 从 CSV 提取人车轨迹
python scenario_reproduction/obj_info.py

# Step 2a: 从 OSM 提取地图特征（含 lanelet 多边形）
python scenario_reproduction/osm_scenario_west_polygon.py

# Step 2b: 从 OSM 提取基础地图特征（不含多边形）
python scenario_reproduction/osm_scenario_dongfeng.py

# Step 3: 用 ScenarioDatasetBuilder 合并（直接 import 使用）
```

## 文件清单

### 核心文件

#### `dongfeng.py` — 主流程脚本

包含三个核心类：

| 类 | 行号 | 功能 | 输入 → 输出 |
|----|------|------|-------------|
| `ObjectInfoProcessor` | 63–161 | 解析 CSV 中 actor 位置/速度/航向/尺寸 | CSV → `obj_info.pkl`。字段：`actor_pos.x/y/z`、`actor_vel.x/y/z`、`actor_psi`、`actor_width/length` |
| `OSMMapProcessor` | 164–312 | 解析 OSM 中的 way/lanelet，转换 GPS 到 Carla 坐标 | OSM → `map_features.pkl`。特征含：polyline、speed_limit、type、polygon（车道多边形） |
| `ScenarioDatasetBuilder` | 315–525 | 合并 obj_info + map_features 为 ScenarioNet 标准格式 | obj_info + map_features → scenario.pkl + dataset_summary.pkl |
| `DongfengPipeline` | 528–598 | 一键串联三个步骤，支持 CLI 参数 | |

**场景数据结构**（`scenario.pkl`）：

```python
{
  "id": str,                    # 场景ID
  "version": str,               # 版本号
  "length": int,                # 时间步数
  "tracks": {
    "ego": {                    # 自车（或"0"）
      "type": "VEHICLE" | "PEDESTRIAN",
      "state": {
        "position": np.ndarray,   # (N, 3)  Carla坐标
        "length": np.ndarray,     # (N,)
        "width": np.ndarray,      # (N,)
        "height": np.ndarray,     # (N,)  默认1.6
        "heading": np.ndarray,    # (N,)  弧度，归一化到 [-π, π)
        "velocity": np.ndarray,   # (N, 3)
        "valid": np.ndarray,      # (N,)  bool
      },
      "metadata": { "type": ..., "track_length": ..., "object_id": ... }
    },
    "1": { ... },               # 其他车辆
    ...
  },
  "map_features": { ... },    # 地图特征
  "dynamic_map_states": {},   # 信号灯状态（当前为空）
  "metadata": {
    "coordinate": "metadrive",
    "sdc_id": "ego",
    "ts": np.ndarray,           # 时间戳序列
    ...
  }
}
```

#### 中间步骤脚本

| 脚本 | 内部实质 | 说明 |
|------|----------|------|
| `obj_info.py` | 调用 `ObjectInfoProcessor(CURRENT_DIR/"Obj_info_batch_1.csv").run(...)` | 仅生成 obj_info 中间 pkl |
| `osm_scenario_west_polygon.py` | 调用 `OSMMapProcessor(..., include_polygons=True)` | 生成带车道多边形的 map_features |
| `osm_scenario_dongfeng.py` | 调用 `OSMMapProcessor(..., include_polygons=False)` | 生成基础地图特征（不含多边形） |

### 坐标转换支持

#### `GNSS_info_process.py` — GPS → 高斯投影

| 类 | 功能 |
|----|------|
| `GPS_to_Global` | `GPS(lat, lon)` — WGS84 经纬度 → 高斯-克吕格投影（3°带，中央子午线 120°E），再偏移 `(615000, 3463000)` 得到 Global 坐标 |
| `GNSS_UDP_receive` | 惯导 UDP 数据接收与解析（本项目未直接使用） |

#### `GNSS_Transform.py` — 高斯投影 → Carla 坐标

| 方法 | 功能 |
|------|------|
| `GNSS_Global_to_CarlaWest(global_x, global_y, yaw)` | Global → Carla 西区坐标。旋转 `-180.8°` + 平移 `(169.3, 846.6)`，Y 取反 |
| `GNSS_CarlaWest_to_Global(carla_x, carla_y, yaw)` | **逆变换**，Carla → Global |
| `GNSS_Global_to_CarlaEast(...)` | 东区坐标转换（本项目未使用） |

坐标转换参数通过**地图匹配**现场标定，硬编码在类中。

#### `carla_to_gps.py` — 反向坐标转换工具

将场景中的 Carla 坐标及航向角转换为 GPS 经纬度 + 地理航向角。

**依赖：** 标准库 + `numpy`，无第三方 GIS 库。

**使用方式：**

```bash
# Demo：Carla原点→GPS，heading 转换对照表
python scenario_reproduction/carla_to_gps.py

# 单点转换
python scenario_reproduction/carla_to_gps.py --x 10.0 --y -20.0

# 从场景 pkl 批量导出所有物体起点 GPS + 航向角
python scenario_reproduction/carla_to_gps.py \
    --scenario tjwest260528obj_info_1/sd_tjwest260528obj_info_1.pkl
```

**核心函数：**

```python
carla_to_gps(carla_x, carla_y, z=0.0)
  # → (latitude, longitude)

stored_heading_to_geographic(stored_heading_rad)
  # → (geo_deg, carla_psi_deg)
```

### 后处理

#### `split.py` — 长场景裁剪

将长场景按滑动窗口切分为多个子场景。

```bash
# 滑动窗口切分（重叠30帧）
python split.py \
    --input tjwest260528obj_info_1/sd_tjwest260528obj_info_1.pkl \
    --output_dir ./output/splits \
    --step 200
```

含已注释的历史方法（按指定 `--starts` 列表切分）和 merged_summary 工具。

### 其他工具

| 文件 | 用途 |
|------|------|
| `osm_scenario.py` | 用 `pyproj` 做 UTM 投影的旧版地图处理（不再使用） |
| `geojson_tif.py` | 可视化 GeoJSON 地图文件 |

## 坐标系说明

### 坐标转换链

```
GPS (lat, lon)
    │ Gauss-Kruger投影 (3°带, CM=120°E, WGS84)
    ▼
Global (x, y)     ← 偏移 (-615000, -3463000)
    │ GNSS_Global_to_CarlaWest (旋转-180.8°, 平移169.3/846.6)
    ▼
Carla坐标 (x, y)  ← Y轴镜像(右舵驾驶修正), 最终存储
```

### origin and heading

| 坐标/角度 | 约定 |
|-----------|------|
| Carla 坐标 | X≈东, Y≈北（Y镜像后Y≈南）, 原点≈同济西校区某点 |
| Carla 原点 GPS | `carla_to_gps(0,0)` ≈ **(31.2965°N, 121.4113°E)** |
| 场景 heading | 弧度, `[-π, π)`，经过 `2π-psi` 变换，**非原始 `actor_psi`** |
| 地理航向角 | **0°=北**，顺时针增加，90°=东 |

### heading 反向

```python
geo_deg, _ = stored_heading_to_geographic(tracks[id]["state"]["heading"][0])
```

输入场景中存的 heading（经 `2π-psi` 变换后的值），输出地理航向角（°）。

## 可视化/验证

```bash
# 2D 渲染
python -m scenarionet.sim \
    -d scenario_reproduction/tjwest260528obj_info_1 \
    --render 2D

# 3D 渲染
python -m scenarionet.sim \
    -d scenario_reproduction/tjwest260528obj_info_1 \
    --render 3D
```
