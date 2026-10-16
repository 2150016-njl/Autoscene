import math
import numpy as np
import pickle as pkl
from collections import defaultdict
import xml.etree.ElementTree as ET
import re
from pyproj import CRS, Transformer

with open('/home/mj/mdsn/scenarionet/dataset/test/map_features.pkl', 'rb') as f:
    map_features = pkl.load(f)

wgs84 = CRS('EPSG:4326')
utm = CRS(f'EPSG:3405')


def wgs_to_utm(lat_lons):
    # 创建转换器
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)
    # 使用矢量化操作进行批量转换
    lats, lons = np.array(lat_lons).T
    utm_coords = transformer.transform(lons, lats)
    # 转换后的坐标点可以构成polyline
    polyline_xy = np.column_stack((utm_coords[0], utm_coords[1]))
    # 给polyline增加一个全为零的z坐标维度
    zeros_z = np.zeros_like(polyline_xy[:, 0])  # 创建与x坐标相同的全零数组
    polyline = np.column_stack((polyline_xy, zeros_z))  # 在第三列添加全零z坐标
    return polyline, utm_coords


osm_file = '/home/mj/mdsn/scenarionet/dataset/test/TesT_Field_Tongji_West.osm'
tree = ET.parse(osm_file)
root = tree.getroot()
nodes = {}
# position = np.zeros((500, 3), dtype=np.float64)
# position2 = np.zeros((500, 3), dtype=np.float64)
# heading0 = np.full(500, 1.0, dtype=np.float64)
# heading2 = np.zeros(500, dtype=np.float64)
position = np.zeros((500, 3))
position2 = np.zeros((500, 3))
heading0 = np.full(500, 1.0)
heading2 = np.zeros(500)
# 遍历解析节点
for node in root.findall('.//node'):
    node_id = node.attrib['id']
    lat = float(node.attrib['lat'])
    lon = float(node.attrib['lon'])
    nodes[node_id] = (lat, lon)
# 选取一条路进行循迹
for way in root.findall('way'):
    way_id = way.attrib['id']
    way_nodes = []
    if way_id == '-1500':
        for nd in way.findall('./nd'):
            ref = nd.attrib['ref']
            way_nodes.append(nodes[ref])
        polyline, utm_coords = wgs_to_utm(way_nodes)
        position[:, 0] = np.linspace(utm_coords[0][0], utm_coords[0][-1], num=500, dtype=np.float64)
        position[:, 1] = np.linspace(utm_coords[1][0], utm_coords[1][-1], num=500, dtype=np.float64)
        # position[:, 0] = np.linspace(0, 100, num=500)
        # position[:, 1] = np.linspace(0, 100, num=500)
        # print(utm_coords[0], utm_coords[1], position)
        heading = heading0 * math.atan2((utm_coords[1][-1] - utm_coords[1][0]),
                                        (utm_coords[0][-1] - utm_coords[0][0]))
        print(heading[0])
    if way_id == '-1499':
        for nd in way.findall('./nd'):
            ref = nd.attrib['ref']
            way_nodes.append(nodes[ref])
        _, utm_coords = wgs_to_utm(way_nodes)
        position2[300:, 0] = np.linspace(utm_coords[0][-1], utm_coords[0][7], num=200, dtype=np.float64)
        position2[300:, 1] = np.linspace(utm_coords[1][-1], utm_coords[1][7], num=200, dtype=np.float64)
        # position2[300:, 0] = np.linspace(utm_coords[0][-1], utm_coords[0][7], num=200)
        # position2[300:, 1] = np.linspace(utm_coords[1][-1], utm_coords[1][7], num=200)
        # print(utm_coords[0], utm_coords[1], position)
        heading2[300:] = heading0[300:] * math.atan2((utm_coords[1][7] - utm_coords[1][-1]),
                                                     (utm_coords[0][7] - utm_coords[0][-1]))
        print(heading2[400])
# position = np.zeros((500, 3), dtype=np.float32)
# position[:, 0] = np.arange(2051504, 02051499, -0.01, dtype=np.float32)
# position[:, 1] = np.arange(3577705, 3577700, -0.01, dtype=np.float32)

# position_2 = np.zeros((500, 3), dtype=np.float32)
# position_2[300:, 0] = np.arange(2051394, 2051457, 0.315, dtype=np.float32)
# position_2[300:, 1] = 3577777

vehicle_2 = np.zeros(500)
vehicle_2[300:] = 1
vehicle_2_valid = vehicle_2 != 0

velocity = np.zeros((500, 2))  # 速度是xy方向还是车的纵横向？车一直在左右晃
velocity[:, 0] = 1.273
velocity[:, 1] = 1.485
velocity_2 = np.zeros((500, 2))
velocity_2[300:, 0] = 0.125
velocity_2[300:, 0] = 0.4
#  scenario.pkl
scenario = {}
scenario["id"] = "1"
scenario["version"] = "2024-08-13"[300:]
scenario["length"] = 500
scenario["tracks"] = {"1": {"type": "VEHICLE",
                            "state": {"position": position,
                                      "length": 4 * np.ones(500),
                                      "width": 2 * np.ones(500),
                                      "height": 1.8 * np.ones(500),
                                      "heading": np.float64(heading),
                                      #   "heading": np.degrees(0.5*np.zeros(500, dtype=np.float32)),
                                      "velocity": velocity,
                                      "valid": np.ones(500, dtype=bool)},
                            "metadata": {}},
                      "2": {"type": "VEHICLE",
                            "state": {"position": position2,
                                      "length": 4 * vehicle_2,
                                      "width": 2 * vehicle_2,
                                      "height": 1.8 * vehicle_2,
                                      "heading": np.float64(heading2),
                                      "velocity": velocity_2,
                                      "valid": vehicle_2_valid},
                            "metadata": {}}}
print(scenario["tracks"]["1"]["state"]["heading"][0])
scenario["dynamic_map_states"] = {}
scenario["map_features"] = map_features
scenario["metadata"] = {}

# summary.pkl
dataset_summary = {}
data_summary = {}
data_summary['id'] = "1"
data_summary['coordinate'] = "test"
data_summary["ts"] = np.arange(0, 50, 0.1)
data_summary["metadrive_processed"] = False
data_summary["sdc_id"] = "1"
data_summary["dataset"] = "test"
data_summary["scenario_id"] = "1"
data_summary["source_file"] = ""
data_summary["track_length"] = 500
data_summary["current_time_index"] = 0
data_summary["sdc_track_index"] = 100
data_summary["objects_of_interest"] = []
data_summary["tracks_to_predict"] = {}
data_summary["object_summary"] = {"1": {"type": "VEHICLE",
                                        "object_id": 1,
                                        "track_length": 500,
                                        "moving_distance": 50,  # 代表什么
                                        "valid_length": 500,
                                        "continuous_valid_length": 500},
                                  "2": {"type": "VEHICLE",
                                        "object_id": 2,
                                        "track_length": 500,
                                        "moving_distance": 80,
                                        "valid_length": 200,
                                        "continuous_valid_length": 200}}
data_summary["number_summary"] = {"num_objects": 2,
                                  "object_type": {"VEHICLE"},
                                  "num_objects_each_type": {"VEHICLE": 2},
                                  "num_moving_objects": 2,
                                  "num_moving_objects_each_type": defaultdict(int, {'VEHICLE': 2}),
                                  "num_traffic_lights": 0,
                                  "num_traffic_light_types": {},
                                  "num_traffic_light_each_step": {},
                                  "num_map_features": 3,
                                  "map_height_diff": 0}

scenario["metadata"] = data_summary

with open("/home/karim/mdsn/scenarionet/dataset/test1/sd_scenario_test.pkl", "wb") as f:
    pkl.dump(scenario, f)

dataset_summary["sd_scenario_test.pkl"] = data_summary
with open("/home/karim/mdsn/scenarionet/dataset/test1/dataset_summary.pkl", "wb") as f:
    pkl.dump(dataset_summary, f)
