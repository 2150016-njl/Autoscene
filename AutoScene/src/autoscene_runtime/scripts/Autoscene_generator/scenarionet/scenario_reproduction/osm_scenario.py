import xml.etree.ElementTree as ET
import re
from pyproj import CRS, Transformer
import numpy as np
import pickle as pkl


def extract_speed_and_convert(unit_str):
    # 使用正则表达式匹配数字部分
    match = re.match(r'(\d+(\.\d+)?)([^\d]+)?', unit_str)
    if match:
        # 提取数字部分并转换为浮点数
        speed_mph = float(match.group(1))
        # 转换为公里每小时
        speed_kph = speed_mph * 0.621371
        return speed_mph, speed_kph
    else:
        raise ValueError("无法解析速度限制值")


# 定义WGS84坐标系
wgs84 = CRS('EPSG:4326')
# 定义UTM坐标系，使用EPSG代码
# zone为50，hemisphere为'north'，可以根据实际需要调整
utm = CRS(f'EPSG:3405')  # 嘉定：3405是UTM区域50，北半球


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
    return polyline


# 假设osm_file是包含OSM数据的文件路径
osm_file = 'TesT_Field_Tongji_West.osm'

# 解析OSM文件
tree = ET.parse(osm_file)
root = tree.getroot()

# 创建一个字典来存储转换后的数据
map_features = {}
nodes = {}
# ways = []

# 遍历解析节点
for node in root.findall('.//node'):
    node_id = node.attrib['id']
    lat = float(node.attrib['lat'])
    lon = float(node.attrib['lon'])
    nodes[node_id] = (lat, lon)

# 遍历OSM文件中的所有道路元素
for way in root.findall('way'):
    way_id = way.attrib['id']
    way_nodes = []
    for nd in way.findall('./nd'):
        ref = nd.attrib['ref']
        way_nodes.append(nodes[ref])
    polyline = wgs_to_utm(way_nodes)

    # 提取道路的特定属性，例如限速
    # 假设maxspeed_tag是包含速度限制的Element对象
    maxspeed_tag = way.find('.//tag[@k="SpeedLimit"]')
    if maxspeed_tag is not None:
        speed_limit_str = maxspeed_tag.get('v')
        try:
            speed_limit_mph, speed_limit_kph = extract_speed_and_convert(speed_limit_str)
        except ValueError as e:
            # 处理错误，例如使用默认值或记录日志
            speed_limit_mph = speed_limit_kph = 0  # 或者其他适当的默认速度限制
    else:
        speed_limit_mph = speed_limit_kph = 0

    # 初始化道路类型
    road_type = None
    # 检查道路类型，只处理Lane和LaneBoundary
    highway_tag = way.find('.//tag[@k="Type"]')
    if highway_tag is not None:
        road_type_value = highway_tag.get('v')
        if road_type_value == 'LaneBoundary':
            road_type = 'ROAD_EDGE_BOUNDARY'
        if road_type_value == 'Lane':
            road_type = 'LANE_SURFACE_STREET'

    # 根据道路类型和属性创建字典条目
    map_features[way_id] = {
        "speed_limit_mph": speed_limit_mph,
        'speed_limit_kph': speed_limit_kph,
        'type': road_type,
        "polyline": polyline
        # "left_neighbor":
        # "right_neighbor":

    }

# 打印或保存转换后的数据
# print(map_features)
# print(list(map_features.items())[:2])
#  使用pickle序列化字典
pickle_data = pkl.dumps(map_features)

# 保存到文件
with open('map_features.pkl', 'wb') as f:
    f.write(pickle_data)
