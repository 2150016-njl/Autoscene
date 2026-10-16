import xml.etree.ElementTree as ET
import re
from pyproj import CRS, Transformer
import numpy as np
import pickle as pkl
from scenario_reproduction.GNSS_info_process import GPS_to_Global
from scenario_reproduction.GNSS_Transform import GNSS_Transform


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


converter_1 = GPS_to_Global()
converter_2 = GNSS_Transform()  # 为解决右舵驾驶问题地图关于x轴镜像，Carla_Y取负

# osm_file是包含OSM数据的文件路径
osm_file = 'TesT_Field_Tongji_West_with_relations.osm'

# 解析OSM文件
tree = ET.parse(osm_file)
root = tree.getroot()

# 创建一个字典来存储转换后的数据
map_features = {}
nodes = {}
ways = {}

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
    polyline_xy = []
    for nd in way.findall('./nd'):
        ref = nd.attrib['ref']
        way_nodes.append(nodes[ref])
    for way_node in way_nodes:
        global_x, global_y = converter_1.GPS(way_node[0], way_node[1])
        Carla_X, Carla_Y, _ = converter_2.GNSS_Global_to_CarlaWest(global_x, global_y, 0)
        polyline_xy.append((Carla_X, Carla_Y))
    polyline_xy = np.array(polyline_xy)
    zeros_z = np.zeros_like(polyline_xy[:, 0])  # 创建与x坐标相同的全零数组
    polyline = np.column_stack((polyline_xy, zeros_z))

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
        "polyline": polyline,
        # "left_boundaries": left_boundaries,
        # "right_boundaries": right_boundaries,
        # "left_neighbor": left_neighbor,
        # "right_neighbor": right_neighbor,
        # "entry_lanes": entry_lanes,
        # "exit_lanes": exit_lanes

    }

    # ====== 新增：将 way 信息存储到 ways 字典 ======
    ways[way_id] = {
        'nodes': way_nodes,  # 存储原始节点坐标
        'polyline': polyline  # 存储转换后的坐标
    }


def convert_nodes_to_polyline(node_ids, is_local=False, with_z=False):
    """将节点ID列表转换为坐标数组"""
    if not node_ids:
        return np.array([])

    polyline_xy = []
    for node_id in node_ids:
        if node_id in nodes:
            lat, lon = nodes[node_id]
            global_x, global_y = converter_1.GPS(lat, lon)
            Carla_X, Carla_Y, _ = converter_2.GNSS_Global_to_CarlaWest(global_x, global_y, 0)
            polyline_xy.append([Carla_X, Carla_Y])

    if len(polyline_xy) == 0:
        return np.array([])

    polyline_xy = np.array(polyline_xy, dtype=np.float32)

    if with_z:
        zeros_z = np.zeros((len(polyline_xy), 1), dtype=np.float32)
        return np.concatenate([polyline_xy, zeros_z], axis=1)
    else:
        return polyline_xy
# 3. 新增：处理 relation 元素以获取 polygon
# 遍历所有 relation 元素
for relation in root.findall('./relation'):
    # 提取关系标签
    tags = {}
    for tag in relation.findall('./tag'):
        tags[tag.get('k')] = tag.get('v')

    # 只处理 type='lanelet' 的关系
    if tags.get('type') != 'lanelet':
        continue

    relation_id = relation.attrib['id']

    # 收集左右边界
    left_boundary_nodes = []
    right_boundary_nodes = []
    center_way_id = None

    for member in relation.findall('./member'):
        member_type = member.get('type')
        member_ref = member.get('ref')
        member_role = member.get('role')

        if member_type == 'way':
            if member_role == 'left':
                if member_ref in ways:
                    left_nodes = ways[member_ref]['nodes']
                    left_boundary_nodes.extend(left_nodes)
            elif member_role == 'right':
                if member_ref in ways:
                    right_nodes = ways[member_ref]['nodes']
                    right_boundary_nodes.extend(right_nodes)
            elif member_role == 'center':
                center_way_id = member_ref

    # 生成 polygon：左边界 + 右边界（逆序）
    if left_boundary_nodes and right_boundary_nodes:
        polygon_nodes = left_boundary_nodes + right_boundary_nodes[::-1]

        # 去重但保持顺序
        seen = set()
        unique_nodes = []
        for node in polygon_nodes:
            if node not in seen:
                seen.add(node)
                unique_nodes.append(node)

        # 转换为 polygon
        # polygon = convert_nodes_to_polyline(unique_nodes, is_local=False, with_z=False)
        if unique_nodes:
            polygon_xy = []
            for coord in unique_nodes:
                lat, lon = coord
                global_x, global_y = converter_1.GPS(lat, lon)
                Carla_X, Carla_Y, _ = converter_2.GNSS_Global_to_CarlaWest(global_x, global_y, 0)
                polygon_xy.append([Carla_X, Carla_Y])

            polygon = np.array(polygon_xy, dtype=np.float32)

            # 确保 polygon 闭合
            if len(polygon) > 2 and not np.allclose(polygon[0], polygon[-1]):
                polygon = np.vstack([polygon, polygon[0:1]])

        # 如果 polygon 是闭合的，确保第一个点和最后一个点相同
        if len(polygon) > 2 and not np.allclose(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0:1]])

        # 将 polygon 添加到对应的中心线 way
        if center_way_id and center_way_id in map_features:
            map_features[center_way_id]['polygon'] = polygon

        # 也添加到左右边界 way
        for member in relation.findall('./member'):
            member_type = member.get('type')
            member_ref = member.get('ref')
            member_role = member.get('role')

            if member_type == 'way' and member_ref in map_features:
                if 'polygon' not in map_features[member_ref]:
                    map_features[member_ref]['polygon'] = polygon

# 打印或保存转换后的数据
# print(map_features)
# print(list(map_features.items())[:2])
#  使用pickle序列化字典
pickle_data = pkl.dumps(map_features)

# 保存到文件
with open('map_features_polygon.pkl', 'wb') as f:
    f.write(pickle_data)