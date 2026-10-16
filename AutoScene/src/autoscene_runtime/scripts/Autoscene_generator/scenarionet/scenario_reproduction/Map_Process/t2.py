import time
import xml.etree.ElementTree as ET


def parse_osm(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    nodes = {}
    ways = []

    # 遍历解析节点
    for node in root.findall('.//node'):
        node_id = node.attrib['id']
        lat = float(node.attrib['lat'])
        lon = float(node.attrib['lon'])
        nodes[node_id] = (lat, lon)

    # 遍历解析道路
    for way in root.findall('.//way'):
        way_id = way.attrib['id']
        way_nodes = []
        for nd in way.findall('./nd'):
            ref = nd.attrib['ref']
            way_nodes.append(nodes[ref])
        ways.append(way_nodes)

    return ways


def find_matching_way(ways, lat, lon):
    for way in ways:
        for i in range(len(way) - 1):
            # 检查给定的经纬度是否在道路的范围内
            if is_point_on_line(lat, lon, way[i], way[i + 1]):
                return way
    return None


def is_point_on_line(lat, lon, point1, point2):
    # 检查给定的经纬度是否在两点之间的线段上
    lat1, lon1 = point1
    lat2, lon2 = point2
    if min(lat1, lat2) <= lat <= max(lat1, lat2) and min(lon1, lon2) <= lon <= max(lon1, lon2):
        return True
    return False


# 解析OSM文件
ways = parse_osm('TesT_Field_Tongji_West.osm')
print(ways)
t1 = time.time()
# 给定经纬度
lat = 31.291509
lon = 121.203442319

# 查找匹配的道路
matching_way = find_matching_way(ways, lat, lon)

if matching_way:
    print("Matching way:", matching_way)
else:
    print("No matching way found.")
t2 = time.time()
print(t2 - t1)
