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


def get_path_for_road(ways, road_name):
    for way in ways:
        # 在这里可以根据道路的特征，例如名称，进行过滤
        # 在这个简单的示例中，我们假设道路名称保存在<tag k="name" v="RoadName"/> 中
        for tag in way.findall('./tag'):
            if tag.attrib['k'] == 'name' and tag.attrib['v'] == road_name:
                return way
    return None


# 解析OSM文件
ways = parse_osm('TesT_Field_Tongji_West.osm')

# 获取特定道路的路径
road_name = 'Your Road Name Here'
road_path = get_path_for_road(ways, road_name)

if road_path:
    print("Road path for", road_name, ":", road_path)
else:
    print("Road path for", road_name, "not found.")
