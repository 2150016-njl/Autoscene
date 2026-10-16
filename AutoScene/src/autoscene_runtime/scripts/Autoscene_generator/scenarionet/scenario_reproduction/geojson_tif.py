# import fiona
# import geopandas as gpd
# # 使用 fiona 打开并加载 GeoJSON 文件
# with fiona.open('/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/Map_Process/scene.geojson', 'r') as src:
#     # 将文件内容转换为 geopandas GeoDataFrame
#     schema = src.schema
#     crs = src.crs
#     features = [feature for feature in src]
#     gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
# # 打印转换后的 GeoDataFrame
# print(gdf.head())

import geopandas as gpd
import matplotlib.pyplot as plt
# 使用 geopandas 直接读取 GeoJSON 文件
gdf = gpd.read_file('/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/Map_Process/scene.geojson')
# 打印 GeoDataFrame 的头部信息
# print(gdf.head())
# 打印 GeoDataFrame 的几何类型
# print(gdf.geometry.type)
# 检查CRS
# print(gdf.crs)
# 可视化几何对象
# 保存高分辨率图片（可选）
plt.figure(dpi=300, figsize=(30, 20))  # 设置为 300 DPI，并指定尺寸
# 绘制矢量图（这里的 GeoDataFrame 默认是矢量格式）
gdf.plot()
# 显示图形
plt.show()
# 保存为矢量格式（SVG 或 PDF）
plt.savefig('high_resolution_map.svg', dpi=300)  # 保存为 SVG 格式，适合矢量图
# 检查几何对象是否为空
empty_geometries = gdf[gdf.geometry.is_empty]
if not empty_geometries.empty:
    print("存在空的几何对象")




import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from shapely.geometry import box
# 设置栅格化参数
resolution = 1  # 栅格分辨率 (单位：米)
north, south, east, west = gdf.total_bounds  # GeoDataFrame 的边界

# 创建一个代表栅格边界的矩形几何对象
grid_bounds = box(west, south, east, north)

# 过滤掉空的、无效的或不在栅格边界内的几何对象
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid & gdf.geometry.within(grid_bounds)]
print(gdf.geometry.head())
# 计算栅格大小
width = int((east - west) / resolution)
height = int((north - south) / resolution)

# 创建一个空白的栅格图像（默认值为 0）
raster = np.zeros((height, width), dtype=np.uint8)

# 创建栅格化的 GeoTransform
transform = rasterio.transform.from_origin(west, north, resolution, resolution)

# 打印栅格大小信息
print(f"栅格宽度: {width}, 栅格高度: {height}")



# 使用 rasterio.features.rasterize 函数进行栅格化
rasterized = rasterio.features.rasterize(
    [(geom, 1) for geom in gdf.geometry if not geom.is_empty],  # (几何对象, 值) 对
    out=raster,
    transform=transform,
    all_touched=True,
    fill=0,
    dtype='uint8'
)

# 使用 rasterio 保存栅格图像
with rasterio.open(
    "output_raster1.tif",
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=raster.dtype,
    crs=gdf.crs,  # 确保CRS与GeoDataFrame一致
    transform=transform
) as dst:
    dst.write(rasterized, 1)

# 可视化栅格化结果
plt.imshow(rasterized, cmap='gray')
plt.show()

# from rasterio.features import geometry_mask
#
# # 选择只处理多边形或面状数据
# geometries = gdf.geometry
#
# # 使用 geometry_mask 创建一个栅格掩膜
# mask = geometry_mask([mapping(geom) for geom in geometries], transform=transform, invert=True, out_shape=(height, width))
#
# # 填充栅格
# raster[mask] = 1  # 1 代表地物，0 代表空地
#
#
# # 使用 rasterio 保存栅格图像
# with rasterio.open("output_raster.tif", 'w', driver='GTiff', count=1, dtype=raster.dtype,
#                    width=width, height=height, crs='EPSG:4326', transform=transform) as dst:
#     dst.write(raster, 1)
#
#
# # 显示栅格图像
# plt.imshow(raster, cmap='gray', origin='upper')
# plt.colorbar(label='Building Area')
# plt.title('GeoJSON Rasterized Map')
# plt.show()
