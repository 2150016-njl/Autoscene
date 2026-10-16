# import osmnx as ox
# G = ox.graph_from_xml('TesT_Field_Tongji_West.osm', bidirectional=True, simplify=True)
# ox.save_graph_shapefile(G, '/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/shapefile')

import osmnx as ox
import geopandas as gpd
# 加载本地的 OSM 文件
osm_file_path = "path_to_your_osm_file.osm"
gdf = ox.io.graph_to_gdfs(ox.graph_from_xml(osm_file_path), nodes=False, edges=True)
# 打印一些信息
print(gdf.head())


from rasterio.features import shapes
import rasterio
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
from rasterio.features import rasterize
# 打开Shapefile
file_path = '/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/shapefile/nodes.shp'
driver = ogr.GetDriverByName('ESRI Shapefile')
ds = driver.Open(file_path, gdal.GA_ReadOnly)
if ds is None:
    print("无法打开文件：", file_path)
else:
    layer = ds.GetLayer()

    x_min, x_max, y_min, y_max = layer.GetExtent()
    if x_min >= x_max or y_min >= y_max:
        print("图层边界不正确")
    else:
        pixel_size = 0.0001  # 栅格单元的大小
        width = int((x_max - x_min) / pixel_size)
        height = int((y_max - y_min) / pixel_size)

    if width > 0 and height > 0:
        transform = rasterio.transform.from_origin(x_min, y_max, pixel_size, -pixel_size)
        with rasterio.Env():
            with rasterio.open('TJ_West.tif', 'w', driver='GTiff',
                               width=width, height=height, count=1, dtype='uint8',
                               crs=layer.GetSpatialRef().ExportToWkt(),
                               transform=transform) as dst:
                # 烧录几何对象到栅格数据
                rasterize(
                    [(feature.GetGeometryRef(), 1) for feature in layer],
                    out=dst,
                    transform=transform,
                    all_touched=True,
                    fill=0,
                    dtype='uint8'
                )
                dst.flush_cache()  # 确保数据写入文件
    else:
        print("计算出的宽度或高度为0")


with rasterio.open('TJ_West.tif') as src:
    plt.imshow(src.read(1), cmap='gray')  # 假设你的TIFF是单波段的
    plt.show()

dataset = gdal.Open('TF_West.tif')
band = dataset.GetRasterBand(1)
image = band.ReadAsArray()
print(image)
#
# import geopandas as gpd
# import matplotlib.pyplot as plt
# # import contextily as ctx
# # 加载Shapefile文件
# shp_file = '/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/shapefile/edges.shp'
# gdf = gpd.read_file(shp_file)
# # 绘制地图
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# gdf.plot(ax=ax, color='blue')  # 根据需要调整颜色
# # 添加底图
# # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
# # 设置坐标轴的显示范围
# ax.set_xlim(gdf.total_bounds[0], gdf.total_bounds[2])
# ax.set_ylim(gdf.total_bounds[1], gdf.total_bounds[3])
# # 隐藏坐标轴
# ax.set_axis_off()
# # 显示地图
# plt.show()