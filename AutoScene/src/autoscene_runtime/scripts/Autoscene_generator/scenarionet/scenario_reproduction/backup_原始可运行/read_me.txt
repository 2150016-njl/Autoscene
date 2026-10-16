运行顺序：
1、obj_info.py从bag文件获取人车信息
2、osm_scenario_dongfeng.py从osm文件获取地图信息
osm_scenario_dongfeng.py >> map_features_dongfeng.pkl
osm_scenario_west_polygon.py >> map_features_polygon.pkl
3、dongfeng.py整合地图及物体信息得到summary文件
4、运行sim.py进行metadrive可视化(命令如下)
metadrive渲染命令：
python -m scenarionet.sim -d /home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/dataset --render 2D
python -m scenarionet.sim -d /home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/dataset --render 3D
