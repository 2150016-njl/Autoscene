"""
更改了road_network_type中的EdgeRoadNetwork为EdgeRoadNetworkWithPlanning
"""
import logging
import numpy as np

from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.plugins.component.road_network.edge_road_network_with_planning import EdgeRoadNetworkWithPlanning
from metadrive.plugins.component.scenario_block.scenario_block_in_osm import ScenarioBlockInOsm
from metadrive.engine.asset_loader import AssetLoader
from metadrive.type import MetaDriveType
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.math import resample_polyline, get_polyline_length


class ScenarioMapInOsm(ScenarioMap):
    def __init__(self, map_index, map_data, random_seed=None):
        self.map_index = map_index
        self.map_data = map_data
        self.need_lane_localization = self.engine.global_config["need_lane_localization"]
        super(ScenarioMapInOsm, self).__init__(map_index=self.map_index, map_data=self.map_data, random_seed=random_seed)

    def _generate(self):
        block = ScenarioBlockInOsm(
            block_index=0,
            global_network=self.road_network,
            random_seed=0,
            map_index=self.map_index,
            map_data=self.map_data,
            need_lane_localization=self.need_lane_localization
        )
        self.crosswalks = block.crosswalks
        self.sidewalks = block.sidewalks
        block.construct_block(self.engine.worldNP, self.engine.physics_world, attach_to_world=True)
        self.blocks.append(block)

    @property
    def road_network_type(self):
        return EdgeRoadNetworkWithPlanning