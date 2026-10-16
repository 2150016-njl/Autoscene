import numpy as np
from metadrive.component.scenario_block.scenario_block import ScenarioBlock
from metadrive.plugins.component.road_network.edge_road_network_with_planning import EdgeRoadNetworkWithPlanning
from metadrive.plugins.component.lane.scenario_lane_in_osm import ScenarioLane # 是从scenario_lane_in_osm中导入ScenarioLane类，而不是从scenario_lane中导入ScenarioLane类
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.type import MetaDriveType


class ScenarioBlockInOsm(ScenarioBlock):
    def _sample_topology(self) -> bool:
        for object_id, data in self.map_data.items():
            if MetaDriveType.is_lane(data.get("type", False)):
                if len(data[ScenarioDescription.POLYLINE]) <= 1:
                    continue
                lane = ScenarioLane(object_id, self.map_data, self.need_lane_localization)
                self.block_network.add_lane(lane)
            elif MetaDriveType.is_sidewalk(data["type"]):
                self.sidewalks[object_id] = {
                    ScenarioDescription.TYPE: MetaDriveType.BOUNDARY_SIDEWALK,
                    ScenarioDescription.POLYGON: np.asarray(data[ScenarioDescription.POLYGON])[..., :2]
                }
            elif MetaDriveType.is_crosswalk(data["type"]):
                self.crosswalks[object_id] = {
                    ScenarioDescription.TYPE: MetaDriveType.CROSSWALK,
                    ScenarioDescription.POLYGON: np.asarray(data[ScenarioDescription.POLYGON])[..., :2]
                }
            else:
                pass
        return True
    
    @property
    def block_network_type(self):
        return EdgeRoadNetworkWithPlanning
