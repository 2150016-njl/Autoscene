import logging

import numpy as np

from metadrive.constants import PGDrivableAreaProperty
import math

from metadrive.component.lane.point_lane import PointLane
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.math import norm, mph_to_kmh
from metadrive.scenario.utils import read_scenario_data


# return the nearest point"s index of the line
def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)


def _normalize_lane_relations(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [_normalize_lane_reference(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_lane_reference(item) for item in value]
    return [_normalize_lane_reference(value)]


def _normalize_lane_reference(value):
    if isinstance(value, dict):
        normalized = value.copy()
        if "feature_id" in normalized and normalized["feature_id"] is not None:
            normalized["feature_id"] = str(normalized["feature_id"])
        if "id" in normalized and normalized["id"] is not None:
            normalized["id"] = str(normalized["id"])
        return normalized
    if isinstance(value, (str, int, np.integer)):
        return str(value)
    return value


def _resolve_lane_key(lane_id, map_data):
    if lane_id in map_data:
        return lane_id
    lane_id_str = str(lane_id)
    if lane_id_str in map_data:
        return lane_id_str
    try:
        lane_id_int = int(lane_id)
    except (TypeError, ValueError):
        lane_id_int = None
    if lane_id_int in map_data:
        return lane_id_int
    return lane_id


class ScenarioLane(PointLane):
    VIS_LANE_WIDTH = 3.5 # 最初为6
    MAX_SPEED_LIMIT = 100

    def __init__(self, lane_id: int, map_data: dict, need_lane_localization):
        """
        Extract the lane information of one lane, and do coordinate shift if required
        """
        lane_key = _resolve_lane_key(lane_id, map_data)
        lane_data = map_data[lane_key]
        center_line_points = np.asarray(lane_data[ScenarioDescription.POLYLINE])
        self.center_line_points = center_line_points[..., :2].copy()
        if ScenarioDescription.POLYGON in lane_data and len(lane_data[ScenarioDescription.POLYGON]) > 3:
            polygon = np.asarray(lane_data[ScenarioDescription.POLYGON])
        else:
            polygon = None
        
        # 这里的问题是地图信息中包括了车道的中心线polyline和由左右边界line组成的polygon，但是在车道merging或narrowing的时候存在重叠部分时，
        # 那重叠部分只会属于一个车道，使得在导航时判断车道在哪一个车道上时总是选择那一个而不会选择为另一个导致部分路径永远不会走
        # 所以如果设置为polygon=None会以中心线为准，车道宽度为VIS_LANE_WIDTH来表示车道的宽度，能够解决重叠部分的归属问题，缺点就是polygon显示不准确
        polygon = None    
        if "speed_limit_kmh" in lane_data or "speed_limit_kph" in lane_data or "speed_limit_mph" in lane_data:
            speed_limit_kmh = lane_data.get("speed_limit_kmh", lane_data.get("speed_limit_kph", None))
            if speed_limit_kmh is None:
                speed_limit_kmh = mph_to_kmh(lane_data["speed_limit_mph"])
        else:
            speed_limit_kmh = self.MAX_SPEED_LIMIT
        super(ScenarioLane, self).__init__(
            center_line_points=center_line_points,
            width=self.VIS_LANE_WIDTH,
            polygon=polygon,
            speed_limit=speed_limit_kmh,
            need_lane_localization=need_lane_localization,
            metadrive_type=lane_data[ScenarioDescription.TYPE]
        )
        self.junction = lane_data.get('junction', None) # cht
        self.index = str(lane_key)
        self.lane_type = lane_data["type"]
        self.entry_lanes = _normalize_lane_relations(lane_data.get(ScenarioDescription.ENTRY))
        self.exit_lanes = _normalize_lane_relations(lane_data.get(ScenarioDescription.EXIT))
        self.left_lanes = _normalize_lane_relations(lane_data.get(ScenarioDescription.LEFT_NEIGHBORS))
        self.right_lanes = _normalize_lane_relations(lane_data.get(ScenarioDescription.RIGHT_NEIGHBORS))
        

    @staticmethod
    def try_get_polygon(map_data, lane_id):
        """
        This method tries to infer polygon from the boundaries
        """
        raise DeprecationWarning("This function can not infer the polygon still")
        lane_info = map_data[lane_id]
        center_line = lane_info["polyline"]
        left_boundary_points = []
        for boundary_info in lane_info[ScenarioDescription.LEFT_BOUNDARIES]:
            boundary = map_data[boundary_info["boundary_feature_id"]][ScenarioDescription.POLYLINE]
            start = center_line[int(boundary_info["lane_start_index"])]
            end = center_line[int(boundary_info["lane_end_index"])]

            start_on_boundary = nearest_point(start, boundary)
            end_on_boundary = nearest_point(end, boundary)

            points = boundary[start_on_boundary:end_on_boundary]
            left_boundary_points.append(points)

        left_boundary_points = np.concatenate(left_boundary_points, axis=0) if len(left_boundary_points) > 0 else []

        right_boundary_points = []
        for boundary_info in lane_info[ScenarioDescription.RIGHT_BOUNDARIES]:
            boundary = map_data[boundary_info["boundary_feature_id"]][ScenarioDescription.POLYLINE]
            start = center_line[int(boundary_info["lane_start_index"])]
            end = center_line[int(boundary_info["lane_end_index"])]

            start_on_boundary = nearest_point(start, boundary)
            end_on_boundary = nearest_point(end, boundary)

            points = boundary[start_on_boundary:end_on_boundary]
            right_boundary_points.append(points)

        right_boundary_points = np.concatenate(right_boundary_points, axis=0) if len(right_boundary_points) > 0 else []
        if len(left_boundary_points) == 0 or len(right_boundary_points) == 0:
            return None
        else:
            polygon = np.concatenate([left_boundary_points, right_boundary_points], axis=0)[..., :2]
            return np.asarray(polygon)

    def get_lane_width(self, lane_id, map_data):
        """
        We use this function to get possible lane width from raw data
        """
        return self.VIS_LANE_WIDTH
        if not (ScenarioDescription.RIGHT_NEIGHBORS in map_data[lane_id]
                and ScenarioDescription.LEFT_NEIGHBORS in map_data[lane_id]):
            return self.VIS_LANE_WIDTH
        right_lanes = map_data[lane_id][ScenarioDescription.RIGHT_NEIGHBORS]
        left_lanes = map_data[lane_id][ScenarioDescription.LEFT_NEIGHBORS]
        if len(right_lanes) + len(left_lanes) == 0:
            return max(sum(map_data[lane_id].get("width", 0)), self.VIS_LANE_WIDTH)
        dist_to_left_lane = 0
        dist_to_right_lane = 0
        if len(right_lanes) > 0 and "feature_id" in right_lanes[0]:
            right_lane = map_data[right_lanes[0]["feature_id"]]
            self_start = int(right_lanes[0]["self_start_index"])
            neighbor_start = int(right_lanes[0]["neighbor_start_index"])
            n_point = right_lane[ScenarioDescription.POLYLINE][neighbor_start]
            self_point = map_data[lane_id][ScenarioDescription.POLYLINE][self_start]
            dist_to_right_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        if len(left_lanes) > 0 and "feature_id" in left_lanes[0]:
            left_lane = map_data[left_lanes[-1]["feature_id"]]
            self_start = int(left_lanes[-1]["self_start_index"])
            neighbor_start = int(left_lanes[-1]["neighbor_start_index"])
            n_point = left_lane[ScenarioDescription.POLYLINE][neighbor_start]
            self_point = map_data[lane_id][ScenarioDescription.POLYLINE][self_start]
            dist_to_left_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        return max(dist_to_left_lane, dist_to_right_lane, self.VIS_LANE_WIDTH)

    def __del__(self):
        logging.debug("ScenarioLane is released")

    def destroy(self):
        self.index = None
        self.entry_lanes = None
        self.exit_lanes = None
        self.left_lanes = None
        self.right_lanes = None
        super(ScenarioLane, self).destroy()


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "test.pkl", unix_style=False)
    data = read_scenario_data(file_path)
    print(data)
    lane = ScenarioLane(108, data["map_features"])
