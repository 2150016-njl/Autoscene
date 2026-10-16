"""
能够直接解析OSM文件或pkl文件并生成地图的 MapManager
"""
import copy
import sys
sys.path.append("../")  # add project root dir to path
from metadrive.plugins.component.map.scenario_map_in_osm import ScenarioMapInOsm
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.engine.logger import get_logger, set_log_level
from metadrive.scenario.parse_object_state import get_idm_route
from metadrive.type import MetaDriveType

import xml.etree.ElementTree as ET # 解析XML文件
import json
import re
from pyproj import CRS, Transformer
import numpy as np
import pickle as pkl
import os
import math

logger = get_logger()

class OSMMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority
    DEFAULT_DATA_BUFFER_SIZE = 200

    def __init__(self):
        super(OSMMapManager, self).__init__()
        self.store_map = self.engine.global_config.get("store_map", False)
        self.current_map = None
        self._no_map = self.engine.global_config["no_map"]
        self.map_num = self.engine.global_config["num_scenarios"]
        self.start_scenario_index = self.engine.global_config["start_scenario_index"]
        self._stored_maps = {
            i: None
            for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)
        }
        self._stored_route_points = {
            i: None
            for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)
        }

        # we put the route searching function here
        self.sdc_start_point = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None
        self._managed_default_spawn_velocity = False

    def _set_route_from_points(self, route_points):
        self.current_sdc_route = None
        self.sdc_start_point = None
        self.sdc_dest_point = None

        if route_points is None:
            return None

        route_points = np.asarray(route_points, dtype=float)
        if route_points.ndim != 2 or route_points.shape[0] < 2 or route_points.shape[1] < 2:
            return None

        route_points = route_points[:, :2].copy()
        self.current_sdc_route = get_idm_route(route_points)
        self.sdc_start_point = copy.deepcopy(route_points[0])
        self.sdc_dest_point = copy.deepcopy(route_points[-1])
        return route_points

    def _sync_default_agent_spawn_from_route(self, spawn_velocity=None, fallback_heading=None):
        agent_configs = copy.deepcopy(self.engine.global_config.get("agent_configs", {}))
        default_cfg = copy.deepcopy(agent_configs.get(DEFAULT_AGENT, {}))

        use_route_spawn = bool(self.engine.global_config.get("use_replay_ego_trj", False))
        if use_route_spawn and self.sdc_start_point is not None:
            if self.current_sdc_route is not None:
                spawn_heading = float(self.current_sdc_route.heading_theta_at(0.0))
            elif fallback_heading is not None:
                spawn_heading = float(fallback_heading)
            else:
                spawn_heading = None

            if spawn_heading is not None:
                default_cfg["spawn_position_heading"] = (copy.deepcopy(self.sdc_start_point), spawn_heading)
                if spawn_velocity is not None:
                    default_cfg["spawn_velocity"] = copy.deepcopy(spawn_velocity)
                    self._managed_default_spawn_velocity = True
                elif self._managed_default_spawn_velocity:
                    if "spawn_velocity" in default_cfg:
                        default_cfg.pop("spawn_velocity")
                    self._managed_default_spawn_velocity = False
            elif "spawn_position_heading" in default_cfg:
                default_cfg.pop("spawn_position_heading")
        else:
            if "spawn_position_heading" in default_cfg:
                default_cfg.pop("spawn_position_heading")
            if self._managed_default_spawn_velocity:
                if "spawn_velocity" in default_cfg:
                    default_cfg.pop("spawn_velocity")
                self._managed_default_spawn_velocity = False

        agent_configs[DEFAULT_AGENT] = default_cfg
        self.engine.global_config["agent_configs"] = agent_configs

    @staticmethod
    def _normalize_lane_id(lane_or_index):
        if lane_or_index is None:
            return None
        if isinstance(lane_or_index, (str, int, np.integer)):
            return str(lane_or_index)
        if hasattr(lane_or_index, "index"):
            return str(lane_or_index.index)
        return str(lane_or_index)

    def _resolve_lane(self, lane_index):
        if self.current_map is None:
            return None
        try:
            return self.current_map.road_network.get_lane(lane_index)
        except Exception:
            return None

    def _collect_downstream_lane_ids(self, start_lane_ids, hops: int):
        frontier = {self._normalize_lane_id(lane_id) for lane_id in (start_lane_ids or [])}
        frontier = {lane_id for lane_id in frontier if lane_id is not None}
        if not frontier:
            return []

        last_non_empty = set(frontier)
        for _ in range(max(0, int(hops))):
            next_frontier = set()
            for lane_id in frontier:
                lane = self._resolve_lane(lane_id)
                if lane is None:
                    continue
                next_frontier.update(
                    self._normalize_lane_id(next_lane_id) for next_lane_id in list(getattr(lane, "exit_lanes", []) or [])
                )
            next_frontier = {lane_id for lane_id in next_frontier if lane_id is not None}
            if not next_frontier:
                break
            frontier = next_frontier
            last_non_empty = set(frontier)
        return sorted(last_non_empty)

    def _align_default_agent_spawn_with_entry_successor(self):
        if self.current_map is None:
            return
        if bool(self.engine.global_config.get("use_replay_ego_trj", False)):
            return

        agent_configs = copy.deepcopy(self.engine.global_config.get("agent_configs", {}))
        default_cfg = copy.deepcopy(agent_configs.get(DEFAULT_AGENT, {}))
        downstream_hops = max(0, int(default_cfg.get("entry_spawn_downstream_hops", 0)))
        if downstream_hops <= 0:
            return

        target_lane_id = None
        spawn_lane_index = default_cfg.get("spawn_lane_index", None)
        spawn_position_heading = default_cfg.get("spawn_position_heading", None)

        if spawn_position_heading is not None:
            spawn_position, spawn_heading = spawn_position_heading
            try:
                lane_id, on_lane = self.current_map.road_network.get_closest_lane_index(
                    spawn_position, float(spawn_heading), None, use_heading_filter=False
                )
            except Exception:
                lane_id, on_lane = None, False
            if on_lane:
                lane = self._resolve_lane(lane_id)
                if lane is not None and len(getattr(lane, "entry_lanes", []) or []) == 0:
                    downstream_lane_ids = self._collect_downstream_lane_ids([lane.index], downstream_hops)
                    if downstream_lane_ids:
                        target_lane_id = downstream_lane_ids[0]
        elif spawn_lane_index is not None:
            lane = self._resolve_lane(spawn_lane_index)
            if lane is not None and len(getattr(lane, "entry_lanes", []) or []) == 0:
                downstream_lane_ids = self._collect_downstream_lane_ids([lane.index], downstream_hops)
                if downstream_lane_ids:
                    target_lane_id = downstream_lane_ids[0]

        if target_lane_id is None:
            return

        target_lane = self._resolve_lane(target_lane_id)
        if target_lane is None:
            return

        lane_length = float(getattr(target_lane, "length", 0.0) or 0.0)
        spawn_longitude = float(default_cfg.get("spawn_longitude", 5.0))
        spawn_longitude = min(max(1.0, spawn_longitude), max(1.0, lane_length - 1.0))

        default_cfg["spawn_lane_index"] = str(target_lane.index)
        default_cfg["spawn_longitude"] = float(spawn_longitude)
        default_cfg["spawn_position_heading"] = (
            np.asarray(target_lane.position(spawn_longitude, 0.0), dtype=float),
            float(target_lane.heading_theta_at(spawn_longitude)),
        )
        agent_configs[DEFAULT_AGENT] = default_cfg
        self.engine.global_config["agent_configs"] = agent_configs

    def reset(self):
        if not self._no_map:
            seed = self.engine.global_random_seed
            assert self.start_scenario_index <= seed < self.start_scenario_index + self.map_num

            self.sdc_start_point = None
            self.current_sdc_route = None
            self.sdc_dest_point = None

            route_points = None
            if self._stored_maps[seed] is None:
                # 使用共享的 OSMMapIO 解析并生成 map_features
                try:
                    parser = OSMMapIO(self.engine.global_config["osm_file_path"], use_projection=False)
                    m_data = parser.parse()
                    route_points = self._set_route_from_points(parser.ego_route_points)
                except Exception as e:
                    logger.warning(f"OSMMapIO.parse failed: {e}. Fallback to scenario map_features.")
                    raise e
                new_map = ScenarioMapInOsm(map_index=seed, map_data=m_data)
                if self.store_map:
                    self._stored_maps[seed] = new_map
                    self._stored_route_points[seed] = copy.deepcopy(route_points)
            else:
                new_map = self._stored_maps[seed]
                route_points = self._stored_route_points.get(seed)
                self._set_route_from_points(route_points)
            self._sync_default_agent_spawn_from_route()
            self.load_map(new_map)
            self._align_default_agent_spawn_with_entry_successor()

    def filter_path(self, start_lanes, end_lanes):
        for start in start_lanes:
            for end in end_lanes:
                path = self.current_map.road_network.shortest_path(start[0].index, end[0].index)
                if len(path) > 0:
                    return (start[0].index, end[0].index)
        return None

    def spawn_object(self, object_class, *args, **kwargs):
        raise ValueError("Please create ScenarioMap instance directly without calling spawn_object function.")
        # map = self.engine.spawn_object(object_class, auto_fill_random_seed=False, *args, **kwargs)
        # self.spawned_objects[map.id] = map
        # return map

    def load_map(self, map):
        map.attach_to_world()
        self.current_map = map

    def unload_map(self, map):
        map.detach_from_world()
        self.current_map = None
        if not self.engine.global_config["store_map"]:
            map.destroy()
            assert len(self.spawned_objects) == 0

    def destroy(self):
        self.clear_stored_maps()
        self._stored_maps = None
        self._stored_route_points = None
        self.current_map = None

        self.sdc_start_point = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

        super(OSMMapManager, self).destroy()

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)

        self.sdc_start_point = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None
        self.current_map = None
        self._sync_default_agent_spawn_from_route()
        self._sync_default_agent_spawn_from_route()

    def clear_stored_maps(self):
        for m in self._stored_maps.values():
            if m is not None:
                m.detach_from_world()
                m.destroy()
        self._stored_maps = {
            i: None
            for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)
        }
        self._stored_route_points = {
            i: None
            for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)
        }

    @property
    def num_stored_maps(self):
        return sum([1 if m is not None else 0 for m in self.engine.map_manager._stored_maps.values()])
    




class OSMMapIO:
    """
    共享的 OSM / PKL 地图读写类。

    目前支持：
    - `.osm` -> `map_features`
    - `.pkl` -> `map_features`
    - `.pkl` / `map_features` -> 自定义 `.osm`

    导出的 `.osm` 会把局部坐标写入 node 的 `local_x/local_y/local_z` tag，
    这样仍然能被本类无损读回，不依赖真实经纬度投影。
    """

    LOCAL_COORD_SCALE = 100000.0
    LOCAL_EXPORT_ANCHOR_LATLON = (30.337203188525226, 121.23513662771698)
    LIST_TAG_KEYS = ("entry_lanes", "exit_lanes", "left_neighbor", "right_neighbor")
    ROUTE_TAG_KEY = "md_kind"
    ROUTE_TAG_VALUE = "ego_route"
    TRUE_VALUES = {"1", "true", "yes"}
    FALSE_VALUES = {"0", "false", "no"}
    CUSTOM_ID_TAG = "Id"
    CUSTOM_TYPE_TAG = "Type"
    CUSTOM_LANE_TYPE_TAG = "LaneType"
    CUSTOM_PREDECESSORS_TAG = "Predecessors"
    CUSTOM_SUCCESSORS_TAG = "Successors"
    CUSTOM_JUNCTION_LANES_TAG = "Lanes"
    CUSTOM_JUNCTION_CROSSWALKS_TAG = "Crosswalks"
    CUSTOM_REF_KEYS = ("Id", "feature_id", "id", "ref")

    def __init__(
        self,
        file_path: str = 'straight_300.osm',
        use_projection: bool = True,
        lane_width: float = 3.5,
    ): 
        file_path = os.fspath(file_path)
        if os.path.isabs(file_path) or os.path.exists(file_path):
            self.file_path = file_path
        else:
            root_path = os.environ.get('ROOT_PATH')
            if root_path is None:
                root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            self.file_path = os.path.join(root_path, 'metadrive/plugins/map_files', file_path)
        self.use_projection = use_projection
        self.lane_width = lane_width
        self.ego_route_points = None
        self.export_anchor_latlon = self.LOCAL_EXPORT_ANCHOR_LATLON

        if self.use_projection:
            epsg_wgs84 = 'EPSG:4326'
            epsg_utm = 'EPSG:32651'
            self.wgs84 = CRS(epsg_wgs84)
            self.utm = CRS(epsg_utm)
            self.transformer = Transformer.from_crs(self.wgs84, self.utm, always_xy=True)
            self.inverse_transformer = Transformer.from_crs(self.utm, self.wgs84, always_xy=True)

            anchor_latlon = (30.337203188525226, 121.23513662771698)
            anchor_utm = (-191.07278681232128, 38.10431034769863)
            self.anchor_latlon = anchor_latlon
            self.anchor_utm = anchor_utm
        else:
            self.wgs84 = None
            self.utm = None
            self.transformer = None
            self.inverse_transformer = None
            self.anchor_latlon = None
            self.anchor_utm = None
            self.local_anchor_latlon = None
            self.earth_radius_m = 6378137.0

    @staticmethod
    def extract_speed_and_convert(unit_str: str):
        """
        返回 (mph, kph)
        兼容:
        - '50'
        - '50 mph'
        - '80 km/h' / '80 kph'
        - 其它字符串解析失败时返回 (0.0, 0.0)
        """
        if not unit_str:
            return 0.0, 0.0
        s = unit_str.strip().lower()
        m = re.match(r'^\s*([0-9]+(?:\.[0-9]*)?)\s*([a-z/ ]+)?\s*$', s)
        if not m:
            return 0.0, 0.0
        value = float(m.group(1))
        unit = (m.group(2) or '').strip()
        if 'mph' in unit or 'mi' in unit:
            mph = value
            kph = value * 1.60934
        else:
            kph = value
            mph = value * 0.621371
        return mph, kph

    @staticmethod
    def _safe_float(value, default=None):
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _parse_bool_tag(cls, value):
        if value is None:
            return None
        lowered = value.strip().lower()
        if lowered in cls.TRUE_VALUES:
            return True
        if lowered in cls.FALSE_VALUES:
            return False
        return None

    @staticmethod
    def _stringify_feature_id(feature_id):
        return str(feature_id)

    @classmethod
    def _extract_feature_key_from_tags(cls, tags, fallback_feature_id):
        return cls._stringify_feature_id(tags.get("feature_id") or fallback_feature_id)

    @classmethod
    def _normalize_feature_reference(cls, ref):
        if ref is None:
            return None
        if isinstance(ref, dict):
            normalized = dict(ref)
            if "feature_id" in normalized and normalized["feature_id"] is not None:
                normalized["feature_id"] = cls._stringify_feature_id(normalized["feature_id"])
            if "id" in normalized and normalized["id"] is not None:
                normalized["id"] = cls._stringify_feature_id(normalized["id"])
            return normalized
        return cls._stringify_feature_id(ref)

    @classmethod
    def _normalize_map_features(cls, map_features):
        normalized_map_features = {}
        for feature_key, feature in map_features.items():
            normalized_key = cls._stringify_feature_id(feature_key)
            if not isinstance(feature, dict):
                normalized_map_features[normalized_key] = feature
                continue

            normalized_feature = dict(feature)
            for key in cls.LIST_TAG_KEYS:
                refs = normalized_feature.get(key) or []
                if not isinstance(refs, (list, tuple)):
                    refs = [refs]
                normalized_feature[key] = [
                    normalized_ref for normalized_ref in (cls._normalize_feature_reference(ref) for ref in refs)
                    if normalized_ref is not None
                ]
            normalized_map_features[normalized_key] = normalized_feature
        return normalized_map_features

    @classmethod
    def _parse_feature_refs(cls, raw_value):
        if not raw_value:
            return []
        return [item.strip() for item in raw_value.split(';') if item.strip()]

    @classmethod
    def _extract_custom_ref_value(cls, raw_item):
        if raw_item is None:
            return None
        if isinstance(raw_item, dict):
            for key in cls.CUSTOM_REF_KEYS:
                value = raw_item.get(key)
                if value not in (None, ""):
                    return cls._stringify_feature_id(value)
            return None

        normalized = cls._normalize_feature_reference(raw_item)
        if isinstance(normalized, dict):
            for key in ("feature_id", "id"):
                value = normalized.get(key)
                if value not in (None, ""):
                    return cls._stringify_feature_id(value)
            return None
        if normalized is None:
            return None
        return cls._stringify_feature_id(normalized)

    @classmethod
    def _parse_json_feature_refs(cls, raw_value, feature_key_by_custom_id=None):
        if not raw_value:
            return []

        try:
            parsed = json.loads(raw_value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return cls._parse_feature_refs(raw_value)

        if not isinstance(parsed, list):
            parsed = [parsed]

        refs = []
        for item in parsed:
            ref = cls._extract_custom_ref_value(item)
            if ref is None:
                continue
            if feature_key_by_custom_id is not None:
                ref = feature_key_by_custom_id.get(ref, ref)
            if ref not in refs:
                refs.append(ref)
        return refs

    @staticmethod
    def _extend_unique_refs(existing_refs, new_refs):
        for ref in new_refs:
            if ref not in existing_refs:
                existing_refs.append(ref)

    @staticmethod
    def _extract_xml_tags(element):
        return {
            tag.get('k'): tag.get('v')
            for tag in element.findall('./tag')
            if tag.get('k') is not None and tag.get('v') is not None
        }

    @staticmethod
    def _float_to_str(value):
        formatted = f"{float(value):.12f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted or "0"

    @staticmethod
    def _indent_xml(element, level=0):
        indent = "\n" + level * "  "
        if len(element):
            if not element.text or not element.text.strip():
                element.text = indent + "  "
            for child in element:
                OSMMapIO._indent_xml(child, level + 1)
            if not element[-1].tail or not element[-1].tail.strip():
                element[-1].tail = indent
        elif level and (not element.tail or not element.tail.strip()):
            element.tail = indent

    def wgs_to_utm(self, lat_lons, with_z: bool = True):
        """
        批量将 [(lat, lon), ...] 转为以米为单位的平面坐标。
        - use_projection=True: 使用 WGS84->UTM 投影并按锚点平移对齐。
        - use_projection=False: 使用简化球面近似，以首个点为原点做局部平面投影。
        with_z=True 时返回 (N,3)，否则返回 (N,2)
        """
        if not lat_lons:
            return np.zeros((0, 3 if with_z else 2))

        arr = np.asarray(lat_lons, dtype=float)

        if not self.use_projection:
            if self.local_anchor_latlon is None:
                self.local_anchor_latlon = (arr[0, 0], arr[0, 1])

            lat0, lon0 = self.local_anchor_latlon
            R = getattr(self, "earth_radius_m", 6378137.0)

            lat_rad = np.radians(arr[:, 0])
            lon_rad = np.radians(arr[:, 1])
            lat0_rad = math.radians(lat0)
            lon0_rad = math.radians(lon0)

            dlat = lat_rad - lat0_rad
            dlon = lon_rad - lon0_rad

            cos_lat0 = math.cos(lat0_rad)
            x = R * cos_lat0 * dlon
            y = R * dlat

            xy = np.column_stack((x, y))
            if with_z:
                z = np.zeros_like(x)
                return np.column_stack((xy, z))
            return xy

        lats = arr[:, 0]
        lons = arr[:, 1]
        x, y = self.transformer.transform(lons, lats)

        anchor_lat, anchor_lon = self.anchor_latlon
        ax, ay = self.transformer.transform(anchor_lon, anchor_lat)
        bias_x = ax - self.anchor_utm[0]
        bias_y = ay - self.anchor_utm[1]

        x_adj = x - bias_x
        y_adj = y - bias_y
        xy = np.column_stack((x_adj, y_adj))
        if with_z:
            z = np.zeros_like(x_adj)
            return np.column_stack((xy, z))
        return xy

    @staticmethod
    def poly_get(points, with_z: bool = True):
        """
        保持输入坐标并按需补 z=0（不做投影转换）
        """
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return np.zeros((0, 3 if with_z else 2))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D polyline array, got shape {arr.shape}")

        if arr.shape[1] >= 3:
            return arr[:, :3] if with_z else arr[:, :2]
        if arr.shape[1] == 2:
            if with_z:
                zeros_z = np.zeros((arr.shape[0],), dtype=float)
                return np.column_stack((arr, zeros_z))
            return arr
        raise ValueError(f"Unsupported point dimension: {arr.shape[1]}")

    def _coords_to_polyline(self, coords, is_local: bool, with_z: bool = True):
        if is_local:
            return self.poly_get(coords, with_z=with_z)
        return self.wgs_to_utm(coords, with_z=with_z)

    def _local_xy_to_wgs84(self, x: float, y: float):
        if self.use_projection and self.inverse_transformer is not None and self.anchor_latlon and self.anchor_utm:
            anchor_lat, anchor_lon = self.anchor_latlon
            ax, ay = self.transformer.transform(anchor_lon, anchor_lat)
            bias_x = ax - self.anchor_utm[0]
            bias_y = ay - self.anchor_utm[1]
            lon, lat = self.inverse_transformer.transform(x + bias_x, y + bias_y)
            return lat, lon

        anchor_lat, anchor_lon = self.export_anchor_latlon
        lat0_rad = math.radians(anchor_lat)
        radius = getattr(self, "earth_radius_m", 6378137.0)
        lat = anchor_lat + math.degrees(y / radius)
        lon = anchor_lon + math.degrees(x / (radius * math.cos(lat0_rad)))
        return lat, lon

    @classmethod
    def _extract_speed_from_tags(cls, tags):
        mph = cls._safe_float(tags.get("speed_limit_mph"))
        kph = cls._safe_float(tags.get("speed_limit_kph"))
        if kph is None:
            kph = cls._safe_float(tags.get("speed_limit_kmh"))

        if mph is not None or kph is not None:
            if mph is None and kph is not None:
                mph = kph * 0.621371
            if kph is None and mph is not None:
                kph = mph * 1.60934
            return mph or 0.0, kph or 0.0

        for key in ('SpeedLimit', 'speed_limit', 'maxspeed'):
            if key in tags:
                return cls.extract_speed_and_convert(tags[key])
        return 0.0, 0.0

    @staticmethod
    def _infer_road_type(tags):
        road_type = None

        if tags.get("highway") is not None:
            road_type = MetaDriveType.LANE_SURFACE_STREET

        subtype = tags.get("subtype")
        if subtype == 'curbstone':
            road_type = MetaDriveType.BOUNDARY_LINE
        elif subtype == 'dashed':
            road_type = MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        elif subtype == 'solid':
            road_type = MetaDriveType.BOUNDARY_LINE
        elif subtype == 'centerlane':
            road_type = MetaDriveType.LANE_SURFACE_STREET

        type_tag = tags.get("type")
        if type_tag == 'stop_line':
            road_type = "STOP_LINE"
        elif type_tag == 'traffic_light':
            road_type = MetaDriveType.TRAFFIC_LIGHT
        elif (
            MetaDriveType.is_lane(type_tag)
            or MetaDriveType.is_road_line(type_tag)
            or MetaDriveType.is_road_boundary_line(type_tag)
            or type_tag in {"STOP_LINE", MetaDriveType.TRAFFIC_LIGHT}
        ):
            road_type = type_tag

        if road_type is None and tags.get("road_edge") in {"1", "true", "yes"}:
            road_type = MetaDriveType.BOUNDARY_LINE

        if road_type is None:
            custom_type = tags.get(OSMMapIO.CUSTOM_TYPE_TAG)
            lane_type = str(tags.get(OSMMapIO.CUSTOM_LANE_TYPE_TAG) or "").strip().casefold()
            if custom_type == "Lane":
                if lane_type == "biking":
                    road_type = MetaDriveType.LANE_BIKE_LANE
                elif lane_type == "shoulder":
                    road_type = MetaDriveType.LANE_SURFACE_UNSTRUCTURE
                elif lane_type == "none":
                    road_type = MetaDriveType.LANE_UNKNOWN
                else:
                    road_type = MetaDriveType.LANE_SURFACE_STREET
            elif custom_type == "LaneBoundary":
                road_type = MetaDriveType.BOUNDARY_LINE

        return road_type

    def _parse_node(self, node):
        node_id = node.attrib['id']
        tags = self._extract_xml_tags(node)
        if 'local_x' in tags and 'local_y' in tags:
            coords = [
                float(tags['local_x']),
                float(tags['local_y']),
            ]
            if 'local_z' in tags:
                coords.append(float(tags['local_z']))
            return node_id, {"coords": tuple(coords), "is_local": True}

        lat = float(node.attrib['lat'])
        lon = float(node.attrib['lon'])
        return node_id, {"coords": (lat, lon), "is_local": False}

    def _build_feature_from_way(self, way, way_coords, is_local, tags=None):
        tags = self._extract_xml_tags(way) if tags is None else tags
        feature_key = self._extract_feature_key_from_tags(tags, way.attrib['id'])
        speed_limit_mph, speed_limit_kph = self._extract_speed_from_tags(tags)
        feature = {
            "speed_limit_mph": speed_limit_mph,
            "speed_limit_kph": speed_limit_kph,
            "type": self._infer_road_type(tags),
            "polyline": self._coords_to_polyline(way_coords, is_local=is_local),
        }

        for key in self.LIST_TAG_KEYS:
            feature[key] = self._parse_feature_refs(tags.get(key))

        if tags.get("turn_direction"):
            feature["turn_direction"] = tags["turn_direction"]

        junction = self._parse_bool_tag(tags.get("junction"))
        if junction is not None:
            feature["junction"] = junction

        stop_line = self._parse_bool_tag(tags.get("stop_line"))
        if stop_line is not None:
            feature["stop_line"] = stop_line

        return feature_key, feature

    @classmethod
    def _apply_custom_way_tag_semantics(cls, map_features, way_tags_by_feature_key, feature_key_by_custom_id):
        junction_lane_refs = []
        crosswalk_refs = []

        for feature_key, tags in way_tags_by_feature_key.items():
            feature = map_features.get(feature_key)
            if not isinstance(feature, dict):
                continue

            custom_type = tags.get(cls.CUSTOM_TYPE_TAG)
            if custom_type == "Lane":
                entry_lanes = cls._parse_json_feature_refs(
                    tags.get(cls.CUSTOM_PREDECESSORS_TAG), feature_key_by_custom_id
                )
                exit_lanes = cls._parse_json_feature_refs(
                    tags.get(cls.CUSTOM_SUCCESSORS_TAG), feature_key_by_custom_id
                )

                if entry_lanes:
                    feature.setdefault("entry_lanes", [])
                    cls._extend_unique_refs(
                        feature["entry_lanes"], [lane_ref for lane_ref in entry_lanes if lane_ref != feature_key]
                    )
                if exit_lanes:
                    feature.setdefault("exit_lanes", [])
                    cls._extend_unique_refs(
                        feature["exit_lanes"], [lane_ref for lane_ref in exit_lanes if lane_ref != feature_key]
                    )
            elif custom_type == "Junction":
                cls._extend_unique_refs(
                    junction_lane_refs,
                    cls._parse_json_feature_refs(tags.get(cls.CUSTOM_JUNCTION_LANES_TAG), feature_key_by_custom_id),
                )
                cls._extend_unique_refs(
                    crosswalk_refs,
                    cls._parse_json_feature_refs(
                        tags.get(cls.CUSTOM_JUNCTION_CROSSWALKS_TAG), feature_key_by_custom_id
                    ),
                )

        for lane_ref in junction_lane_refs:
            feature = map_features.get(lane_ref)
            if isinstance(feature, dict):
                feature["junction"] = True

        for crosswalk_ref in crosswalk_refs:
            feature = map_features.get(crosswalk_ref)
            if isinstance(feature, dict) and not feature.get("type"):
                feature["type"] = MetaDriveType.CROSSWALK

    def parse(self):
        """
        解析 `file_path`，返回 `map_features`
        """
        if not self.file_path:
            raise ValueError("file_path is required")

        self.ego_route_points = None

        suffix = os.path.splitext(self.file_path)[1].lower()
        if suffix == '.osm':
            return self._parse_osm()
        if suffix == '.pkl':
            return self._parse_pkl()
        raise ValueError(f"Unsupported map file suffix: {suffix}")

    def _parse_osm(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        nodes = {}
        for node in root.findall('./node'):
            node_id, node_info = self._parse_node(node)
            nodes[node_id] = node_info

        ways = {}
        ways_id = {}
        way_coord_modes = {}
        way_feature_keys = {}
        way_tags_by_feature_key = {}
        feature_key_by_custom_id = {}
        map_features = {}

        for way in root.findall('./way'):
            way_id = way.attrib['id']
            way_nodes = []
            way_nodes_id = []
            for nd in way.findall('./nd'):
                ref = nd.attrib['ref']
                if ref in nodes:
                    way_nodes.append(nodes[ref])
                    way_nodes_id.append(ref)
            if not way_nodes:
                continue

            coord_modes = {node["is_local"] for node in way_nodes}
            if len(coord_modes) != 1:
                raise ValueError(f"Way {way_id} mixes local and geographic coordinates")

            is_local = coord_modes.pop()
            way_coords = [node["coords"] for node in way_nodes]
            way_tags = self._extract_xml_tags(way)
            if way_tags.get(self.ROUTE_TAG_KEY) == self.ROUTE_TAG_VALUE:
                self.ego_route_points = self._coords_to_polyline(way_coords, is_local=is_local, with_z=False)
                continue

            feature_key, feature = self._build_feature_from_way(way, way_coords, is_local=is_local, tags=way_tags)

            ways[way_id] = way_coords
            ways_id[way_id] = way_nodes_id
            way_coord_modes[way_id] = is_local
            way_feature_keys[way_id] = feature_key
            way_tags_by_feature_key[feature_key] = way_tags
            custom_feature_id = way_tags.get(self.CUSTOM_ID_TAG)
            if custom_feature_id:
                feature_key_by_custom_id[self._stringify_feature_id(custom_feature_id)] = feature_key
            map_features[feature_key] = feature

        map_features_relation = self._parse_lanelet_relations(
            root=root,
            ways=ways,
            ways_id=ways_id,
            way_coord_modes=way_coord_modes,
            way_feature_keys=way_feature_keys,
        )

        for key, value in map_features_relation.items():
            if key in map_features and isinstance(map_features[key], dict) and isinstance(value, dict):
                merged = dict(map_features[key])
                merged.update(value)
                map_features[key] = merged
            else:
                map_features[key] = value

        self._apply_custom_way_tag_semantics(map_features, way_tags_by_feature_key, feature_key_by_custom_id)
        return map_features

    def _parse_lanelet_relations(self, root, ways, ways_id, way_coord_modes, way_feature_keys):
        map_features_relation = {}

        for relation in root.findall('./relation'):
            if relation.get('action') == 'delete':
                continue

            relation_tags = self._extract_xml_tags(relation)
            if relation_tags.get("type") != 'lanelet':
                continue

            is_junction = relation_tags.get("junction") == 'yes'

            left_boundary_nodes = []
            left_boundary_nodes_id = []
            left_boundary_way_id = []
            right_boundary_nodes = []
            right_boundary_nodes_id = []
            right_boundary_way_id = []
            center_line_nodes = []
            center_line_nodes_id = []
            center_role = None
            stop_line = False
            relation_coord_modes = set()

            for member in relation.findall('./member'):
                member_type = member.get('type')
                member_ref = member.get('ref')
                member_role = member.get('role')

                if member_type == 'way' and member_ref in ways:
                    relation_coord_modes.add(way_coord_modes[member_ref])
                    way_nodes = ways[member_ref]
                    way_nodes_ids = ways_id[member_ref]
                    feature_ref = way_feature_keys.get(member_ref, member_ref)

                    if member_role == 'left':
                        left_boundary_nodes.extend(way_nodes)
                        left_boundary_nodes_id.extend(way_nodes_ids)
                        left_boundary_way_id.append(feature_ref)
                    elif member_role == 'right':
                        right_boundary_nodes.extend(way_nodes)
                        right_boundary_nodes_id.extend(way_nodes_ids)
                        right_boundary_way_id.append(feature_ref)
                    elif member_role == 'center':
                        center_role = feature_ref
                        center_line_nodes.extend(way_nodes)
                        center_line_nodes_id.extend(way_nodes_ids)
                elif member_type == 'relation' and member_role == 'regulatory_element':
                    stop_line = True

            if not center_line_nodes or center_role is None:
                continue

            if len(relation_coord_modes) > 1:
                raise ValueError(f"Lanelet relation {relation.attrib.get('id')} mixes coordinate systems")
            relation_uses_local = next(iter(relation_coord_modes), False)

            polygon_nodes = left_boundary_nodes + right_boundary_nodes[::-1]
            seen = set()
            unique_nodes = []
            for node in polygon_nodes:
                if node not in seen:
                    seen.add(node)
                    unique_nodes.append(node)

            polygon = self._coords_to_polyline(unique_nodes, is_local=relation_uses_local, with_z=False)

            turn_direction = relation_tags.get("turn_direction", 'straight')
            map_features_relation[center_role] = {
                "polygon": polygon,
                "turn_direction": turn_direction,
                "center_line_nodes": center_line_nodes,
                "center_line_nodes_id": center_line_nodes_id,
                "left_boundary_nodes": left_boundary_nodes,
                "left_boundary_nodes_id": left_boundary_nodes_id,
                "left_boundary_way_id": left_boundary_way_id,
                "right_boundary_nodes": right_boundary_nodes,
                "right_boundary_nodes_id": right_boundary_nodes_id,
                "right_boundary_way_id": right_boundary_way_id,
                "stop_line": stop_line,
                "junction": is_junction,
                "exit_lanes": [],
                "entry_lanes": [],
                "left_neighbor": [],
                "right_neighbor": []
            }

        relation_ids = list(map_features_relation.keys())
        for rid1 in relation_ids:
            for rid2 in relation_ids:
                if rid1 == rid2:
                    continue
                n1_last = map_features_relation[rid1]["center_line_nodes"][-1]
                n2_first = map_features_relation[rid2]["center_line_nodes"][0]
                if n1_last == n2_first:
                    map_features_relation[rid1]["exit_lanes"].append(rid2)
                    map_features_relation[rid2]["entry_lanes"].append(rid1)

        for rid in relation_ids:
            curr_feat = map_features_relation[rid]

            curr_l_ways = set(curr_feat['left_boundary_way_id'])
            curr_r_ways = set(curr_feat['right_boundary_way_id'])
            curr_l_nodes = set(curr_feat['left_boundary_nodes_id'])
            curr_r_nodes = set(curr_feat['right_boundary_nodes_id'])

            for other in relation_ids:
                if other == rid:
                    continue

                other_feat = map_features_relation[other]

                other_l_ways = set(other_feat['left_boundary_way_id'])
                other_r_ways = set(other_feat['right_boundary_way_id'])
                other_l_nodes = set(other_feat['left_boundary_nodes_id'])
                other_r_nodes = set(other_feat['right_boundary_nodes_id'])

                is_left_neighbor = False
                if curr_l_ways & other_r_ways:
                    is_left_neighbor = True
                elif len(curr_l_nodes & other_r_nodes) >= 2:
                    is_left_neighbor = True

                if is_left_neighbor:
                    if other not in curr_feat['left_neighbor']:
                        curr_feat['left_neighbor'].append(other)
                    if rid not in other_feat['right_neighbor']:
                        other_feat['right_neighbor'].append(rid)

                is_right_neighbor = False
                if curr_r_ways & other_l_ways:
                    is_right_neighbor = True
                elif len(curr_r_nodes & other_l_nodes) >= 2:
                    is_right_neighbor = True

                if is_right_neighbor:
                    if other not in curr_feat['right_neighbor']:
                        curr_feat['right_neighbor'].append(other)
                    if rid not in other_feat['left_neighbor']:
                        other_feat['left_neighbor'].append(rid)

        for rid in relation_ids:
            left_neighbors = map_features_relation[rid]['left_neighbor']
            right_neighbors = map_features_relation[rid]['right_neighbor']
            map_features_relation[rid]['left_neighbor'] = list(dict.fromkeys(left_neighbors))
            map_features_relation[rid]['right_neighbor'] = list(dict.fromkeys(right_neighbors))

        return map_features_relation

    def _parse_pkl(self):
        obj_info = self._load_pickle_object()
        self.ego_route_points = self.extract_route_points_from_pickle_object(obj_info)
        return self.map_features_from_pickle_object(obj_info)

    def _load_pickle_object(self):
        with open(self.file_path, 'rb') as handle:
            return pkl.load(handle)

    @classmethod
    def _normalize_route_points(cls, route_like):
        if route_like is None:
            return None

        if isinstance(route_like, dict):
            for key in ("polyline", "points", "waypoints", "position", "positions", "route"):
                if key in route_like:
                    route = cls._normalize_route_points(route_like[key])
                    if route is not None:
                        return route
            state = route_like.get("state")
            if isinstance(state, dict) and "position" in state:
                return cls._normalize_route_points(state["position"])
            return None

        if isinstance(route_like, (list, tuple)) and route_like and isinstance(route_like[0], dict):
            points = []
            for item in route_like:
                if "position" in item:
                    points.append(item["position"])
                elif "x" in item and "y" in item:
                    points.append([item["x"], item["y"]])
            if not points:
                return None
            route_like = points

        try:
            arr = np.asarray(route_like, dtype=float)
        except (TypeError, ValueError):
            return None

        if arr.ndim != 2:
            return None

        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T

        if arr.shape[1] >= 2:
            return arr[:, :2]

        return None

    @classmethod
    def extract_route_points_from_pickle_object(cls, obj_info):
        if not isinstance(obj_info, dict):
            return None

        for key in ("route", "ego_route", "global_route"):
            if key in obj_info:
                route = cls._normalize_route_points(obj_info[key])
                if route is not None and len(route) >= 2:
                    return route
        return None

    def get_ego_route(self):
        if self.ego_route_points is None or len(self.ego_route_points) < 2:
            return None
        return get_idm_route(np.asarray(self.ego_route_points)[..., :2])

    @staticmethod
    def _empty_lane_relations():
        return {
            "entry_lanes": [],
            "exit_lanes": [],
            "left_neighbor": [],
            "right_neighbor": [],
        }

    @classmethod
    def _normalize_road_connection_types(cls, road_connection_types, lane_count):
        try:
            relation_array = np.asarray(road_connection_types, dtype=float)
        except (TypeError, ValueError):
            return None

        if relation_array.size == 0:
            return None

        if relation_array.ndim == 1:
            if relation_array.size % 6 != 0:
                return None
            pair_count = relation_array.size // 6
            side = int(round(math.sqrt(pair_count)))
            if side * side != pair_count:
                return None
            relation_array = relation_array.reshape(side, side, 6)
        elif relation_array.ndim == 2:
            if relation_array.shape[1] == 6:
                pair_count = relation_array.shape[0]
                side = int(round(math.sqrt(pair_count)))
                if side * side != pair_count:
                    return None
                relation_array = relation_array.reshape(side, side, 6)
            elif relation_array.shape[0] == 6:
                pair_count = relation_array.shape[1]
                side = int(round(math.sqrt(pair_count)))
                if side * side != pair_count:
                    return None
                relation_array = relation_array.T.reshape(side, side, 6)
            else:
                return None
        elif relation_array.ndim == 3:
            if relation_array.shape[-1] == 6:
                pass
            elif relation_array.shape[0] == 6:
                relation_array = np.moveaxis(relation_array, 0, -1)
            else:
                return None
        else:
            return None

        if relation_array.shape[0] != relation_array.shape[1] or relation_array.shape[-1] != 6:
            return None

        valid_lane_count = min(lane_count, relation_array.shape[0], relation_array.shape[1])
        if valid_lane_count <= 0:
            return None

        if relation_array.shape[0] != lane_count:
            logger.warning(
                "road_connection_types lane count %s does not match road_points lane count %s. Use first %s lanes.",
                relation_array.shape[0], lane_count, valid_lane_count
            )

        return relation_array[:valid_lane_count, :valid_lane_count, :]

    @classmethod
    def _lane_relations_from_pickle_object(cls, obj_info, lane_ids):
        lane_ids = [cls._stringify_feature_id(lane_id) for lane_id in lane_ids]
        lane_relations = {lane_id: cls._empty_lane_relations() for lane_id in lane_ids}

        if not isinstance(obj_info, dict) or not lane_ids:
            return lane_relations

        relation_array = cls._normalize_road_connection_types(obj_info.get("road_connection_types"), len(lane_ids))
        if relation_array is None:
            return lane_relations

        def append_unique(lane_id, relation_key, neighbor_lane_id):
            lane_id = cls._stringify_feature_id(lane_id)
            neighbor_lane_id = cls._stringify_feature_id(neighbor_lane_id)
            relation_list = lane_relations[lane_id][relation_key]
            if neighbor_lane_id not in relation_list:
                relation_list.append(neighbor_lane_id)

        valid_lane_count = min(len(lane_ids), relation_array.shape[0])
        for lane_idx in range(valid_lane_count):
            src_lane_id = lane_ids[lane_idx]
            for other_idx in range(valid_lane_count):
                if lane_idx == other_idx:
                    continue

                relation_code = int(np.argmax(relation_array[lane_idx, other_idx]))
                dst_lane_id = lane_ids[other_idx]

                if relation_code == 1:
                    append_unique(src_lane_id, "entry_lanes", dst_lane_id)
                    append_unique(dst_lane_id, "exit_lanes", src_lane_id)
                elif relation_code == 2:
                    append_unique(src_lane_id, "exit_lanes", dst_lane_id)
                    append_unique(dst_lane_id, "entry_lanes", src_lane_id)
                elif relation_code == 3:
                    append_unique(src_lane_id, "left_neighbor", dst_lane_id)
                    append_unique(dst_lane_id, "right_neighbor", src_lane_id)
                elif relation_code == 4:
                    append_unique(src_lane_id, "right_neighbor", dst_lane_id)
                    append_unique(dst_lane_id, "left_neighbor", src_lane_id)

        return lane_relations

    @classmethod
    def _infer_lane_keys_from_map_features(cls, map_features):
        lane_key_pairs = []
        for feature_key, feature in map_features.items():
            if not isinstance(feature, dict):
                continue

            feature_type = str(feature.get("type", ""))
            if not feature_type.startswith("LANE_"):
                continue

            try:
                numeric_key = int(feature_key)
            except (TypeError, ValueError):
                return None
            lane_key_pairs.append((numeric_key, feature_key))

        if not lane_key_pairs:
            return None

        lane_key_pairs.sort(key=lambda item: item[0])
        numeric_keys = [item[0] for item in lane_key_pairs]
        if numeric_keys != list(range(len(numeric_keys))):
            return None

        return [item[1] for item in lane_key_pairs]

    @classmethod
    def _merge_pickle_lane_relations_into_map_features(cls, obj_info, map_features):
        lane_keys = cls._infer_lane_keys_from_map_features(map_features)
        if lane_keys is None:
            return map_features

        lane_relations = cls._lane_relations_from_pickle_object(obj_info, lane_keys)
        for lane_key in lane_keys:
            map_features[lane_key].update(lane_relations[lane_key])
        return map_features

    def map_features_from_pickle_object(self, obj_info):
        if 'map_features' in obj_info:
            map_features = obj_info['map_features']
            if isinstance(map_features, dict) and obj_info.get("road_connection_types") is not None:
                map_features = self._normalize_map_features(copy.deepcopy(map_features))
                map_features = self._merge_pickle_lane_relations_into_map_features(obj_info, map_features)
            elif isinstance(map_features, dict):
                map_features = self._normalize_map_features(copy.deepcopy(map_features))
            return map_features

        if 'road_points' not in obj_info:
            raise ValueError("Pickle file does not contain map_features or road_points")

        map_features = {}
        lane_ids = [self._stringify_feature_id(i) for i in range(len(obj_info['road_points']))]
        lane_relations = self._lane_relations_from_pickle_object(obj_info, lane_ids)

        for way_id, raw_centerline in enumerate(obj_info['road_points']):
            lane_id = lane_ids[way_id]
            centerline = self.poly_get(raw_centerline)
            map_features[lane_id] = {
                "speed_limit_mph": 40,
                "speed_limit_kph": 40,
                "type": 'LANE_SURFACE_STREET',
                "polyline": centerline,
                **lane_relations.get(lane_id, self._empty_lane_relations()),
            }

            left_boundary, right_boundary = self.generate_lane_boundaries(centerline, lane_width=self.lane_width)
            map_features[f'{lane_id}_left'] = {
                "speed_limit_mph": 40,
                "speed_limit_kph": 40,
                "type": 'ROAD_EDGE_BOUNDARY',
                "polyline": left_boundary,
            }
            map_features[f'{lane_id}_right'] = {
                "speed_limit_mph": 40,
                "speed_limit_kph": 40,
                "type": 'ROAD_EDGE_BOUNDARY',
                "polyline": right_boundary,
            }

        return map_features

    def export_pickle_to_osm(self, output_osm_path: str, overwrite: bool = False):
        if os.path.splitext(self.file_path)[1].lower() != '.pkl':
            raise ValueError("export_pickle_to_osm only supports `.pkl` input")
        map_features = self._parse_pkl()
        return self.export_map_features_to_osm(
            map_features,
            output_osm_path,
            overwrite=overwrite,
            ego_route_points=self.ego_route_points,
        )

    @classmethod
    def convert_pickle_to_osm(cls, input_pkl_path: str, output_osm_path: str, overwrite: bool = False, lane_width: float = 3.5):
        converter = cls(file_path=input_pkl_path, use_projection=False, lane_width=lane_width)
        return converter.export_pickle_to_osm(output_osm_path=output_osm_path, overwrite=overwrite)

    def export_map_features_to_osm(self, map_features, output_osm_path: str, overwrite: bool = False, ego_route_points=None):
        output_osm_path = os.path.abspath(output_osm_path)
        if os.path.exists(output_osm_path) and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_osm_path}")

        os.makedirs(os.path.dirname(output_osm_path), exist_ok=True)

        root = ET.Element("osm", attrib={"version": "0.6", "generator": "MetaDrive OSMMapIO"})
        next_node_id = -1
        next_way_id = -1
        exported_count = 0

        for feature_key, feature in map_features.items():
            if not isinstance(feature, dict) or "polyline" not in feature:
                continue

            polyline = self.poly_get(feature["polyline"])
            if len(polyline) < 2:
                continue

            node_refs = []
            for point in polyline:
                x = float(point[0])
                y = float(point[1])
                z = float(point[2]) if len(point) > 2 else 0.0
                lat, lon = self._local_xy_to_wgs84(x, y)

                node = ET.SubElement(
                    root,
                    "node",
                    attrib={
                        "id": str(next_node_id),
                        "visible": "true",
                        "lat": self._float_to_str(lat),
                        "lon": self._float_to_str(lon),
                    }
                )
                ET.SubElement(node, "tag", attrib={"k": "local_x", "v": self._float_to_str(x)})
                ET.SubElement(node, "tag", attrib={"k": "local_y", "v": self._float_to_str(y)})
                ET.SubElement(node, "tag", attrib={"k": "local_z", "v": self._float_to_str(z)})
                node_refs.append(str(next_node_id))
                next_node_id -= 1

            way = ET.SubElement(root, "way", attrib={"id": str(next_way_id), "visible": "true"})
            for ref in node_refs:
                ET.SubElement(way, "nd", attrib={"ref": ref})

            tags = {"feature_id": self._stringify_feature_id(feature_key)}
            if feature.get("type"):
                tags["type"] = str(feature["type"])
            if feature.get("speed_limit_mph") is not None:
                tags["speed_limit_mph"] = self._float_to_str(feature["speed_limit_mph"])
            if feature.get("speed_limit_kph") is not None:
                tags["speed_limit_kph"] = self._float_to_str(feature["speed_limit_kph"])
            if feature.get("turn_direction"):
                tags["turn_direction"] = str(feature["turn_direction"])
            if feature.get("junction") is not None:
                tags["junction"] = "yes" if feature["junction"] else "no"
            if feature.get("stop_line") is not None:
                tags["stop_line"] = "yes" if feature["stop_line"] else "no"

            for key in self.LIST_TAG_KEYS:
                values = feature.get(key)
                if values:
                    tags[key] = ";".join(self._stringify_feature_id(value) for value in values)

            for key, value in tags.items():
                ET.SubElement(way, "tag", attrib={"k": key, "v": value})

            next_way_id -= 1
            exported_count += 1

        route_points = self._normalize_route_points(ego_route_points)
        if route_points is not None and len(route_points) >= 2:
            node_refs = []
            route_polyline = self.poly_get(route_points)
            for point in route_polyline:
                x = float(point[0])
                y = float(point[1])
                z = float(point[2]) if len(point) > 2 else 0.0
                lat, lon = self._local_xy_to_wgs84(x, y)

                node = ET.SubElement(
                    root,
                    "node",
                    attrib={
                        "id": str(next_node_id),
                        "visible": "true",
                        "lat": self._float_to_str(lat),
                        "lon": self._float_to_str(lon),
                    }
                )
                ET.SubElement(node, "tag", attrib={"k": "local_x", "v": self._float_to_str(x)})
                ET.SubElement(node, "tag", attrib={"k": "local_y", "v": self._float_to_str(y)})
                ET.SubElement(node, "tag", attrib={"k": "local_z", "v": self._float_to_str(z)})
                node_refs.append(str(next_node_id))
                next_node_id -= 1

            way = ET.SubElement(root, "way", attrib={"id": str(next_way_id), "visible": "true"})
            for ref in node_refs:
                ET.SubElement(way, "nd", attrib={"ref": ref})
            ET.SubElement(way, "tag", attrib={"k": self.ROUTE_TAG_KEY, "v": self.ROUTE_TAG_VALUE})
            ET.SubElement(way, "tag", attrib={"k": "type", "v": "route"})
            next_way_id -= 1

        self._indent_xml(root)
        tree = ET.ElementTree(root)
        tree.write(output_osm_path, encoding="utf-8", xml_declaration=True)
        return exported_count

    def generate_lane_boundaries(self, centerline, lane_width=None):
        """
        根据中心线生成左右边界线
        """
        lane_width = self.lane_width if lane_width is None else lane_width
        centerline = np.asarray(centerline, dtype=float)
        centerline_xy = self.poly_get(centerline, with_z=False)
        n_points = len(centerline_xy)

        if n_points < 2:
            boundary = self.poly_get(centerline)
            return boundary, boundary

        half_width = lane_width / 2
        left_boundary = np.zeros_like(centerline_xy)
        right_boundary = np.zeros_like(centerline_xy)

        for i in range(n_points):
            if i == 0:
                direction = centerline_xy[1] - centerline_xy[0]
            elif i == n_points - 1:
                direction = centerline_xy[-1] - centerline_xy[-2]
            else:
                dir1 = centerline_xy[i] - centerline_xy[i - 1]
                dir2 = centerline_xy[i + 1] - centerline_xy[i]
                direction = (dir1 + dir2) / 2

            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0])

            normal = np.array([-direction[1], direction[0]])
            left_boundary[i] = centerline_xy[i] + normal * half_width
            right_boundary[i] = centerline_xy[i] - normal * half_width

        if centerline.ndim == 2 and centerline.shape[1] >= 3:
            z = centerline[:, 2]
            left_boundary = np.column_stack((left_boundary, z))
            right_boundary = np.column_stack((right_boundary, z))
            return left_boundary, right_boundary

        return self.poly_get(left_boundary), self.poly_get(right_boundary)


OSMMapParser = OSMMapIO
