import argparse
import csv
import math
import os
import pickle as pkl
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_reproduction.GNSS_info_process import GPS_to_Global
from scenario_reproduction.GNSS_Transform import GNSS_Transform


try:
    from metadrive.scenario.scenario_description import ScenarioDescription as MetaDriveScenarioDescription
except ModuleNotFoundError:
    MetaDriveScenarioDescription = None


class ScenarioKeys:
    TRACKS = "tracks"
    VERSION = "version"
    ID = "id"
    DYNAMIC_MAP_STATES = "dynamic_map_states"
    MAP_FEATURES = "map_features"
    LENGTH = "length"
    METADATA = "metadata"

    class SUMMARY:
        OBJECT_SUMMARY = "object_summary"
        NUMBER_SUMMARY = "number_summary"
        TYPE = "type"
        OBJECT_ID = "object_id"
        TRACK_LENGTH = "track_length"
        MOVING_DIST = "moving_distance"
        VALID_LENGTH = "valid_length"
        CONTINUOUS_VALID_LENGTH = "continuous_valid_length"
        OBJECT_TYPES = "object_types"
        NUM_OBJECTS = "num_objects"
        NUM_MOVING_OBJECTS = "num_moving_objects"
        NUM_OBJECTS_EACH_TYPE = "num_objects_each_type"
        NUM_MOVING_OBJECTS_EACH_TYPE = "num_moving_objects_each_type"
        NUM_TRAFFIC_LIGHTS = "num_traffic_lights"
        NUM_TRAFFIC_LIGHT_TYPES = "num_traffic_light_types"
        NUM_TRAFFIC_LIGHTS_EACH_STEP = "num_traffic_light_each_step"
        NUM_MAP_FEATURES = "num_map_features"
        MAP_HEIGHT_DIFF = "map_height_diff"


SD = MetaDriveScenarioDescription or ScenarioKeys


class ObjectInfoProcessor:
    """Convert the object CSV exported from bag data into scenario object arrays."""

    FIELD_PATTERN = re.compile(r"field\.(vehicle|walker)_info_batch(\d+)\.actor_pos\.x")
    OBJECT_FIELDS = {
        "position": ("actor_pos.x", "actor_pos.y", "actor_pos.z"),
        "velocity": ("actor_vel.x", "actor_vel.y", "actor_vel.z"),
        "psi": ("actor_psi",),
        "width": ("actor_width",),
        "length": ("actor_length",),
        "valid": ("valid",),
    }

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

    def run(self, output_path=None):
        obj_info = self.parse()
        if output_path is not None:
            self.dump(obj_info, output_path)
        return obj_info

    def parse(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"object csv not found: {self.csv_path}")

        with self.csv_path.open("r", newline="") as file:
            reader = csv.reader(file)
            headers = next(reader)
            header_index = {name: index for index, name in enumerate(headers)}
            obj_data = self._init_objects(headers)

            for row in reader:
                self._append_row(obj_data["vehicles"], "vehicle", row, header_index)
                self._append_row(obj_data["walkers"], "walker", row, header_index)

        obj_data["vehicles"] = self._finalize(obj_data["vehicles"])
        obj_data["walkers"] = self._finalize(obj_data["walkers"])
        return obj_data

    @staticmethod
    def dump(obj_info, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pkl.dump(obj_info, f)

    @classmethod
    def _init_objects(cls, headers):
        obj_data = {"vehicles": {}, "walkers": {}}
        for header in headers:
            match = cls.FIELD_PATTERN.match(header)
            if not match:
                continue
            actor_type, actor_id = match.group(1), int(match.group(2))
            bucket = obj_data["vehicles"] if actor_type == "vehicle" else obj_data["walkers"]
            bucket[actor_id] = {
                "position": [],
                "velocity": [],
                "psi": [],
                "width": [],
                "length": [],
                "valid": [],
            }
        return obj_data

    def _append_row(self, objects, actor_type, row, header_index):
        for actor_id, info in objects.items():
            prefix = f"field.{actor_type}_info_batch{actor_id}."
            position_x = self._get_float(row, header_index, prefix + "actor_pos.x")
            position_y = self._get_float(row, header_index, prefix + "actor_pos.y")
            velocity_x = self._get_float(row, header_index, prefix + "actor_vel.x")
            velocity_y = self._get_float(row, header_index, prefix + "actor_vel.y")
            velocity_z = self._get_float(row, header_index, prefix + "actor_vel.z")
            psi = self._get_float(row, header_index, prefix + "actor_psi")
            width = self._get_float(row, header_index, prefix + "actor_width")
            length = self._get_float(row, header_index, prefix + "actor_length")
            valid = (
                np.isfinite(position_x)
                and np.isfinite(position_y)
                and np.isfinite(psi)
                and np.isfinite(width)
                and np.isfinite(length)
                and width > 0
                and length > 0
            )

            info["position"].append((position_x, -position_y, 0.0))
            info["velocity"].append((velocity_x, -velocity_y, velocity_z))
            angle = 2 * np.pi - psi if np.isfinite(psi) else 0.0
            info["psi"].append((angle + np.pi) % (2 * np.pi) - np.pi)
            info["width"].append(width)
            info["length"].append(length)
            info["valid"].append(valid)

    @staticmethod
    def _get_float(row, header_index, header):
        try:
            value = row[header_index[header]]
        except KeyError as exc:
            raise KeyError(f"missing required csv column: {header}") from exc
        value = value.strip()
        if value == "":
            return 0.0
        try:
            return float(value)
        except ValueError:
            return np.nan

    @staticmethod
    def _finalize(objects):
        result = {}
        for actor_id, info in objects.items():
            valid = np.asarray(info["valid"], dtype=bool)
            if not np.any(valid):
                continue
            finalized = {}
            for key, value in info.items():
                array = np.asarray(value)
                if key == "valid":
                    finalized[key] = valid
                    continue
                if np.issubdtype(array.dtype, np.number):
                    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
                    array[~valid] = 0
                finalized[key] = array
            result[actor_id] = finalized
        return result


class OSMMapProcessor:
    """Convert OSM ways and lanelet relations into MetaDrive map features."""

    SPEED_PATTERN = re.compile(r"(\d+(\.\d+)?)([^\d]+)?")

    def __init__(self, osm_path, include_polygons=True):
        self.osm_path = Path(osm_path)
        self.include_polygons = include_polygons
        self.gps_converter = GPS_to_Global()
        self.gnss_transform = GNSS_Transform()

    def run(self, output_path=None):
        map_features = self.parse()
        if output_path is not None:
            self.dump(map_features, output_path)
        return map_features

    def parse(self):
        if not self.osm_path.exists():
            raise FileNotFoundError(f"osm file not found: {self.osm_path}")

        root = ET.parse(self.osm_path).getroot()
        nodes = self._parse_nodes(root)
        map_features = {}
        ways = {}

        for way in root.findall("way"):
            way_id = way.attrib["id"]
            way_nodes = [nodes[nd.attrib["ref"]] for nd in way.findall("./nd")]
            polyline = self._coords_to_polyline(way_nodes, with_z=True)
            if len(polyline) < 2:
                continue
            map_features[way_id] = {
                "speed_limit_mph": 0,
                "speed_limit_kph": 0,
                "type": self._road_type(way),
                "polyline": polyline,
            }
            speed_limit_mph, speed_limit_kph = self._speed_limit(way)
            map_features[way_id]["speed_limit_mph"] = speed_limit_mph
            map_features[way_id]["speed_limit_kph"] = speed_limit_kph
            ways[way_id] = {"nodes": way_nodes, "polyline": polyline}

        if self.include_polygons:
            self._attach_lanelet_polygons(root, map_features, ways)

        return map_features

    @staticmethod
    def dump(map_features, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pkl.dump(map_features, f)

    @staticmethod
    def _parse_nodes(root):
        return {
            node.attrib["id"]: (float(node.attrib["lat"]), float(node.attrib["lon"]))
            for node in root.findall(".//node")
        }

    def _coords_to_polyline(self, coords, with_z=False):
        polyline_xy = []
        for lat, lon in coords:
            global_x, global_y = self.gps_converter.GPS(lat, lon)
            carla_x, carla_y, _ = self.gnss_transform.GNSS_Global_to_CarlaWest(global_x, global_y, 0)
            polyline_xy.append((carla_x, carla_y))
        polyline_xy = np.asarray(polyline_xy, dtype=np.float32)
        polyline_xy = self._clean_polyline(polyline_xy)
        if with_z:
            return np.column_stack((polyline_xy, np.zeros(len(polyline_xy), dtype=np.float32)))
        return polyline_xy

    @staticmethod
    def _clean_polyline(polyline, min_segment_length=1e-4):
        polyline = np.asarray(polyline, dtype=np.float32)
        if len(polyline) == 0:
            return polyline

        finite_mask = np.isfinite(polyline[..., :2]).all(axis=1)
        polyline = polyline[finite_mask]
        if len(polyline) <= 1:
            return polyline

        segment_length = np.linalg.norm(np.diff(polyline[..., :2], axis=0), axis=1)
        keep_mask = np.concatenate(([True], segment_length > min_segment_length))
        return polyline[keep_mask]

    @classmethod
    def _extract_speed_and_convert(cls, unit_str):
        match = cls.SPEED_PATTERN.match(unit_str)
        if not match:
            raise ValueError(f"cannot parse speed limit: {unit_str}")
        speed_mph = float(match.group(1))
        speed_kph = speed_mph * 0.621371
        return speed_mph, speed_kph

    def _speed_limit(self, way):
        maxspeed_tag = way.find('.//tag[@k="SpeedLimit"]')
        if maxspeed_tag is None:
            return 0, 0
        try:
            return self._extract_speed_and_convert(maxspeed_tag.get("v"))
        except ValueError:
            return 0, 0

    @staticmethod
    def _road_type(way):
        type_tag = way.find('.//tag[@k="Type"]')
        if type_tag is None:
            return None
        road_type = type_tag.get("v")
        if road_type == "LaneBoundary":
            return "ROAD_EDGE_BOUNDARY"
        if road_type == "Lane":
            return "LANE_SURFACE_STREET"
        return None

    def _attach_lanelet_polygons(self, root, map_features, ways):
        for relation in root.findall("./relation"):
            tags = {tag.get("k"): tag.get("v") for tag in relation.findall("./tag")}
            if tags.get("type") != "lanelet":
                continue

            left_boundary_nodes = []
            right_boundary_nodes = []
            center_way_id = None

            for member in relation.findall("./member"):
                if member.get("type") != "way":
                    continue
                member_ref = member.get("ref")
                member_role = member.get("role")
                if member_role == "left" and member_ref in ways:
                    left_boundary_nodes.extend(ways[member_ref]["nodes"])
                elif member_role == "right" and member_ref in ways:
                    right_boundary_nodes.extend(ways[member_ref]["nodes"])
                elif member_role == "center":
                    center_way_id = member_ref

            if not left_boundary_nodes or not right_boundary_nodes:
                continue

            polygon_nodes = self._dedupe_coords(left_boundary_nodes + right_boundary_nodes[::-1])
            polygon = self._coords_to_polyline(polygon_nodes, with_z=False)
            if len(polygon) > 2 and not np.allclose(polygon[0], polygon[-1]):
                polygon = np.vstack([polygon, polygon[0:1]])

            if center_way_id and center_way_id in map_features:
                map_features[center_way_id]["polygon"] = polygon

            for member in relation.findall("./member"):
                member_ref = member.get("ref")
                if member.get("type") == "way" and member_ref in map_features:
                    map_features[member_ref].setdefault("polygon", polygon)

    @staticmethod
    def _dedupe_coords(coords):
        seen = set()
        unique = []
        for coord in coords:
            if coord in seen:
                continue
            seen.add(coord)
            unique.append(coord)
        return unique


class ScenarioDatasetBuilder:
    """Merge object info and map features into a ScenarioNet dataset directory."""

    def __init__(
        self,
        scenario_id="tjwest260528obj_info_1",
        dataset_name="tjwest",
        version="2024-08-13",
        sdc_id="ego",
        hz=10,
        ego_vehicle_parameters=None,
    ):
        self.scenario_id = scenario_id
        self.dataset_name = dataset_name
        self.version = version
        self.sdc_id = sdc_id
        self.hz = hz
        self.ego_vehicle_parameters = ego_vehicle_parameters or {
            "width": 3.07,
            "wheel_base": 3.089,
            "length": 6.076,
            "cog_position_from_rear_axle": 1.67,
            "height": 1.777,
        }

    def run(self, map_features, obj_info, output_dir):
        output_dir = Path(output_dir)
        scene_length = self._infer_scene_length(obj_info)
        scenario = self.build_scenario(map_features, obj_info, scene_length)
        data_summary = scenario["metadata"]
        scenario_file = output_dir / f"sd_{self.scenario_id}.pkl"
        summary_file = output_dir / "dataset_summary.pkl"

        output_dir.mkdir(parents=True, exist_ok=True)
        with scenario_file.open("wb") as f:
            pkl.dump(scenario, f)

        with summary_file.open("wb") as f:
            pkl.dump({scenario_file.name: data_summary}, f)

        return {
            "output_dir": output_dir,
            "scenario_file": scenario_file,
            "summary_file": summary_file,
            "scenario": scenario,
            "dataset_summary": {scenario_file.name: data_summary},
        }

    def build_scenario(self, map_features, obj_info, scene_length):
        scenario = {
            "id": self.scenario_id,
            "version": self.version,
            "length": scene_length,
            "tracks": {},
            "dynamic_map_states": {},
            "map_features": map_features,
            "metadata": self._metadata(scene_length),
        }

        next_id = self._add_tracks(
            scenario=scenario,
            objects=obj_info.get("vehicles", {}),
            start_index=1,
            object_type="VEHICLE",
            scene_length=scene_length,
            first_id=self.sdc_id,
        )
        self._add_tracks(
            scenario=scenario,
            objects=obj_info.get("walkers", {}),
            start_index=next_id,
            object_type="PEDESTRIAN",
            scene_length=scene_length,
        )

        self._update_summaries(scenario)
        return scenario

    def _metadata(self, scene_length):
        return {
            "id": self.scenario_id,
            "ts": np.arange(0, scene_length / self.hz, 1 / self.hz, dtype=np.float32),
            "metadrive_processed": False,
            "coordinate": "metadrive",
            "sdc_id": self.sdc_id,
            "ego_vehicle_parameters": self.ego_vehicle_parameters,
            "dataset": self.dataset_name,
            "scenario_id": self.scenario_id,
            "track_length": scene_length,
            "current_time_index": 20,
            "object_summary": {},
            "number_summary": {},
        }

    @staticmethod
    def _infer_scene_length(obj_info):
        lengths = []
        for bucket in ("vehicles", "walkers"):
            for info in obj_info.get(bucket, {}).values():
                lengths.append(len(info["position"]))
        if not lengths:
            raise ValueError("no valid vehicles or walkers found in object info")
        if len(set(lengths)) != 1:
            raise ValueError(f"inconsistent object track lengths: {sorted(set(lengths))}")
        return lengths[0]

    def _add_tracks(self, scenario, objects, start_index, object_type, scene_length, first_id=None):
        current_index = start_index
        for _, info in objects.items():
            object_id = first_id if first_id and current_index == start_index else str(current_index)
            scenario["tracks"][object_id] = {
                "type": object_type,
                "state": {
                    "position": self._fit_length(info["position"], scene_length),
                    "length": self._fit_length(info["length"], scene_length),
                    "width": self._fit_length(info["width"], scene_length),
                    "height": self._fit_length(info.get("height", np.ones(scene_length) * 1.6), scene_length),
                    "heading": self._fit_length(info["psi"], scene_length),
                    "velocity": self._fit_length(info["velocity"], scene_length),
                    "valid": self._fit_length(info["valid"], scene_length).astype(bool),
                },
                "metadata": {"type": object_type, "track_length": scene_length, "object_id": object_id},
            }
            current_index += 1
        return current_index

    @staticmethod
    def _fit_length(values, scene_length):
        array = np.asarray(values)
        if len(array) != scene_length:
            raise ValueError(f"track length {len(array)} does not match scene length {scene_length}")
        return array

    def _update_summaries(self, scenario):
        object_summary = {}
        for track_id, track in scenario["tracks"].items():
            object_summary[track_id] = self.get_object_summary(track, track_id)
        scenario["metadata"]["object_summary"] = object_summary
        scenario["metadata"]["number_summary"] = self.get_number_summary(scenario)

    @staticmethod
    def get_object_summary(object_dict, object_id):
        if MetaDriveScenarioDescription is not None:
            return MetaDriveScenarioDescription.get_object_summary(object_dict=object_dict, object_id=object_id)

        object_type = object_dict["type"]
        state_dict = object_dict["state"]
        track = state_dict["position"]
        valid_track = track[np.where(state_dict["valid"].astype(int))][..., :2]
        distance = float(
            sum(np.linalg.norm(valid_track[i] - valid_track[i + 1]) for i in range(valid_track.shape[0] - 1))
        )
        valid_length = int(sum(state_dict["valid"]))
        continuous_valid_length = 0
        for valid in state_dict["valid"]:
            if valid:
                continuous_valid_length += 1
            if continuous_valid_length > 0 and not valid:
                break

        return {
            "type": object_type,
            "object_id": str(object_id),
            "track_length": int(len(track)),
            "moving_distance": float(distance),
            "valid_length": int(valid_length),
            "continuous_valid_length": int(continuous_valid_length),
        }

    @staticmethod
    def get_number_summary(scenario):
        number_summary = {
            "num_objects": len(scenario["tracks"]),
            "object_types": set(v["type"] for v in scenario["tracks"].values()),
            "num_traffic_lights": len(scenario["dynamic_map_states"]),
            "num_traffic_light_types": set(),
            "num_traffic_light_each_step": {},
            "num_map_features": len(scenario["map_features"]),
            "map_height_diff": ScenarioDatasetBuilder.map_height_diff(scenario["map_features"]),
        }
        object_types_counter = defaultdict(int)
        moving_objects_counter = defaultdict(int)
        moving_objects = 0
        for track in scenario["tracks"].values():
            object_types_counter[track["type"]] += 1
        for summary in scenario["metadata"]["object_summary"].values():
            if summary["moving_distance"] > 1:
                moving_objects += 1
                moving_objects_counter[summary["type"]] += 1

        number_summary["num_objects_each_type"] = dict(object_types_counter)
        number_summary["num_moving_objects"] = moving_objects
        number_summary["num_moving_objects_each_type"] = dict(moving_objects_counter)
        return number_summary

    @staticmethod
    def map_height_diff(map_features, target=10):
        z_max = -math.inf
        z_min = math.inf
        for feature in map_features.values():
            polyline = feature.get("polyline")
            if polyline is None or len(polyline) == 0 or len(polyline[0]) < 3:
                continue
            z = np.asarray(polyline)[..., -1]
            z_max = max(z_max, np.max(z))
            z_min = min(z_min, np.min(z))
            if z_max - z_min > target:
                break
        if z_max == -math.inf or z_min == math.inf:
            return 0.0
        return float(z_max - z_min)


class DongfengPipeline:
    """One-call pipeline matching scenario_reproduction/Read_Me."""

    def __init__(
        self,
        csv_path=CURRENT_DIR / "Obj_info_batch_1.csv",
        osm_path=CURRENT_DIR / "TesT_Field_Tongji_West_with_relations.osm",
        output_dir=None,
        scenario_id="tjwest260528obj_info_1",
        dataset_name="tjwest",
        obj_info_path=CURRENT_DIR / "obj_info_1.pkl",
        map_features_path=CURRENT_DIR / "map_features_polygon.pkl",
        include_polygons=True,
    ):
        self.csv_path = Path(csv_path)
        self.osm_path = Path(osm_path)
        self.scenario_id = scenario_id
        self.output_dir = Path(output_dir) if output_dir is not None else CURRENT_DIR / scenario_id
        self.obj_info_path = Path(obj_info_path) if obj_info_path is not None else None
        self.map_features_path = Path(map_features_path) if map_features_path is not None else None
        self.include_polygons = include_polygons
        self.dataset_builder = ScenarioDatasetBuilder(scenario_id=scenario_id, dataset_name=dataset_name)

    def run(self):
        obj_info = ObjectInfoProcessor(self.csv_path).run(self.obj_info_path)
        map_features = OSMMapProcessor(self.osm_path, include_polygons=self.include_polygons).run(self.map_features_path)
        return self.dataset_builder.run(map_features=map_features, obj_info=obj_info, output_dir=self.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Build the Dongfeng ScenarioNet dataset in one command.")
    parser.add_argument("--csv", default=CURRENT_DIR / "Obj_info_intersection.csv", type=Path, help="object csv path")
    parser.add_argument(
        "--osm",
        default="/data_set/UniTraj_benchmark/metadrive-main/metadrive/plugins/map_files/TesT_Field_Tongji_West_with_relations_name.osm",
        # default=CURRENT_DIR / "TesT_Field_Tongji_West_with_relations.osm",
        type=Path,
        help="osm map path",
    )
    parser.add_argument("--scenario-id", default="tjwest260725intersection", help="scenario id and output folder name")
    parser.add_argument("--dataset-name", default="tjwest", help="dataset name stored in metadata")
    parser.add_argument("--output-dir", default=None, type=Path, help="output dataset directory")
    parser.add_argument("--obj-info-pkl", default=CURRENT_DIR / "Obj_info_intersection.pkl", type=Path, help="object pkl cache")
    parser.add_argument(
        "--map-features-pkl",
        # default=CURRENT_DIR / "map_features_polygon.pkl",
        default=CURRENT_DIR / "map_features_intersection.pkl",
        type=Path,
        help="map features pkl cache",
    )
    parser.add_argument("--no-polygons", action="store_true", help="skip lanelet polygon extraction")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = DongfengPipeline(
        csv_path=args.csv,
        osm_path=args.osm,
        output_dir=args.output_dir,
        scenario_id=args.scenario_id,
        dataset_name=args.dataset_name,
        obj_info_path=args.obj_info_pkl,
        map_features_path=args.map_features_pkl,
        include_polygons=not args.no_polygons,
    )
    result = pipeline.run()
    print(f"scenario saved: {result['scenario_file']}")
    print(f"dataset summary saved: {result['summary_file']}")


if __name__ == "__main__":
    main()
