import math
import numpy as np
import pickle as pkl
from collections import defaultdict

position = np.zeros((500, 3), dtype=np.float32)
position[:, 0] = np.arange(0, 50, 0.1, dtype=np.float32)

position_2 = np.zeros((500, 3), dtype=np.float32)
position_2[300:, 0] = np.arange(100, 20, -0.4, dtype=np.float32)
position_2[300:, 1] = 4.0

vehicle_2 = np.zeros((500), dtype=np.float32)
vehicle_2[300:] = 1
vehicle_2_valid = vehicle_2 != 0

center_lane = np.zeros((100, 3), dtype=np.float32)
center_lane[:, 0] = np.arange(0, 100, 1, dtype=np.float32)
center_lane_2 = center_lane.copy()
center_lane_2[:, 1] = 4.0

boundary_up = np.zeros((100, 3), dtype=np.float32)
boundary_up[:, 0] = np.arange(0, 100, 1, dtype=np.float32)
boundary_up[:, 1] = 10.0
solid_white_1 = boundary_up.copy()
solid_white_1[:, 1] = 2.0
boundary_down = boundary_up.copy()
boundary_down[:, 1] = -2.0
width = 4 * np.ones((100, 2), dtype=np.float32)

velocity = np.zeros((500, 2), dtype=np.float32)
velocity[:, 0] = 1

velocity_2 = np.zeros((500, 2), dtype=np.float32)
velocity_2[300:, 0] = 0.4

#  scenario.pkl
scenario = {}
scenario["id"] = "1"
scenario["version"] = "2024-08-13"
scenario["length"] = 500
scenario["tracks"] = {"1": {"type": "VEHICLE",
                            "state": {"position": position,
                                      "length": 4 * np.ones(500, dtype=np.float32),
                                      "width": 2 * np.ones(500, dtype=np.float32),
                                      "height": 1.8 * np.ones(500, dtype=np.float32),
                                      "heading": np.zeros(500, dtype=np.float32),
                                      "velocity": velocity,
                                      "valid": np.ones(500, dtype=bool)},
                            "metadata": {}},
                      "2": {"type": "VEHICLE",
                            "state": {"position": position_2,
                                      "length": 4 * vehicle_2,
                                      "width": 2 * vehicle_2,
                                      "height": 1.8 * vehicle_2,
                                      "heading": math.pi * vehicle_2,
                                      "velocity": velocity_2,
                                      "valid": vehicle_2_valid},
                            "metadata": {}}}
scenario["dynamic_map_states"] = {}
scenario["map_features"] = {"101": {"type": "ROAD_EDGE_BOUNDARY",
                                    "polyline": boundary_up},
                            "102": {"type": "ROAD_EDGE_BOUNDARY",
                                    "polyline": boundary_down},
                            "103": {"type": "ROAD_LINE_SOLID_SINGLE_WHITE",
                                    "polyline": solid_white_1},
                            "201": {"speed_limit_mph": 37.28227,
                                    "speed_limit_kph": 60.0,
                                    "type": "LANE_SURFACE_STREET",
                                    "polyline": center_lane,
                                    "interpolating": False,
                                    "entry_lanes": [],
                                    "exit_lanes": [],
                                    "left_boundaries": [],
                                    "right_boundaries": [],
                                    "left_neighbor": [],
                                    "right_neighbor": [],
                                    "width": width},
                            "202": {"speed_limit_mph": 37.28227,
                                    "speed_limit_kph": 60.0,
                                    "type": "LANE_SURFACE_STREET",
                                    "polyline": center_lane_2,
                                    "interpolating": False,
                                    "entry_lanes": [],
                                    "exit_lanes": [],
                                    "left_boundaries": [{"lane_start_index": "0",
                                                         "lane_end_index": "100",
                                                         "boundary_type": "ROAD_EDGE_BOUNDARY",
                                                         "boundary_feature_id": "101"}],
                                    "right_boundaries": [],
                                    "left_neighbor": [],
                                    "right_neighbor": [],
                                    "width": width * 2}}
# scenario["map_features"] = {"101": {"type": "ROAD_EDGE_BOUNDARY",
#                                     "polyline": boundary_up},
#                             "102": {"type": "ROAD_EDGE_BOUNDARY",
#                                     "polyline": boundary_down},
#                             "103": {"type": "ROAD_LINE_SOLID_SINGLE_WHITE",
#                                     "polyline": solid_white_1},
#                             "201": {"speed_limit_mph": 37.28227,
#                                     "speed_limit_kph": 60.0,
#                                     "type": "LANE_SURFACE_STREET",
#                                     "polyline": center_lane,
#                                     "interpolating": False,
#                                     "entry_lanes": [],
#                                     "exit_lanes": [],
#                                     "left_boundaries": [{"lane_start_index": "200",
#                                                          "lane_end_index": "300",
#                                                          "boundary_type": "ROAD_LINE_SOLID_SINGLE_WHITE",
#                                                          "boundary_feature_id": "103"}],
#                                     "right_boundaries": [{"lane_start_index": "0",
#                                                           "lane_end_index": "50",
#                                                           "boundary_type": "ROAD_EDGE_BOUNDARY",
#                                                           "boundary_feature_id": "102"}],
#                                     "left_neighbor": [{"feature_id": "202",
#                                                        "self_start_index": "0",
#                                                        "self_end_index": "100",
#                                                        "neighbor_start_index": "0",
#                                                        "neighbor_end_index": "10000",
#                                                        "boundaries": [{"lane_start_index": "0",
#                                                                        "lane_end_index": "100",
#                                                                        "boundary_type": "ROAD_LINE_SOLID_SINGLE_WHITE",
#                                                                        "boundary_feature_id": "103"}]}],
#                                     "right_neighbor": [],
#                                     "width": width},
#                             "202": {"speed_limit_mph": 37.28227,
#                                     "speed_limit_kph": 60.0,
#                                     "type": "LANE_SURFACE_STREET",
#                                     "polyline": center_lane_2,
#                                     "interpolating": False,
#                                     "entry_lanes": [],
#                                     "exit_lanes": [],
#                                     "left_boundaries": [{"lane_start_index": "200",
#                                                          "lane_end_index": "300",
#                                                          "boundary_type": "ROAD_EDGE_BOUNDARY",
#                                                          "boundary_feature_id": "101"}],
#                                     "right_boundaries": [{"lane_start_index": "0",
#                                                           "lane_end_index": "50",
#                                                           "boundary_type": "ROAD_LINE_SOLID_SINGLE_WHITE",
#                                                           "boundary_feature_id": "103"}],
#                                     "left_neighbor": [],
#                                     "right_neighbor": [{"feature_id": "201",
#                                                         "self_start_index": "0",
#                                                         "self_end_index": "100",
#                                                         "neighbor_start_index": "0",
#                                                         "neighbor_end_index": "100",
#                                                         "boundaries": [{"lane_start_index": "0",
#                                                                         "lane_end_index": "100",
#                                                                         "boundary_type": "ROAD_LINE_SOLID_SINGLE_WHITE",
#                                                                         "boundary_feature_id": "103"}]}],
#                                     "width": width}}
scenario["metadata"] = {}

## summary.pkl
dataset_summary = {}
data_summary = {}
data_summary['id'] = "1"
data_summary['coordinate'] = "test"
data_summary["ts"] = np.arange(0, 50, 0.1, dtype=np.float32)
data_summary["metadrive_processed"] = False
data_summary["sdc_id"] = "1"
data_summary["dataset"] = "test"
data_summary["scenario_id"] = "1"
data_summary["source_file"] = ""
data_summary["track_length"] = 500
data_summary["current_time_index"] = 0
data_summary["sdc_track_index"] = 100
data_summary["objects_of_interest"] = []
data_summary["tracks_to_predict"] = {}
data_summary["object_summary"] = {"1": {"type": "VEHICLE",
                                        "object_id": 1,
                                        "track_length": 500,
                                        "moving_distance": 50,
                                        "valid_length": 500,
                                        "continuous_valid_length": 500},
                                  "2": {"type": "VEHICLE",
                                        "object_id": 2,
                                        "track_length": 500,
                                        "moving_distance": 80,
                                        "valid_length": 200,
                                        "continuous_valid_length": 200}}
data_summary["number_summary"] = {"num_objects": 2,
                                  "object_type": {"VEHICLE"},
                                  "num_objects_each_type": {"VEHICLE": 2},
                                  "num_moving_objects": 2,
                                  "num_moving_objects_each_type": defaultdict(int, {'VEHICLE': 2}),
                                  "num_traffic_lights": 0,
                                  "num_traffic_light_types": {},
                                  "num_traffic_light_each_step": {},
                                  "num_map_features": 3,
                                  "map_height_diff": 0}

scenario["metadata"] = data_summary

with open("/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/test/sd_scenario_test.pkl", "wb") as f:
    pkl.dump(scenario, f)

dataset_summary["sd_scenario_test.pkl"] = data_summary
with open("/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/test/dataset_summary.pkl", "wb") as f:
    pkl.dump(dataset_summary, f)

