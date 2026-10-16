import os
from os import mkdir

import numpy as np
import pickle as pkl
from collections import defaultdict
from metadrive.scenario.scenario_description import ScenarioDescription

with open('map_features_polygon.pkl', 'rb') as f:
    map_features = pkl.load(f)
with open('obj_info_1.pkl', 'rb') as f:
    obj_info = pkl.load(f)

def get_number_summary(scenario):
    """Return the stats of all objects in a scenario.

    Examples:
        {'num_objects': 211,
         'object_types': {'CYCLIST', 'PEDESTRIAN', 'VEHICLE'},
         'num_objects_each_type': {'VEHICLE': 184, 'PEDESTRIAN': 25, 'CYCLIST': 2},
         'num_moving_objects': 69,
         'num_moving_objects_each_type': defaultdict(int, {'VEHICLE': 52, 'PEDESTRIAN': 15, 'CYCLIST': 2}),
         'num_traffic_lights': 8,
         'num_traffic_light_types': {'LANE_STATE_STOP', 'LANE_STATE_UNKNOWN'},
         'num_traffic_light_each_step': {'LANE_STATE_UNKNOWN': 164, 'LANE_STATE_STOP': 564},
         'num_map_features': 358,
         'map_height_diff': 2.4652252197265625}

    Args:
        scenario: The input scenario.

    Returns:
        A dict describing the number of different kinds of data.
    """
    number_summary_dict = {}

    # object
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_OBJECTS] = len(scenario[ScenarioDescription.TRACKS])
    number_summary_dict[ScenarioDescription.SUMMARY.OBJECT_TYPES] = \
        set(v["type"] for v in scenario[ScenarioDescription.TRACKS].values())
    object_types_counter = defaultdict(int)
    for v in scenario[ScenarioDescription.TRACKS].values():
        object_types_counter[v["type"]] += 1
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_OBJECTS_EACH_TYPE] = dict(object_types_counter)

    # # If object summary does not exist, fill them here
    # object_summaries = {}
    # for track_id, track in scenario[SD.TRACKS].items():
    #     object_summaries[track_id] = scenario.get_object_summary(object_dict=track, object_id=track_id)
    # scenario[scenario.METADATA][scenario.SUMMARY.OBJECT_SUMMARY] = object_summaries

    # moving object
    number_summary_dict.update(ScenarioDescription._calculate_num_moving_objects(scenario))

    # Number of different dynamic object states
    dynamic_object_states_types = set()
    dynamic_object_states_counter = defaultdict(int)
    for v in scenario[ScenarioDescription.DYNAMIC_MAP_STATES].values():
        for step_state in v["state"]["object_state"]:
            if step_state is None:
                continue
            dynamic_object_states_types.add(step_state)
            dynamic_object_states_counter[step_state] += 1
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHTS
                        ] = len(scenario[ScenarioDescription.DYNAMIC_MAP_STATES])
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHT_TYPES] = dynamic_object_states_types
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHTS_EACH_STEP
                        ] = dict(dynamic_object_states_counter)

    # map
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_MAP_FEATURES
                        ] = len(scenario[ScenarioDescription.MAP_FEATURES])
    number_summary_dict[ScenarioDescription.SUMMARY.MAP_HEIGHT_DIFF
                        ] = ScenarioDescription.map_height_diff(scenario[ScenarioDescription.MAP_FEATURES])
    return number_summary_dict

#  scenario.pkl
scene_length=1739
scenario_id = 'tjwest260528obj_info_1'
sdc_id = 'ego'

scenario = {"id": scenario_id,
            "version": "2024-08-13",
            "length": scene_length,
            "tracks": {},
            "dynamic_map_states": {},
            "map_features": map_features,
            "metadata": {}}
num_vehicle = 1
for _, info in obj_info['vehicles'].items():
    vehicle_id = str(num_vehicle)
    # 如果是第一辆车，ID设为'ego'，否则用数字
    if num_vehicle == 1:
        vehicle_id = sdc_id

    vehicle_info = {vehicle_id: {"type": "VEHICLE",
                                       "state": {'position': info['position'],
                                                 "length": info['length'],
                                                 "width": info['width'],
                                                 "height": info.get('height', np.ones(scene_length)*1.6),
                                                 'heading': info['psi'],
                                                 'velocity': info['velocity'],
                                                 'valid': np.ones(scene_length, dtype=bool)},
                                       "metadata": {}}}
    scenario['tracks'].update(vehicle_info)
    num_vehicle = num_vehicle + 1

num_walker = num_vehicle
for _, info in obj_info['walkers'].items():
    walker_info = {str(num_walker): {"type": "PEDESTRIAN",
                                     "state": {'position': info['position'],
                                               "length": info['length'],
                                               "width": info['width'],
                                               "height": info.get('height', np.ones(scene_length)*1.6),
                                               'heading': info['psi'],
                                               'velocity': info['velocity'],
                                               'valid': np.ones(scene_length, dtype=bool)},
                                     "metadata": {}},
                   }
    scenario['tracks'].update(walker_info)
    num_walker = num_walker + 1


ego_vehicle_parameters={'width':3.07,
                        'wheel_base':3.089,
                        'length':6.076,
                        'cog_position_from_rear_axle':1.67,
                        'height':1.777}
data_summary = {'id': scenario_id,
                # 'coordinate': "test",
                "ts": np.arange(0, scene_length/10, 0.1),
                "metadrive_processed": False,
                "sdc_id": sdc_id,  # 主车为'ego'，用于定位到主车
                'ego_vehicle_parameters': ego_vehicle_parameters,
                "dataset": "tjwest",
                "scenario_id": scenario_id,
                "track_length": scene_length,
                "current_time_index": 0,
                # "sdc_track_index": 100,
                # "objects_of_interest": [],
                # "tracks_to_predict": {},
                'object_summary':{},
                'number_summary':{}
                }
scenario["metadata"] = data_summary

SD = ScenarioDescription
summary_dict = {}
for track_id, track in scenario[SD.TRACKS].items():
    summary_dict[track_id] = SD.get_object_summary(object_dict=track, object_id=track_id)

scenario["metadata"]['object_summary'] = summary_dict
data_summary['object_summary'] = summary_dict

scenario["metadata"]['number_summary'] = get_number_summary(scenario)
data_summary['number_summary'] = get_number_summary(scenario)

os.makedirs(f'/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/{scenario_id}', exist_ok=True)
with open(f"/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/{scenario_id}/sd_{scenario_id}.pkl", "wb") as f:
    pkl.dump(scenario, f)

dataset_summary = {f"sd_{scenario_id}.pkl": data_summary}
with open(f"/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/{scenario_id}/dataset_summary.pkl", "wb") as f:
    pkl.dump(dataset_summary, f)