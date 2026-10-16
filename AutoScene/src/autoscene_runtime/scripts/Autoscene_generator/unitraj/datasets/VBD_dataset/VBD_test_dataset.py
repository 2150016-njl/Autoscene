import glob
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from .VBDdata_utils import data_collate_fn, data_process_scenario


class VBDTestDataset(Dataset):
    """ScenarioNet -> VBD online inference adapter.

    VBD uses 10 historical states + the current state. The neural model still
    predicts its checkpoint-native 80 future steps; AutoScene truncation to 50
    future points is performed only after model inference.
    """

    def __init__(
        self,
        data_dir=None,
        anchor_path="data/cluster_64_center_dict.pkl",
        max_object=16,
        max_map_points=3000,
        max_polylines=256,
        history_length=11,
        future_steps=80,
        num_points_polyline=30,
        max_traffic_lights=16,
        map_radius_m=150.0,
    ):
        self.data_list = glob.glob(os.path.join(data_dir, "*")) if data_dir else []
        if not anchor_path:
            raise ValueError("VBD anchor_path is required")
        if not os.path.isfile(anchor_path):
            raise FileNotFoundError(f"VBD anchor file not found: {anchor_path}")
        with open(anchor_path, "rb") as stream:
            self.anchors = pickle.load(stream)
        required_anchor_keys = ("TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST")
        missing = [key for key in required_anchor_keys if key not in self.anchors]
        if missing:
            raise KeyError(f"VBD anchor file missing keys: {missing}")

        self.max_object = int(max_object)
        self.max_map_points = int(max_map_points)
        self.max_polylines = int(max_polylines)
        self.history_length = int(history_length)
        self.future_steps = int(future_steps)
        self.num_points_polyline = int(num_points_polyline)
        self.max_traffic_lights = int(max_traffic_lights)
        self.map_radius_m = float(map_radius_m)
        self.base_path = os.path.dirname(os.path.abspath(self.data_list[0])) if self.data_list else None
        self.__collate_fn__ = data_collate_fn

    def __len__(self):
        return len(self.data_list)

    def _process(self, types):
        """Convert VBD agent type ids to the checkpoint anchor dictionary.

        ScenarioNet adapter uses the same ids as official VBD/Waymax:
        1=vehicle, 2=pedestrian, 3=cyclist, 0=padding/unknown.
        """
        anchors = []
        for type_id in np.asarray(types).reshape(-1):
            type_id = int(type_id)
            if type_id == 1:
                anchors.append(self.anchors["TYPE_VEHICLE"])
            elif type_id == 2:
                anchors.append(self.anchors["TYPE_PEDESTRIAN"])
            elif type_id == 3:
                anchors.append(self.anchors["TYPE_CYCLIST"])
            elif type_id == 0:
                anchors.append(np.zeros_like(self.anchors["TYPE_VEHICLE"]))
            else:
                raise ValueError(f"unsupported VBD agent type id={type_id}")
        return np.asarray(anchors, dtype=np.float32)

    def gen_tensor(self, data):
        numeric_keys = (
            "agents_history",
            "agents_interested",
            "agents_future",
            "agents_type",
            "traffic_light_points",
            "polylines",
            "polylines_valid",
            "relations",
        )
        tensors = {key: torch.from_numpy(np.asarray(data[key])) for key in numeric_keys}
        tensors["anchors"] = torch.from_numpy(self._process(data["agents_type"]))
        return tensors

    def process_scenario(
        self,
        scenario_raw,
        current_index=10,
        use_log=True,
        selected_agents=None,
        remove_history=False,
        required_agent_ids=None,
    ):
        data_dict = data_process_scenario(
            scenario_raw,
            current_index=int(current_index),
            max_num_objects=self.max_object,
            max_polylines=self.max_polylines,
            num_points_polyline=self.num_points_polyline,
            use_log=use_log,
            selected_agents=selected_agents,
            remove_history=remove_history,
            history_length=self.history_length,
            future_steps=self.future_steps,
            required_agent_ids=required_agent_ids,
            max_traffic_lights=self.max_traffic_lights,
            map_radius_m=self.map_radius_m,
        )
        data_dict["anchors"] = self._process(data_dict["agents_type"])
        return data_dict

    def reset_agent_length(self, max_object):
        self.max_object = int(max_object)

    def get_scenario_by_id(self, scenario_id, current_index=10, use_log=True, remove_history=False):
        if self.base_path is None:
            raise RuntimeError("VBDTestDataset has no data_dir; get_scenario_by_id is unavailable")
        file_path = os.path.join(self.base_path, f"scenario_{scenario_id}.pkl")
        with open(file_path, "rb") as stream:
            data = pickle.load(stream)
        scenario_raw = data.get("scenario_raw", data.get("scenario"))
        if scenario_raw is None:
            raise ValueError(f"scenario data not found in {file_path}")
        return scenario_id, scenario_raw, self.process_scenario(
            scenario_raw,
            current_index=current_index,
            use_log=use_log,
            remove_history=remove_history,
        )

    def get_scenario_by_index(self, index, current_index=10, use_log=True, remove_history=False):
        filename = self.data_list[index]
        with open(filename, "rb") as stream:
            data = pickle.load(stream)
        if "scenario_raw" in data:
            scenario_raw = data["scenario_raw"]
            scenario_id = data.get("scenario_id", index)
        elif "scenario" in data:
            scenario_raw = data["scenario"]
            scenario_id = os.path.splitext(os.path.basename(filename))[0]
        else:
            raise ValueError(f"scenario_raw/scenario not found in {filename}")
        return scenario_id, scenario_raw, self.process_scenario(
            scenario_raw,
            current_index=current_index,
            use_log=use_log,
            remove_history=remove_history,
        )

    def __getitem__(self, idx):
        _, _, data_dict = self.get_scenario_by_index(idx)
        return data_dict