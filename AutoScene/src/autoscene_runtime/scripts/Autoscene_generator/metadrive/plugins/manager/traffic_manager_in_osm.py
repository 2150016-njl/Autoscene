"""
添加了closest_vehicle属性，返回当前自车最近的交通车辆和trans_adbv属性，表示是否使用trans_adbv
"""
import copy
from collections import namedtuple
from typing import Dict

import math
import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.scenario import ScenarioDescription as SD
from metadrive.utils import merge_dicts

from metadrive.engine.logger import get_logger
logger = get_logger()

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class TrafficMode:
    # Traffic vehicles will be respawned, once they arrive at the destinations
    Respawn = "respawn"

    # Traffic vehicles will be triggered only once
    Trigger = "trigger"

    # Hybrid, some vehicles are triggered once on map and disappear when arriving at destination, others exist all time
    Hybrid = "hybrid"

class TrafficManagerInOsm(BaseManager):
    VEHICLE_GAP = 10  # m
    CLOSEST_VEHICLE_SWITCH_GAP = 10.0
    ENTRY_SPAWN_MARGIN = 10.0
    MAX_SPAWN_DISTANCE = 120.0
    MAX_JUNCTION_SPAWNS = 10
    RESPAWN_TARGET_VEHICLE_COUNT = 10

    def __init__(self):
        """
        控制整个交通流
        """
        super(TrafficManagerInOsm, self).__init__()

        self._traffic_vehicles = []

        # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = self.engine.global_config["traffic_mode"]
        self.random_traffic = self.engine.global_config["random_traffic"]
        self.density = self.engine.global_config["traffic_density"]
        self.respawn_lanes = None
        self.cv = None
        self.v_to_remove = None
        self.risk_field = None
        self.name_id=0
        self._obj_id_to_original_id = {}

    def _get_ego_vehicle(self):
        for vehicle in self.engine.agent_manager.active_agents.values():
            if vehicle.class_name == "DefaultVehicle":
                return vehicle
        for vehicle in self.vehicles:
            if vehicle.class_name == "DefaultVehicle":
                return vehicle
        return None

    def _activate_triggered_vehicles(self):
        if self.mode == TrafficMode.Respawn or not self.block_triggered_vehicles:
            return

        for vehicle in self.engine.agent_manager.active_agents.values():
            ego_lane_idx = vehicle.lane_index[:-1]
            ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
            if ego_road == self.block_triggered_vehicles[-1].trigger_road:
                block_vehicles = self.block_triggered_vehicles.pop()
                self._traffic_vehicles += list(self.get_objects(block_vehicles.vehicles).values())
                break

    def _build_existing_vehicle_positions(self):
        existing_vehicle_positions = {}
        for vehicle in self.vehicles:
            if vehicle.lane is None:
                continue
            lane_index = vehicle.lane.index
            existing_vehicle_positions.setdefault(lane_index, []).append(vehicle.lane.local_coordinates(vehicle.position)[0])
        return existing_vehicle_positions

    def _next_vehicle_name(self):
        while str(self.name_id) in self.engine._spawned_objects:
            self.name_id += 1
        return str(self.name_id)

    def _split_vehicle_spawn_config(self, vehicle_config):
        base_vehicle_keys = set(self.engine.global_config["vehicle_config"].keys())
        spawn_vehicle_config = {}
        extra_vehicle_config = {}
        for key, value in vehicle_config.items():
            if key in base_vehicle_keys:
                spawn_vehicle_config[key] = value
            else:
                extra_vehicle_config[key] = value
        return spawn_vehicle_config, extra_vehicle_config

    def _spawn_traffic_vehicle(self, vehicle_type, vehicle_config, policy_cls, name=None, **spawn_kwargs):
        from metadrive.utils.error_class import NavigationError

        spawn_vehicle_config, extra_vehicle_config = self._split_vehicle_spawn_config(vehicle_config)
        try:
            random_v = self.spawn_object(vehicle_type, vehicle_config=spawn_vehicle_config, name=name, **spawn_kwargs)
        except NavigationError:
            return None
        if extra_vehicle_config:
            random_v.update_config(extra_vehicle_config, allow_add_new_key=True)
        self.add_policy(random_v.id, policy_cls, random_v, self.generate_seed())
        self._traffic_vehicles.append(random_v)
        return random_v

    def _create_custom_adversaries(self):
        """
        Spawn user-specified adversarial vehicles from global_config["custom_adversaries"].

        Each item can contain any normal vehicle_config key plus:
            name: optional object name.
            metadata: optional bookkeeping dict; ignored by the simulator.
        """
        custom_adversaries = self.engine.global_config.get("custom_adversaries", []) or []
        if not custom_adversaries:
            return

        from metadrive.component.vehicle.vehicle_type import SVehicle
        from metadrive.plugins.policy.idm_policy_in_osm import IDMPolicyInOsm

        spawned_names = []
        for index, custom_config in enumerate(custom_adversaries):
            traffic_v_config = copy.deepcopy(self.engine.global_config["traffic_vehicle_config"])
            traffic_v_config.update(copy.deepcopy(custom_config))
            name = str(traffic_v_config.get("name", f"adv_{index}"))
            metadata = copy.deepcopy(traffic_v_config.get("metadata", {}))
            if "name" in traffic_v_config:
                traffic_v_config.pop("name")
            if "metadata" in traffic_v_config:
                traffic_v_config.pop("metadata")

            spawn_kwargs = {}
            spawn_position_heading = traffic_v_config.get("spawn_position_heading", None)
            if spawn_position_heading is not None:
                spawn_kwargs["position"] = spawn_position_heading[0]
                spawn_kwargs["heading"] = float(spawn_position_heading[1])

            vehicle = self._spawn_traffic_vehicle(
                SVehicle, traffic_v_config, IDMPolicyInOsm, name=name, **spawn_kwargs
            )
            if vehicle is None:
                message = f"Failed to spawn required custom adversary {name}."
                if self.engine.global_config.get("require_custom_adversaries", True):
                    raise ValueError(message)
                logger.warning(f"Skip custom adversary {name}: failed to spawn.")
            else:
                spawned_names.append(vehicle.name)
                original_id = metadata.get("vehicle_id", name)
                self._obj_id_to_original_id[vehicle.name] = str(original_id)
                # print(
                #     f"[custom adversary] {name}: target={spawn_kwargs.get('position')} "
                #     f"actual={vehicle.position} speed={vehicle.speed:.3f} m/s"
                # )
        if self.engine.global_config.get("debug_osm_route", False):
            print(
                "[osm traffic][custom_adversaries]",
                "required=", len(custom_adversaries),
                "spawned=", len(spawned_names),
                "names=", spawned_names,
            )

    def _traffic_remove_reason(self, vehicle):
        if self.engine.global_config.get("keep_vehicle_on_crash", True):
            pass  # skip crash removal
        else:
            if vehicle.crash_vehicle:
                return "crash_vehicle"
        if vehicle.crash_object:
            return "crash_object"
        if vehicle.crash_sidewalk:
            return "crash_sidewalk"
        if vehicle.crash_building:
            return "crash_building"
        if not vehicle.on_lane:
            is_custom_adversary = (
                str(getattr(vehicle, "name", "")) in self._obj_id_to_original_id
                or str(getattr(vehicle, "id", "")) in self._obj_id_to_original_id
            )
            if is_custom_adversary and self.engine.global_config.get("keep_custom_adversaries_on_off_lane", True):
                return None
            if self.engine.global_config.get("relax_osm_traffic_off_lane_removal", True):
                try:
                    lateral = abs(float(vehicle.navigation.current_lateral))
                except Exception:
                    lateral = float("inf")
                max_lateral = float(self.engine.global_config.get("max_lateral_dist", 20))
                if lateral <= max_lateral:
                    return None
                return f"off_lane_lateral>{max_lateral:.1f}"
            return "off_lane"
        return None

    def _should_remove_vehicle(self, vehicle):
        return self._traffic_remove_reason(vehicle) is not None

    def _lane_speed_limit_km_h(self, lane):
        speed_limit = getattr(lane, "speed_limit", None)
        try:
            speed_limit = float(speed_limit)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(speed_limit) or speed_limit <= 0:
            return None
        return speed_limit

    def _sample_respawn_speed_profile(self, lane):
        traffic_cfg = self.engine.global_config.get("traffic_vehicle_config", {})
        min_speed = float(traffic_cfg.get("idm_target_speed_min_km_h", 15.0))
        max_speed = float(traffic_cfg.get("idm_target_speed_max_km_h", 40.0))
        if max_speed < min_speed:
            min_speed, max_speed = max_speed, min_speed

        lane_limit = self._lane_speed_limit_km_h(lane)
        respect_limit = bool(traffic_cfg.get("idm_respect_lane_speed_limit", True))
        low = min_speed
        high = max_speed
        if respect_limit and lane_limit is not None:
            low = min(min_speed, lane_limit)
            high = min(max_speed, lane_limit)

        target_speed_km_h = low if abs(high - low) < 1e-6 else self.np_random.uniform(low, high)
        spawn_scale = self.np_random.uniform(0.75, 0.95)
        spawn_speed_m_s = max(0.0, float(target_speed_km_h) * spawn_scale / 3.6)
        return float(target_speed_km_h), float(spawn_speed_m_s)

    def reset(self):
        """
        根据生成模式和生成密度，在地图上生成交通
        :return: List of Traffic vehicles
        """
        map = self.current_map
        logger.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []
        self._obj_id_to_original_id = {}

        use_custom_adversaries = bool(self.engine.global_config.get("use_custom_adversaries", False))
        if use_custom_adversaries:
            self._create_custom_adversaries()
            if bool(self.engine.global_config.get("disable_random_traffic_when_custom_adversaries", True)):
                return

        traffic_density = self.density
        if abs(traffic_density) < 1e-2:
            return
        self.respawn_lanes = self.engine.current_map.road_network.get_all_lanes()
        if self.mode == TrafficMode.Respawn:
            self._create_respawn_vehicles(map, traffic_density,False)
            # self._create_vehicles_once(map, traffic_density)
        elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
            self._create_vehicles_once(map, traffic_density)
        else:
            raise ValueError("No such mode named {}".format(self.mode))

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """

        # trigger vehicles
        self._activate_triggered_vehicles()

        # closest_vehicle = self.closest_vehicle
        # self.cv = closest_vehicle
        # use_trans_adbv = (
        #     closest_vehicle is not None and getattr(self.engine.global_config, "trans_adbv", False)
        #     and getattr(self.engine, "top_down_renderer", None) is not None
        # )

        # for v in self._traffic_vehicles:
        #     if use_trans_adbv and v is closest_vehicle:
        #         p = MyEnvInputPolicy(v, self.generate_seed())
        #         self.engine.top_down_renderer.closet_vehicle = v.name
        #         v.before_step(p.act("bv_agent"))
        #     else:
        #         p = self.engine.get_policy(v.name)
        #         v.before_step(p.act(v.id))
            
        # return dict()
        for v in self._traffic_vehicles:
            p = self.engine.get_policy(v.name)
            v.before_step(p.act())
        return dict()
    
    def step(self):
        if self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
            self._create_respawn_vehicles(self.current_map, self.density, True)

    def _get_record_id_mapping(self):
        obj_id_to_original_id = {}
        for obj_id in self.engine.get_objects().keys():
            obj_id = str(obj_id)
            obj_id_to_original_id[obj_id] = self._obj_id_to_original_id.get(obj_id, obj_id)
        original_id_to_obj_id = {str(original_id): obj_id for obj_id, original_id in obj_id_to_original_id.items()}
        return {
            SD.ORIGINAL_ID_TO_OBJ_ID: original_id_to_obj_id,
            SD.OBJ_ID_TO_ORIGINAL_ID: obj_id_to_original_id,
        }

    @property
    def closest_vehicle(self):
        ego_vehicle = self._get_ego_vehicle()
        if ego_vehicle is None:
            return None

        min_vehicle = None
        min_distance = float("inf")
        distance = {}
        for vehicle in self._traffic_vehicles:
            if vehicle.class_name in {"AdbvVehicle", "DefaultVehicle"}:
                continue
            dist = np.linalg.norm(ego_vehicle.position - vehicle.position)
            distance[vehicle.name] = dist
            if dist < min_distance:
                min_distance = dist
                min_vehicle = vehicle

        if min_vehicle is None:
            return None

        if self.cv is not None and self.cv.name in distance and distance[self.cv.name] - min_distance <= self.CLOSEST_VEHICLE_SWITCH_GAP:
            return self.cv
        return min_vehicle

    def after_step(self, *args, **kwargs):
        """
        方法主要用于更新交通系统中每辆车的状态，并根据车辆*是否在车道上*以及当前的交通模式决定是否移除车辆。
        如果车辆被移除且当前模式允许重生，则会在随机选择的车道上生成新的车辆。
        该方法确保交通流的动态管理，涉及车辆状态更新和重生机制。
        """
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if self._should_remove_vehicle(v):
                v_to_remove.append(v)
        self.v_to_remove = v_to_remove

        for v in v_to_remove:
            if self.engine.global_config.get("debug_osm_route", False):
                print(
                    "[osm traffic][remove]",
                    "name=", v.name,
                    "id=", v.id,
                    "reason=", self._traffic_remove_reason(v),
                    "on_lane=", v.on_lane,
                    "lane_index=", getattr(v, "lane_index", None),
                    "pos=", v.position,
                )
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)
            # if self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
            #     # 只保留 self.respawn_lanes 中不在 occupied_lanes 中的元素
            #     occupied_lanes = set([vehicle.lane for vehicle in self.vehicles])
            #     respawn_lanes = list(set(self.respawn_lanes) - set(occupied_lanes))
            #     lane = respawn_lanes[self.np_random.randint(0, len(respawn_lanes))]
            #     lane_idx = lane.index
            #     long = self.np_random.rand() * lane.length / 2
            #     traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
            #     traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
            #     new_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
            #     from metadrive.policy.idm_policy import IDMPolicy
            #     self.add_policy(new_v.id, IDMPolicy, new_v, self.generate_seed())
            #     self._traffic_vehicles.append(new_v)

        return dict()

    def before_reset(self) -> None:
        """
        Clear the scene and then reset the scene to empty
        :return: None
        """
        super(TrafficManagerInOsm, self).before_reset()
        self.density = self.engine.global_config["traffic_density"]
        self.block_triggered_vehicles = []
        self._traffic_vehicles = []
        self.name_id = 0

    def get_vehicle_num(self):
        """
        Get the vehicles on road
        :return:
        """
        if self.mode == TrafficMode.Respawn:
            return len(self._traffic_vehicles)
        return sum(len(block_vehicle_set.vehicles) for block_vehicle_set in self.block_triggered_vehicles)

    def get_global_states(self) -> Dict:
        """
        有问题
        Return all traffic vehicles' states
        :return: States of all vehicles
        """
        states = dict()
        traffic_states = {vehicle.id: vehicle.get_state() for vehicle in self._traffic_vehicles}
        states[TRAFFIC_VEHICLES] = traffic_states
        active_obj = copy.copy(self.engine.agent_manager._active_objects)
        pending_obj = copy.copy(self.engine.agent_manager._pending_objects)
        dying_obj = copy.copy(self.engine.agent_manager._dying_objects)
        states[TARGET_VEHICLES] = {k: v.get_state() for k, v in active_obj.items()}
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v.get_state()
                                      for k, v in pending_obj.items()}, allow_new_keys=True
        )
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v_count[0].get_state()
                                      for k, v_count in dying_obj.items()},allow_new_keys=True
        )

        states[OBJECT_TO_AGENT] = copy.deepcopy(self.engine.agent_manager._object_to_agent)
        states[AGENT_TO_OBJECT] = copy.deepcopy(self.engine.agent_manager._agent_to_object)
        return states

    def get_global_init_states(self) -> Dict:
        """
        Special handling for first states of traffic vehicles
        :return: States of all vehicles
        """
        vehicles = dict()
        for vehicle in self._traffic_vehicles:
            init_state = vehicle.get_state()
            init_state["index"] = vehicle.index
            init_state["type"] = vehicle.class_name
            init_state["enable_respawn"] = vehicle.enable_respawn
            vehicles[vehicle.index] = init_state

        # collect other vehicles
        if self.mode != TrafficMode.Respawn:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    init_state = vehicle.get_state()
                    init_state["type"] = vehicle.class_name
                    init_state["index"] = vehicle.index
                    init_state["enable_respawn"] = vehicle.enable_respawn
                    vehicles[vehicle.index] = init_state
        return vehicles

    def _propose_vehicle_configs(self, lane: AbstractLane):
        """
        在一条车道上根据车道长度和车辆间距提出车辆生成配置
        """
        start_offset =1.0  # 设置最小纵向位置为 1 米
        
        # 如果车道太短，连 1 米都没有，则不生成
        if lane.length <= start_offset:
            return []
        # 计算有效可生成长度 (总长 - 起始偏移)
        available_length = lane.length - start_offset
        potential_vehicle_configs = []
        # 根据有效长度计算能放下多少辆车
        total_num = int(available_length / self.VEHICLE_GAP)
        vehicle_longs = [start_offset + i * self.VEHICLE_GAP for i in range(total_num)]
        # Only choose given number of vehicles
        for long in vehicle_longs:
            random_vehicle_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False, "junction": lane.junction}
            potential_vehicle_configs.append(random_vehicle_config)
        return potential_vehicle_configs

    # def _create_respawn_vehicles(self, map: BaseMap, traffic_density: float, spawn_only_from_entry: bool = False):
    #     '''
    #     改进版：增加了自车雷达范围过滤，防止突然“空降”车辆干扰自车。
    #     '''
    #     # 1. 基础准备
    #     all_lanes = sorted(map.road_network.get_all_lanes(), key=lambda l: l.index)
    #     ego_pos = None
    #     lidar_range = 50
    #     # 不能用self.engine.agent_manager.active_agents，因为在after_reset中才会更新，影响不到这里的reset
    #     if self.engine.agent_manager.episode_created_agents:
    #         ego_pos = self.engine.agent_manager.episode_created_agents['default_agent'].position
    #         lidar_range = self.engine.agent_manager.episode_created_agents['default_agent'].config["lidar"]["distance"]
    #     # 自车雷达感知半径，用于过滤生成点，防止生成在自车视野内影响状态
    #     MIN_SPAWN_DISTANCE = lidar_range + 10.0 
    #     # ------------------------

    #     # 2. 现有车辆位置映射 (用于避免重叠)
    #     existing_vehicle_positions = {}
    #     for v in self.vehicles:
    #         if v.lane is None: continue
    #         l_index = v.lane.index
    #         if l_index not in existing_vehicle_positions:
    #             existing_vehicle_positions[l_index] = []
    #         long_pos = v.lane.local_coordinates(v.position)[0]
    #         existing_vehicle_positions[l_index].append(long_pos)

    #     # 3. 确定目标车道
    #     if spawn_only_from_entry:
    #         target_lanes = [l for l in all_lanes if l.entry_lanes is None or len(l.entry_lanes) == 0]
    #     else:
    #         target_lanes = [l for l in all_lanes if l.junction is False]

    #     # 4. 生成并过滤配置
    #     valid_spawn_configs = []
        
    #     for l in target_lanes:
    #         # 生成原始点
    #         if spawn_only_from_entry:
    #             raw_configs = [{"spawn_lane_index": l.index, "spawn_longitude": 1.0, "enable_reverse": False}]
    #         else:
    #             raw_configs = self._propose_vehicle_configs(l)

    #         # --- 调用过滤函数 (传入 ego_pos) ---
    #         lane_valid_configs = self._filter_configs(
    #             l, raw_configs, existing_vehicle_positions, 
    #             ego_pos=ego_pos, min_spawn_dist=MIN_SPAWN_DISTANCE
    #         )
    #         valid_spawn_configs.extend(lane_valid_configs)
    #     # 5. 计算数量并 Shuffle (逻辑不变)
    #     total_length = sum([lane.length for lane in all_lanes])
    #     total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
    #     target_total_vehicles = int(math.floor(total_spawn_points * traffic_density))
    #     target_total_vehicles = 20
    #     current_vehicle_count = len(self._traffic_vehicles)
    #     needed_vehicles = target_total_vehicles - current_vehicle_count

    #     if needed_vehicles <= 0 or len(valid_spawn_configs) == 0:
    #         return
    #     valid_spawn_configs.sort(key=lambda x: str(x.get('spawn_lane_index', 0)) + str(x.get('spawn_longitude', 0)))
    #     self.np_random.shuffle(valid_spawn_configs)
    #     real_spawn_count = min(needed_vehicles, len(valid_spawn_configs))
    #     selected = valid_spawn_configs[:real_spawn_count]
    #     # 6. 实例化 (逻辑不变)
    #     from metadrive.CHT.policy.idm_policy_in_osm import IDMPolicyInOsm
    #     for traffic_v_config in selected: 
    #         while str(self.name_id) in self.engine._spawned_objects:
    #             self.name_id += 1
            
    #         # vehicle_type = self.random_vehicle_type()
    #         from metadrive.component.vehicle.vehicle_type import SVehicle
    #         vehicle_type = SVehicle
    #         traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
            
    #         random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config, name=str(self.name_id))
    #         self.add_policy(random_v.id, IDMPolicyInOsm, random_v, self.generate_seed())
    #         self._traffic_vehicles.append(random_v)
    #         self.name_id += 1
    def _create_respawn_vehicles(self, map: BaseMap, traffic_density: float, spawn_only_from_entry: bool = False, enable_max_distance_filter: bool = False):
        '''
        改进版：
        1. enable_max_distance_filter: 可选开启，只在自车周围一定范围内生成车辆。
        2. MIN_SPAWN_DISTANCE: 逻辑修改，仅在 spawn_only_from_entry=True (动态生成) 时生效。
        '''
        # 1. 基础准备
        all_lanes = sorted(map.road_network.get_all_lanes(), key=lambda l: l.index)
        ego_pos = None
        lidar_range = 50
        
        # 获取自车位置
        if self.engine.agent_manager.episode_created_agents:
            default_agent = self.engine.agent_manager.episode_created_agents['default_agent']
            ego_pos = default_agent.position
            lidar_range = default_agent.config["lidar"]["distance"]
            
        # --- 距离阈值设置 ---
        # 动态生成默认避开雷达范围；初始全图生成可通过配置避开 ego 附近。
        if spawn_only_from_entry:
            configured_min_spawn_dist = self.engine.global_config.get(
                "traffic_respawn_min_spawn_distance_to_ego",
                None,
            )
            if configured_min_spawn_dist is None:
                configured_min_spawn_dist = lidar_range + self.ENTRY_SPAWN_MARGIN
            current_min_spawn_dist = float(configured_min_spawn_dist)
        else:
            configured_min_spawn_dist = self.engine.global_config.get(
                "traffic_initial_min_spawn_distance_to_ego",
                0.0,
            )
            current_min_spawn_dist = 0.0 if configured_min_spawn_dist is None else float(configured_min_spawn_dist)
        
        # 最大生成距离（仅当 enable_max_distance_filter=True 时使用）
        max_spawn_distance = self.MAX_SPAWN_DISTANCE
        # ------------------------

        # 2. 现有车辆位置映射
        existing_vehicle_positions = self._build_existing_vehicle_positions()

        # 3. 确定目标车道
        if spawn_only_from_entry:
            target_lanes = [l for l in all_lanes if l.entry_lanes is None or len(l.entry_lanes) == 0]
        else:
            allow_junction_spawn = bool(self.engine.global_config.get("traffic_spawn_on_junction_lanes", False))
            target_lanes = all_lanes if allow_junction_spawn else [l for l in all_lanes if l.junction is not True]

        # 4. 生成并过滤配置
        valid_spawn_configs = []
        junction_count = 0
        for l in target_lanes:
            # 4.1 生成原始候选点
            if spawn_only_from_entry:
                raw_configs = [{"spawn_lane_index": l.index, "spawn_longitude": 1.0, "enable_reverse": False}]
            else:
                raw_configs = self._propose_vehicle_configs(l)
            
            # --- 逻辑修改：最大距离预过滤 (可选) ---
            if enable_max_distance_filter and ego_pos is not None:
                in_range_configs = []
                for config in raw_configs:
                    spawn_long = config.get("spawn_longitude", 0.0)
                    spawn_point = l.position(spawn_long, 0.0)
                    dist = np.linalg.norm(spawn_point - ego_pos)
                    
                    if dist <= max_spawn_distance:
                        in_range_configs.append(config)
                raw_configs = in_range_configs
            # -----------------------------------

            # 4.2 调用过滤函数
            # 注意：这里传入的 current_min_spawn_dist 根据 spawn_only_from_entry 动态变化
            lane_valid_configs = self._filter_configs(
                l, raw_configs, existing_vehicle_positions, 
                ego_pos=ego_pos, min_spawn_dist=current_min_spawn_dist
            )
            if lane_valid_configs and lane_valid_configs[0].get("junction") is True:
                junction_count += 1
                lane_valid_configs = [lane_valid_configs[0]]
                if junction_count > self.MAX_JUNCTION_SPAWNS:
                    continue
            valid_spawn_configs.extend(lane_valid_configs)

        # 5. 计算数量并 Shuffle
        total_length = sum([lane.length for lane in all_lanes])
        total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
        target_total_vehicles = int(math.floor(total_spawn_points * traffic_density))
        configured_total_vehicle_count = self.engine.global_config.get("total_vehicle_count", None)
        if configured_total_vehicle_count is not None:
            # _traffic_vehicles does not include ego, but total_vehicle_count does.
            target_total_vehicles = max(0, int(configured_total_vehicle_count) - 1)
        else:
            target_total_vehicles = self.RESPAWN_TARGET_VEHICLE_COUNT
        current_vehicle_count = len(self._traffic_vehicles)
        needed_vehicles = target_total_vehicles - current_vehicle_count
        if self.engine.global_config.get("debug_osm_route", False):
            print(
                "[osm traffic][spawn_count]",
                "total_vehicle_count=", configured_total_vehicle_count,
                "target_traffic_vehicles=", target_total_vehicles,
                "existing_traffic_vehicles=", current_vehicle_count,
                "needed_random_vehicles=", needed_vehicles,
            )

        if needed_vehicles <= 0 or len(valid_spawn_configs) == 0:
            return
            
        valid_spawn_configs.sort(key=lambda x: (str(x.get("spawn_lane_index", 0)), x.get("spawn_longitude", 0.0)))
        
        self.np_random.shuffle(valid_spawn_configs)
        real_spawn_count = min(needed_vehicles, len(valid_spawn_configs))
        selected = valid_spawn_configs[:real_spawn_count]

        # 6. 实例化
        from metadrive.plugins.policy.idm_policy_in_osm import IDMPolicyInOsm
        from metadrive.component.vehicle.vehicle_type import SVehicle
        
        for traffic_v_config in selected: 
            vehicle_type = SVehicle
            try:
                spawn_lane = map.road_network.get_lane(traffic_v_config["spawn_lane_index"])
            except Exception:
                spawn_lane = None
            target_speed_km_h, spawn_speed_m_s = self._sample_respawn_speed_profile(spawn_lane)
            traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
            traffic_v_config["idm_target_speed_km_h"] = target_speed_km_h
            if spawn_lane is not None:
                lane_speed_limit = self._lane_speed_limit_km_h(spawn_lane)
                if lane_speed_limit is not None:
                    traffic_v_config["idm_lane_speed_limit_km_h"] = lane_speed_limit
            traffic_v_config["spawn_velocity"] = [spawn_speed_m_s, 0]
            traffic_v_config["spawn_velocity_car_frame"] = True
            self._spawn_traffic_vehicle(
                vehicle_type,
                traffic_v_config,
                IDMPolicyInOsm,
                name=self._next_vehicle_name(),
            )
            self.name_id += 1

    def _filter_configs(self, lane, configs, existing_positions_map, ego_pos=None, min_spawn_dist=50.0):
        """
        双重过滤：
        1. 检查同车道是否有车 (纵向距离)
        2. 检查与自车的全局距离 (欧氏距离),会过滤在雷达范围内的生成点，防止“空降”干扰自车状态
        """
        valid_configs = []
        
        # 1. 准备同车道现有车辆数据
        current_lane_vehicles = existing_positions_map.get(lane.index, [])
        safe_gap = self.VEHICLE_GAP * 0.9 

        for config in configs:
            spawn_long = config['spawn_longitude']
            
            # --- Check A: 同车道重叠检测 ---
            is_overlap = False
            for exist_long in current_lane_vehicles:
                if abs(spawn_long - exist_long) < safe_gap:
                    is_overlap = True
                    break
            if is_overlap:
                continue # 跳过该点

            # --- Check B: 自车雷达范围检测 (新增) ---
            if ego_pos is not None:
                # 将 (Lane, Longitude) 转换为全局坐标 (x, y)
                # lane.position(long, lat) 返回的是 numpy array [x, y]
                spawn_point_global = lane.position(spawn_long, 0)
                
                # 计算欧氏距离
                dist = np.linalg.norm(spawn_point_global - ego_pos)
                
                if dist < min_spawn_dist:
                    # 距离太近，在自车视野内，跳过生成
                    continue

            # 通过所有检查，加入候选列表
            valid_configs.append(config)
        
        return valid_configs

    def get_near_lanes(self, ego_lane, max_depth):
        ego_lane_index = ego_lane.index
        processed_lanes = set()  # 已处理的车道
        lanes_to_process = set([ego_lane_index])  # 待处理的车道
        near_lanes = set([ego_lane_index])  # 结果集：所有相邻车道

        current_depth = 0
        
        while lanes_to_process and current_depth < max_depth:
            next_lanes = set()  # 下一轮要处理的车道
        
            for lane_index in lanes_to_process:
                if lane_index in processed_lanes:
                    continue
                
                processed_lanes.add(lane_index)
                lane = self.engine.map_manager.current_map.road_network.get_lane(lane_index)
                
                # 添加左右车道
                next_lanes.update(lane.left_lanes)
                next_lanes.update(lane.right_lanes)
                
                # 添加入口和出口车道
                next_lanes.update(lane.entry_lanes)
                next_lanes.update(lane.exit_lanes)
            
            # 更新近邻车道集合
            near_lanes.update(next_lanes)
            lanes_to_process = next_lanes - processed_lanes
            current_depth += 1

        near_lanes = [self.engine.map_manager.current_map.road_network.get_lane(idx) for idx in near_lanes]
        return near_lanes
        
    
    def calculate_centerline_using_medial_axis(self, polygon):
        """
        使用多边形中轴线算法计算车道中心线，并计算每个点的航向角
        
        参数:
        - polygon: 车道多边形点列表，按顺时针或逆时针顺序排列
        
        返回:
        - centerline: 中心线点列表，每个点为 [x, y]
        - headings: 对应每个中心线点的航向角列表（弧度）
        """
        import numpy as np
        from skimage.morphology import medial_axis
        import cv2
        import matplotlib.pyplot as plt
        import threading
        
        # 确定多边形边界框
        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_y = max(p[1] for p in polygon)
        
        # 调整分辨率以获得更准确的中轴线
        resolution = 1.0  # 每米的像素数
        width = int((max_x - min_x) * resolution) + 10
        height = int((max_y - min_y) * resolution) + 10
        
        # 创建二值图像
        img = np.zeros((height, width), dtype=np.uint8)
        
        # 将多边形转换为图像坐标
        polygon_img = []
        for p in polygon:
            x = int((p[0] - min_x) * resolution) + 5
            y = int((p[1] - min_y) * resolution) + 5
            polygon_img.append([x, y])
        
        # 填充多边形
        cv2.fillPoly(img, [np.array(polygon_img, dtype=np.int32)], 255)
        
        # 计算中轴线
        skel, dist = medial_axis(img, return_distance=True)
        
        # 提取中轴线点
        centerline_img = np.argwhere(skel)
        
        # 将图像坐标转换回实际坐标
        centerline_raw = []
        for p in centerline_img:
            y, x = p
            real_x = (x - 5) / resolution + min_x
            real_y = (y - 5) / resolution + min_y
            centerline_raw.append([real_x, real_y])
        
        # 对中轴线点进行排序，使其按照路径的连续性排列
        if len(centerline_raw) > 1:
            centerline = self._sort_centerline_points(centerline_raw)
        else:
            centerline = centerline_raw
        
        # 计算每个点的航向角
        headings = self._calculate_headings(centerline)
        
        return np.array(centerline), np.array(headings)
    
    def _sort_centerline_points(self, points):
        """
        对中轴线点进行排序，使其按照路径的连续性排列
        
        参数:
        - points: 未排序的中轴线点列表
        
        返回:
        - sorted_points: 排序后的点列表
        """
        if len(points) <= 1:
            return points
        
        # 使用贪心算法，从一个端点开始，依次选择最近的点
        sorted_points = [points[0]]
        remaining_points = points[1:]
        
        while remaining_points:
            current_point = sorted_points[-1]
            # 找到距离当前点最近的点
            distances = [np.linalg.norm(np.array(p) - np.array(current_point)) for p in remaining_points]
            nearest_idx = np.argmin(distances)
            
            # 将最近的点添加到排序列表中
            sorted_points.append(remaining_points[nearest_idx])
            remaining_points.pop(nearest_idx)
        
        return sorted_points
    
    def _calculate_headings(self, centerline):
        """
        计算中心线上每个点的航向角
        
        参数:
        - centerline: 排序后的中心线点列表
        
        返回:
        - headings: 每个点的航向角列表（弧度）
        """
        if len(centerline) <= 1:
            return [0.0] * len(centerline)
        
        headings = []
        
        for i in range(len(centerline)):
            if i == 0:
                # 第一个点：使用当前点到下一个点的方向
                dx = centerline[i+1][0] - centerline[i][0]
                dy = centerline[i+1][1] - centerline[i][1]
            elif i == len(centerline) - 1:
                # 最后一个点：使用前一个点到当前点的方向
                dx = centerline[i][0] - centerline[i-1][0]
                dy = centerline[i][1] - centerline[i-1][1]
            else:
                # 中间点：使用前一个点到下一个点的方向（更平滑）
                dx = centerline[i+1][0] - centerline[i-1][0]
                dy = centerline[i+1][1] - centerline[i-1][1]
            
            # 计算航向角（弧度）
            heading = np.arctan2(dy, dx)
            headings.append(heading)
        
        return headings
    

    # def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
    #     """
    #     Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
    #     :param map: Map
    #     :param traffic_density: it can be adjusted each episode
    #     :return: None
    #     """
    #     vehicle_num = 0
    #     type = 'ADBV'
    #     for block in map.blocks[1:]: # block即地图分块，如X，C，O
    #         trigger_lanes = block.get_intermediate_spawn_lanes() # 提出候选位置供新车辆生成，X和其他block的该函数不，写在不同的文件中
    #         if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
    #             neg_lanes = block.block_network.get_negative_lanes()
    #             self.np_random.shuffle(neg_lanes)
    #             trigger_lanes += neg_lanes
    #         potential_vehicle_configs = []
    #         for lanes in trigger_lanes: # 大车道road
    #             for l in lanes: # 大车道中的每条车道lane
    #                 if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
    #                     continue
    #                 potential_vehicle_configs += self._propose_vehicle_configs(l)

    #         # 在一个block中生成的车辆数目
    #         total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes]) # 所有车道长度总和
    #         total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP)) # 所有车道可生成的车辆位置总数
    #         total_vehicles = int(math.floor(total_spawn_points * traffic_density)) # 在一个block中实际生成车辆数目

    #         # Generate vehicles!
    #         vehicles_on_block = []
    #         self.np_random.shuffle(potential_vehicle_configs) # 打乱顺序，操作等同于随机选取
    #         selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))] # 在打乱的可生成位置中取前x位，其中x为所有车道中可生成车辆数目和要求生成车辆数目中的最小值
    #         # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))

    #         n=0
    #         for v_config in selected:
    #             if type == 'ADBV':
    #                 print("ADBV!!")
    #                 # v_config['spawn_longitude']=-30 # 周车生成时的起点长度start设置为50，故就算这里设置为0也还是会在道路中端生成车辆。但是自车的start不是50。
    #                 # v_config['spawn_lateral']=5.25
    #                 vehicle_type = self.random_vehicle_type() # 根据设置概率随机选择车型,同样是是BaseVehicle基类
    #                 v_config.update(self.engine.global_config["traffic_vehicle_config"]) # 为v_config添加'enable_reverse'，'show_navi_mark'等属性
    #                 random_v = self.spawn_object(vehicle_type, vehicle_config=v_config,TYPE=type) # 返回的是车辆对象。通过添加position=[10,10.5],heading=0(弧度值)指定周车生成位置
    #                 self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
    #                 vehicles_on_block.append(random_v.name) # 只记录名字。name和id好像是一样的
    #                 type = 'BV'
    #             else:
    #                 vehicle_type = self.random_vehicle_type() # 根据设置概率随机选择车型BaseVehicle基类
    #                 v_config.update(self.engine.global_config["traffic_vehicle_config"]) # 为v_config添加'enable_reverse'，'show_navi_mark'等属性
    #                 random_v = self.spawn_object(vehicle_type, vehicle_config=v_config) # 返回的是车辆对象。通过添加position=[50,10.5],heading=0指定周车生成位置
    #                 self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
    #                 vehicles_on_block.append(random_v.name) # 只记录名字。name和id好像是一样的

    #         trigger_road = block.pre_block_socket.positive_road # 当前block之前的连接block的右向车道。可能用于多block首尾连接
    #         block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)
    #         self.block_triggered_vehicles.append(block_vehicles) # 所有block中生成的车辆与其road的信息

    #         vehicle_num += len(vehicles_on_block) # 每一个block中最终生成的车辆数目
    #     self.block_triggered_vehicles.reverse()# 列表反向排序

    def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
        # 选择合适的道路作为新车辆生成的候选位置
        trigger_lanes = self.respawn_lanes
        potential_vehicle_configs = []
        for l in trigger_lanes:
            # 如果该车道在事故车道中，则跳过
            if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                continue
            potential_vehicle_configs += self._propose_vehicle_configs(l)

        # 生成数量
        total_length = sum([lane.length for lane in trigger_lanes])
        total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
        total_vehicles = int(math.floor(total_spawn_points * traffic_density))

        # Generate vehicles!
        self.np_random.shuffle(potential_vehicle_configs)
        selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]
        from metadrive.policy.idm_policy import MyEgoPolicy
        # print("===== We are initializing {} vehicles =====".format(len(selected)))
        # print("Current seed: ", self.engine.global_random_seed)
        for v_config in selected:
            vehicle_type = self.random_vehicle_type()
            v_config.update(self.engine.global_config["traffic_vehicle_config"])
            self._spawn_traffic_vehicle(vehicle_type, v_config, MyEgoPolicy)

        # 因为原本的地图是结构化的，是block组成的，所以trigger模式可以实现，当车辆到trigger车辆所在block的前一个block时，触发该block的车辆生成
        # 但是真实的地图没有block的概念，所以没法简单实现。若是直接使用respawn_lanes作为触发道路，那么并列车道的还是不会被触发
        # TODO
        # trigger_road = block.pre_block_socket.positive_road
        # block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

        # self.block_triggered_vehicles.append(block_vehicles)
        # vehicle_num += len(vehicles_on_block)
        # self.block_triggered_vehicles.reverse()


    def _get_available_respawn_lanes(self, map: BaseMap) -> list:
        """
        Used to find some respawn lanes
        :param map: select spawn lanes from this map
        :return: respawn_lanes
        """
        respawn_lanes = []
        respawn_roads = []
        for block in map.blocks:
            roads = block.get_respawn_roads()
            for road in roads:
                if road in respawn_roads: # 如果存在，则将其移除（这意味着该道路在多个区块中可能被重复引用）。如果不存在，则将其添加。
                    respawn_roads.remove(road)
                else:
                    respawn_roads.append(road)
        for road in respawn_roads:
            respawn_lanes += road.get_lanes(map.road_network)
        return respawn_lanes

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
        return vehicle_type

    def destroy(self) -> None:
        """
        Destory func, release resource
        :return: None
        """
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        # current map

        # traffic vehicle list
        self._traffic_vehicles = None
        self.block_triggered_vehicles = None

        # traffic property
        self.mode = None
        self.random_traffic = None
        self.density = None

    def __del__(self):
        logger.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self._traffic_vehicles.__repr__()

    @property
    def vehicles(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    def seed(self, random_seed):
        if not self.random_traffic:
            super(TrafficManagerInOsm, self).seed(random_seed)

    @property
    def current_map(self):
        return self.engine.map_manager.current_map

    def get_state(self):
        ret = super(TrafficManagerInOsm, self).get_state()
        ret.update(self._get_record_id_mapping())
        ret["_traffic_vehicles"] = [v.name for v in self._traffic_vehicles]
        flat = []
        for b_v in self.block_triggered_vehicles:
            flat.append((b_v.trigger_road.start_node, b_v.trigger_road.end_node, b_v.vehicles))
        ret["block_triggered_vehicles"] = flat
        return ret

    def set_state(self, state: dict, old_name_to_current=None):
        super(TrafficManagerInOsm, self).set_state(state, old_name_to_current)
        self._traffic_vehicles = list(
            self.get_objects([old_name_to_current[name] for name in state["_traffic_vehicles"]]).values()
        )
        self.block_triggered_vehicles = [
            BlockVehicles(trigger_road=Road(s, e), vehicles=[old_name_to_current[name] for name in v])
            for s, e, v in state["block_triggered_vehicles"]
        ]


    # def adopt_vehicle_as_traffic(self, vehicle, policy_cls):
    #     """
    #     接收一个已从agent降级的车辆，使其成为交通车辆并附加交通策略。
    #     """
    #     if vehicle in self._traffic_vehicles:
    #         return
    #     self._traffic_vehicles.append(vehicle)
    #     self.add_policy(vehicle.id, policy_cls, vehicle, self.generate_seed())
    #     vehicle.set_static(False)
