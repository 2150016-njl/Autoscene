from collections import deque
from types import SimpleNamespace
from metadrive.constants import CamMask

import numpy as np
from panda3d.core import NodePath, Material
from metadrive.engine.logger import get_logger
from metadrive.component.lane.point_lane import PointLane
from metadrive.component.navigation_module.base_navigation import BaseNavigation
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.coordinates_shift import panda_vector
from metadrive.utils.math import norm, clip
from metadrive.utils.math import panda_vector
from metadrive.utils.math import wrap_to_pi
from metadrive.utils.pg.utils import ray_localization
from metadrive.utils import get_np_random
from metadrive.type import MetaDriveType
# from metadrive.CHT.policy.rL_planning_policy import RLPlanningPolicy


def _reference_trajectory_to_render_path(reference_trajectory, sample_interval=2):
    if reference_trajectory is None:
        return None

    interval = max(float(sample_interval), 0.1)
    points = np.asarray(reference_trajectory.get_polyline(interval=interval, lateral=0), dtype=float)
    if points.ndim != 2 or len(points) == 0 or points.shape[1] < 2:
        return None

    points = points[:, :2]
    if len(points) >= 2:
        deltas = np.diff(points, axis=0)
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        headings = np.append(headings, headings[-1])
    else:
        headings = np.zeros(1, dtype=float)

    curvature = np.zeros(len(points), dtype=float)
    return [points[:, 0].tolist(), points[:, 1].tolist(), headings.tolist(), curvature.tolist()]


class OsmTrajectoryNavigation(BaseNavigation):
    """
    此类适用于类似OSM地图格式的地图数据
    需包含车道之间的连接关系(前后左右)和车道的polyline信息
    以生成的全局路径为索引
    为了和trajectory_navigation保持一致，修改osm的维度，这是基于edge_navigation和trajectory_navigation的结合体
    """
    DISCRETE_LEN = 2  # m 把连续车道/参考轨迹按固定距离切成点
    CHECK_POINT_INFO_DIM = 5 # 2 指定每个检查点所包含的导航信息的数量或维度
    NUM_WAY_POINT = 4 # 10 作为obs的参数，用作导航信息的检查点个数，会影响navi_info的维度，维度=NUM_WAY_POINT*CHECK_POINT_INFO_DIM
    NUM_NAVI_NODE = 5 # 原代码是使用NUM_WAY_POINT作为显示的检查点个数，但是会影响模型参数个数，故使用新变量
    NAVI_POINT_DIST = 30  # m, used to clip value, should be greater than DISCRETE_LEN * MAX_NUM_WAY_POINT

    def __init__(
        self,
        show_navi_mark: bool = False,
        show_dest_mark=False,
        show_line_to_dest=False,
        panda_color=None,
        name=None,
        vehicle_config=None
    ):
        super(OsmTrajectoryNavigation, self).__init__(
            show_navi_mark=show_navi_mark,
            show_dest_mark=show_dest_mark, # 需要把在这的False修改掉，不然无法传递参数
            show_line_to_dest=show_line_to_dest,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        if self.origin is not None:
            self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)

        self.rng = get_np_random()

        self._route_completion = 0
        self.checkpoints = None  # All check points
        self._dest_mode = None  # 记录当前导航模式
        self._global_route_locked = False  # 一次性全局路径规划后锁定，禁止重规划
        self._reference_trajectory = None # 点坐标序列全局路径
        self._initial_reference_trajectory = None
        self._initial_debug_navigation_global_path = None
        self._initial_route_lane_ids = None
        self._fixed_start_lane_index = None
        self._fixed_destination_lane_index = None
        self.debug_navigation_global_path = None
        self._debug_path_vehicle = None
        # for compatibility
        self.next_ref_lanes = None

        # override the show navi mark function here
        self._navi_point_model = None # 导航点模型，就是方块模型
        self._ckpt_vis_models = None # 检查点可视化模型
        if show_navi_mark and self._show_navi_info and self.vehicle_config['vehicle_model'] == 'default':
            # self._ckpt_vis_models = [NodePath(str(i)) for i in range(self.NUM_WAY_POINT)]
            self._ckpt_vis_models = [NodePath(str(i)) for i in range(self.NUM_NAVI_NODE)]
            for model in self._ckpt_vis_models:
                if self._navi_point_model is None:
                    self._navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                    self._navi_point_model.setScale(0.5)
                    # if self.engine.use_render_pipeline:
                    material = Material()
                    # material.setBaseColor((19 / 255, 212 / 255, 237 / 255, 1))
                    material.setShininess(16)
                    material.setEmission((0.2, 0.2, 0.2, 0.2))
                    self._navi_point_model.setMaterial(material, True)
                    # self._navi_point_model.setColor(1, 0, 0, 1) # 红色。若设置了材质，通过节点设置颜色需要在材质设置之后
                self._navi_point_model.instanceTo(model) # 将导航点模型实例化到当前的可视化模型上，就是在检查点处生成方块模型
                model.reparentTo(self.origin) # 将可视化模型重新添加到场景中的一个父节点

        # should be updated every step after calling update_localization
        self.last_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_current_heading_theta_at_long = deque([0.0, 0.0], maxlen=2)
        self.record_lane_index='0'
        self.set_force_calculate_lane_index(True) # 会设置self.FORCE_CALCULATE = True，强制每一步都计算当前车道索引，适用于在复杂路口容易定位错误的情况

    def _get_excluded_navigation_lane_types(self):
        if self.vehicle_config.get("filter_bike_lanes_in_navigation", True):
            return {MetaDriveType.LANE_BIKE_LANE}
        return set()

    def _filter_navigation_lane_candidates(self, possible_lanes):
        excluded_lane_types = self._get_excluded_navigation_lane_types()
        if not excluded_lane_types:
            return possible_lanes
        return [
            lane_candidate for lane_candidate in possible_lanes
            if not self.map.road_network.is_lane_excluded(lane_candidate[1], excluded_lane_types)
        ]

    def _get_navigation_exit_lanes(self, lane):
        exit_lanes = self.map.road_network.filter_lane_indices(
            getattr(lane, "exit_lanes", []), excluded_lane_types=self._get_excluded_navigation_lane_types()
        )
        valid_exits = []
        for lane_index in exit_lanes:
            next_lane = self.map.road_network.get_lane(lane_index)
            if self._is_valid_exit_lane_transition(lane, next_lane):
                valid_exits.append(lane_index)
        return valid_exits

    def _get_navigation_peer_lanes(self, lane_index):
        return self.map.road_network.get_peer_lanes_from_index(
            lane_index, excluded_lane_types=self._get_excluded_navigation_lane_types()
        )

    @staticmethod
    def _lane_index_matches(lane_index, lane):
        return str(lane_index) == str(getattr(lane, "index", None))

    def _lane_index_in_lanes(self, lane_index, lanes):
        return any(self._lane_index_matches(lane_index, lane) for lane in (lanes or []))

    def _is_lane_index_on_route(self, lane_index):
        if lane_index in self.checkpoints:
            return True
        if self._lane_index_in_lanes(lane_index, self.current_ref_lanes):
            return True
        if self._lane_index_in_lanes(lane_index, self.next_ref_lanes):
            return True
        return False

    def _debug_route_enabled(self, vehicle):
        if not self.engine.global_config.get("debug_osm_route", False):
            return False
        if self.engine.global_config.get("debug_osm_route_all_vehicles", False):
            return True
        agents = getattr(self.engine, "agents", {}) or {}
        return getattr(vehicle, "id", None) in agents

    def _is_agent_vehicle(self, vehicle):
        agents = getattr(self.engine, "agents", {}) or {}
        vehicle_id = getattr(vehicle, "id", None)
        vehicle_name = getattr(vehicle, "name", None)
        return vehicle in agents.values() or vehicle_id in agents or vehicle_name in agents

    def _debug_route_log(self, vehicle, event, **kwargs):
        if not self._debug_route_enabled(vehicle):
            return
        parts = [
            f"[osm route][{event}]",
            f"vehicle={getattr(vehicle, 'name', getattr(vehicle, 'id', None))}",
            f"id={getattr(vehicle, 'id', None)}",
            f"step={getattr(self.engine, 'episode_step', None)}",
        ]
        for key, value in kwargs.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts))

    def _has_valid_route(self):
        return bool(self.checkpoints) and self._target_checkpoints_index is not None

    def reset(self, vehicle):
        """
        获取导航点；设置self._current_lane；
        """
        self.route_completion = 0
        self._debug_path_vehicle = vehicle
        self._fixed_start_lane_index = str(vehicle.config.get("spawn_lane_index", "")) or None
        destination = vehicle.config.get("destination", None)
        self._fixed_destination_lane_index = str(destination) if destination is not None else None
        current_lane = self.get_checkpoints(vehicle, allow_off_lane_fallback=True)
        if current_lane is None or not self.checkpoints:
            from metadrive.utils.error_class import NavigationError
            raise NavigationError("Can't initialize OSM navigation because the vehicle is off-road.")

        # 若没有下一个车道，则当前车道是最后一个车道，下一个导航车道也是当前车道，为了格式统一
        if len(self.checkpoints) == 1:
            self.checkpoints.append(self.checkpoints[0])
        # super(OsmTrajectoryNavigation, self).reset(current_lane=self.reference_trajectory)
        super(OsmTrajectoryNavigation, self).reset(current_lane)
        self.set_route()
        lock_route = self.vehicle_config.get("lock_destination_route", None)
        if lock_route is None:
            lock_route = self._dest_mode == 1
        self._global_route_locked = bool(lock_route)
        self._capture_initial_route()
        self._refresh_debug_navigation_path()
        self._attach_debug_navigation_path(vehicle)
        self._debug_route_log(
            vehicle,
            "reset",
            spawn_lane=vehicle.config.get("spawn_lane_index", None),
            destination=self._fixed_destination_lane_index,
            current_lane=getattr(current_lane, "index", None),
            checkpoints=self.checkpoints[:8],
            ref_len=round(self.reference_trajectory.length, 2) if self.reference_trajectory else None,
            initial_route=(self._initial_route_lane_ids or [])[:8],
        )

    def get_current_lane_modified(self, vehicle, return_all=False, use_heading_filter=True, allow_off_lane_fallback=False):
        possible_lanes, on_lane = ray_localization(
            vehicle.heading, vehicle.position, self.engine, use_heading_filter=use_heading_filter,return_on_lane=True
        )
        possible_lanes = self._filter_navigation_lane_candidates(possible_lanes)

        if len(possible_lanes) == 0:
            closest_lane_index, on_lane = self.map.road_network.get_closest_lane_index(
                vehicle.position,
                vehicle.heading_theta,
                vehicle,
                use_heading_filter=use_heading_filter,
                return_all=return_all,
                excluded_lane_types=self._get_excluded_navigation_lane_types()
            )
            # # 绘制射线和车道边界的多边形，帮助调试定位问题
            # if on_lane is False:
            #     closest_lane_index, on_lane = self.map.road_network.get_closest_lane_index(
            #         vehicle.position,
            #         vehicle.heading_theta,
            #         vehicle,
            #         use_heading_filter=use_heading_filter,
            #         return_all=return_all,
            #         draw_polygon=True,
            #         excluded_lane_types=self._get_excluded_navigation_lane_types()
            #     )
            if closest_lane_index is None:
                return ([], False) if return_all else (None, None, False)
            if on_lane is False and not allow_off_lane_fallback:
                return ([], False) if return_all else (None, None, False)

            lane_indices = closest_lane_index if return_all else [closest_lane_index]
            ranked_lanes = []
            for lane_index in lane_indices:
                lane = self.map.road_network.get_lane(lane_index)
                dist = lane.distance(vehicle.position)
                angle_diff = 0.0
                if use_heading_filter:
                    long, _ = lane.local_coordinates(vehicle.position)
                    angle_diff = abs(wrap_to_pi(lane.heading_theta_at(long) - vehicle.heading_theta))
                ranked_lanes.append((lane, lane_index, dist, angle_diff))

            ranked_lanes.sort(key=lambda item: (item[3], item[2]) if use_heading_filter else (item[2], ))
            possible_lanes = [(lane, lane_index, dist) for lane, lane_index, dist, _ in ranked_lanes]
            possible_lanes = self._filter_navigation_lane_candidates(possible_lanes)

        if return_all:
            return possible_lanes, on_lane
        if len(possible_lanes) == 0:
            return None, None, on_lane

        current_lane, current_index = possible_lanes[0][:-1]
        return current_lane, current_index, on_lane

    def _get_reset_fallback_lane(self, vehicle):
        fallback_indices = []
        spawn_lane_index = vehicle.config.get("spawn_lane_index", None)
        if spawn_lane_index is not None:
            fallback_indices.append(spawn_lane_index)
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is not None and lane_index not in fallback_indices:
            fallback_indices.append(lane_index)

        for lane_index in fallback_indices:
            try:
                lane = self.map.road_network.get_lane(lane_index)
            except Exception:
                lane = None
            if lane is not None:
                return lane, lane_index
        return None, None

    def get_checkpoints(self, vehicle, allow_off_lane_fallback=False):
        """
        生成车辆行驶路径保存在self.checkpoints中；获取当前所在车道current_lane；获取当前车道和下一个车道的相邻车道
        """
        current_lane = None
        current_index = None
        if allow_off_lane_fallback:
            current_lane, current_index = self._get_reset_fallback_lane(vehicle)
        if current_lane is None:
            current_lane, current_index, on_lane = self.get_current_lane_modified(
                vehicle, allow_off_lane_fallback=allow_off_lane_fallback
            )
        if current_lane is None:
            if allow_off_lane_fallback:
                current_lane, current_index = self._get_reset_fallback_lane(vehicle)
            if current_lane is None:
                vehicle.on_lane = False
                vehicle.out_of_route = True
                self.checkpoints = None
                self.final_lane = None
                return None
        
        self._dest_mode = self.vehicle_config.get('dest_mode', 1)
        if self._dest_mode == 1: # dest_mode=1表示给定目的地寻找路径，所以需要传入目的地参数
            destination = self._fixed_destination_lane_index or self.vehicle_config["destination"]
            ret = self.mode_2_navigation(self._dest_mode,current_lane,current_index,destination)
        else:
            ret = self.mode_2_navigation(self._dest_mode,current_lane,current_index)
        self._debug_route_log(
            vehicle,
            "get_checkpoints",
            dest_mode=self._dest_mode,
            start=current_index,
            fixed_start=self._fixed_start_lane_index,
            destination=self._fixed_destination_lane_index or self.vehicle_config.get("destination", None),
            checkpoints=(self.checkpoints or [])[:8],
        )
        return ret

    def _mark_vehicle_navigation_invalid(self, vehicle):
        vehicle.on_lane = False
        vehicle.out_of_route = True
        self.checkpoints = None
        self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._target_checkpoints_index = None
        self._route_completion = 0.0
        if self._reference_trajectory is not None:
            self._reference_trajectory.destroy()
        self._reference_trajectory = None
        self._refresh_debug_navigation_path()

    def mode_2_navigation(self,dest_mode,current_lane,current_index,dest=None):
        previous_checkpoints = list(self.checkpoints or [])
        self.checkpoints = []

        # ----------不同的导航方式----------
        if dest_mode == 1:
            # 方法一：给定目的地寻找路径，有时候会很慢
            new_checkpoints = self.map.road_network.shortest_path(
                current_lane.index,
                dest,
                excluded_lane_types=self._get_excluded_navigation_lane_types()
            )
            if new_checkpoints:
                self.checkpoints = new_checkpoints
            elif previous_checkpoints:
                self.checkpoints = previous_checkpoints
            else:
                self.checkpoints = [current_index]
        elif dest_mode == 2:
            # 每到一个新路径时才选择下一个路径，更合理
            self.checkpoints.append(current_index)
            last_lane = current_lane
            exit_lanes = self._get_navigation_exit_lanes(last_lane)
            if exit_lanes:
                current_index = str(self.rng.choice(exit_lanes))
                # current_index = last_lane.exit_lanes[0]
                current_lane = self.map.road_network.get_lane(current_index)
                if current_index in self.map.road_network.graph and current_index not in self.checkpoints:
                    self.checkpoints.append(current_index)
                    last_lane = current_lane

        elif dest_mode == 3:
            # 一次性添加5条路径到导航路径中
            self.checkpoints.append(current_index)
            last_lane = current_lane
            while len(self.checkpoints) < 5:
                exit_lanes = self._get_navigation_exit_lanes(last_lane)
                if not exit_lanes:
                    break
                occ_lane = False
                exit_nums = len(exit_lanes)
                for i in range(exit_nums):
                    current_index = str(exit_lanes[i])
                    if current_index in self.map.road_network.graph and current_index not in self.checkpoints:
                        self.checkpoints.append(current_index)
                        last_lane = self.map.road_network.get_lane(current_index)
                        occ_lane = True
                        break
                if not occ_lane:
                    break

        elif dest_mode == 4:
            # 一直添加路径直到断头路
            self.checkpoints.append(current_index)
            last_lane = current_lane
            while last_lane is not None:
                exit_lanes = self._get_navigation_exit_lanes(last_lane)
                if not exit_lanes:
                    break
                occ_lane = False
                for next_lane_index in self.rng.permutation(exit_lanes):
                    current_index = str(next_lane_index)
                    if current_index in self.map.road_network.graph and current_index not in self.checkpoints:
                        self.checkpoints.append(current_index)
                        last_lane = self.map.road_network.get_lane(current_index)
                        occ_lane = True
                        break
                if not occ_lane:
                    break

        elif dest_mode == 5:
            # 始终以自车下一个checkpoint作为对抗车导航终点。存在问题
            self.checkpoints = self.map.road_network.shortest_path(
                current_lane.index,
                next(iter(self.engine.agent_manager.spawned_objects.values())).navigation.next_checkpoint_lane_index,
                excluded_lane_types=self._get_excluded_navigation_lane_types()
            )
            if not self.checkpoints:
                self.checkpoints = [current_index]

        elif dest_mode == 6:
            # 对抗车的导航路径 = 对抗车到自车的导航路径 + 自车的导航路径
            ego_checkpoints = next(iter(self.engine.agent_manager.spawned_objects.values())).navigation.checkpoints

            self.checkpoints = self.map.road_network.shortest_path(
                current_lane.index,
                ego_checkpoints[0],
                excluded_lane_types=self._get_excluded_navigation_lane_types()
            )
            if not self.checkpoints:
                self.checkpoints = ego_checkpoints
            else:
                self.checkpoints += ego_checkpoints
        if not self.checkpoints:
            self.checkpoints = [current_index]
        self.final_lane = self.map.road_network.get_lane(self.checkpoints[-1])
        return current_lane
    

    @property
    def reference_trajectory(self):
        return self._reference_trajectory

    def _attach_debug_navigation_path(self, vehicle=None):
        if vehicle is not None:
            self._debug_path_vehicle = vehicle
        if self._debug_path_vehicle is not None:
            self._debug_path_vehicle.navigation_global_path = self.debug_navigation_global_path

    def _refresh_debug_navigation_path(self):
        if self.vehicle_config.get("render_initial_route", True) and self._initial_debug_navigation_global_path is not None:
            self.debug_navigation_global_path = self._initial_debug_navigation_global_path
        else:
            self.debug_navigation_global_path = _reference_trajectory_to_render_path(
                self.reference_trajectory, sample_interval=self.DISCRETE_LEN
            )
        self._attach_debug_navigation_path()
        return self.debug_navigation_global_path

    def _capture_initial_route(self):
        self._initial_route_lane_ids = list(self.checkpoints or [])
        self._initial_reference_trajectory = self.reference_trajectory
        self._initial_debug_navigation_global_path = _reference_trajectory_to_render_path(
            self._initial_reference_trajectory, sample_interval=self.DISCRETE_LEN
        )

    def _append_route_points(self, route_points, new_points):
        if new_points is None or len(new_points) == 0:
            return

        if isinstance(new_points, np.ndarray):
            new_points = new_points.tolist()

        if len(route_points) > 0 and np.allclose(route_points[-1], new_points[0], atol=1e-3):
            new_points = new_points[1:]
        route_points.extend(new_points)

    def _get_lane_graph_info(self, checkpoint):
        road_network = getattr(self.map, "road_network", None)
        if road_network is None or not hasattr(road_network, "graph"):
            return None

        graph_key = checkpoint
        if hasattr(road_network, "_resolve_graph_key"):
            graph_key = road_network._resolve_graph_key(checkpoint)
        elif checkpoint is not None:
            graph_key = str(checkpoint)
        return road_network.graph.get(graph_key)

    def _is_lateral_transition(self, from_checkpoint, to_checkpoint):
        lane_info = self._get_lane_graph_info(from_checkpoint)
        if lane_info is None:
            return False

        candidate = str(to_checkpoint)
        left_lanes = {str(lane_id) for lane_id in (lane_info.left_lanes or [])}
        right_lanes = {str(lane_id) for lane_id in (lane_info.right_lanes or [])}
        return candidate in left_lanes or candidate in right_lanes

    def _are_lanes_endpoint_connected(self, from_lane, to_lane):
        if from_lane is None or to_lane is None:
            return False

        from_end = np.asarray(getattr(from_lane, "end", from_lane.position(from_lane.length, 0.0)), dtype=float)[:2]
        to_start = np.asarray(getattr(to_lane, "start", to_lane.position(0.0, 0.0)), dtype=float)[:2]
        return np.linalg.norm(from_end - to_start) <= max(1.0, 0.5 * self.DISCRETE_LEN)

    def _is_valid_exit_lane_transition(self, from_lane, to_lane):
        if from_lane is None or to_lane is None:
            return False

        from_end = np.asarray(getattr(from_lane, "end", from_lane.position(from_lane.length, 0.0)), dtype=float)[:2]
        to_start = np.asarray(getattr(to_lane, "start", to_lane.position(0.0, 0.0)), dtype=float)[:2]
        gap = float(np.linalg.norm(from_end - to_start))
        max_gap = float(self.engine.global_config.get("osm_exit_lane_max_connection_gap_m", 8.0))
        if gap > max_gap:
            return False

        from_long = max(float(getattr(from_lane, "length", 0.0) or 0.0) - 0.1, 0.0)
        to_long = min(0.1, float(getattr(to_lane, "length", 0.0) or 0.0))
        from_heading = float(from_lane.heading_theta_at(from_long))
        to_heading = float(to_lane.heading_theta_at(to_long))
        max_heading_diff = float(
            self.engine.global_config.get("osm_exit_lane_max_heading_diff_rad", np.deg2rad(120.0))
        )
        return abs(wrap_to_pi(from_heading - to_heading)) <= max_heading_diff

    def _have_similar_heading(self, from_lane, to_lane):
        if from_lane is None or to_lane is None:
            return False

        from_long = max(float(getattr(from_lane, "length", 0.0) or 0.0) - 0.1, 0.0)
        to_long = min(0.1, float(getattr(to_lane, "length", 0.0) or 0.0))
        from_heading = float(from_lane.heading_theta_at(from_long))
        to_heading = float(to_lane.heading_theta_at(to_long))
        return abs(wrap_to_pi(from_heading - to_heading)) <= np.deg2rad(35.0)

    def _should_use_smooth_transition(self, from_checkpoint, to_checkpoint):
        from_lane = self.map.road_network.get_lane(from_checkpoint)
        to_lane = self.map.road_network.get_lane(to_checkpoint)
        if from_lane is None or to_lane is None:
            return False

        if self._are_lanes_endpoint_connected(from_lane, to_lane):
            return False

        if self._is_lateral_transition(from_checkpoint, to_checkpoint):
            return True

        return False

    def _build_lane_change_transition_points(self, start_lane, end_lane):
        if start_lane is None or end_lane is None:
            return []

        transition_length = max(
            float(getattr(start_lane, "length", 0.0) or 0.0),
            float(getattr(end_lane, "length", 0.0) or 0.0),
        )
        sample_count = max(int(np.ceil(max(transition_length, self.DISCRETE_LEN) / self.DISCRETE_LEN)), 2)

        points = []
        for idx in range(sample_count + 1):
            t = idx / sample_count
            weight = t * t * (3.0 - 2.0 * t)
            start_point = np.asarray(start_lane.position(t * start_lane.length, 0.0), dtype=float)[:2]
            end_point = np.asarray(end_lane.position(t * end_lane.length, 0.0), dtype=float)[:2]
            blended = (1.0 - weight) * start_point + weight * end_point
            if len(points) == 0 or not np.allclose(points[-1], blended, atol=1e-3):
                points.append(blended.tolist())
        return points

    def _build_reference_trajectory(self):
        if not self.checkpoints:
            return None

        route_points = []
        lane_widths = []
        checkpoints = []
        for checkpoint in self.checkpoints:
            if len(checkpoints) == 0 or checkpoint != checkpoints[-1]:
                checkpoints.append(checkpoint)

        idx = 0
        while idx < len(checkpoints):
            checkpoint = checkpoints[idx]
            lane = self.map.road_network.get_lane(checkpoint)
            if lane is None:
                idx += 1
                continue

            lane_change_end_idx = idx
            while (
                lane_change_end_idx + 1 < len(checkpoints)
                and self._should_use_smooth_transition(
                    checkpoints[lane_change_end_idx], checkpoints[lane_change_end_idx + 1]
                )
            ):
                lane_change_end_idx += 1

            if lane_change_end_idx > idx:
                end_lane = self.map.road_network.get_lane(checkpoints[lane_change_end_idx])
                lane_change_points = self._build_lane_change_transition_points(lane, end_lane)
                self._append_route_points(route_points, lane_change_points)
                lane_widths.extend([lane.width, end_lane.width])
                idx = lane_change_end_idx + 1
                continue

            lane_points = lane.get_polyline(self.DISCRETE_LEN, lateral=0)
            self._append_route_points(route_points, lane_points)
            lane_widths.append(lane.width)
            idx += 1

        if len(route_points) == 0:
            return None

        width = float(np.mean(lane_widths)) if lane_widths else self.final_lane.width
        trajectory = PointLane(route_points, width)
        trajectory.index = self.checkpoints[0]
        return trajectory

    def _project_to_reference_trajectory(self, position, default_heading):
        """
        
        """
        if self.reference_trajectory is None:
            return 0.0, 0.0, default_heading

        long, lat = self.reference_trajectory.local_coordinates(position)
        clipped_long = float(np.clip(long, 0.0, self.reference_trajectory.length))
        heading_theta_at_long = self.reference_trajectory.heading_theta_at(clipped_long)
        return clipped_long, lat, heading_theta_at_long


    def set_route(self):
        """
        设置一些基础变量，从给定的车道索引获取相邻的车道信息[当前车道，左边车道，右边车道]
        :param current_lane_index: start road node
        :param destination: end road node or end lane index
        :return: None
        """
        self._target_checkpoints_index = [0, 1] # 当前检查点和下一个检查点的序号
        self._navi_info.fill(0.0) # 导航状态信息

        # 从给定的车道索引获取相邻的车道信息[当前车道，左边车道，右边车道]
        if self.current_checkpoint_lane_index is None or self.next_checkpoint_lane_index is None:
            self.current_ref_lanes = []
            self.next_ref_lanes = []
            self._reference_trajectory = None
            self._refresh_debug_navigation_path()
            return
        self.current_ref_lanes = self._get_navigation_peer_lanes(self.current_checkpoint_lane_index)
        self.next_ref_lanes = self._get_navigation_peer_lanes(self.next_checkpoint_lane_index)
        self._reference_trajectory = self._build_reference_trajectory()
        self._refresh_debug_navigation_path()

    def show_dest_node(self,vehicle_mode):
        """
        在路径终点处显示红色方块，仅针对默认自车
        """
        if self._dest_node_path is not None and vehicle_mode == 'default':
            # 设置终点的节点位置
            if self.reference_trajectory is not None:
                check_point = self.reference_trajectory.end
            else:
                ref_lane = self.final_lane
                later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * ref_lane.width
                check_point = ref_lane.position(ref_lane.length, later_middle)
            self._dest_node_path.setPos(check_point[0], check_point[1],self.MARK_HEIGHT) # check_point[0], check_point[1],self.MARK_HEIGHT
            self._dest_node_path.setColor(1, 0, 0, 0.7)

    def show_target_node(self,check_point,vehicle_mode):
        """
        在指定位置处显示蓝色方块
        for debug
        """
        if self._dest_node_path is not None and vehicle_mode == 'default':
            self._goal_node_path.setPos(check_point[0], check_point[1],3) # check_point[0], check_point[1],self.MARK_HEIGHT
            self._goal_node_path.setColor(0, 0, 1, 0.7)


    @property
    def current_checkpoint_lane_index(self):
        if not self._has_valid_route():
            return None
        return self.checkpoints[self._target_checkpoints_index[0]]
    @property
    def next_checkpoint_lane_index(self):
        if not self._has_valid_route():
            return None
        return self.checkpoints[self._target_checkpoints_index[1]]

    def update_localization(self, ego_vehicle):
        if not self._has_valid_route():
            if self.get_checkpoints(ego_vehicle, allow_off_lane_fallback=True) is None:
                self._mark_vehicle_navigation_invalid(ego_vehicle)
                return
            if len(self.checkpoints) == 1:
                self.checkpoints.append(self.checkpoints[0])
            self._target_checkpoints_index = [0, 1]
            self.set_route()

        position = ego_vehicle.position
        lane, lane_index = self._update_current_lane(ego_vehicle) # 根据车辆位置更新当前所在车道

        # 在到达一个新的路段时会触发need_update = True
        need_update = self._update_target_checkpoints(lane_index) # 检查是否存在下一个检查点，是返回True，False表示位于最后的检查点
        route_replanned = False

        # 让导航路径点一直保持在一定数量
        # while self.vehicle_config['vehicle_model'] == 'default' and len(self.checkpoints) <= 30:
        #     self.checkpoints = self.checkpoints[:-1] + self.map.road_network.random_path(self.checkpoints[-1])
        # 表示车辆已经到达了最后一个检查点，需要重新选择新的目的地进行轨迹规划
        # if self._target_checkpoints_index[0] == self._target_checkpoints_index[1] and (self.vehicle_config['vehicle_model'] == 'adbv' or self.vehicle_config['vehicle_model'] == 'default'):
        #     print("ARRIVE!") # 到达目的地

        # ========== 重新选择新的目的地进行轨迹规划 ==========
        # 1、车辆已经到达了最后一个检查点
        long, lat, heading_theta_at_long = self._project_to_reference_trajectory(position, ego_vehicle.heading_theta)
        on_final_checkpoint = self.current_checkpoint_lane_index == self.next_checkpoint_lane_index
        near_destination = False
        if self.reference_trajectory is not None:
            destination_threshold = max(self.reference_trajectory.length - self.CKPT_UPDATE_RANGE, 0.0)
            near_destination = long >= destination_threshold
        arrived_destination = on_final_checkpoint and near_destination
        # 长度表示车辆已经到达了终点附近，且车辆位置在参考轨迹的最后一个检查点附近，此时可以认为车辆已经到达了目的地
        hold_route_after_arrival = (
            arrived_destination and ego_vehicle.id in ego_vehicle.engine.agents and self._dest_mode in {1, 4}
        )

        # 到达目的地后并且不保持当前路线，针对总是需要重新规划路径的mode
        if not self._global_route_locked and arrived_destination and not hold_route_after_arrival:
            _ = self.get_checkpoints(ego_vehicle)
            if not self.checkpoints:
                self._mark_vehicle_navigation_invalid(ego_vehicle)
                return
            if len(self.checkpoints) == 1:
                self.checkpoints.append(self.checkpoints[0])
            self._target_checkpoints_index = [0, 1]
            need_update = True
            route_replanned = True

        # 2、车辆驶出了规划路径范围
        if not self._global_route_locked and not self._is_lane_index_on_route(lane_index):
            replan_threshold = float(
                self.engine.global_config.get(
                    "osm_replan_lateral_threshold",
                    self.engine.global_config.get("max_lateral_dist", 20) * 0.5
                )
            )
            allow_destination_replan = bool(self.engine.global_config.get("osm_allow_destination_replan", False))
            is_agent_vehicle = self._is_agent_vehicle(ego_vehicle)
            should_replan = (
                self._dest_mode != 1
                or (allow_destination_replan and abs(lat) > replan_threshold)
            )
            self._debug_route_log(
                ego_vehicle,
                "off_route",
                localized_lane=lane_index,
                current_checkpoint=self.current_checkpoint_lane_index,
                next_checkpoint=self.next_checkpoint_lane_index,
                lat=round(float(lat), 2),
                threshold=round(float(replan_threshold), 2),
                should_replan=should_replan,
                allow_destination_replan=allow_destination_replan,
                is_agent_vehicle=is_agent_vehicle,
            )
            if should_replan:
                # 如果车辆驶出了规划路径范围，且找不到当前所在的车道，则删去车辆
                if self.get_checkpoints(ego_vehicle) is None:
                    self._mark_vehicle_navigation_invalid(ego_vehicle)
                    return
                # 若没有下一个车道，则当前车道是最后一个车道，下一个导航车道也是当前车道
                if len(self.checkpoints) == 1:
                    self.checkpoints.append(self.checkpoints[0])
                self._target_checkpoints_index = [0, 1] # 重置当前检查点和下一个检查点的序号
                need_update = True
                route_replanned = True
                self._debug_route_log(
                    ego_vehicle,
                    "replanned",
                    localized_lane=lane_index,
                    checkpoints=(self.checkpoints or [])[:8],
                )

        if need_update: # 当车道进入一个新的路段时，周边的车道也变化，所以需要更新
            # 主要是更新临近的车道关系，从给定的车道索引获取相邻的车道信息
            self.current_ref_lanes = self._get_navigation_peer_lanes(self.current_checkpoint_lane_index) # [当前车道，所有左边车道，所有右边车道]
            self.next_ref_lanes = self._get_navigation_peer_lanes(self.next_checkpoint_lane_index)
            if route_replanned:
                self._reference_trajectory = self._build_reference_trajectory()
                self._refresh_debug_navigation_path()

        long, lat, heading_theta_at_long = self._project_to_reference_trajectory(position, ego_vehicle.heading_theta)
        self._attach_debug_navigation_path(ego_vehicle)
        self.last_current_heading_theta_at_long.append(heading_theta_at_long)
        self.last_current_long.append(long)
        self.last_current_lat.append(lat)
        if self.reference_trajectory is not None and self.reference_trajectory.length > 0:
            self._route_completion = clip(long / self.reference_trajectory.length, 0.0, 1.0)
        else:
            self._route_completion = 0.0
        if hold_route_after_arrival:
            self._route_completion = 1.0

        # self._navi_info可以作为观察的一部分
        self._navi_info.fill(0.0)
        half = self.CHECK_POINT_INFO_DIM # 用来分割导航信息数组_navi_info，每个检查点对应一段导航信息
        if self.current_checkpoint_lane_index is None or self.next_checkpoint_lane_index is None:
            return
        self._navi_info[:half], lanes_heading1, checkpoint = self._get_info_for_checkpoint(
            lanes_id=0,
            ref_lane=self.map.road_network.get_lane(self.current_checkpoint_lane_index),
            ego_vehicle=ego_vehicle
        )

        self._navi_info[half:half+self.CHECK_POINT_INFO_DIM], lanes_heading2, _ = self._get_info_for_checkpoint(
            lanes_id=1,
            ref_lane=self.map.road_network.get_lane(self.next_checkpoint_lane_index),
            ego_vehicle=ego_vehicle
        )
        self.show_dest_node(self.vehicle_config['vehicle_model']) # 原本是在set_route中设置，每次reset显示，对于需要经常变化终点的需求在此更新


        # -------------------显示的是导航路径中每条道路的end_point-------------------
        # 每次显示self.NUM_NAVI_NODE个导航点，如果最后的导航点不足self.NUM_NAVI_NODE个
        # 那么将多出的self._ckpt_vis_models模型重合放置在最后一个检查点
        # next_idx = self._target_checkpoints_index[0]
        # end_idx = min(next_idx + self.NUM_NAVI_NODE, len(self.checkpoints)) # 保证不超出序号
        # ckpts = self.checkpoints[next_idx:end_idx]
        # diff = self.NUM_NAVI_NODE - len(ckpts)
        # assert diff >= 0, "Number of Navigation points error!"
        # if diff > 0:
        #     ckpts += [self.checkpoints[-1] for _ in range(diff)]
        # self._navi_info.fill(0.0) # 重置导航信息

        # for k, ckpt in enumerate(ckpts[:]):
        #     start = k * self.CHECK_POINT_INFO_DIM
        #     end = (k + 1) * self.CHECK_POINT_INFO_DIM
        #     # self._navi_info[start:end], lanes_heading = self._get_info_for_checkpoint(ckpt, ego_vehicle)
        #     # 更新检查点的可视化效果
        #     if self._show_navi_info and self._ckpt_vis_models is not None and self.vehicle_config['vehicle_model'] == 'default':
        #         pos_of_goal = self.map.road_network.get_lane(ckpt).end # 检查点所在路径的end端坐标
        #         lane=self.map.road_network.get_lane(ckpt)
        #         self._ckpt_vis_models[k].setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
        #         # self._ckpt_vis_models[k].setZ(self._ckpt_vis_models[k].getZ() + 3) # 修改高度
        #         # self._ckpt_vis_models[k].setColor(1, 0, 0, 1) # 修改颜色

        # -------------------显示的是导航路径中车辆面前的几个点-------------------
        # if ego_vehicle.class_name == 'DefaultVehicle' and self.reference_trajectory is not None:
        #     checkpoints = self.reference_trajectory.get_polyline(5)
        #     next_idx = min(max(int(long / 5) + 1, 0), len(checkpoints) - 1)
        #     end_idx = min(next_idx + self.NUM_NAVI_NODE, len(checkpoints))
        #     ckpts = checkpoints[next_idx:end_idx]
        #     diff = self.NUM_NAVI_NODE - len(ckpts)
        #     assert diff >= 0, "Number of Navigation points error!"
        #     if diff > 0:
        #         ckpts = np.append(ckpts, [checkpoints[-1] for _ in range(diff)], axis=0)

        #     for k, ckpt in enumerate(ckpts[:]):
        #         # 
        #         # start = k * self.CHECK_POINT_INFO_DIM
        #         # end = (k + 1) * self.CHECK_POINT_INFO_DIM
        #         # self._navi_info[start:end], lanes_heading = self._get_info_for_checkpoint(ckpt, ego_vehicle)
        #         # 更新检查点的可视化效果
        #         if self._show_navi_info and self._ckpt_vis_models is not None and self.vehicle_config['vehicle_model'] == 'default':
        #             pos_of_goal = ckpt
        #             self._ckpt_vis_models[k].setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
        #             # self._ckpt_vis_models[k].setZ(self._ckpt_vis_models[k].getZ() + 3) # 修改高度
        #             # self._ckpt_vis_models[k].setColor(1, 0, 0, 1) # 修改颜色

    def _get_info_for_checkpoint(self, lanes_id, ref_lane, ego_vehicle):
        """
        检查点位置映射到目标车辆的坐标系中，其中 +x 是车辆的航向，+y 是车辆的右侧。
        """
        navi_information = []
        check_point = ref_lane.end
        dir_vec = check_point - ego_vehicle.position  # get the vector from center of vehicle to checkpoint
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > self.NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * self.NAVI_POINT_DIST
        # 将dir_vec向量投影到目标车辆的坐标系中，沿着行驶方向为y轴，右侧为x轴
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
            dir_vec, 0.0
        )

        # Dim 1:将检查点在目标车辆的航向方向上的相对位置添加到导航信息中
        navi_information.append(clip((ckpt_in_heading / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        # Dim 2:将检查点在目标车辆的右侧方向上的横向相对位置添加到导航信息中
        navi_information.append(clip((ckpt_in_rhs / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        if lanes_id == 0:
            # 计算当前车辆所在位置处车道的航向角
            lanes_heading = ref_lane.heading_theta_at(ref_lane.local_coordinates(ego_vehicle.position)[0])
        else:
            # 计算下一个车道的航向角，因为没有车辆位置所以通过车道的长度和预设值的最小值来计算
            lanes_heading = ref_lane.heading_theta_at(min(self.PRE_NOTIFY_DIST, ref_lane.length))

        # 尝试将当前车道的信息包含到导航信息中，如果当前车道是环形车道，那么将环形车道的信息包含到导航信息中，否则为0
        bendradius = 0.0
        dir = 0.0
        angle = 0.0
        from metadrive.component.lane.circular_lane import CircularLane
        from metadrive.component.pg_space import Parameter, BlockParameterSpace
        if isinstance(ref_lane, CircularLane):
            bendradius = ref_lane.radius / (
                BlockParameterSpace.CURVE[Parameter.radius].max +
                self.get_current_lane_num() * self.get_current_lane_width()
            )
            dir = -ref_lane.direction
            angle = ref_lane.angle
        # Dim 3: The bending radius of current lane
        navi_information.append(clip(bendradius, 0.0, 1.0))

        # Dim 4: The bending direction of current lane (+1 for clockwise, -1 for counterclockwise)
        navi_information.append(clip((dir + 1) / 2, 0.0, 1.0))

        # Dim 5: The angular difference between the heading in lane ending position and
        # the heading in lane starting position
        navi_information.append(
            clip((np.rad2deg(angle) / BlockParameterSpace.CURVE[Parameter.angle].max + 1) / 2, 0.0, 1.0)
        )
        return navi_information, lanes_heading, check_point
    
    def _update_current_lane(self, ego_vehicle):
        """
        具体逻辑就是根据车辆的位置，计算车辆当前所在的车道，因为在一个step中车辆的位置会发生变化。
        但如果根据目前位置无法确定车道，那么就根据车辆之前的位置来确定车道。
        ego_vehicle.lane调用的还是self.navigation.current_lane来获得车道信息，
        所以在没有执行self.current_lane = lane这一句时还保留了之前的车道信息。
        不论通过哪种方式获得的lane都在self.current_lane = lane这一句中更新了当前车道。
        """
        lane, lane_index, on_lane = self._get_current_lane(ego_vehicle)
        ego_vehicle.on_lane = on_lane
        if lane is not None: 
            assert lane_index == lane.index, "lane index mismatch!" # 确保车道索引与车道的实际索引匹配
        if lane is None: # 先将之前的车道作为现在车道，但是on_lane已经是False了，方便后续更新
            lane, lane_index = ego_vehicle.lane, ego_vehicle.lane_index
        self.current_lane = lane # 在这更新了self._current_lane
        return lane, lane_index

    def _update_target_checkpoints(self, ego_lane_index) -> bool:
        """
        更新检查点，如果车辆当前车道在之后的导航路径中则返回 True以便更新ref_lanes
        否则表示此时已经在最后一段路径，返回 False
        具体逻辑：
        刚开始时，self._target_checkpoints_index = [0, 1]，即当前检查点索引为 0，下一个检查点索引为 1。
        通过检查车辆当前车道ego_lane_index是否在从self.next_checkpoint_lane_index直到最后的检查点(实际就是规划路径中所有车道的索引)中
        来判断是否到达了下一个检查点，就是是否更换了车道。那么就更新self._target_checkpoints_index为 [idx，idx+1]。
        如果idx + 1 == len(self.checkpoints)，因为idx从0开始，表示已经到最后一段路，没有下一个idx了。
        """
        if self.current_checkpoint_lane_index == self.next_checkpoint_lane_index:  # on last road
            return False

        # arrive to second checkpoint
        new_index = ego_lane_index
        if new_index in self.checkpoints[self._target_checkpoints_index[1]:]:
            idx = self.checkpoints.index(new_index, self._target_checkpoints_index[1]) # index() 方法用于查找 new_index 在 self.checkpoints 列表中的位置（索引）。self._target_checkpoints_index[1] 是 index() 方法的起始查找位置，这意味着搜索将从这个索引开始
            self._target_checkpoints_index = [idx]
            if idx + 1 == len(self.checkpoints):
                self._target_checkpoints_index.append(idx)
            else:
                self._target_checkpoints_index.append(idx + 1)
            return True
        if self._lane_index_in_lanes(new_index, self.next_ref_lanes):
            idx = self._target_checkpoints_index[1]
            self._target_checkpoints_index = [idx]
            if idx + 1 == len(self.checkpoints):
                self._target_checkpoints_index.append(idx)
            else:
                self._target_checkpoints_index.append(idx + 1)
            return True
        return False
    
    def _get_current_lane(self, ego_vehicle):
        possible_lanes, on_lane = self.get_current_lane_modified(
            ego_vehicle, return_all=True, use_heading_filter=True
        )
        current_ref_lanes = self.current_ref_lanes or []
        for lane, index, l_1_dist in possible_lanes:
            if lane in current_ref_lanes:
                return lane, index, on_lane
        if not self.checkpoints or self._target_checkpoints_index is None:
            return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)
        nx_ckpt = self._target_checkpoints_index[-1]
        if nx_ckpt == self.checkpoints[-1] or self.next_ref_lanes is None:
            return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)

        next_ref_lanes = self.next_ref_lanes
        for lane, index, l_1_dist in possible_lanes:
            if lane in next_ref_lanes:
                return lane, index, on_lane
        return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)
    
    def get_current_lateral_range(self, current_position, engine) -> float:
        return self.current_lane.width * max(len(self.current_ref_lanes or []), 1)

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1

    # @classmethod
    # def _get_info_for_checkpoint(cls, checkpoint, ego_vehicle):
    #     # 和edge_road_navigation中的_get_info_for_checkpoint方法一样，只是用了相隔CHECK_POINT_INFO_DIM个后的checkpoint代替了参考路径的lane.end点
    #     navi_information = []
    #     # Project the checkpoint position into the target vehicle's coordination, where
    #     # +x is the heading and +y is the right hand side.
    #     # 检查点位置映射到目标车辆的坐标系中，其中 +x 是车辆的航向，+y 是车辆的右侧。
    #     dir_vec = checkpoint - ego_vehicle.position # 获取从车辆中心到检查点的向量
    #     dir_norm = norm(dir_vec[0], dir_vec[1])
    #     if dir_norm > cls.NAVI_POINT_DIST: # 如果检查点太远，将方向向量缩放到最大距离 NAVI_POINT_DIST，以避免过大的值影响后续计算
    #         dir_vec = dir_vec / dir_norm * cls.NAVI_POINT_DIST
    #     ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
    #         dir_vec, 0.0
    #     )  # 将方向向量 dir_vec 转换到以 ego 车辆为中心的局部坐标系中。返回值包括沿车辆前进方向的距离 (ckpt_in_heading) 和右侧方向的距离 (ckpt_in_rhs)

    #     # Dim 1: the relative position of the checkpoint in the target vehicle's heading direction.
    #     navi_information.append(clip((ckpt_in_heading / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

    #     # Dim 2: the relative position of the checkpoint in the target vehicle's right hand side direction.
    #     navi_information.append(clip((ckpt_in_rhs / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

    #     return navi_information

    def destroy(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        self.debug_navigation_global_path = None
        self._attach_debug_navigation_path()
        self._debug_path_vehicle = None
        if self._reference_trajectory is not None:
            self._reference_trajectory.destroy()
        self._reference_trajectory = None
        super(OsmTrajectoryNavigation, self).destroy()

    def before_reset(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        self.debug_navigation_global_path = None
        self._attach_debug_navigation_path()
        self._debug_path_vehicle = None
        if self._reference_trajectory is not None:
            self._reference_trajectory.destroy()
        self._reference_trajectory = None

    @property
    def route_completion(self):
        return self._route_completion
    
    @route_completion.setter
    def route_completion(self, value):
        self._route_completion = float(value)

    @classmethod
    def get_navigation_info_dim(cls):
        # 在update_localization中设置self._navi_info，总体维度 = 每个检查点的信息维度之和 + 这两个全局导航量（横向位置与航向差）
        return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM + 2
    
    @property
    def current_lane(self):
        return self._current_lane
    
    @current_lane.setter
    def current_lane(self, value):
        self._current_lane = value  # 设置属性的逻辑

    @property
    def last_longitude(self):
        return self.last_current_long[0]

    @property
    def current_longitude(self):
        return self.last_current_long[1]

    @property
    def last_lateral(self):
        return self.last_current_lat[0]

    @property
    def current_lateral(self):
        return self.last_current_lat[1]

    @property
    def last_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[0]

    @property
    def current_heading_theta_at_long(self): # 这个值是车辆当前位置在参考路径上的投影点的航向角，反映了车辆当前的行驶方向与参考路径的匹配程度
        return self.last_current_heading_theta_at_long[1]


class IdmReplayTrajectoryNavigation(BaseNavigation):
    """
    This module enabling follow a given reference trajectory given a map
    原代码中的设置，trajectory_navigation和(edge_navigation和node_navigation的维度设置不同，这两者是基于base_navigation的）
    DISCRETE_LEN = 2  # m
    CHECK_POINT_INFO_DIM = 2
    NUM_WAY_POINT = 10
    NAVI_POINT_DIST = 30  # m, used to clip value, should be greater than DISCRETE_LEN * MAX_NUM_WAY_POINT
    为了和osm_trajectory_navigation保持一致，修改osm的维度，那是基于edge_navigation和trajectory_navigation的结合体
    """
    
    DISCRETE_LEN = 2  # 2,mj1231  # m
    CHECK_POINT_INFO_DIM = 2  # 2,mj1231
    NUM_WAY_POINT = 10  # 10,mj1231
    NAVI_POINT_DIST = 30  # 30 m, used to clip value, should be greater than DISCRETE_LEN * MAX_NUM_WAY_POINT

    def __init__(
        self,
        show_navi_mark: bool = False,
        show_dest_mark=False,
        show_line_to_dest=False,
        panda_color=None,
        name=None,
        vehicle_config=None
    ):
        if show_dest_mark or show_line_to_dest:
            get_logger().warning("show_dest_mark and show_line_to_dest are not supported in TrajectoryNavigation")
        super(IdmReplayTrajectoryNavigation, self).__init__(
            show_navi_mark=False,
            show_dest_mark=False,
            show_line_to_dest=False,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        if self.origin is not None:
            self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)

        self._route_completion = 0
        self.checkpoints = None  # All check points
        self.debug_navigation_global_path = None
        self._debug_path_vehicle = None

        # for compatibility
        self.next_ref_lanes = None

        # override the show navi mark function here
        self._navi_point_model = None
        self._ckpt_vis_models = None
        if show_navi_mark and self._show_navi_info:
            self._ckpt_vis_models = [NodePath(str(i)) for i in range(self.NUM_WAY_POINT)]
            for model in self._ckpt_vis_models:
                if self._navi_point_model is None:
                    self._navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                    self._navi_point_model.setScale(0.5)
                    # if self.engine.use_render_pipeline:
                    material = Material()
                    material.setBaseColor((19 / 255, 212 / 255, 237 / 255, 1))
                    material.setShininess(16)
                    material.setEmission((0.2, 0.2, 0.2, 0.2))
                    self._navi_point_model.setMaterial(material, True)
                self._navi_point_model.instanceTo(model)
                model.reparentTo(self.origin)

        # should be updated every step after calling update_localization
        self.last_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_current_heading_theta_at_long = deque([0.0, 0.0], maxlen=2)

    def _reference_trajectory_to_global_path(self, reference_trajectory, sample_interval=2):
        if reference_trajectory is None:
            return None

        interval = max(float(sample_interval), 0.1)
        points = np.asarray(reference_trajectory.get_polyline(interval=interval, lateral=0), dtype=float)
        if points.ndim != 2 or len(points) == 0 or points.shape[1] < 2:
            return None

        points = points[:, :2]
        if len(points) >= 2:
            deltas = np.diff(points, axis=0)
            headings = np.arctan2(deltas[:, 1], deltas[:, 0])
            headings = np.append(headings, headings[-1])
        else:
            headings = np.zeros(1, dtype=float)
        curvature = np.zeros(len(points), dtype=float)
        return SimpleNamespace(x=points[:, 0], y=points[:, 1], heading=headings, curvature=curvature)

    def _attach_debug_navigation_path(self, vehicle=None):
        if vehicle is not None:
            self._debug_path_vehicle = vehicle
        if self._debug_path_vehicle is not None:
            self._debug_path_vehicle.navigation_global_path = self.debug_navigation_global_path

    def _refresh_debug_navigation_path(self):
        self.debug_navigation_global_path = _reference_trajectory_to_render_path(
            self.reference_trajectory, sample_interval=self.DISCRETE_LEN
        )
        self._attach_debug_navigation_path()
        return self.debug_navigation_global_path

    def reset(self, vehicle):
        self._debug_path_vehicle = vehicle
        super(IdmReplayTrajectoryNavigation, self).reset(current_lane=self.reference_trajectory)
        self.set_route()
        self._attach_debug_navigation_path(vehicle)
        if self.name == 'default_agent':
            return self._reference_trajectory_to_global_path(self.engine.map_manager.current_sdc_route)

    @property
    def reference_trajectory(self):
        return self.engine.map_manager.current_sdc_route

    @property
    def current_ref_lanes(self):
        return [self.reference_trajectory]

    def set_route(self):
        self.checkpoints = self.discretize_reference_trajectory()

        num_way_point = min(len(self.checkpoints), self.NUM_WAY_POINT)

        self._navi_info.fill(0.0)
        self.next_ref_lanes = None
        if self._dest_node_path is not None:
            check_point = self.reference_trajectory.end
            self._dest_node_path.setPos(panda_vector(check_point[0], check_point[1], 1))
        self._refresh_debug_navigation_path()

    def discretize_reference_trajectory(self):
        ret = []
        length = self.reference_trajectory.length
        num = int(length / self.DISCRETE_LEN)
        for i in range(num):
            ret.append(self.reference_trajectory.position(i * self.DISCRETE_LEN, 0))
        ret.append(self.reference_trajectory.end)
        return ret

    def update_localization(self, ego_vehicle):
        """
        It is called every step
        """
        if self.reference_trajectory is None:
            return
        self._attach_debug_navigation_path(ego_vehicle)
        excluded_lane_types = (
            {MetaDriveType.LANE_BIKE_LANE}
            if self.vehicle_config.get("filter_bike_lanes_in_navigation", True) else set()
        )
        
        possible_lanes = ray_localization(ego_vehicle.heading, ego_vehicle.position, self.engine, use_heading_filter=True)
        possible_lanes = [
            lane_candidate for lane_candidate in possible_lanes
            if not self.map.road_network.is_lane_excluded(lane_candidate[1], excluded_lane_types)
        ]
        if len(possible_lanes) > 0:
            on_lane = True
        else:
            _,on_lane = self.map.road_network.get_closest_lane_index(
                ego_vehicle.position,
                ego_vehicle.heading_theta,
                ego_vehicle,
                excluded_lane_types=excluded_lane_types
            )

        ego_vehicle.on_lane = on_lane
        # Update ckpt index
        long, lat = self.reference_trajectory.local_coordinates(ego_vehicle.position)
        possible_lane = ray_localization(
            ego_vehicle.heading,
            ego_vehicle.position,
            ego_vehicle.engine,
            use_heading_filter=True)
        possible_lane = [
            lane_candidate for lane_candidate in possible_lane
            if not self.map.road_network.is_lane_excluded(lane_candidate[1], excluded_lane_types)
        ]
        
        heading_theta_at_long = self.reference_trajectory.heading_theta_at(long)
        self.last_current_heading_theta_at_long.append(heading_theta_at_long)
        self.last_current_long.append(long)
        self.last_current_lat.append(lat)

        next_idx = max(int(long / self.DISCRETE_LEN) + 1, 0)
        next_idx = min(next_idx, len(self.checkpoints) - 1)
        end_idx = min(next_idx + self.NUM_WAY_POINT, len(self.checkpoints))
        ckpts = self.checkpoints[next_idx:end_idx]
        diff = self.NUM_WAY_POINT - len(ckpts)
        assert diff >= 0, "Number of Navigation points error!"
        if diff > 0:
            ckpts += [self.checkpoints[-1] for _ in range(diff)] # 如果最后的检查点个数小于self.NUM_WAY_POINT，那么就让diff个检查点位置重合以达到消失的目的

        # target_road_1 is the road segment the vehicle is driving on.
        self._navi_info.fill(0.0)
        for k, ckpt in enumerate(ckpts[:]):
            start = k * self.CHECK_POINT_INFO_DIM
            end = (k + 1) * self.CHECK_POINT_INFO_DIM
            self._navi_info[start:end], lanes_heading = self._get_info_for_checkpoint(ckpt, ego_vehicle)

            if self._show_navi_info and self._ckpt_vis_models is not None:
                pos_of_goal = ckpt
                self._ckpt_vis_models[k].setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
                self._ckpt_vis_models[k].setH(self._goal_node_path.getH() + 3)

        self._navi_info[end] = clip((lat / self.engine.global_config["max_lateral_dist"] + 1) / 2, 0.0, 1.0)
        self._navi_info[end + 1] = clip(
            (wrap_to_pi(heading_theta_at_long - ego_vehicle.heading_theta) / np.pi + 1) / 2, 0.0, 1.0
        )

        self._route_completion = long / self.reference_trajectory.length
        if len(possible_lane) != 0:
            current_lane = possible_lane[0][0]
            self.reference_trajectory.index = current_lane.index
        else:
            return True

    def get_current_lateral_range(self, current_position, engine) -> float:
        return self.current_lane.width * 2

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1

    @classmethod
    def _get_info_for_checkpoint(cls, checkpoint, ego_vehicle):
        navi_information = []
        dir_vec = checkpoint - ego_vehicle.position
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > cls.NAVI_POINT_DIST:
            dir_vec = dir_vec / dir_norm * cls.NAVI_POINT_DIST
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
            dir_vec, 0.0
        )

        navi_information.append(clip((ckpt_in_heading / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))
        navi_information.append(clip((ckpt_in_rhs / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))
        return navi_information

    def destroy(self):
        self.checkpoints = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        self.debug_navigation_global_path = None
        self._attach_debug_navigation_path()
        self._debug_path_vehicle = None
        super(IdmReplayTrajectoryNavigation, self).destroy()

    def before_reset(self):
        self.checkpoints = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        self.debug_navigation_global_path = None
        self._attach_debug_navigation_path()
        self._debug_path_vehicle = None

    @property
    def current_lane(self):
        return self._current_lane

    @current_lane.setter
    def current_lane(self, value):
        self._current_lane = value

    @property
    def route_completion(self):
        return self._route_completion

    @classmethod
    def get_navigation_info_dim(cls):
        return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM + 2

    @property
    def last_longitude(self):
        return self.last_current_long[0]

    @property
    def current_longitude(self):
        return self.last_current_long[1]

    @property
    def last_lateral(self):
        return self.last_current_lat[0]

    @property
    def current_lateral(self):
        return self.last_current_lat[1]

    @property
    def last_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[0]

    @property
    def current_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[1]
