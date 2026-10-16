import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math import not_zero, wrap_to_pi, norm
import time

class FrontBackObjects:
    """
    用于处理车辆或物体在车道前后位置的检测和管理。
    检查是否有物体在前方、后方或旁边的车道上，并返回对应的物体和距离信息。
    """
    def __init__(self, front_ret, back_ret, front_dist, back_dist):
        self.front_objs = front_ret
        self.back_objs = back_ret
        self.front_dist = front_dist
        self.back_dist = back_dist

    def left_lane_exist(self):
        # 不存在的相邻车道就是None
        return True if self.front_dist[0] is not None else False

    def right_lane_exist(self):
        return True if self.front_dist[-1] is not None else False

    def has_front_object(self):
        return True if self.front_objs[1] is not None else False

    def has_back_object(self):
        return True if self.back_objs[1] is not None else False

    def has_left_front_object(self):
        return True if self.front_objs[0] is not None else False

    def has_left_back_object(self):
        return True if self.back_objs[0] is not None else False

    def has_right_front_object(self):
        return True if self.front_objs[-1] is not None else False

    def has_right_back_object(self):
        return True if self.back_objs[-1] is not None else False

    def front_object(self):
        return self.front_objs[1]

    def left_front_object(self):
        return self.front_objs[0]

    def right_front_object(self):
        return self.front_objs[-1]

    def back_object(self):
        return self.back_objs[1]

    def left_back_object(self):
        return self.back_objs[0]

    def right_back_object(self):
        return self.back_objs[-1]

    def left_front_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.front_dist[0]

    def right_front_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.front_dist[-1]

    def front_min_distance(self):
        return self.front_dist[1]

    def left_back_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.back_dist[0]

    def right_back_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.back_dist[-1]

    def back_min_distance(self):
        return self.back_dist[1]

    @classmethod
    def get_find_front_back_objs(cls, objs, lane, position, max_distance, ref_lanes=None):
        """
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        """
        if ref_lanes is not None:
            assert lane in ref_lanes
        idx = lane.index[-1] if ref_lanes is not None else None
        left_lane = ref_lanes[idx - 1] if ref_lanes is not None and idx > 0 else None
        right_lane = ref_lanes[idx + 1] if ref_lanes is not None and idx + 1 < len(ref_lanes) else None
        lanes = [left_lane, lane, right_lane]

        min_front_long = [max_distance if lane is not None else None for lane in lanes]
        min_back_long = [max_distance if lane is not None else None for lane in lanes]

        front_ret = [None, None, None]
        back_ret = [None, None, None]

        find_front_in_current_lane = [False, False, False]
        find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes]
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in enumerate(lanes)]

        # 计算车辆和障碍物之间的距离
        for i, lane in enumerate(lanes):
            if lane is None:
                continue
            for obj in objs:
                if obj.lane is lane:
                    long = lane.local_coordinates(obj.position)[0] - current_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                        find_front_in_current_lane[i] = True
                    if long < 0 and abs(long) < min_back_long[i]:
                        min_back_long[i] = abs(long)
                        back_ret[i] = obj
                        find_back_in_current_lane[i] = True

                elif not find_front_in_current_lane[i] and lane.is_previous_lane_of(obj.lane):
                    long = obj.lane.local_coordinates(obj.position)[0] + left_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                elif not find_back_in_current_lane[i] and obj.lane.is_previous_lane_of(lane):
                    long = obj.lane.length - obj.lane.local_coordinates(obj.position)[0] + current_long[i]
                    if min_back_long[i] > long:
                        min_back_long[i] = long
                        back_ret[i] = obj

        return cls(front_ret, back_ret, min_front_long, min_back_long)
    
    @classmethod
    def get_find_front_back_objs_id(cls, objs, lane, position, max_distance, ref_lanes=None):
        """
        只考虑同一个道路下不同车道的障碍物
        因为原先的lane.index是按照区块设置的，以字母等命名
        但是在地图采用数字命名道路时，原先代码会报错故重写
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        return:front_objs(前方所有同向车道障碍物对象), back_objs(后方障碍物对象), front_dist(距离前方障碍物最近距离), back_dist(距离后方障碍物最近距离)
        """
        left_lane = None
        right_lane = None
        if ref_lanes is not None:
            assert lane in ref_lanes
            for l in ref_lanes:
                if l != lane:
                    if l.index in lane.left_lanes and left_lane == None:
                        left_lane = l
                    elif l.index in lane.right_lanes and right_lane == None:
                        right_lane = l
        lanes = [left_lane, lane, right_lane]
        # 列表用于存储各车道前方和后方物体的最小距离，初始值设为 max_distance（表示未检测到物体）
        front_dist = [max_distance if lane is not None else None for lane in lanes] 
        back_dist = [max_distance if lane is not None else None for lane in lanes]
        # 用于存储每个车道检测到的前方和后方物体
        front_objs = [None, None, None]
        back_objs = [None, None, None]
        # 用于标记在当前车道中是否已找到前方或后方物体
        find_front_in_current_lane = [False, False, False]
        find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes] # 存储车辆在各车道坐标系上的当前纵向位置
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in enumerate(lanes)] # 计算剩余的车道长度

        for i, lane in enumerate(lanes):
            if lane is None: # 遍历每个车道，对于存在的车道（不为 None），再遍历所有物体 objs
                continue
            for obj in objs:
                # 障碍物在当前车道 
                # <-----o----v----|-----
                if obj.lane is lane: 
                    long = lane.local_coordinates(obj.position)[0] - current_long[i] # 物体和车辆在当前车道坐标系下的纵向距离
                    if front_dist[i] > long > 0: # 只考虑前方障碍物
                        front_dist[i] = long
                        front_objs[i] = obj
                        find_front_in_current_lane[i] = True
                    if long < 0 and abs(long) < back_dist[i]:
                        back_dist[i] = abs(long)
                        back_objs[i] = obj
                        find_back_in_current_lane[i] = True
                # 若当前车道的end是障碍物所在车道的start
                # <-----o------------------|---------------v----|
                #       |<--obj.position-->|<--left_long-->|
                # 但是会存在这种情况：障碍物在当前车道的下下个车道，但是下个车道的length很短，所以会检测不到障碍物
                elif not find_front_in_current_lane[i] and lane.is_previous_lane_of(obj.lane): 
                    long = obj.lane.local_coordinates(obj.position)[0] + left_long[i] # o-v之间的距离
                    if front_dist[i] > long > 0:
                        front_dist[i] = long
                        front_objs[i] = obj
                # 若当前车道的start是障碍物所在车道的end
                # <-----v---|----o----|
                elif not find_back_in_current_lane[i] and obj.lane.is_previous_lane_of(lane): 
                    long = obj.lane.length - obj.lane.local_coordinates(obj.position)[0] + current_long[i]
                    if back_dist[i] > long:
                        back_dist[i] = long
                        back_objs[i] = obj

        return cls(front_objs, back_objs, front_dist, back_dist)
    
    @classmethod
    def get_find_front_back_objs_id_with_lane_index(cls, network, objs, lane, position, max_distance, ref_lanes=None):
        """
        与get_find_front_back_objs_id不同之处在于将obj.lane替换为network.get_lane(obj.lane_index)
        因为此时自车的lane是给定的reference_lane，是一整条历史轨迹且没有index信息的
        所以通过在IdmReplayTrajectoryNavigation中的update_localization每一步根据车辆位置获取当前车道的index来定位
        """
        left_lane = None
        right_lane = None
        if ref_lanes is not None:
            assert lane in ref_lanes
            for l in ref_lanes:
                if l != lane:
                    if l.index in lane.left_lanes and left_lane == None:
                        left_lane = l
                    elif l.index in lane.right_lanes and right_lane == None:
                        right_lane = l
        lanes = [left_lane, lane, right_lane]
        # 列表用于存储各车道前方和后方物体的最小距离，初始值设为 max_distance（表示未检测到物体）
        front_dist = [max_distance if lane is not None else None for lane in lanes] 
        back_dist = [max_distance if lane is not None else None for lane in lanes]
        # 用于存储每个车道检测到的前方和后方物体
        front_objs = [None, None, None]
        back_objs = [None, None, None]
        # 用于标记在当前车道中是否已找到前方或后方物体
        find_front_in_current_lane = [False, False, False]
        find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes] # 存储车辆在各车道坐标系上的当前纵向位置
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in enumerate(lanes)] # 计算剩余的车道长度

        for i, lane in enumerate(lanes):
            if lane is None: # 遍历每个车道，对于存在的车道（不为 None），再遍历所有物体 objs
                continue
            for obj in objs:
                obj_lane_index = getattr(obj, "lane_index", None)
                if obj_lane_index is None:
                    obj_lane = getattr(obj, "lane", None)
                    obj_lane_index = getattr(obj_lane, "index", None)
                if obj_lane_index is None:
                    continue
                try:
                    obj_lane = network.get_lane(obj_lane_index)
                except Exception:
                    continue
                # 障碍物在当前车道 
                # <-----o----v----|-----
                if obj_lane is lane: 
                    long = lane.local_coordinates(obj.position)[0] - current_long[i] # 物体和车辆在当前车道坐标系下的纵向距离
                    if front_dist[i] > long > 0: # 只考虑前方障碍物
                        front_dist[i] = long
                        front_objs[i] = obj
                        find_front_in_current_lane[i] = True
                    if long < 0 and abs(long) < back_dist[i]:
                        back_dist[i] = abs(long)
                        back_objs[i] = obj
                        find_back_in_current_lane[i] = True
                # 若当前车道的end是障碍物所在车道的start
                # <-----o------------------|---------------v----|
                #       |<--obj.position-->|<--left_long-->|
                # 但是会存在这种情况：障碍物在当前车道的下下个车道，但是下个车道的length很短，所以会检测不到障碍物
                elif not find_front_in_current_lane[i] and lane.is_previous_lane_of(obj_lane): 
                    long = obj_lane.local_coordinates(obj.position)[0] + left_long[i] # o-v之间的距离
                    if front_dist[i] > long > 0:
                        front_dist[i] = long
                        front_objs[i] = obj
                # 若当前车道的start是障碍物所在车道的end
                # <-----v---|----o----|
                elif not find_back_in_current_lane[i] and obj_lane.is_previous_lane_of(lane): 
                    long = obj_lane.length - obj_lane.local_coordinates(obj.position)[0] + current_long[i]
                    if back_dist[i] > long:
                        back_dist[i] = long
                        back_objs[i] = obj

        return cls(front_objs, back_objs, front_dist, back_dist)

    @classmethod
    def get_find_front_back_objs_single_lane(cls, objs, lane, position, max_distance):
        """
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        """
        lanes = [None, lane, None]

        min_front_long = [max_distance if lane is not None else None for lane in lanes]
        min_back_long = [max_distance if lane is not None else None for lane in lanes]

        front_ret = [None, None, None]
        back_ret = [None, None, None]

        find_front_in_current_lane = [False, False, False]
        # find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes]

        for i, lane in enumerate(lanes):
            if lane is None:
                continue
            for obj in objs:
                _d = obj.position - position
                if norm(_d[0], _d[1]) > max_distance:
                    continue
                if hasattr(obj, "bounding_box") and all([not lane.point_on_lane(p) for p in obj.bounding_box]):
                    continue
                elif not hasattr(obj, "bounding_box") and not lane.point_on_lane(obj.position):
                    continue

                long, _ = lane.local_coordinates(obj.position)
                # if abs(lat) > lane.width / 2:
                #     continue
                long = long - current_long[i]
                if min_front_long[i] > long > 0:
                    min_front_long[i] = long
                    front_ret[i] = obj
                    find_front_in_current_lane[i] = True

        return cls(front_ret, back_ret, min_front_long, min_back_long)

class _LegacyIDMPolicyInOsm(BasePolicy):
    """
    旧版IDMPolicy，已被下面的 IDMPolicyInOsm 替代。保留仅作参考。
    """

    DEBUG_MARK_COLOR = (219, 3, 252, 255)

    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.3  # [s]
    TAU_LATERAL = 0.8  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    DISTANCE_WANTED = 10.0 # 到前车所希望的间隔距离
    TIME_WANTED = 1.5  # [s] 到前车所希望的间隔时间
    DELTA = 10.0  # [] Exponent of the velocity term 
    DELTA_RANGE = [3.5, 4.5] # Range of delta when chosen randomly

    # 侧向策略参数
    LANE_CHANGE_FREQ = 50  # [step]
    LANE_CHANGE_SPEED_INCREASE = 10
    SAFE_LANE_CHANGE_DISTANCE = 15
    MAX_LONG_DIST = 30
    MAX_SPEED = 100  # km/h

    # 正常速度，目标速度
    NORMAL_SPEED = 5  # km/h

    # Creep Speed
    CREEP_SPEED = 5

    # acc factor
    ACC_FACTOR = 1.0
    DEACC_FACTOR = -5

    SAFE_DISTANCE = 10.0  # 安全距离（米）
    EMERGENCY_DISTANCE = 5.0  # 紧急制动距离（米）
    LANE_CHANGE_BLOCK_DISTANCE = 18.0
    LANE_CHANGE_MIN_SPEED_ADVANTAGE = 2.0
    LANE_CHANGE_STUCK_SPEED = 8.0  # km/h
    DEFAULT_TARGET_SPEED_MIN = 15.0  # km/h
    DEFAULT_TARGET_SPEED_MAX = 40.0  # km/h
    DEFAULT_TARGET_SPEED_JITTER = 2.0  # km/h
    DEFAULT_LANE_SPEED_SCALE_MIN = 0.55
    DEFAULT_LANE_SPEED_SCALE_MAX = 0.90
    DEFAULT_LANE_SPEED_CAP_RATIO = 1.00
    DEFAULT_TARGET_SPEED_BLEND = 0.35
    DEFAULT_TIME_WANTED_RANGE = (1.2, 2.0)
    DEFAULT_LANE_CHANGE_ADVANTAGE_RANGE = (1.0, 4.0)


    def __init__(self, control_object, random_seed):
        super(IDMPolicyInOsm, self).__init__(control_object=control_object, random_seed=random_seed)
        self.target_speed = None
        self.cruise_speed_km_h = None
        self.min_target_speed_km_h = None
        self.max_target_speed_km_h = None
        self.respect_lane_speed_limit = True
        self.lane_speed_cap_ratio = self.DEFAULT_LANE_SPEED_CAP_RATIO
        self.target_speed_blend = self.DEFAULT_TARGET_SPEED_BLEND
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = self.engine.global_config.get("enable_idm_lane_change", True)
        # self.enable_lane_change = False
        self.disable_idm_deceleration = self.engine.global_config.get("disable_idm_deceleration", False)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)
        self._initialize_driver_profile(force_resample=True)

    def act(self, *args, **kwargs):
        self._initialize_driver_profile(force_resample=False)
        # success的作用是判断车辆是否在正确的车道上，首先确定在初始位置在正确道路上，然后可以执行正确动作；
        # 若是发现执行完动作后车辆不在正确道路上，则一直执行沿着当前道路行驶的动作，self.routing_target_lane始终不会更新了
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            if success and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # 采用get_find_front_back_objs_id_with_lane_index
                surrounding_objects = FrontBackObjects.get_find_front_back_objs_id_with_lane_index(
                    self.engine.map_manager.current_map.road_network,
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except:
            # error fallback
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane
            # logging.warning("IDM bug! fall back")
            # print("IDM bug! fall back")

        # if success and self.enable_lane_change:
        #     # perform lane change due to routing
        #     acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
        # else:
        #     # can not find routing target lane
        #     surrounding_objects = FrontBackObjects.get_find_front_back_objs_id(
        #         all_objects,
        #         self.routing_target_lane,
        #         self.control_object.position,
        #         max_distance=self.MAX_LONG_DIST
        #     )
        #     acc_front_obj = surrounding_objects.front_object()
        #     acc_front_dist = surrounding_objects.front_min_distance()
        #     steering_target_lane = self.routing_target_lane

        # 以上并不考虑交叉路口中车道交错的情况，应该针对此作出反应，并实现转弯让直行，在steering较大时才减速
        
        # ============ control by PID and IDM ============
        steering = self.steering_control(steering_target_lane)
        idm_acc = self.acceleration(acc_front_obj, acc_front_dist)

        # 找到前方最近的障碍物,基于前方障碍物计算加速度,会在路口避让，但是可能造成交通堵塞
        closest_front_obj, closest_distance = self._find_closest_front_object(all_objects)
        acc = self._merge_longitudinal_control(
            idm_acc,
            closest_front_obj,
            closest_distance,
            steering_target_lane,
        )
        if abs(steering) > 0.45:
            acc = min(acc, 0.0)
        if abs(steering) > 0.70:
            acc = min(acc, -0.5)

        action = [steering, acc]
        self.action_info["action"] = action
        return action

    def move_to_next_road(self):
        """Follow the checkpoint lane sequence from navigation (global path from shortest_path).

        Advances routing_target_lane through checkpoints: when vehicle nears the end
        of the current target lane, move to the next lane in the checkpoint sequence.
        This naturally handles lane changes (if the next checkpoint is a neighbor lane)
        and turns (if the next checkpoint is an exit lane).
        """
        nav = self.control_object.navigation
        checkpoints = getattr(nav, "checkpoints", None)
        if not checkpoints:
            # fallback: original lane-based routing
            current_lanes = nav.current_ref_lanes
            if self.routing_target_lane is None:
                self.routing_target_lane = self.control_object.lane
                return True if self.routing_target_lane in current_lanes else False
            return True

        road_network = self.engine.map_manager.current_map.road_network

        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            # 找到车辆实际所在 lane 在 checkpoints 中的位置
            self._checkpoint_idx = 0
            for idx, cp_id in enumerate(checkpoints):
                if str(cp_id) == str(getattr(self.control_object.lane, "index", None)):
                    self._checkpoint_idx = idx
                    break
            return True

        checkpoint_ids = [str(cp_id) for cp_id in checkpoints]
        current_lane_id = str(getattr(self.control_object.lane, "index", None))
        target_lane_id = str(getattr(self.routing_target_lane, "index", None))
        if target_lane_id not in checkpoint_ids or getattr(self, "_checkpoint_idx", 0) >= len(checkpoints):
            self.routing_target_lane = self.control_object.lane
            self._checkpoint_idx = checkpoint_ids.index(current_lane_id) if current_lane_id in checkpoint_ids else 0
            return True
        if current_lane_id in checkpoint_ids:
            current_idx = checkpoint_ids.index(current_lane_id)
            if current_idx > getattr(self, "_checkpoint_idx", 0):
                self._checkpoint_idx = current_idx
                self.routing_target_lane = self.control_object.lane
                return True

        # Find vehicle position on current routing_target_lane
        long, _ = self.routing_target_lane.local_coordinates(self.control_object.position)
        advance_ratio = float(self._vehicle_cfg_get("idm_checkpoint_advance_ratio", 0.65))
        advance_ratio = float(np.clip(advance_ratio, 0.3, 0.95))
        near_end = long > self.routing_target_lane.length * advance_ratio

        if not near_end:
            return True

        # Advance to next checkpoint in sequence
        if not hasattr(self, "_checkpoint_idx"):
            self._checkpoint_idx = 0

        next_idx = self._checkpoint_idx + 1
        while next_idx < len(checkpoints):
            next_lane = road_network.get_lane(checkpoints[next_idx])
            if next_lane is not None:
                self._checkpoint_idx = next_idx
                self.routing_target_lane = next_lane
                return True
            next_idx += 1

        return True

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance contro
        ego_vehicle = self.control_object
        lane_to_follow = target_lane
        navigation = getattr(ego_vehicle, "navigation", None)
        reference_trajectory = getattr(navigation, "reference_trajectory", None)
        if reference_trajectory is not None and float(getattr(reference_trajectory, "length", 0.0) or 0.0) > 1.0:
            lane_to_follow = reference_trajectory

        long, lat = lane_to_follow.local_coordinates(ego_vehicle.position)
        lane_length = float(getattr(lane_to_follow, "length", 0.0) or 0.0)
        speed_mps = max(float(getattr(ego_vehicle, "speed", 0.0) or 0.0), 0.0)
        lookahead = float(np.clip(1.0 + 0.3 * speed_mps, 1.0, 4.0))
        heading_long = float(np.clip(long + lookahead, 0.0, max(lane_length, 0.0)))
        lane_heading = lane_to_follow.heading_theta_at(heading_long)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def _vehicle_cfg_get(self, key, default):
        vehicle_cfg = getattr(self.control_object, "config", None)
        if vehicle_cfg is None:
            return default
        getter = getattr(vehicle_cfg, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except Exception:
                return default
        try:
            return vehicle_cfg[key]
        except Exception:
            return default

    def _resolve_lane_speed_limit(self, lane=None):
        lane = lane or self.routing_target_lane or getattr(self.control_object, "lane", None)
        speed_limit = getattr(lane, "speed_limit", None)
        if speed_limit is None:
            speed_limit = self._vehicle_cfg_get("idm_lane_speed_limit_km_h", None)
        try:
            speed_limit = float(speed_limit)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(speed_limit) or speed_limit <= 0:
            return None
        return float(min(speed_limit, self.MAX_SPEED))

    def _compute_target_speed_bounds(self):
        min_speed = float(self._vehicle_cfg_get("idm_target_speed_min_km_h", self.DEFAULT_TARGET_SPEED_MIN))
        max_speed = float(self._vehicle_cfg_get("idm_target_speed_max_km_h", self.DEFAULT_TARGET_SPEED_MAX))
        if max_speed < min_speed:
            min_speed, max_speed = max_speed, min_speed
        max_speed = max(max_speed, min_speed + 1e-3)
        return min_speed, max_speed

    def _sample_cruise_speed(self, lane=None):
        configured_speed = self._vehicle_cfg_get("idm_target_speed_km_h", None)
        if configured_speed is not None:
            try:
                configured_speed = float(configured_speed)
            except (TypeError, ValueError):
                configured_speed = None

        min_speed, max_speed = self._compute_target_speed_bounds()
        lane_speed_limit = self._resolve_lane_speed_limit(lane)

        if configured_speed is None:
            low = min_speed
            high = max_speed
            if self.respect_lane_speed_limit and lane_speed_limit is not None:
                scale_min = float(self._vehicle_cfg_get("idm_lane_speed_scale_min", self.DEFAULT_LANE_SPEED_SCALE_MIN))
                scale_max = float(self._vehicle_cfg_get("idm_lane_speed_scale_max", self.DEFAULT_LANE_SPEED_SCALE_MAX))
                if scale_max < scale_min:
                    scale_min, scale_max = scale_max, scale_min
                low = max(min_speed, lane_speed_limit * scale_min)
                high = min(max_speed, lane_speed_limit * scale_max)
            if high < low:
                fallback_speed = lane_speed_limit if lane_speed_limit is not None else self.NORMAL_SPEED
                low = high = float(np.clip(fallback_speed, min_speed, max_speed))
            configured_speed = low if abs(high - low) < 1e-6 else self.np_random.uniform(low, high)
            speed_jitter = float(self._vehicle_cfg_get("idm_target_speed_jitter_km_h", self.DEFAULT_TARGET_SPEED_JITTER))
            if speed_jitter > 0:
                configured_speed += self.np_random.uniform(-speed_jitter, speed_jitter)

        return float(np.clip(configured_speed, min_speed, max_speed))

    def _initialize_driver_profile(self, force_resample=False):
        if not force_resample and self.cruise_speed_km_h is not None:
            if self.target_speed is None:
                self.target_speed = self._get_cruise_target_speed()
            return

        self.min_target_speed_km_h, self.max_target_speed_km_h = self._compute_target_speed_bounds()
        self.DISTANCE_WANTED = float(self._vehicle_cfg_get("idm_distance_wanted_m", self.DISTANCE_WANTED))
        self.SAFE_DISTANCE = float(self._vehicle_cfg_get("idm_safe_distance_m", self.SAFE_DISTANCE))
        self.EMERGENCY_DISTANCE = float(self._vehicle_cfg_get("idm_emergency_distance_m", self.EMERGENCY_DISTANCE))
        self.MAX_LONG_DIST = float(self._vehicle_cfg_get("idm_max_long_dist_m", self.MAX_LONG_DIST))
        self.FORWARD_LOOKAHEAD_BASE = float(
            self._vehicle_cfg_get("idm_forward_lookahead_base_m", self.FORWARD_LOOKAHEAD_BASE)
        )
        self.FORWARD_LOOKAHEAD_GAIN = float(
            self._vehicle_cfg_get("idm_forward_lookahead_gain_s", self.FORWARD_LOOKAHEAD_GAIN)
        )
        self.FORWARD_LOOKAHEAD_MAX = float(
            self._vehicle_cfg_get("idm_forward_lookahead_max_m", self.FORWARD_LOOKAHEAD_MAX)
        )
        self.INTERSECTION_LOOKAHEAD = float(
            self._vehicle_cfg_get("idm_intersection_lookahead_m", self.INTERSECTION_LOOKAHEAD)
        )
        self.INTERSECTION_DEADLOCK_WAIT_STEPS = int(
            self._vehicle_cfg_get("idm_intersection_deadlock_wait_steps", self.INTERSECTION_DEADLOCK_WAIT_STEPS)
        )
        self.INTERSECTION_RELEASE_TTC_MIN = float(
            self._vehicle_cfg_get("idm_intersection_release_ttc_min_s", self.INTERSECTION_RELEASE_TTC_MIN)
        )
        self.INTERSECTION_CREEP_SPEED_KM_H = float(
            self._vehicle_cfg_get("idm_intersection_creep_speed_km_h", self.INTERSECTION_CREEP_SPEED_KM_H)
        )
        self.respect_lane_speed_limit = bool(self._vehicle_cfg_get("idm_respect_lane_speed_limit", True))
        self.lane_speed_cap_ratio = float(
            self._vehicle_cfg_get("idm_lane_speed_cap_ratio", self.DEFAULT_LANE_SPEED_CAP_RATIO)
        )
        self.target_speed_blend = float(
            np.clip(self._vehicle_cfg_get("idm_target_speed_blend", self.DEFAULT_TARGET_SPEED_BLEND), 0.0, 1.0)
        )
        self.cruise_speed_km_h = self._sample_cruise_speed()

        time_min = float(self._vehicle_cfg_get("idm_time_wanted_min_s", self.DEFAULT_TIME_WANTED_RANGE[0]))
        time_max = float(self._vehicle_cfg_get("idm_time_wanted_max_s", self.DEFAULT_TIME_WANTED_RANGE[1]))
        if time_max < time_min:
            time_min, time_max = time_max, time_min
        self.TIME_WANTED = float(self.np_random.uniform(time_min, time_max))

        adv_min = float(
            self._vehicle_cfg_get(
                "idm_lane_change_adv_min_km_h", self.DEFAULT_LANE_CHANGE_ADVANTAGE_RANGE[0]
            )
        )
        adv_max = float(
            self._vehicle_cfg_get(
                "idm_lane_change_adv_max_km_h", self.DEFAULT_LANE_CHANGE_ADVANTAGE_RANGE[1]
            )
        )
        if adv_max < adv_min:
            adv_min, adv_max = adv_max, adv_min
        self.LANE_CHANGE_MIN_SPEED_ADVANTAGE = float(self.np_random.uniform(adv_min, adv_max))
        self.target_speed = self._get_cruise_target_speed()

    def _get_cruise_target_speed(self, lane=None, speed_scale=1.0):
        if self.cruise_speed_km_h is None:
            self._initialize_driver_profile(force_resample=False)

        target_speed = float(self.cruise_speed_km_h)
        lane_speed_limit = self._resolve_lane_speed_limit(lane)
        if self.respect_lane_speed_limit and lane_speed_limit is not None:
            target_speed = min(target_speed, lane_speed_limit * self.lane_speed_cap_ratio)
        target_speed *= speed_scale
        return float(np.clip(target_speed, self.min_target_speed_km_h, self.max_target_speed_km_h))

    def _set_cruise_target_speed(self, lane=None, speed_scale=1.0):
        cruise_target = self._get_cruise_target_speed(lane=lane, speed_scale=speed_scale)
        if self.target_speed is None:
            self.target_speed = cruise_target
        else:
            blend = float(np.clip(self.target_speed_blend, 0.0, 1.0))
            self.target_speed = float((1.0 - blend) * self.target_speed + blend * cruise_target)
        return self.target_speed

    def _preferred_speed_reference(self, lane=None):
        return self._get_cruise_target_speed(lane=lane)

    def _get_adjacent_lane(self, current_lanes, direction):
        """从当前参考车道中找出与 routing_target_lane 相邻的目标车道。"""
        if self.routing_target_lane is None or not current_lanes:
            return None

        if direction == "left":
            neighbor_indices = set(getattr(self.routing_target_lane, "left_lanes", []) or [])
            offset = -1
        elif direction == "right":
            neighbor_indices = set(getattr(self.routing_target_lane, "right_lanes", []) or [])
            offset = 1
        else:
            return None

        for lane in current_lanes:
            if lane is self.routing_target_lane:
                continue
            if lane.index in neighbor_indices:
                return lane

        try:
            current_idx = current_lanes.index(self.routing_target_lane)
        except ValueError:
            return None

        target_idx = current_idx + offset
        if 0 <= target_idx < len(current_lanes):
            return current_lanes[target_idx]
        return None

    def _resolve_object_lane(self, obj):
        """尽量解析物体所属车道，兼容 `lane` 与 `lane_index` 两种来源。"""
        lane = getattr(obj, "lane", None)
        if lane is not None:
            return lane

        lane_index = getattr(obj, "lane_index", None)
        if lane_index is None:
            return None

        try:
            return self.engine.map_manager.current_map.road_network.get_lane(lane_index)
        except Exception:
            return None

    def _object_matches_lane_path(self, obj, target_lane):
        """判断障碍物是否仍在当前目标车道的前进路径上。"""
        if obj is None or target_lane is None:
            return False

        obj_lane = self._resolve_object_lane(obj)
        if obj_lane is None:
            return False

        return (
            obj_lane is target_lane
            or target_lane.is_previous_lane_of(obj_lane)
            or obj_lane.is_previous_lane_of(target_lane)
        )

    def _merge_longitudinal_control(self, idm_acc, hazard_obj, hazard_distance, steering_target_lane):
        """融合目标车道 IDM 跟驰与几何前向避障。

        换道时优先相信目标车道上的 IDM 跟驰结果，只有在障碍物仍位于目标车道路径上
        或已经逼近紧急距离时，才让几何避障覆盖。
        """
        acc = float(idm_acc)
        if hazard_obj is None or not np.isfinite(hazard_distance):
            return acc

        hazard_acc = float(self._calculate_acceleration(hazard_obj, hazard_distance))
        lane_change_in_progress = (
            steering_target_lane is not None
            and self.routing_target_lane is not None
            and steering_target_lane is not self.routing_target_lane
        )

        if not lane_change_in_progress:
            return min(acc, hazard_acc)

        if hazard_distance < self.EMERGENCY_DISTANCE:
            return min(acc, hazard_acc)

        if self._object_matches_lane_path(hazard_obj, steering_target_lane):
            return min(acc, hazard_acc)

        # 已经明确向相邻车道并线时，不再让当前车道的慢车把换道完全刹停。
        if hazard_distance < self.SAFE_DISTANCE:
            return min(acc, 0.0)
        return acc

    # def acceleration(self, front_obj, dist_to_front) -> float:
    #     """
    #     计算当前车辆的加速度。

    #     参数:
    #         front_obj: 前方物体（如另一辆车）的对象。
    #         dist_to_front: 当前车辆与前方物体之间的距离（单位：米）。

    #     返回:
    #         float: 计算得到的加速度值。
    #     """
    #     ego_vehicle = self.control_object
    #     ego_target_speed = not_zero(self.target_speed, 0)
    #     acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed_km_h, 0) / ego_target_speed,1)) # np.power计算(当前速度 / 目标速度)的DELTA次幂。DELTA值越大，车辆对当前速度与目标速度的差异的反应越平缓；值越小，可以使得反应更为敏感
    #     if front_obj and (not self.disable_idm_deceleration):
    #         d = dist_to_front
    #         speed_diff = self.desired_gap(ego_vehicle, front_obj) / not_zero(d)
    #         acceleration -= self.ACC_FACTOR * (speed_diff**2)
    #     return acceleration

    # def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
    #     d0 = self.DISTANCE_WANTED
    #     tau = self.TIME_WANTED
    #     ab = -self.ACC_FACTOR * self.DEACC_FACTOR
    #     dv = np.dot(ego_vehicle.velocity_km_h - front_obj.velocity_km_h, ego_vehicle.heading) if projected \
    #         else ego_vehicle.speed_km_h - front_obj.speed_km_h
    #     d_star = d0 + ego_vehicle.speed_km_h * tau + ego_vehicle.speed_km_h * dv / (2 * np.sqrt(ab))
    #     return d_star
    
    def acceleration(self, front_obj, dist_to_front) -> float:
        """
        改进的IDM加速度计算，确保能产生负加速度
        """
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0.1)
        ego_speed = max(ego_vehicle.speed_km_h, 0)
        
        # 自由流加速度项
        free_flow_term = np.power(ego_speed / ego_target_speed, self.DELTA)
        
        if front_obj and (not self.disable_idm_deceleration):
            # 确保最小距离
            d = max(dist_to_front, 0.1)
            d_star = self.desired_gap(ego_vehicle, front_obj)
            
            # 交互项
            interaction_term = (d_star / d) ** 2
            
            # 完整IDM公式：a = a_max * [1 - (v/v_0)^δ - (d*/d)^2]
            acceleration = self.ACC_FACTOR * (1 - free_flow_term - interaction_term)
            
            # 调试输出
            # if dist_to_front < 15.0:
            #     print(f"Debug: dist={dist_to_front:.2f}, d_star={d_star:.2f}, "
            #         f"free_flow={free_flow_term:.3f}, interaction={interaction_term:.3f}, "
            #         f"acc={acceleration:.3f}")
            
            # 紧急制动逻辑
            if dist_to_front < 5.0:
                # 当距离很近时，强制制动
                emergency_factor = max(0, (5.0 - dist_to_front) / 5.0)
                max_decel = -abs(self.ACC_FACTOR * self.DEACC_FACTOR)
                emergency_decel = max_decel * emergency_factor
                acceleration = min(acceleration, emergency_decel)
            
            # 静止车辆检测
            if hasattr(front_obj, 'speed_km_h') and front_obj.speed_km_h < 2.0:
                if dist_to_front < 10.0 and ego_speed > 2.0:
                    # 计算停车所需的减速度
                    # v² = u² + 2as => a = (v² - u²) / 2s
                    stop_distance = max(dist_to_front - 1.5, 0.5)  # 保留1.5米安全距离
                    ego_speed_ms = ego_speed / 3.6
                    required_decel = -(ego_speed_ms ** 2) / (2 * stop_distance)
                    acceleration = min(acceleration, required_decel)
                    
                    # print(f"Static object detected: required_decel={required_decel:.3f}")
        else:
            # 无前车，使用自由流加速度
            acceleration = self.ACC_FACTOR * (1 - free_flow_term)
        
        # 限制加速度范围
        max_decel = -abs(self.ACC_FACTOR * self.DEACC_FACTOR)
        max_accel = self.ACC_FACTOR
        acceleration = np.clip(acceleration, max_decel, max_accel)
        
        return acceleration
    
    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        """
        改进的期望间距计算
        """
        d0 = max(self.DISTANCE_WANTED, 2.0)  # 最小保持2米距离
        tau = self.TIME_WANTED
        ab = abs(self.ACC_FACTOR * self.DEACC_FACTOR)  # 确保为正值
        
        # 速度差计算
        if hasattr(front_obj, 'velocity_km_h') and hasattr(front_obj, 'speed_km_h'):
            if projected:
                dv = np.dot(ego_vehicle.velocity_km_h - front_obj.velocity_km_h, ego_vehicle.heading)
            else:
                dv = ego_vehicle.speed_km_h - front_obj.speed_km_h
        else:
            # 假设前方物体静止
            dv = ego_vehicle.speed_km_h
        
        # IDM期望间距公式
        ego_speed_ms = ego_vehicle.speed_km_h / 3.6  # 转换为m/s
        dv_ms = dv / 3.6  # 转换为m/s
        
        if ab > 0:
            d_star = d0 + max(0, ego_speed_ms * tau + ego_speed_ms * dv_ms / (2 * np.sqrt(ab)))
        else:
            d_star = d0 + ego_speed_ms * tau
        
        return max(d_star, d0)  # 确保不小于最小距离

    def reset(self):
        self.heading_pid.reset()
        self.lateral_pid.reset()
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self._initialize_driver_profile(force_resample=True)

    def lane_change_policy(self, all_objects):
        """
        适用于ScenarioMap和PGMap的车道变更策略。
        """
        current_lanes = self.control_object.navigation.current_ref_lanes
        surrounding_objects = FrontBackObjects.get_find_front_back_objs_id(
            all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
        )
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0
        from metadrive.component.map.scenario_map import ScenarioMap
        from metadrive.component.map.pg_map import PGMap
        # 获取当前车道在列表中的索引
        current_lane_idx = current_lanes.index(self.routing_target_lane)

        def lane_follow():
            self._set_cruise_target_speed(self.routing_target_lane)
            self.overtake_timer += 1
            return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane

        # 根据地图类型进行不同的处理
        if isinstance(self.engine.map_manager.current_map,ScenarioMap):
            # 如果下一个导航车道在当前车道的两侧
            if next_lanes and self.routing_target_lane in next_lanes:
                # 如果下一个车道在当前车道的右边
                if next_lanes[0].index in self.routing_target_lane.right_lanes:
                    if self._safe_to_change_lane(surrounding_objects, "right"):
                        self._set_cruise_target_speed(next_lanes[0])
                        self.overtake_timer = 0
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                        next_lanes[0]
                # 如果下一个车道在当前车道的左边
                if next_lanes[0].index in self.routing_target_lane.left_lanes:
                    if self._safe_to_change_lane(surrounding_objects, "left"):
                        self._set_cruise_target_speed(next_lanes[0])
                        self.overtake_timer = 0
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                        next_lanes[0]

            
            # # 如果下一条道路的车道数比当前道路少，则需要进行换道
            #TODO
            # # 1. 强制换道：当车道数减少时
            # if lane_num_diff > 0:
            #     # 计算目标车道范围
            #     if next_lanes and len(next_lanes) > 0:
            #         # 简化逻辑：目标是保持在有效的车道范围内
            #         target_range = list(range(len(next_lanes)))
            #         if current_lane_idx not in target_range:
            #             # 需要换道到有效范围内
            #             if current_lane_idx >= len(next_lanes):
            #                 # 需要向左换道
            #                 target_idx = len(next_lanes) - 1
            #                 if target_idx >= 0 and self._safe_to_change_lane(surrounding_objects, "left"):
            #                     self.target_speed = self.NORMAL_SPEED
            #                     return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
            #                         current_lanes[target_idx]
            #                 else:
            #                     # 不安全，减速等待
            #                     self.target_speed = self.CREEP_SPEED
            #                     return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), \
            #                         self.routing_target_lane

            # 2. 主动换道：超车或避让慢车
            if self._should_attempt_lane_change(surrounding_objects):
                best_choice = None
                # 检查左侧换道
                if self._safe_to_change_lane(surrounding_objects, "left"):
                    left_benefit = self._calculate_lane_change_benefit(surrounding_objects, "left")
                    left_lane = self._get_adjacent_lane(current_lanes, "left")
                    if left_lane is not None:
                        expect_lane_idx = current_lanes.index(left_lane)
                        if expect_lane_idx in self.available_routing_index_range and left_benefit > self.LANE_CHANGE_MIN_SPEED_ADVANTAGE:
                            best_choice = (
                                left_benefit,
                                surrounding_objects.left_front_object(),
                                surrounding_objects.left_front_min_distance(),
                                left_lane,
                            )
                
                # 检查右侧换道
                if self._safe_to_change_lane(surrounding_objects, "right"):
                    right_benefit = self._calculate_lane_change_benefit(surrounding_objects, "right")
                    right_lane = self._get_adjacent_lane(current_lanes, "right")
                    if right_lane is not None:
                        expect_lane_idx = current_lanes.index(right_lane)
                        if expect_lane_idx in self.available_routing_index_range and right_benefit > self.LANE_CHANGE_MIN_SPEED_ADVANTAGE:
                            candidate = (
                                right_benefit,
                                surrounding_objects.right_front_object(),
                                surrounding_objects.right_front_min_distance(),
                                right_lane,
                            )
                            if best_choice is None or candidate[0] > best_choice[0]:
                                best_choice = candidate

                if best_choice is not None:
                    _, front_obj, front_dist, target_lane = best_choice
                    self._set_cruise_target_speed(target_lane)
                    self.overtake_timer = 0
                    return front_obj, front_dist, target_lane

        if isinstance(self.engine.map_manager.current_map,PGMap):
            # 如果下一条道路的车道数比当前道路少，则需要进行换道
            if lane_num_diff > 0:
                # lane num decreasing happened in left road or right road
                if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                    index_range = [i for i in range(len(next_lanes))]
                else:
                    index_range = [i for i in range(lane_num_diff, len(current_lanes))]
                self.available_routing_index_range = index_range
                if self.routing_target_lane.index[-1] not in index_range:
                    # not on suitable lane do lane change !!!
                    if self.routing_target_lane.index[-1] > index_range[-1]:
                        # change to left
                        if surrounding_objects.left_back_min_distance(
                        ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                            # creep to wait
                            self.target_speed = self.CREEP_SPEED
                            return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                            ), self.routing_target_lane
                        else:
                            # it is time to change lane!
                            self._set_cruise_target_speed(current_lanes[self.routing_target_lane.index[-1] - 1])
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[self.routing_target_lane.index[-1] - 1]
                    else:
                        # change to right
                        if surrounding_objects.right_back_min_distance(
                        ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                            # unsafe, creep and wait
                            self.target_speed = self.CREEP_SPEED
                            return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                            ), self.routing_target_lane,
                        else:
                            # change lane
                            self._set_cruise_target_speed(current_lanes[self.routing_target_lane.index[-1] + 1])
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[self.routing_target_lane.index[-1] + 1]

            # lane follow or active change lane/overtake for high driving speed
            preferred_speed = self._preferred_speed_reference(self.routing_target_lane)
            if abs(self.control_object.speed_km_h - preferred_speed) > 3 and surrounding_objects.has_front_object(
            ) and abs(surrounding_objects.front_object().speed_km_h -
                    preferred_speed) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
                # may lane change
                right_front_speed = surrounding_objects.right_front_object().speed_km_h if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                    if surrounding_objects.right_lane_exist() and surrounding_objects.right_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.right_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
                front_speed = surrounding_objects.front_object().speed_km_h if surrounding_objects.has_front_object(
                ) else self.MAX_SPEED
                left_front_speed = surrounding_objects.left_front_object().speed_km_h if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                    if surrounding_objects.left_lane_exist() and surrounding_objects.left_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.left_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    # left overtake has a high priority
                    expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                    if expect_lane_idx in self.available_routing_index_range:
                        self._set_cruise_target_speed(current_lanes[expect_lane_idx])
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                            current_lanes[expect_lane_idx]
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                    if expect_lane_idx in self.available_routing_index_range:
                        self._set_cruise_target_speed(current_lanes[expect_lane_idx])
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                            current_lanes[expect_lane_idx]

        # fall back to lane follow
        return lane_follow()
        
    def _should_attempt_lane_change(self, surrounding_objects):
        """
        判断是否应该尝试换道:
        1、当前速度与正常速度差异较大 且 前方有车辆且其速度与正常速度差异较大
        2、前车很近且很慢
        
        """
        if not surrounding_objects.has_front_object():
            return False

        ego_speed = float(self.control_object.speed_km_h)
        front_obj = surrounding_objects.front_object()
        front_speed = float(getattr(front_obj, "speed_km_h", 0.0))
        front_distance = float(surrounding_objects.front_min_distance())
        preferred_speed = self._preferred_speed_reference(self.routing_target_lane)

        speed_condition = abs(ego_speed - preferred_speed) > 2
        front_speed_condition = front_speed < preferred_speed - 2 or front_speed < ego_speed - 3
        timer_condition = self.overtake_timer > self.LANE_CHANGE_FREQ * 0.5

        close_slow_condition = (
            front_distance < self.LANE_CHANGE_BLOCK_DISTANCE
            and front_speed < ego_speed - 2
        )
        stuck_condition = (
            ego_speed < self.LANE_CHANGE_STUCK_SPEED
            and front_distance < self.LANE_CHANGE_BLOCK_DISTANCE
            and front_speed < preferred_speed - 2
        )

        if close_slow_condition or stuck_condition:
            return True

        if not (speed_condition and front_speed_condition and timer_condition):
            return False

        # 相邻车道只要有一侧存在，就允许进入收益评估。
        return surrounding_objects.left_lane_exist() or surrounding_objects.right_lane_exist()

    def _lane_change_gap_requirement(self, direction):
        """根据当前速度动态调整换道前后向安全距离。"""
        ego_speed_mps = max(float(self.control_object.speed_km_h) / 3.6, 0.0)
        front_gap = max(self.SAFE_LANE_CHANGE_DISTANCE, 8.0 + ego_speed_mps * 0.8)
        back_gap = max(self.SAFE_LANE_CHANGE_DISTANCE, 6.0 + ego_speed_mps * 0.6)
        return front_gap, back_gap

    def _safe_to_change_lane(self, surrounding_objects, direction):
        """检查换道是否安全，根据前后车的距离和车道存在性"""
        front_gap, back_gap = self._lane_change_gap_requirement(direction)
        if direction == "left":
            if not surrounding_objects.left_lane_exist():
                return False
            safe_front = surrounding_objects.left_front_min_distance() > front_gap
            safe_back = surrounding_objects.left_back_min_distance() > back_gap
            return safe_front and safe_back
        elif direction == "right":
            if not surrounding_objects.right_lane_exist():
                return False
            safe_front = surrounding_objects.right_front_min_distance() > front_gap
            safe_back = surrounding_objects.right_back_min_distance() > back_gap
            return safe_front and safe_back
        return False

    def _can_change_left(self, surrounding_objects, current_idx, current_lanes):
        """检查是否可以向左换道"""
        return (current_idx > 0 and 
                current_idx - 1 in self.available_routing_index_range and
                self._safe_to_change_lane(surrounding_objects, "left"))

    def _can_change_right(self, surrounding_objects, current_idx, current_lanes):
        """检查是否可以向右换道"""
        return (current_idx < len(current_lanes) - 1 and 
                current_idx + 1 in self.available_routing_index_range and
                self._safe_to_change_lane(surrounding_objects, "right"))

    def _calculate_lane_change_benefit(self, surrounding_objects, direction):
        """计算换道收益"""
        front_speed = surrounding_objects.front_object().speed_km_h if surrounding_objects.has_front_object() else self.MAX_SPEED
        front_dist = surrounding_objects.front_min_distance()
        
        if direction == "left":
            target_speed = surrounding_objects.left_front_object().speed_km_h if surrounding_objects.has_left_front_object() else self.MAX_SPEED
            target_dist = surrounding_objects.left_front_min_distance()
        elif direction == "right":
            target_speed = surrounding_objects.right_front_object().speed_km_h if surrounding_objects.has_right_front_object() else self.MAX_SPEED
            target_dist = surrounding_objects.right_front_min_distance()
        else:
            return 0
        
        distance_bonus = max(0.0, target_dist - front_dist) * 0.2
        return (target_speed - front_speed) + distance_bonus
   
    # def _find_closest_front_object(self, all_objects):
    #     """
    #     找到前方最近的障碍物
    #     """
    #     ego_pos = self.control_object.position
    #     ego_heading = self.control_object.heading_theta
        
    #     # 前进方向向量
    #     forward_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        
    #     closest_obj = None
    #     min_distance = float('inf')
        
    #     for obj in all_objects:
    #         if hasattr(obj, 'position'):
    #             # 计算相对位置
    #             rel_pos = np.array(obj.position[:2]) - np.array(ego_pos[:2])
    #             distance = np.linalg.norm(rel_pos)
                
    #             # 检查是否在前方（投影到前进方向 > 0）
    #             forward_distance = np.dot(rel_pos, forward_vec)
                
    #             if forward_distance > 0.5 and distance < min_distance:
    #                 # 检查是否在合理的横向范围内（可选）
    #                 lateral_distance = abs(np.dot(rel_pos, np.array([-forward_vec[1], forward_vec[0]])))
    #                 if lateral_distance < 3.0:  # 3米横向范围内
    #                     closest_obj = obj
    #                     min_distance = distance
        
    #     return closest_obj, min_distance
    def _find_closest_front_object(self, all_objects):
        """
        基于几何投影寻找前方障碍物（适用于直道或微弯道）
        """
        # 1. 获取自车信息
        ego_pos = np.array(self.control_object.position[:2])
        ego_heading = self.control_object.heading_theta
        ego_id = self.control_object.id  # 假设对象有ID属性
        
        # 2. 预计算方向向量
        # 前进方向
        forward_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        # 右侧横向向量 (旋转90度: x,y -> y,-x) -> (cos, sin) -> (sin, -cos)
        # 你的写法 [-sin, cos] 是向左还是向右取决于坐标系定义，但在取 abs() 后不影响结果
        right_vec = np.array([-np.sin(ego_heading), np.cos(ego_heading)]) 
        
        closest_obj = None
        min_longitudinal_dist = float('inf') # 改为记录最小纵向距离
        
        for obj in all_objects:
            # 1. 显式过滤掉自己
            if obj.id == ego_id:
                continue
            
            if not hasattr(obj, 'position'):
                continue

            # 2. 计算相对位置向量
            obj_pos = np.array(obj.position[:2])
            rel_pos = obj_pos - ego_pos
            
            # 3. 投影计算纵向和横向距离
            # dot(a, b) = |a|*|b|*cos(theta), 因为 forward_vec 是单位向量，所以就是投影长度
            long_dist = np.dot(rel_pos, forward_vec)
            lat_dist = np.dot(rel_pos, right_vec) # 带符号的横向距离
            
            # 4. 筛选逻辑
            # - long_dist > 0: 在前方
            # - long_dist < min_longitudinal_dist: 比当前找到的更近
            # - abs(lat_dist) < 3.0: 在车道宽度内
            if 0 < long_dist < min_longitudinal_dist and abs(lat_dist) < 3.0:
                min_longitudinal_dist = long_dist
                closest_obj = obj
        
        return closest_obj, min_longitudinal_dist
    
    # def _calculate_acceleration(self, front_obj, distance):
    #     """
    #     基于前方障碍物距离计算加速度，包含静止车辆检测
    #     """
    #     ego_speed = self.control_object.speed_km_h
        
    #     # 添加完全停车逻辑
    #     if ego_speed < 1.0 and front_obj is not None and distance < self.SAFE_DISTANCE:
    #         return 0.0  # 强制停车需要设置为0加速度
        
    #     if front_obj is None:
    #         # 没有前方障碍物，正常加速到目标速度
    #         if ego_speed < self.NORMAL_SPEED:
    #             return self.ACC_FACTOR
    #         else:
    #             return 0.0
        
    #     # 有前方障碍物的情况
    #     if distance < self.EMERGENCY_DISTANCE:
    #         # 紧急制动 - 增强制动力
    #         return -20.0  # 增加制动力到-20
    #     elif distance < self.SAFE_DISTANCE:
    #         # 渐进减速
    #         if ego_speed < 2.0:  # 降低阈值到2.0
    #             return -15.0   # 增强制动力
    #         else:
    #             decel_factor = (self.SAFE_DISTANCE - distance) / (self.SAFE_DISTANCE - self.EMERGENCY_DISTANCE)
    #             acceleration = self.DEACC_FACTOR  * decel_factor * 2.0  # 增强减速效果
    #             return acceleration
    #     else:
    #         # 距离较远时的处理
    #         if ego_speed < self.NORMAL_SPEED:
    #             return self.ACC_FACTOR * 0.5
    #         else:
    #             return 0.0
    def _calculate_acceleration(self, front_obj, distance):
        """
        基于前方障碍物距离计算加速度，包含静止车辆检测和朝向判断
        """
        ego_speed = self.control_object.speed_km_h
        
        # 添加完全停车逻辑
        if ego_speed < 1.0 and front_obj is not None and distance < self.SAFE_DISTANCE:
            # 检查朝向和静止状态
            if self._should_ignore_static_object(front_obj, distance):
                # 忽略该静止物体，继续前进
                return self.ACC_FACTOR * 0.3  # 缓慢前进
            return 0.0  # 强制停车
        # if self._should_ignore_static_object(front_obj, distance):
        #     # print(f"忽略静止物体: 距离={distance:.2f}m, 朝向差={self._get_heading_difference(front_obj):.1f}°")
        #     # 忽略该物体，按无障碍物处理
        #     if ego_speed < self.target_speed:
        #         return self.ACC_FACTOR * 0.3  # 稍微保守的加速
        #     else:
        #         return 0.0
        
        # 有前方障碍物的情况
        if distance < self.EMERGENCY_DISTANCE:
            # 紧急制动 - 增强制动力
            return -5.0  # 增加制动力到-20
        elif distance < self.SAFE_DISTANCE:
            # 渐进减速
            if ego_speed < 2.0:  # 降低阈值到2.0
                return -0.5   # 增强制动力
            else:
                decel_factor = (self.SAFE_DISTANCE - distance) / (self.SAFE_DISTANCE - self.EMERGENCY_DISTANCE)
                acceleration = self.DEACC_FACTOR * decel_factor * 2.0  # 增强减速效果
                
                # 静止车辆检测（原有逻辑保留）
                if hasattr(front_obj, 'speed_km_h') and front_obj.speed_km_h < 2.0:
                    if distance < 10.0 and ego_speed > 1.0:
                        stop_distance = max(distance - 1.0, 0.3)
                        ego_speed_ms = ego_speed / 3.6
                        required_decel = -(ego_speed_ms ** 2) / (2 * stop_distance)
                        required_decel = min(required_decel, -10.0)
                        acceleration = min(acceleration, required_decel)
                        
                        # print(f"Static object detected: distance={distance:.2f}, required_decel={required_decel:.3f}")
                
                return acceleration
        else:
            # 距离较远时的处理
            if ego_speed < self.target_speed:
                return self.ACC_FACTOR * 0.5
            else:
                return 0.0
    
    # def _should_ignore_static_object(self, front_obj, distance):
    #     """
    #     判断是否应该忽略前方的静止物体
    #     条件：
    #     1. 前方物体静止（速度 < 2 km/h）
    #     2. 自车静止（速度 < 2 km/h）
    #     3. 两车朝向角度差较大（> 60度）
    #     4. 距离在合理范围内（2-8米）
    #     """
    #     if front_obj is None:
    #         return False
        
    #     # 检查速度条件
    #     ego_speed = self.control_object.speed_km_h
    #     front_speed = getattr(front_obj, 'speed_km_h', 0)
        
    #     # 两车都静止
    #     if ego_speed < 2.0 and front_speed > 2.0:
    #         return False
        
    #     # # 检查距离范围
    #     # if distance < 2.0 or distance > 8.0:
    #     #     return False
        
    #     # 检查朝向角度差
    #     heading_diff = self._get_heading_difference(front_obj)
    #     if heading_diff < 60.0:  # 60度阈值
    #         return False
        
    #     # 所有条件满足，可以忽略该静止物体
    #     return True
    def _should_ignore_static_object(self, front_obj, distance):
        """
        判断是否应该忽略前方的静止物体
        通常用于解决死锁：当我静止且前方有个横向/对向的静止车辆时，忽略它以便起步。
        
        必须同时满足以下所有条件才返回 True (忽略):
        1. 自车静止 (speed < 2)
        2. 前车静止 (speed < 2)
        3. 两车角度差大 (> 60度，说明不是同向跟车)
        4. 距离适中 (太近了不能忽略，太远了没必要忽略)
        """
        if front_obj is None:
            return False
            
        # 1. 获取速度
        ego_speed = self.control_object.speed_km_h
        front_speed = getattr(front_obj, 'speed_km_h', 0)
        
        # --- 修正逻辑 A：必须两车都静止 ---
        # 如果我在动，绝对不能忽略障碍物（无论它什么角度）-> 安全第一
        if ego_speed > 2.0:
            return False
            
        # 如果前车在动，也不能忽略（哪怕它是横向穿梭的）
        if front_speed > 2.0:
            return False
            
        # --- 修正逻辑 B：距离检查 ---
        # 如果距离太近（比如贴脸 1米），物理上已经撞了或极度危险，不能忽略
        # 如果距离太远（比如 20米），不需要特殊处理，交给正常跟车逻辑即可
        if distance < 2.0 or distance > 10.0:
            return False

        # --- 逻辑 C：角度检查 ---
        heading_diff = self._get_heading_difference(front_obj)
        
        # 如果角度差很小（比如同向车道堵车），不能忽略，必须排队
        # if heading_diff < 60.0:
        #     return False
            
        # 只有同时满足：大家都不动 + 距离合适 + 他是横着的/对向的
        # 这时候才认为他可能是路边的死车或者无关车辆，可以尝试忽略（绕行或起步）
        return True
    
    def _get_heading_difference(self, front_obj):
        """
        计算自车与前方物体的朝向角度差（度）
        """
        if not hasattr(front_obj, 'heading_theta'):
            return 0.0
        
        ego_heading = self.control_object.heading_theta
        front_heading = front_obj.heading_theta
        
        # 计算角度差，转换为度数
        angle_diff = abs(ego_heading - front_heading)
        
        # 处理角度周期性（确保在0-180度范围内）
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        # 转换为度数
        return np.degrees(angle_diff)
    
    def _is_perpendicular_or_opposite(self, front_obj):
        """
        判断前方物体是否与自车垂直或反向
        """
        heading_diff = self._get_heading_difference(front_obj)
        
        # 垂直：80-100度
        is_perpendicular = 80.0 <= heading_diff <= 100.0
        
        # 反向：160-180度  
        is_opposite = 160.0 <= heading_diff <= 180.0
        
        return is_perpendicular or is_opposite
    
    def _classify_static_object(self, front_obj, distance):
        """
        对静止物体进行分类
        """
        if front_obj is None:
            return "none"
        
        ego_speed = self.control_object.speed_km_h
        front_speed = getattr(front_obj, 'speed_km_h', 0)
        
        if ego_speed < 2.0 and front_speed < 2.0:
            heading_diff = self._get_heading_difference(front_obj)
            
            if heading_diff < 30.0:
                return "same_direction"  # 同向静止
            elif 60.0 <= heading_diff <= 120.0:
                return "perpendicular"   # 垂直静止
            elif heading_diff > 150.0:
                return "opposite"        # 反向静止
            else:
                return "angled"          # 斜向静止
        
        return "moving"  # 有一方在移动


class IDMPolicyInOsm(BasePolicy):
    """
    面向 OSM/Scenario 路网的 IDM 策略。

    设计原则：
    1. 车道级 front/back 关系只用于换道收益和安全间隙评估；
    2. 真实纵向避障主要依赖 lidar 的几何前向障碍检测，避免在路口仅凭车道拓扑误判；
    3. 保留原始 IDM 的跟驰结构，但把前向 hazard 仲裁独立出来，方便调试。
    """

    DEBUG_MARK_COLOR = (219, 3, 252, 255)

    TAU_ACC = 0.6
    TAU_HEADING = 0.3
    TAU_LATERAL = 0.8

    TAU_PURSUIT = 0.5 * TAU_HEADING
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL
    MAX_STEERING_ANGLE = np.pi / 3
    DELTA_SPEED = 5

    DISTANCE_WANTED = 10.0
    TIME_WANTED = 1.5
    DELTA = 10.0
    DELTA_RANGE = [3.5, 4.5]

    LANE_CHANGE_FREQ = 50
    LANE_CHANGE_SPEED_INCREASE = 10
    SAFE_LANE_CHANGE_DISTANCE = 15
    MAX_LONG_DIST = 30
    MAX_SPEED = 100

    NORMAL_SPEED = 10
    CREEP_SPEED = 5

    ACC_FACTOR = 1.0
    DEACC_FACTOR = -5

    SAFE_DISTANCE = 10.0
    EMERGENCY_DISTANCE = 5.0
    LANE_CHANGE_BLOCK_DISTANCE = 18.0
    LANE_CHANGE_MIN_SPEED_ADVANTAGE = 2.0
    LANE_CHANGE_STUCK_SPEED = 8.0
    DEFAULT_TARGET_SPEED_MIN = 5 # 15.0
    DEFAULT_TARGET_SPEED_MAX = 5# 40.0
    DEFAULT_TARGET_SPEED_JITTER = 2.0
    DEFAULT_LANE_SPEED_SCALE_MIN = 0.55
    DEFAULT_LANE_SPEED_SCALE_MAX = 0.90
    DEFAULT_LANE_SPEED_CAP_RATIO = 1.00
    DEFAULT_TARGET_SPEED_BLEND = 0.35
    DEFAULT_TIME_WANTED_RANGE = (1.2, 2.0)
    DEFAULT_LANE_CHANGE_ADVANTAGE_RANGE = (1.0, 4.0)

    FORWARD_LOOKAHEAD_BASE = 18.0
    FORWARD_LOOKAHEAD_GAIN = 2.5
    FORWARD_LOOKAHEAD_MAX = 45.0
    INTERSECTION_LOOKAHEAD = 18.0
    FORWARD_LATERAL_MARGIN = 0.8
    INTERSECTION_LATERAL_MARGIN = 2.5
    LANE_CHANGE_REAR_TTC_MIN = 2.0
    LANE_CHANGE_FRONT_TTC_MIN = 1.2
    INTERSECTION_STOP_SPEED_KM_H = 2.5
    INTERSECTION_DEADLOCK_WAIT_STEPS = 45
    INTERSECTION_COMMIT_STEPS = 25
    INTERSECTION_CREEP_SPEED_KM_H = 7.0
    INTERSECTION_RELEASE_DISTANCE_MIN = 2.5
    INTERSECTION_RELEASE_DISTANCE_MAX = 16.0
    INTERSECTION_RELEASE_TTC_MIN = 1.5

    def __init__(self, control_object, random_seed):
        super(IDMPolicyInOsm, self).__init__(control_object=control_object, random_seed=random_seed)
        self.target_speed = None
        self.cruise_speed_km_h = None
        self.min_target_speed_km_h = None
        self.max_target_speed_km_h = None
        self.respect_lane_speed_limit = True
        self.lane_speed_cap_ratio = self.DEFAULT_LANE_SPEED_CAP_RATIO
        self.target_speed_blend = self.DEFAULT_TARGET_SPEED_BLEND
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = self.engine.global_config.get("enable_idm_lane_change", True)
        self.disable_idm_deceleration = self.engine.global_config.get("disable_idm_deceleration", False)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)
        self.intersection_wait_steps = 0
        self.intersection_wait_target = None
        self.intersection_commit_steps = 0
        self.intersection_commit_target = None
        self._initialize_driver_profile(force_resample=True)

    def act(self, *args, **kwargs):
        self._initialize_driver_profile(force_resample=False)
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        # 若有全局 checkpoints，强制用 routing_target_lane（move_to_next_road 已设置）
        nav = self.control_object.navigation
        checkpoints = getattr(nav, "checkpoints", None)
        if checkpoints and self.routing_target_lane is not None:
            steering_target_lane = self.routing_target_lane
            surrounding_objects = FrontBackObjects.get_find_front_back_objs_id_with_lane_index(
                self.engine.map_manager.current_map.road_network,
                all_objects,
                steering_target_lane,
                self.control_object.position,
                max_distance=self.MAX_LONG_DIST,
            )
            route_front_obj = surrounding_objects.front_object()
            route_front_dist = surrounding_objects.front_min_distance()
            self._set_cruise_target_speed(steering_target_lane)
        else:
            try:
                route_front_obj, route_front_dist, steering_target_lane = self._plan_route_context(
                    success, all_objects
                )
            except Exception as exc:
                steering_target_lane = self._default_target_lane()
                route_front_obj = None
                route_front_dist = float("inf")
                self.action_info["planner_error"] = repr(exc)

            if steering_target_lane is None:
                steering_target_lane = self._default_target_lane()

        steering = self.steering_control(steering_target_lane)
        idm_acc = self.acceleration(route_front_obj, route_front_dist)
        lidar_hazard = self._find_lidar_forward_obstacle(all_objects, steering_target_lane)
        acc = self._merge_longitudinal_control(idm_acc, lidar_hazard, steering_target_lane)

        if lidar_hazard is not None:
            self.action_info["hazard_distance"] = float(lidar_hazard["distance"])
            self.action_info["hazard_reason"] = lidar_hazard["reason"]
            self.action_info["hazard_lateral"] = float(lidar_hazard["lateral"])
            self.action_info["hazard_should_yield"] = bool(lidar_hazard.get("should_yield", True))
            self.action_info["hazard_time_gap"] = float(lidar_hazard.get("time_gap", 0.0))
            self.action_info["hazard_dist_conflict"] = float(lidar_hazard.get("ego_dist_conflict", np.inf))

        action = [float(np.clip(steering, -1.0, 1.0)), float(acc)]
        self.action_info["action"] = action
        return action

    def _default_target_lane(self, current_lanes=None):
        current_lanes = list(current_lanes or getattr(self.control_object.navigation, "current_ref_lanes", []) or [])
        if self.routing_target_lane is not None and (
            not current_lanes or self.routing_target_lane in current_lanes
        ):
            return self.routing_target_lane
        if current_lanes:
            return current_lanes[0]
        return getattr(self.control_object, "lane", None)

    def _plan_route_context(self, success, all_objects):
        if success and self.enable_lane_change:
            return self.lane_change_policy(all_objects)

        target_lane = self._default_target_lane()
        if target_lane is None:
            return None, float("inf"), None

        surrounding_objects = FrontBackObjects.get_find_front_back_objs_id_with_lane_index(
            self.engine.map_manager.current_map.road_network,
            all_objects,
            target_lane,
            self.control_object.position,
            max_distance=self.MAX_LONG_DIST
        )
        self._set_cruise_target_speed(target_lane)
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), target_lane

    def move_to_next_road(self):
        """Follow the checkpoint lane sequence from navigation (global path from shortest_path)."""
        nav = self.control_object.navigation
        checkpoints = getattr(nav, "checkpoints", None)
        if not checkpoints:
            # fallback to original logic
            current_lanes = list(getattr(nav, "current_ref_lanes", []) or [])
            if self.routing_target_lane is None:
                self.routing_target_lane = self._default_target_lane(current_lanes)
                return self.routing_target_lane in current_lanes if self.routing_target_lane is not None else False
            return True

        road_network = getattr(self.engine.map_manager.current_map, "road_network", None)
        if road_network is None:
            road_network = getattr(getattr(nav, "map", None), "road_network", None)

        if self.routing_target_lane is None:
            # 跳过车辆已经驶过的 checkpoint（s_noise 可能让车出生在 lane 末尾甚至后续 lane）
            self._checkpoint_idx = 0
            current_lane_idx = str(getattr(self.control_object.lane, "index", None))
            matched = False
            for idx, cp_id in enumerate(checkpoints):
                cp_lane = road_network.get_lane(cp_id) if road_network else None
                if cp_lane is None:
                    continue
                if str(cp_id) == current_lane_idx:
                    try:
                        long, _ = cp_lane.local_coordinates(self.control_object.position)
                    except Exception:
                        long = cp_lane.length
                    self._checkpoint_idx = idx if long < cp_lane.length * 0.5 else idx + 1
                    matched = True
                    break
                try:
                    s, lat = cp_lane.local_coordinates(self.control_object.position)
                    if 0 <= s <= cp_lane.length and abs(lat) < cp_lane.width * 2:
                        self._checkpoint_idx = idx + 1
                        matched = True
                        break
                except Exception:
                    pass
            # 兜底：车辆不在任何 checkpoint lane 上，找第一个能投影且未走完的
            if not matched:
                for idx, cp_id in enumerate(checkpoints):
                    cp_lane = road_network.get_lane(cp_id) if road_network else None
                    if cp_lane is None:
                        continue
                    try:
                        s, _ = cp_lane.local_coordinates(self.control_object.position)
                        if s < cp_lane.length * 0.9:
                            self._checkpoint_idx = idx
                            matched = True
                            break
                    except Exception:
                        pass
            self.routing_target_lane = self.control_object.lane
            if self._checkpoint_idx < len(checkpoints):
                target = road_network.get_lane(checkpoints[self._checkpoint_idx])
                if target is not None:
                    self.routing_target_lane = target
            # print(
            #     f"[idm] {getattr(self.control_object, 'name', self.control_object.id)} "
            #     f"init checkpoint_idx={self._checkpoint_idx} current_lane={current_lane_idx} "
            #     f"routing_target={getattr(self.routing_target_lane, 'index', '?')} "
            #     f"checkpoints={checkpoints[:8]}{'...' if len(checkpoints)>8 else ''}"
            # )
            return True

        # Find vehicle position on current routing_target_lane
        try:
            long, _ = self.routing_target_lane.local_coordinates(self.control_object.position)
            near_end = long > self.routing_target_lane.length * 0.85
        except Exception:
            near_end = True

        if not near_end:
            return True

        # Advance through checkpoint sequence, skipping lanes already passed
        if not hasattr(self, "_checkpoint_idx"):
            self._checkpoint_idx = 0

        next_idx = self._checkpoint_idx + 1
        best_idx = self._checkpoint_idx  # stay put if nothing better
        best_lane = self.routing_target_lane
        while next_idx < len(checkpoints):
            candidate = road_network.get_lane(checkpoints[next_idx]) if road_network else None
            if candidate is not None:
                try:
                    s, lat = candidate.local_coordinates(self.control_object.position)
                    # 能投影上且还没走完 → 可追; 抛异常或投影不上 → 不可追，跳过
                    can_reach = s < candidate.length * 0.9
                except Exception:
                    can_reach = False
                if can_reach:
                    best_idx = next_idx
                    best_lane = candidate
                    break
            next_idx += 1

        if best_idx > self._checkpoint_idx:
            old_idx = self._checkpoint_idx
            self._checkpoint_idx = best_idx
            self.routing_target_lane = best_lane
            # print(
            #     f"[idm] {getattr(self.control_object, 'name', self.control_object.id)} "
            #     f"advance {old_idx}→{best_idx} "
            #     f"lane={checkpoints[old_idx]}→{checkpoints[best_idx]}"
            # )

        return True

    def steering_control(self, target_lane) -> float:
        ego_vehicle = self.control_object
        target_lane = target_lane or self._default_target_lane()
        if target_lane is None:
            return 0.0
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def _vehicle_cfg_get(self, key, default):
        vehicle_cfg = getattr(self.control_object, "config", None)
        if vehicle_cfg is None:
            return default
        getter = getattr(vehicle_cfg, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except Exception:
                return default
        try:
            return vehicle_cfg[key]
        except Exception:
            return default

    def _resolve_lane_speed_limit(self, lane=None):
        lane = lane or self.routing_target_lane or getattr(self.control_object, "lane", None)
        speed_limit = getattr(lane, "speed_limit", None)
        if speed_limit is None:
            speed_limit = self._vehicle_cfg_get("idm_lane_speed_limit_km_h", None)
        try:
            speed_limit = float(speed_limit)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(speed_limit) or speed_limit <= 0:
            return None
        return float(min(speed_limit, self.MAX_SPEED))

    def _compute_target_speed_bounds(self):
        min_speed = float(self._vehicle_cfg_get("idm_target_speed_min_km_h", self.DEFAULT_TARGET_SPEED_MIN))
        max_speed = float(self._vehicle_cfg_get("idm_target_speed_max_km_h", self.DEFAULT_TARGET_SPEED_MAX))
        if max_speed < min_speed:
            min_speed, max_speed = max_speed, min_speed
        max_speed = max(max_speed, min_speed + 1e-3)
        return min_speed, max_speed

    def _sample_cruise_speed(self, lane=None):
        configured_speed = self._vehicle_cfg_get("idm_target_speed_km_h", None)
        if configured_speed is not None:
            try:
                configured_speed = float(configured_speed)
            except (TypeError, ValueError):
                configured_speed = None

        min_speed, max_speed = self._compute_target_speed_bounds()
        lane_speed_limit = self._resolve_lane_speed_limit(lane)

        if configured_speed is None:
            low = min_speed
            high = max_speed
            if self.respect_lane_speed_limit and lane_speed_limit is not None:
                scale_min = float(self._vehicle_cfg_get("idm_lane_speed_scale_min", self.DEFAULT_LANE_SPEED_SCALE_MIN))
                scale_max = float(self._vehicle_cfg_get("idm_lane_speed_scale_max", self.DEFAULT_LANE_SPEED_SCALE_MAX))
                if scale_max < scale_min:
                    scale_min, scale_max = scale_max, scale_min
                low = max(min_speed, lane_speed_limit * scale_min)
                high = min(max_speed, lane_speed_limit * scale_max)
            if high < low:
                fallback_speed = lane_speed_limit if lane_speed_limit is not None else self.NORMAL_SPEED
                low = high = float(np.clip(fallback_speed, min_speed, max_speed))
            configured_speed = low if abs(high - low) < 1e-6 else self.np_random.uniform(low, high)
            speed_jitter = float(self._vehicle_cfg_get("idm_target_speed_jitter_km_h", self.DEFAULT_TARGET_SPEED_JITTER))
            if speed_jitter > 0:
                configured_speed += self.np_random.uniform(-speed_jitter, speed_jitter)

        return float(np.clip(configured_speed, min_speed, max_speed))

    def _initialize_driver_profile(self, force_resample=False):
        if not force_resample and self.cruise_speed_km_h is not None:
            if self.target_speed is None:
                self.target_speed = self._get_cruise_target_speed()
            return

        self.min_target_speed_km_h, self.max_target_speed_km_h = self._compute_target_speed_bounds()
        self.DISTANCE_WANTED = float(self._vehicle_cfg_get("idm_distance_wanted_m", self.DISTANCE_WANTED))
        self.SAFE_DISTANCE = float(self._vehicle_cfg_get("idm_safe_distance_m", self.SAFE_DISTANCE))
        self.EMERGENCY_DISTANCE = float(self._vehicle_cfg_get("idm_emergency_distance_m", self.EMERGENCY_DISTANCE))
        self.MAX_LONG_DIST = float(self._vehicle_cfg_get("idm_max_long_dist_m", self.MAX_LONG_DIST))
        self.FORWARD_LOOKAHEAD_BASE = float(
            self._vehicle_cfg_get("idm_forward_lookahead_base_m", self.FORWARD_LOOKAHEAD_BASE)
        )
        self.FORWARD_LOOKAHEAD_GAIN = float(
            self._vehicle_cfg_get("idm_forward_lookahead_gain_s", self.FORWARD_LOOKAHEAD_GAIN)
        )
        self.FORWARD_LOOKAHEAD_MAX = float(
            self._vehicle_cfg_get("idm_forward_lookahead_max_m", self.FORWARD_LOOKAHEAD_MAX)
        )
        self.INTERSECTION_LOOKAHEAD = float(
            self._vehicle_cfg_get("idm_intersection_lookahead_m", self.INTERSECTION_LOOKAHEAD)
        )
        self.INTERSECTION_DEADLOCK_WAIT_STEPS = int(
            self._vehicle_cfg_get("idm_intersection_deadlock_wait_steps", self.INTERSECTION_DEADLOCK_WAIT_STEPS)
        )
        self.INTERSECTION_RELEASE_TTC_MIN = float(
            self._vehicle_cfg_get("idm_intersection_release_ttc_min_s", self.INTERSECTION_RELEASE_TTC_MIN)
        )
        self.INTERSECTION_CREEP_SPEED_KM_H = float(
            self._vehicle_cfg_get("idm_intersection_creep_speed_km_h", self.INTERSECTION_CREEP_SPEED_KM_H)
        )
        self.respect_lane_speed_limit = bool(self._vehicle_cfg_get("idm_respect_lane_speed_limit", True))
        self.lane_speed_cap_ratio = float(
            self._vehicle_cfg_get("idm_lane_speed_cap_ratio", self.DEFAULT_LANE_SPEED_CAP_RATIO)
        )
        self.target_speed_blend = float(
            np.clip(self._vehicle_cfg_get("idm_target_speed_blend", self.DEFAULT_TARGET_SPEED_BLEND), 0.0, 1.0)
        )
        self.cruise_speed_km_h = self._sample_cruise_speed()

        time_min = float(self._vehicle_cfg_get("idm_time_wanted_min_s", self.DEFAULT_TIME_WANTED_RANGE[0]))
        time_max = float(self._vehicle_cfg_get("idm_time_wanted_max_s", self.DEFAULT_TIME_WANTED_RANGE[1]))
        if time_max < time_min:
            time_min, time_max = time_max, time_min
        self.TIME_WANTED = float(self.np_random.uniform(time_min, time_max))

        adv_min = float(
            self._vehicle_cfg_get("idm_lane_change_adv_min_km_h", self.DEFAULT_LANE_CHANGE_ADVANTAGE_RANGE[0])
        )
        adv_max = float(
            self._vehicle_cfg_get("idm_lane_change_adv_max_km_h", self.DEFAULT_LANE_CHANGE_ADVANTAGE_RANGE[1])
        )
        if adv_max < adv_min:
            adv_min, adv_max = adv_max, adv_min
        self.LANE_CHANGE_MIN_SPEED_ADVANTAGE = float(self.np_random.uniform(adv_min, adv_max))
        self.target_speed = self._get_cruise_target_speed()

    def _get_cruise_target_speed(self, lane=None, speed_scale=1.0):
        if self.cruise_speed_km_h is None:
            self._initialize_driver_profile(force_resample=False)

        target_speed = float(self.cruise_speed_km_h)
        lane_speed_limit = self._resolve_lane_speed_limit(lane)
        if self.respect_lane_speed_limit and lane_speed_limit is not None:
            target_speed = min(target_speed, lane_speed_limit * self.lane_speed_cap_ratio)
        target_speed *= speed_scale
        return float(np.clip(target_speed, self.min_target_speed_km_h, self.max_target_speed_km_h))

    def _set_cruise_target_speed(self, lane=None, speed_scale=1.0):
        cruise_target = self._get_cruise_target_speed(lane=lane, speed_scale=speed_scale)
        if self.target_speed is None:
            self.target_speed = cruise_target
        else:
            blend = float(np.clip(self.target_speed_blend, 0.0, 1.0))
            self.target_speed = float((1.0 - blend) * self.target_speed + blend * cruise_target)
        return self.target_speed

    def _preferred_speed_reference(self, lane=None):
        return self._get_cruise_target_speed(lane=lane)

    def _get_adjacent_lane(self, current_lanes, direction):
        if self.routing_target_lane is None or not current_lanes:
            return None

        if direction == "left":
            neighbor_indices = set(getattr(self.routing_target_lane, "left_lanes", []) or [])
            offset = -1
        elif direction == "right":
            neighbor_indices = set(getattr(self.routing_target_lane, "right_lanes", []) or [])
            offset = 1
        else:
            return None

        for lane in current_lanes:
            if lane is self.routing_target_lane:
                continue
            if lane.index in neighbor_indices:
                return lane

        try:
            current_idx = current_lanes.index(self.routing_target_lane)
        except ValueError:
            return None

        target_idx = current_idx + offset
        if 0 <= target_idx < len(current_lanes):
            return current_lanes[target_idx]
        return None

    def _resolve_object_lane(self, obj):
        lane = getattr(obj, "lane", None)
        if lane is not None:
            return lane

        lane_index = getattr(obj, "lane_index", None)
        if lane_index is None:
            return None
        try:
            return self.engine.map_manager.current_map.road_network.get_lane(lane_index)
        except Exception:
            return None

    def _object_matches_lane_path(self, obj, target_lane):
        if obj is None or target_lane is None:
            return False
        obj_lane = self._resolve_object_lane(obj)
        if obj_lane is None:
            return False
        return (
            obj_lane is target_lane
            or target_lane.is_previous_lane_of(obj_lane)
            or obj_lane.is_previous_lane_of(target_lane)
        )

    def _lane_width(self, lane=None):
        lane = lane or self.routing_target_lane or getattr(self.control_object, "lane", None)
        width = getattr(lane, "width", None)
        try:
            width = float(width)
        except (TypeError, ValueError):
            width = 4.0
        return max(width, 3.0)

    def _object_half_extent(self, obj):
        width = getattr(obj, "WIDTH", getattr(obj, "top_down_width", 2.0))
        length = getattr(obj, "LENGTH", getattr(obj, "top_down_length", 4.0))
        try:
            width = float(width)
        except (TypeError, ValueError):
            width = 2.0
        try:
            length = float(length)
        except (TypeError, ValueError):
            length = 4.0
        return max(width * 0.5, 0.4), max(length * 0.5, 0.6)

    def _relative_kinematics(self, obj):
        ego_pos = np.asarray(self.control_object.position[:2], dtype=np.float64)
        ego_heading = float(self.control_object.heading_theta)
        forward_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)
        right_vec = np.array([-np.sin(ego_heading), np.cos(ego_heading)], dtype=np.float64)

        obj_pos = np.asarray(obj.position[:2], dtype=np.float64)
        rel_pos = obj_pos - ego_pos
        longitudinal = float(np.dot(rel_pos, forward_vec))
        lateral = float(np.dot(rel_pos, right_vec))

        ego_vel = np.asarray(self.control_object.velocity[:2], dtype=np.float64)
        obj_vel = np.asarray(getattr(obj, "velocity", [0.0, 0.0])[:2], dtype=np.float64)
        rel_vel = ego_vel - obj_vel
        closing_speed = float(np.dot(rel_vel, forward_vec))
        return longitudinal, lateral, closing_speed

    def _crossing_conflict_state(self, obj, longitudinal, lateral, target_lane, obj_half_width, obj_half_length):
        lane_width = self._lane_width(target_lane)
        corridor_half = lane_width * 0.5 + self.FORWARD_LATERAL_MARGIN + obj_half_width

        ego_heading = float(self.control_object.heading_theta)
        forward_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)
        right_vec = np.array([-np.sin(ego_heading), np.cos(ego_heading)], dtype=np.float64)
        obj_vel = np.asarray(getattr(obj, "velocity", [0.0, 0.0])[:2], dtype=np.float64)
        obj_long_speed = float(np.dot(obj_vel, forward_vec))
        obj_lat_speed = float(np.dot(obj_vel, right_vec))
        obj_speed_km_h = float(getattr(obj, "speed_km_h", 0.0))
        ego_speed_mps = max(float(self.control_object.speed), 0.0)

        intruding_now = abs(lateral) <= corridor_half and longitudinal <= self.SAFE_DISTANCE
        moving_toward_path = abs(obj_lat_speed) > 0.15 and lateral * obj_lat_speed < 0.0

        if obj_speed_km_h < self.INTERSECTION_STOP_SPEED_KM_H and not intruding_now:
            return {
                "ignore": True,
                "should_yield": False,
                "reason": "intersection_waiting_outside",
                "ego_t_conflict": np.inf,
                "obj_t_conflict": np.inf,
                "time_gap": np.inf,
                "ego_dist_conflict": np.inf,
            }

        if not moving_toward_path:
            if intruding_now:
                return {
                    "ignore": False,
                    "should_yield": True,
                    "reason": "intersection_intruding",
                    "ego_t_conflict": 0.0,
                    "obj_t_conflict": 0.0,
                    "time_gap": 0.0,
                    "ego_dist_conflict": 0.0,
                }
            return {
                "ignore": True,
                "should_yield": False,
                "reason": "intersection_moving_away",
                "ego_t_conflict": np.inf,
                "obj_t_conflict": np.inf,
                "time_gap": np.inf,
                "ego_dist_conflict": np.inf,
            }

        lateral_clearance = max(abs(lateral) - corridor_half, 0.0)
        obj_t_conflict = lateral_clearance / max(abs(obj_lat_speed), 1e-3)
        x_conflict = longitudinal + obj_long_speed * obj_t_conflict
        if x_conflict < -obj_half_length or x_conflict > self.FORWARD_LOOKAHEAD_MAX:
            return {
                "ignore": True,
                "should_yield": False,
                "reason": "intersection_far_conflict",
                "ego_t_conflict": np.inf,
                "obj_t_conflict": obj_t_conflict,
                "time_gap": np.inf,
                "ego_dist_conflict": np.inf,
            }

        ego_front_clearance = max(self.control_object.LENGTH * 0.45, 1.5)
        ego_dist_conflict = max(x_conflict - ego_front_clearance, 0.0)
        ego_t_conflict = ego_dist_conflict / max(ego_speed_mps, 0.1)
        time_gap = obj_t_conflict - ego_t_conflict
        should_yield = obj_t_conflict <= ego_t_conflict + 0.8

        return {
            "ignore": False,
            "should_yield": bool(should_yield),
            "reason": "intersection_corridor",
            "ego_t_conflict": float(ego_t_conflict),
            "obj_t_conflict": float(obj_t_conflict),
            "time_gap": float(time_gap),
            "ego_dist_conflict": float(ego_dist_conflict),
        }

    def _find_lidar_forward_obstacle(self, all_objects, target_lane=None):
        target_lane = target_lane or self.routing_target_lane
        lookahead = min(
            self.FORWARD_LOOKAHEAD_MAX,
            max(self.FORWARD_LOOKAHEAD_BASE, self.control_object.speed / 1.0 * self.FORWARD_LOOKAHEAD_GAIN)
        )
        lane_width = self._lane_width(target_lane)
        half_corridor = lane_width * 0.5 + self.FORWARD_LATERAL_MARGIN
        intersection_half_corridor = half_corridor + self.INTERSECTION_LATERAL_MARGIN

        ego_id = getattr(self.control_object, "id", None)
        best = None
        best_score = float("inf")

        for obj in all_objects:
            if obj is None or not hasattr(obj, "position"):
                continue
            if obj is self.control_object:
                continue
            if ego_id is not None and getattr(obj, "id", None) == ego_id:
                continue

            longitudinal, lateral, closing_speed = self._relative_kinematics(obj)
            if longitudinal <= 0.0 or longitudinal > lookahead:
                continue

            obj_half_width, obj_half_length = self._object_half_extent(obj)
            lane_path_match = self._object_matches_lane_path(obj, target_lane)
            heading_diff = self._get_heading_difference(obj)

            lateral_limit = half_corridor + obj_half_width
            reason = "lane_path"
            if lane_path_match:
                if abs(lateral) > lateral_limit:
                    continue
            else:
                crossing_like = heading_diff > 35.0
                if longitudinal > self.INTERSECTION_LOOKAHEAD:
                    continue
                if abs(lateral) > (intersection_half_corridor + obj_half_width):
                    continue
                if not crossing_like and abs(lateral) > lateral_limit:
                    continue
                if crossing_like:
                    crossing_state = self._crossing_conflict_state(
                        obj,
                        longitudinal,
                        lateral,
                        target_lane,
                        obj_half_width,
                        obj_half_length,
                    )
                    if crossing_state.get("ignore", False):
                        continue
                    reason = crossing_state["reason"]
                else:
                    crossing_state = {
                        "ignore": False,
                        "should_yield": True,
                        "reason": "forward_corridor",
                        "ego_t_conflict": np.inf,
                        "obj_t_conflict": np.inf,
                        "time_gap": 0.0,
                        "ego_dist_conflict": np.inf,
                    }
                    reason = "forward_corridor"

            distance = max(longitudinal - obj_half_length, 0.0)
            score = distance + abs(lateral) * 0.25
            if score < best_score:
                ttc = np.inf
                if closing_speed > 1e-3:
                    ttc = distance / closing_speed
                best_score = score
                best = {
                    "object": obj,
                    "distance": distance,
                    "longitudinal": longitudinal,
                    "lateral": lateral,
                    "closing_speed": closing_speed,
                    "ttc": ttc,
                    "lane_path_match": lane_path_match,
                    "heading_diff": heading_diff,
                    "reason": reason,
                    "should_yield": crossing_state["should_yield"] if not lane_path_match else True,
                    "ego_t_conflict": crossing_state["ego_t_conflict"] if not lane_path_match else np.inf,
                    "obj_t_conflict": crossing_state["obj_t_conflict"] if not lane_path_match else np.inf,
                    "time_gap": crossing_state["time_gap"] if not lane_path_match else 0.0,
                    "ego_dist_conflict": crossing_state["ego_dist_conflict"] if not lane_path_match else np.inf,
                }

        return best

    def _hazard_brake_acceleration(self, hazard):
        if hazard is None:
            return self.ACC_FACTOR

        distance = float(hazard["distance"])
        ego_speed_ms = max(float(self.control_object.speed) , 0.0)
        max_decel = -abs(self.ACC_FACTOR * self.DEACC_FACTOR)
        if distance <= self.EMERGENCY_DISTANCE:
            return max_decel

        clearance = max(1.5, 0.15 * max(self.control_object.LENGTH, 4.0))
        stop_distance = max(distance - clearance, 0.5)
        required_decel = -(ego_speed_ms ** 2) / (2.0 * stop_distance) if ego_speed_ms > 1e-6 else 0.0

        ttc = float(hazard["ttc"])
        brake_acc = min(0.0, required_decel)
        if np.isfinite(ttc):
            if ttc < self.TIME_WANTED:
                brake_acc = min(brake_acc, -2.0)
            if ttc < 0.8:
                brake_acc = min(brake_acc, max_decel)

        if not hazard["lane_path_match"] and distance < self.SAFE_DISTANCE:
            brake_acc = min(brake_acc, -2.5)

        if (
            not hazard["lane_path_match"]
            and hazard.get("reason") in ("intersection_corridor", "intersection_waiting_outside")
            and not hazard.get("should_yield", True)
        ):
            return self.ACC_FACTOR

        if (
            not hazard["lane_path_match"]
            and hazard.get("should_yield", True)
            and np.isfinite(hazard.get("ego_t_conflict", np.inf))
        ):
            dist_to_conflict = max(float(hazard.get("ego_dist_conflict", distance)), 0.0)
            stop_clearance = max(1.0, 0.20 * max(self.control_object.LENGTH, 4.0))
            brake_distance = max(dist_to_conflict - stop_clearance, 0.5)
            conflict_stop_decel = -(ego_speed_ms ** 2) / (2.0 * brake_distance) if ego_speed_ms > 1e-6 else 0.0
            brake_acc = min(brake_acc, conflict_stop_decel)
            ego_t_conflict = float(hazard.get("ego_t_conflict", np.inf))
            obj_t_conflict = float(hazard.get("obj_t_conflict", np.inf))
            if np.isfinite(ego_t_conflict) and np.isfinite(obj_t_conflict):
                if obj_t_conflict - ego_t_conflict < 0.5:
                    brake_acc = min(brake_acc, -3.5)
                if obj_t_conflict - ego_t_conflict < 0.2 or ego_t_conflict < 1.2:
                    brake_acc = min(brake_acc, max_decel)

        return float(np.clip(brake_acc, max_decel, self.ACC_FACTOR))

    def _hazard_token(self, obj):
        if obj is None:
            return None
        obj_id = getattr(obj, "id", None)
        if obj_id is not None:
            return str(obj_id)
        obj_name = getattr(obj, "name", None)
        if obj_name is not None:
            return str(obj_name)
        return str(id(obj))

    def _reset_intersection_deadlock_state(self, keep_commit=False):
        self.intersection_wait_steps = 0
        self.intersection_wait_target = None
        if not keep_commit:
            self.intersection_commit_steps = 0
            self.intersection_commit_target = None

    def _intersection_priority_allows_go(self, hazard):
        hazard_obj = hazard.get("object", None)
        my_token = self._hazard_token(self.control_object)
        other_token = self._hazard_token(hazard_obj)
        if my_token is None or other_token is None:
            return True
        return my_token <= other_token

    def _intersection_creep_acceleration(self):
        ego_speed = float(self.control_object.speed_km_h)
        if ego_speed < self.INTERSECTION_CREEP_SPEED_KM_H - 0.5:
            return min(self.ACC_FACTOR * 0.45, 0.45)
        if ego_speed < self.INTERSECTION_CREEP_SPEED_KM_H + 0.5:
            return 0.05
        return 0.0

    def _intersection_deadlock_release_acceleration(self, base_acc, hazard):
        if hazard is None:
            self._reset_intersection_deadlock_state()
            return None

        is_intersection_hazard = (
            hazard.get("reason") == "intersection_corridor"
            and not bool(hazard.get("lane_path_match", False))
        )
        if not is_intersection_hazard:
            self._reset_intersection_deadlock_state()
            return None

        hazard_obj = hazard.get("object", None)
        hazard_token = self._hazard_token(hazard_obj)
        hazard_distance = float(hazard.get("distance", np.inf))
        hazard_ttc = float(hazard.get("ttc", np.inf))
        ego_speed = float(self.control_object.speed_km_h)
        other_speed = float(getattr(hazard_obj, "speed_km_h", 0.0)) if hazard_obj is not None else 0.0

        moving_conflict = (
            other_speed > self.INTERSECTION_STOP_SPEED_KM_H
            or (np.isfinite(hazard_ttc) and hazard_ttc < self.INTERSECTION_RELEASE_TTC_MIN)
        )
        base_blocked = float(base_acc) < -0.15
        in_release_window = (
            self.INTERSECTION_RELEASE_DISTANCE_MIN <= hazard_distance <= self.INTERSECTION_RELEASE_DISTANCE_MAX
        )
        deadlock_candidate = (
            in_release_window
            and ego_speed < self.INTERSECTION_STOP_SPEED_KM_H
            and other_speed < self.INTERSECTION_STOP_SPEED_KM_H
            and not moving_conflict
            and not base_blocked
        )

        if self.intersection_commit_steps > 0:
            if (
                hazard_token == self.intersection_commit_target
                and hazard_distance > self.EMERGENCY_DISTANCE
                and not moving_conflict
            ):
                self.intersection_commit_steps -= 1
                self.action_info["intersection_mode"] = "commit_cross"
                self.action_info["intersection_commit_steps"] = self.intersection_commit_steps
                return self._intersection_creep_acceleration()
            self.intersection_commit_steps = 0
            self.intersection_commit_target = None

        if not deadlock_candidate:
            self._reset_intersection_deadlock_state(keep_commit=True)
            return None

        if hazard_token == self.intersection_wait_target:
            self.intersection_wait_steps += 1
        else:
            self.intersection_wait_target = hazard_token
            self.intersection_wait_steps = 1

        self.action_info["intersection_mode"] = "wait_cross"
        self.action_info["intersection_wait_steps"] = self.intersection_wait_steps

        if self.intersection_wait_steps < self.INTERSECTION_DEADLOCK_WAIT_STEPS:
            return None

        if not self._intersection_priority_allows_go(hazard):
            self.action_info["intersection_mode"] = "yield_by_priority"
            return None

        self.intersection_wait_steps = 0
        self.intersection_wait_target = None
        self.intersection_commit_steps = self.INTERSECTION_COMMIT_STEPS
        self.intersection_commit_target = hazard_token
        self.action_info["intersection_mode"] = "release_cross"
        self.action_info["intersection_commit_steps"] = self.intersection_commit_steps
        return self._intersection_creep_acceleration()

    def _merge_longitudinal_control(self, idm_acc, hazard, steering_target_lane):
        acc = float(idm_acc)
        release_acc = self._intersection_deadlock_release_acceleration(acc, hazard)
        if release_acc is not None:
            return float(release_acc)
        if hazard is None:
            return acc

        hazard_acc = float(self._hazard_brake_acceleration(hazard))
        lane_change_in_progress = (
            steering_target_lane is not None
            and self.routing_target_lane is not None
            and steering_target_lane is not self.routing_target_lane
        )

        if not lane_change_in_progress:
            return min(acc, hazard_acc)
        if hazard["lane_path_match"]:
            return min(acc, hazard_acc)
        if hazard["distance"] < self.SAFE_DISTANCE:
            return min(acc, hazard_acc)
        return min(acc, max(hazard_acc, 0.0))

    def acceleration(self, front_obj, dist_to_front) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0.1)
        ego_speed = max(float(ego_vehicle.speed_km_h), 0.0)
        free_flow_term = np.power(ego_speed / ego_target_speed, self.DELTA)
        acceleration = self.ACC_FACTOR * (1.0 - free_flow_term)

        if front_obj is not None and (not self.disable_idm_deceleration) and np.isfinite(dist_to_front):
            d = max(float(dist_to_front), 0.1)
            d_star = self.desired_gap(ego_vehicle, front_obj)
            interaction_term = (d_star / d) ** 2
            acceleration -= self.ACC_FACTOR * interaction_term

        max_decel = -abs(self.ACC_FACTOR * self.DEACC_FACTOR)
        max_accel = self.ACC_FACTOR
        return float(np.clip(acceleration, max_decel, max_accel))

    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        d0 = max(self.DISTANCE_WANTED, 2.0)
        tau = self.TIME_WANTED
        ab = abs(self.ACC_FACTOR * self.DEACC_FACTOR)

        if hasattr(front_obj, "velocity_km_h") and hasattr(front_obj, "speed_km_h"):
            if projected:
                dv = np.dot(ego_vehicle.velocity_km_h - front_obj.velocity_km_h, ego_vehicle.heading)
            else:
                dv = ego_vehicle.speed_km_h - front_obj.speed_km_h
        else:
            dv = ego_vehicle.speed_km_h

        ego_speed_ms = ego_vehicle.speed_km_h / 3.6
        dv_ms = dv / 3.6
        if ab > 0:
            d_star = d0 + max(0.0, ego_speed_ms * tau + ego_speed_ms * dv_ms / (2.0 * np.sqrt(ab)))
        else:
            d_star = d0 + ego_speed_ms * tau
        return float(max(d_star, d0))

    def reset(self):
        self.heading_pid.reset()
        self.lateral_pid.reset()
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self._reset_intersection_deadlock_state()
        self._initialize_driver_profile(force_resample=True)

    def lane_change_policy(self, all_objects):
        current_lanes = list(getattr(self.control_object.navigation, "current_ref_lanes", []) or [])
        if not current_lanes:
            target_lane = self._default_target_lane()
            self._set_cruise_target_speed(target_lane)
            return None, float("inf"), target_lane

        if self.routing_target_lane not in current_lanes:
            self.routing_target_lane = self._default_target_lane(current_lanes)

        surrounding_objects = FrontBackObjects.get_find_front_back_objs_id(
            all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
        )
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = list(getattr(self.control_object.navigation, "next_ref_lanes", []) or [])
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes else 0

        def lane_follow():
            self._set_cruise_target_speed(self.routing_target_lane)
            self.overtake_timer += 1
            return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane

        from metadrive.component.map.pg_map import PGMap
        from metadrive.component.map.scenario_map import ScenarioMap

        current_map = self.engine.map_manager.current_map
        if isinstance(current_map, ScenarioMap):
            route_choice = self._try_route_lane_change_for_scenario(next_lanes, surrounding_objects)
            if route_choice is not None:
                return route_choice

            active_choice = self._choose_active_lane_change(current_lanes, surrounding_objects)
            if active_choice is not None:
                return active_choice

        if isinstance(current_map, PGMap):
            forced_choice = self._try_force_lane_change_for_pg(current_lanes, next_lanes, lane_num_diff, surrounding_objects)
            if forced_choice is not None:
                return forced_choice

            active_choice = self._choose_active_lane_change(current_lanes, surrounding_objects)
            if active_choice is not None:
                return active_choice

        return lane_follow()

    def _try_route_lane_change_for_scenario(self, next_lanes, surrounding_objects):
        if not next_lanes or self.routing_target_lane is None or self.routing_target_lane not in next_lanes:
            return None

        if next_lanes[0].index in getattr(self.routing_target_lane, "right_lanes", []):
            if self._safe_to_change_lane(surrounding_objects, "right"):
                self._set_cruise_target_speed(next_lanes[0])
                self.overtake_timer = 0
                return (
                    surrounding_objects.right_front_object(),
                    surrounding_objects.right_front_min_distance(),
                    next_lanes[0],
                )

        if next_lanes[0].index in getattr(self.routing_target_lane, "left_lanes", []):
            if self._safe_to_change_lane(surrounding_objects, "left"):
                self._set_cruise_target_speed(next_lanes[0])
                self.overtake_timer = 0
                return (
                    surrounding_objects.left_front_object(),
                    surrounding_objects.left_front_min_distance(),
                    next_lanes[0],
                )
        return None

    def _try_force_lane_change_for_pg(self, current_lanes, next_lanes, lane_num_diff, surrounding_objects):
        if lane_num_diff <= 0 or not next_lanes or self.routing_target_lane is None:
            return None

        if current_lanes[0].is_previous_lane_of(next_lanes[0]):
            index_range = [i for i in range(len(next_lanes))]
        else:
            index_range = [i for i in range(lane_num_diff, len(current_lanes))]
        self.available_routing_index_range = index_range

        lane_index = self.routing_target_lane.index[-1]
        if lane_index in index_range:
            return None

        if lane_index > index_range[-1]:
            if not self._safe_to_change_lane(surrounding_objects, "left"):
                self.target_speed = self.CREEP_SPEED
                return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane
            target_lane = current_lanes[lane_index - 1]
            self._set_cruise_target_speed(target_lane)
            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), target_lane

        if not self._safe_to_change_lane(surrounding_objects, "right"):
            self.target_speed = self.CREEP_SPEED
            return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane
        target_lane = current_lanes[lane_index + 1]
        self._set_cruise_target_speed(target_lane)
        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), target_lane

    def _choose_active_lane_change(self, current_lanes, surrounding_objects):
        if not self._should_attempt_lane_change(surrounding_objects):
            return None

        best_choice = None
        for direction in ("left", "right"):
            if not self._safe_to_change_lane(surrounding_objects, direction):
                continue
            target_lane = self._get_adjacent_lane(current_lanes, direction)
            if target_lane is None:
                continue
            target_idx = current_lanes.index(target_lane)
            if self.available_routing_index_range is not None and target_idx not in self.available_routing_index_range:
                continue
            benefit = self._calculate_lane_change_benefit(surrounding_objects, direction)
            if benefit <= self.LANE_CHANGE_MIN_SPEED_ADVANTAGE:
                continue
            if direction == "left":
                candidate = (
                    benefit,
                    surrounding_objects.left_front_object(),
                    surrounding_objects.left_front_min_distance(),
                    target_lane,
                )
            else:
                candidate = (
                    benefit,
                    surrounding_objects.right_front_object(),
                    surrounding_objects.right_front_min_distance(),
                    target_lane,
                )
            if best_choice is None or candidate[0] > best_choice[0]:
                best_choice = candidate

        if best_choice is None:
            return None
        _, front_obj, front_dist, target_lane = best_choice
        self._set_cruise_target_speed(target_lane)
        self.overtake_timer = 0
        return front_obj, front_dist, target_lane

    def _should_attempt_lane_change(self, surrounding_objects):
        if not surrounding_objects.has_front_object():
            return False

        ego_speed = float(self.control_object.speed_km_h)
        front_obj = surrounding_objects.front_object()
        front_speed = float(getattr(front_obj, "speed_km_h", 0.0))
        front_distance = float(surrounding_objects.front_min_distance())
        preferred_speed = self._preferred_speed_reference(self.routing_target_lane)

        speed_condition = abs(ego_speed - preferred_speed) > 2.0
        front_speed_condition = front_speed < preferred_speed - 2.0 or front_speed < ego_speed - 3.0
        timer_condition = self.overtake_timer > self.LANE_CHANGE_FREQ * 0.5

        close_slow_condition = front_distance < self.LANE_CHANGE_BLOCK_DISTANCE and front_speed < ego_speed - 2.0
        stuck_condition = (
            ego_speed < self.LANE_CHANGE_STUCK_SPEED
            and front_distance < self.LANE_CHANGE_BLOCK_DISTANCE
            and front_speed < preferred_speed - 2.0
        )

        if close_slow_condition or stuck_condition:
            return True
        if not (speed_condition and front_speed_condition and timer_condition):
            return False
        return surrounding_objects.left_lane_exist() or surrounding_objects.right_lane_exist()

    def _lane_change_gap_requirement(self, direction):
        _ = direction
        ego_speed_mps = max(float(self.control_object.speed_km_h) / 3.6, 0.0)
        front_gap = max(self.SAFE_LANE_CHANGE_DISTANCE, 8.0 + ego_speed_mps * 0.8)
        back_gap = max(self.SAFE_LANE_CHANGE_DISTANCE, 6.0 + ego_speed_mps * 0.6)
        return front_gap, back_gap

    def _lane_change_ttc_ok(self, gap_distance, other_obj, closing_speed_mps, min_ttc):
        if gap_distance is None or other_obj is None:
            return True
        if closing_speed_mps <= 1e-3:
            return True
        return float(gap_distance) / closing_speed_mps > min_ttc

    def _safe_to_change_lane(self, surrounding_objects, direction):
        front_gap_req, back_gap_req = self._lane_change_gap_requirement(direction)
        ego_speed_mps = float(self.control_object.speed_km_h) / 3.6

        if direction == "left":
            if not surrounding_objects.left_lane_exist():
                return False
            front_obj = surrounding_objects.left_front_object()
            back_obj = surrounding_objects.left_back_object()
            front_dist = surrounding_objects.left_front_min_distance()
            back_dist = surrounding_objects.left_back_min_distance()
        elif direction == "right":
            if not surrounding_objects.right_lane_exist():
                return False
            front_obj = surrounding_objects.right_front_object()
            back_obj = surrounding_objects.right_back_object()
            front_dist = surrounding_objects.right_front_min_distance()
            back_dist = surrounding_objects.right_back_min_distance()
        else:
            return False

        front_speed_mps = float(getattr(front_obj, "speed_km_h", self.MAX_SPEED)) / 3.6 if front_obj is not None else 0.0
        back_speed_mps = float(getattr(back_obj, "speed_km_h", 0.0)) / 3.6 if back_obj is not None else 0.0

        front_ok = (
            float(front_dist) > front_gap_req
            and self._lane_change_ttc_ok(front_dist, front_obj, max(ego_speed_mps - front_speed_mps, 0.0), self.LANE_CHANGE_FRONT_TTC_MIN)
        )
        back_ok = (
            float(back_dist) > back_gap_req
            and self._lane_change_ttc_ok(back_dist, back_obj, max(back_speed_mps - ego_speed_mps, 0.0), self.LANE_CHANGE_REAR_TTC_MIN)
        )
        return bool(front_ok and back_ok)

    def _calculate_lane_change_benefit(self, surrounding_objects, direction):
        front_speed = surrounding_objects.front_object().speed_km_h if surrounding_objects.has_front_object() else self.MAX_SPEED
        front_dist = surrounding_objects.front_min_distance()

        if direction == "left":
            target_speed = surrounding_objects.left_front_object().speed_km_h if surrounding_objects.has_left_front_object() else self.MAX_SPEED
            target_dist = surrounding_objects.left_front_min_distance()
        elif direction == "right":
            target_speed = surrounding_objects.right_front_object().speed_km_h if surrounding_objects.has_right_front_object() else self.MAX_SPEED
            target_dist = surrounding_objects.right_front_min_distance()
        else:
            return 0.0

        distance_bonus = max(0.0, float(target_dist) - float(front_dist)) * 0.2
        return float(target_speed - front_speed + distance_bonus)

    def _find_closest_front_object(self, all_objects):
        hazard = self._find_lidar_forward_obstacle(all_objects, self.routing_target_lane)
        if hazard is None:
            return None, float("inf")
        return hazard["object"], float(hazard["distance"])

    def _calculate_acceleration(self, front_obj, distance):
        cruise_acc = self.acceleration(None, float("inf"))
        if front_obj is None or not np.isfinite(distance):
            return cruise_acc

        longitudinal, lateral, closing_speed = self._relative_kinematics(front_obj)
        ttc = np.inf
        if closing_speed > 1e-3:
            ttc = max(float(distance), 0.0) / closing_speed
        hazard = {
            "object": front_obj,
            "distance": float(max(distance, 0.0)),
            "longitudinal": longitudinal,
            "lateral": lateral,
            "closing_speed": closing_speed,
            "ttc": ttc,
            "lane_path_match": self._object_matches_lane_path(front_obj, self.routing_target_lane),
            "heading_diff": self._get_heading_difference(front_obj),
            "reason": "direct_forward",
        }
        return min(cruise_acc, self._hazard_brake_acceleration(hazard))

    def _get_heading_difference(self, front_obj):
        if not hasattr(front_obj, "heading_theta"):
            return 0.0
        ego_heading = self.control_object.heading_theta
        front_heading = front_obj.heading_theta
        angle_diff = abs(ego_heading - front_heading)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        return float(np.degrees(angle_diff))

    def _is_perpendicular_or_opposite(self, front_obj):
        heading_diff = self._get_heading_difference(front_obj)
        return bool((80.0 <= heading_diff <= 100.0) or (160.0 <= heading_diff <= 180.0))

class EgoIDMPolicyInOsm(IDMPolicyInOsm):
    """
    用于自车是遵循replay的全局路径，此时无法判断自车所在的lane_index
    根据雷达获得的周围物体信息，计算加速度和转向角度的策略
    其他的IDM策略是根据车辆所在车道索引关系判断的
    """
    
    def __init__(self, control_object, random_seed):
        super(EgoIDMPolicyInOsm, self).__init__(control_object=control_object, random_seed=random_seed)
        self.target_speed = 20
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = self.engine.global_config.get("enable_idm_lane_change", True)
        self.enable_lane_change = False
        self.disable_idm_deceleration = self.engine.global_config.get("disable_idm_deceleration", False)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

    # def act(self, *args, **kwargs):
    #     # success的作用是判断车辆是否在正确的车道上，首先确定在初始位置在正确道路上，然后可以执行正确动作；
    #     # 若是发现执行完动作后车辆不在正确道路上，则一直执行沿着当前道路行驶的动作，self.routing_target_lane始终不会更新了
    #     success = self.move_to_next_road()
    #     all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
    #     try:
    #         if success and self.enable_lane_change:
    #             # perform lane change due to routing
    #             acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
    #         else:
    #             # can not find routing target lane
    #             surrounding_objects = FrontBackObjects.get_find_front_back_objs_id(
    #                 all_objects,
    #                 self.engine.map_manager.current_map.road_network.get_lane(self.control_object.lane_index),
    #                 self.control_object.position,
    #                 max_distance=self.MAX_LONG_DIST
    #             )
    #             acc_front_obj = surrounding_objects.front_object()
    #             acc_front_dist = surrounding_objects.front_min_distance()
    #             steering_target_lane = self.routing_target_lane
    #     except:
    #         # error fallback
    #         acc_front_obj = None
    #         acc_front_dist = 5
    #         steering_target_lane = self.routing_target_lane

    #     # ============ control by PID and IDM ============
    #     steering = self.steering_control(steering_target_lane)
    #     acc = self.acceleration(acc_front_obj, acc_front_dist)
    #     action = [steering, acc]
    #     self.action_info["action"] = action
    #     return action
    
    def act(self, *args, **kwargs):
        _ = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        target_lane = self.routing_target_lane or self.control_object.lane
        hazard = self._find_lidar_forward_obstacle(all_objects, target_lane)
        if hazard is not None:
            closest_front_obj = hazard["object"]
            closest_distance = float(hazard["distance"])
        else:
            closest_front_obj, closest_distance = None, float("inf")

        base_acc = self._calculate_acceleration(closest_front_obj, closest_distance)
        release_acc = self._intersection_deadlock_release_acceleration(base_acc, hazard)
        acc = float(release_acc) if release_acc is not None else float(base_acc)

        steering = self.steering_control(target_lane)
        action = [steering, acc]

        if hazard is not None:
            self.action_info["hazard_distance"] = float(hazard["distance"])
            self.action_info["hazard_reason"] = hazard["reason"]
            self.action_info["hazard_should_yield"] = bool(hazard.get("should_yield", True))
            self.action_info["hazard_time_gap"] = float(hazard.get("time_gap", 0.0))
            self.action_info["hazard_dist_conflict"] = float(hazard.get("ego_dist_conflict", np.inf))
        self.action_info["action"] = action
        return action

# class AdversarialIDMPolicy(IDMPolicyInOsm):
#     """
#     对抗攻击策略：
#     在保证不碰撞其他车辆的前提下，主动追踪并撞击自车。
#     采用人工势场法 (Potential Field)：
#     1. 对自车产生引力 (Attraction)
#     2. 对其他车辆产生斥力 (Repulsion)
#     """

#     def __init__(self, control_object, random_seed):
#         super(AdversarialIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        
#         # === 参数调优 ===
#         self.ATTACK_GAIN = 1.5        # 攻击欲望强度
#         self.AVOID_GAIN = 10.0        # 避障排斥力强度 (建议设高一点以确保安全)
#         self.SAFE_DIST_THRESHOLD = 12.0 # 开始产生斥力的距离阈值 (米)
#         self.PREDICT_TIME_BASE = 1.0  # 基础预判时间 (秒)
        
#         # 覆盖父类的PID参数，使转向更激进
#         self.heading_pid = PIDController(2.0, 0.01, 4.0) 

#     def act(self, *args, **kwargs):
#         # 1. 获取环境信息
#         # 获取自车对象 (Target)
#         try:
#             ego_agent_id = self.engine.agent_manager.ego_agent_id
#             ego_vehicle = self.engine.agent_manager.active_agents[ego_agent_id]
#         except:
#             # 如果找不到自车，回退到普通IDM逻辑
#             return super().act(*args, **kwargs)

#         # 获取周围所有物体
#         all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        
#         # 2. 计算合成矢量 (Force Vector)
#         # 矢量 A: 攻击矢量 (指向自车未来位置)
#         attack_vec = self._get_attack_vector(ego_vehicle)
        
#         # 矢量 B: 避障矢量 (背离其他车辆)
#         avoid_vec = self._get_avoidance_vector(all_objects, ego_vehicle)
        
#         # 合成最终方向
#         final_vec = attack_vec + avoid_vec
        
#         # 3. 将矢量转换为控制指令 (Steering & Acc)
#         steering = self._vector_to_steering(final_vec)
#         acc = self._vector_to_acc(steering, final_vec, all_objects)

#         action = [steering, acc]
#         self.action_info["action"] = action
#         return action

#     def _get_attack_vector(self, target_vehicle):
#         """
#         计算指向目标的攻击矢量，包含简单的轨迹预测
#         """
#         npc_pos = np.array(self.control_object.position)
#         target_pos = np.array(target_vehicle.position)
#         target_vel = np.array(target_vehicle.velocity) # 速度矢量 (vx, vy)
        
#         # 计算距离
#         dist = np.linalg.norm(target_pos - npc_pos)
        
#         # 动态调整预判时间：距离越远，预判越久，但有上限
#         predict_time = min(dist / 15.0, 2.0) + 0.5 
        
#         # 预测目标未来的位置 (P_pred = P_now + V * t)
#         pred_pos = target_pos + target_vel * predict_time
        
#         # 计算指向向量
#         vec = pred_pos - npc_pos
        
#         # 归一化并应用增益
#         norm_val = np.linalg.norm(vec)
#         if norm_val > 0.1:
#             vec = (vec / norm_val) * self.ATTACK_GAIN
#         else:
#             vec = np.zeros(2)
            
#         return vec

#     def _get_avoidance_vector(self, all_objects, target_vehicle):
#         """
#         计算避障斥力矢量总和
#         """
#         npc_pos = np.array(self.control_object.position)
#         total_avoid_vec = np.array([0.0, 0.0])
        
#         npc_id = self.control_object.id
#         target_id = target_vehicle.id
        
#         for obj in all_objects:
#             # 排除自己和攻击目标(自车)
#             if obj.id == npc_id or obj.id == target_id:
#                 continue
            
#             # 这里的all_objects可能包含非车辆物体，确保有位置属性
#             if not hasattr(obj, 'position'):
#                 continue
                
#             obj_pos = np.array(obj.position)
            
#             # 计算指向障碍物的向量
#             vec_to_obj = obj_pos - npc_pos
#             dist = np.linalg.norm(vec_to_obj)
            
#             # 只有在安全距离内才产生斥力
#             if 0.1 < dist < self.SAFE_DIST_THRESHOLD:
#                 # 斥力方向：从障碍物指向我 (-vec_to_obj)
#                 # 斥力大小：与距离的平方成反比 (1 / dist^2)
#                 repulsion_force = self.AVOID_GAIN / (dist ** 2)
                
#                 # 归一化方向向量
#                 direction = -vec_to_obj / dist
                
#                 total_avoid_vec += direction * repulsion_force
                
#         return total_avoid_vec

#     def _vector_to_steering(self, target_vec):
#         """
#         将目标矢量转换为方向盘转角
#         """
#         if np.linalg.norm(target_vec) < 0.1:
#             return 0.0
            
#         # 计算目标矢量的角度 (Math.atan2)
#         target_heading = np.arctan2(target_vec[1], target_vec[0])
        
#         # 当前车头朝向
#         current_heading = self.control_object.heading_theta
        
#         # 计算角度差 (Target - Current)
#         diff = wrap_to_pi(target_heading - current_heading)
        
#         # 使用PID控制器计算转向
#         steering = self.heading_pid.get_result(diff)
        
#         # 限制幅度
#         return np.clip(steering, -1.0, 1.0)

#     def _vector_to_acc(self, steering, target_vec, all_objects):
#         """
#         计算油门/刹车
#         逻辑：方向对准了就加速，方向偏离大就减速，遇到极近障碍物紧急刹车
#         """
#         # 1. 紧急避障检查
#         # 简单的碰撞预测：如果前方极近距离有非目标车辆，强制刹车
#         closest_obj, dist = self._find_closest_front_object(all_objects) # 复用父类方法
        
#         # 获取目标（自车）ID
#         try:
#             target_id = self.engine.agent_manager.active_agents[self.engine.agent_manager.ego_agent_id].id
#         except:
#             target_id = None

#         # 如果前方最近的物体不是目标自车，且距离很近，说明要撞到路人车了
#         if closest_obj and closest_obj.id != target_id and dist < 8.0:
#              return -1.0 # 紧急刹车 (Full Brake)
        
#         # 2. 正常行驶逻辑
#         # 如果转向角很大，说明需要急转弯，减速
#         if abs(steering) > 0.4:
#             return 0.1 # 维持低速
            
#         # 3. 攻击冲刺
#         # 如果方向基本对准了，全油门
#         return 1.0

import numpy as np
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.utils.math import wrap_to_pi

class AdversarialIDMPolicy(IDMPolicyInOsm):
    """
    随机性格对抗策略 (Random Personality Adversary):
    每次 Reset 时随机生成一种驾驶风格（性格），解决千篇一律的问题。
    包含多种速度策略：恒定高速、正弦波动、随机漫步。
    """

    def __init__(self, control_object, random_seed):
        super(AdversarialIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        
        # 1. 真正的随机源
        self.true_rng = np.random.RandomState(None)
        
        # 2. 控制器
        self.heading_pid = PIDController(2.5, 0.01, 5.0) 
        
        # 3. 初始化性格参数 (会在 reset 中被覆盖)
        self.personality = {}
        self._randomize_personality()

    def reset(self):
        """
        [关键] 每次环境重置时调用。
        在这里彻底改变策略参数，确保每一局的表现都不一样。
        """
        super().reset()
        self._randomize_personality()
        # print(f"对抗车新性格生成: {self.personality}") # 调试用

    def _randomize_personality(self):
        """生成随机的驾驶性格"""
        self.personality = {
            # 速度策略: 
            # 0 = 疯狗模式 (一直全油门)
            # 1 = 变速猎手 (速度呈正弦波变化，忽快忽慢)
            # 2 = 稳健杀手 (保持一个中等速度，试图精准撞击)
            "speed_mode": self.true_rng.choice([0, 1, 1, 2]), 
            
            # 基础目标速度 (仅在模式 2 有效)
            "target_speed": self.true_rng.randint(30, 80), 
            
            # 拦截偏置: 
            # < 0 瞄准车尾 (追尾)
            # 0 瞄准中心
            # > 0 瞄准车头 (截杀)
            "aim_bias": self.true_rng.uniform(-1.0, 2.5), 
            
            # 预判时间因子: 
            # 值越大，预判越远（适合高速），值越小，越看重当前位置
            "predict_factor": self.true_rng.uniform(0.5, 2.0),
            
            # 波动相位 (用于模式 1)
            "wave_phase": self.true_rng.uniform(0, 2 * np.pi),
            "wave_freq": self.true_rng.uniform(0.05, 0.2)
        }

    def act(self, *args, **kwargs):
        # 1. 获取自车信息
        try:
            ego_agent_id = self.engine.agent_manager.ego_agent_id
            ego_vehicle = self.engine.agent_manager.active_agents[ego_agent_id]
        except:
            return super().act(*args, **kwargs)

        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        
        # 2. 计算当前距离
        npc_pos = np.array(self.control_object.position)
        target_pos = np.array(ego_vehicle.position)
        dist_to_target = np.linalg.norm(target_pos - npc_pos)
        
        # 3. 计算矢量
        # 攻击矢量 (受 aim_bias 和 predict_factor 影响)
        attack_vec = self._get_attack_vector(ego_vehicle, dist_to_target)
        
        # 避障矢量 (隧道效应：近距离无视障碍)
        tunnel_factor = 0.1 if dist_to_target < 15.0 else 1.0
        avoid_vec = self._get_avoidance_vector(all_objects, ego_vehicle) * tunnel_factor
        
        final_vec = attack_vec + avoid_vec
        
        # 4. 执行控制
        steering = self._vector_to_steering(final_vec)
        acc = self._calculate_variable_speed(steering, dist_to_target, all_objects, ego_vehicle)

        return [steering, acc]

    def _get_attack_vector(self, target_vehicle, dist_to_target):
        npc_pos = np.array(self.control_object.position)
        target_pos = np.array(target_vehicle.position)
        target_vel = np.array(target_vehicle.velocity)
        target_heading = target_vehicle.heading_theta
        
        # [性格影响] 预判时间
        predict_time = min(dist_to_target / 15.0, 2.0) * self.personality["predict_factor"]
        
        # [性格影响] 瞄准点偏置 (Aim Bias)
        # 如果 aim_bias 是 2.0，它会故意撞向你车头前方 2米处（预判你会往前开）
        bias_distance = self.personality["aim_bias"]
        bias_vec = np.array([np.cos(target_heading), np.sin(target_heading)]) * bias_distance
        
        pred_pos = target_pos + (target_vel * predict_time) + bias_vec
        
        vec = pred_pos - npc_pos
        norm_val = np.linalg.norm(vec)
        if norm_val > 0.1:
            vec = (vec / norm_val) * 2.0 # 固定攻击增益
        else:
            vec = np.zeros(2)
        return vec

    def _calculate_variable_speed(self, steering, dist_to_target, all_objects, ego_vehicle):
        """
        根据性格计算油门
        """
        # A. 绝对安全制动 (避免撞墙/无关车辆)
        closest_obj, dist = self._find_closest_front_object(all_objects)
        try:
            target_id = ego_vehicle.id
        except:
            target_id = None
            
        # 除非离目标很近了，否则还是要避让一下路人，防止卡死
        if closest_obj and closest_obj.id != target_id and dist < 6.0 and dist_to_target > 15.0:
            return -1.0
            
        # B. 末端冲刺 (无论什么性格，贴脸了就全油门)
        if dist_to_target < 15.0:
            return 1.0

        # C. 基于性格的速度控制
        mode = self.personality["speed_mode"]
        
        if mode == 0: 
            # === 疯狗模式 ===
            # 简单粗暴，只要不是大转弯就全油门
            if abs(steering) > 0.6:
                return 0.2
            return 1.0
            
        elif mode == 1:
            # === 变速猎手 (正弦波) ===
            # 使用系统时间产生波浪: sin(t * freq + phase)
            # 结果在 -0.2 (轻微刹车) 到 1.0 (全油门) 之间波动
            t = time.time()
            val = np.sin(t * self.personality["wave_freq"] * 10 + self.personality["wave_phase"])
            # 将 [-1, 1] 映射到 [-0.2, 1.0]
            throttle = np.interp(val, [-1, 1], [-0.2, 1.0])
            return throttle
            
        elif mode == 2:
            # === 稳健杀手 (控速) ===
            # 试图维持在 target_speed (例如 40km/h)
            # 这样可以避免因速度太快而错过路口
            current_speed = self.control_object.speed_km_h
            target = self.personality["target_speed"]
            
            if current_speed < target:
                return 0.6 # 适度加速
            else:
                return -0.1 # 含着刹车滑行
                
        return 1.0

    def _get_avoidance_vector(self, all_objects, target_vehicle):
        # 简化的避障逻辑，保持原样即可
        npc_pos = np.array(self.control_object.position)
        total_avoid_vec = np.array([0.0, 0.0])
        npc_id = self.control_object.id
        target_id = target_vehicle.id
        
        for obj in all_objects:
            if obj.id == npc_id or obj.id == target_id:
                continue
            if not hasattr(obj, 'position'):
                continue
            obj_pos = np.array(obj.position)
            vec_to_obj = obj_pos - npc_pos
            dist = np.linalg.norm(vec_to_obj)
            
            if 0.1 < dist < 10.0:
                repulsion = 5.0 / (dist ** 2)
                total_avoid_vec += (-vec_to_obj / dist) * repulsion
        return total_avoid_vec

    def _vector_to_steering(self, target_vec):
        if np.linalg.norm(target_vec) < 0.1:
            return 0.0
        target_heading = np.arctan2(target_vec[1], target_vec[0])
        current_heading = self.control_object.heading_theta
        diff = wrap_to_pi(target_heading - current_heading)
        steering = self.heading_pid.get_result(diff)
        return np.clip(steering, -1.0, 1.0)
