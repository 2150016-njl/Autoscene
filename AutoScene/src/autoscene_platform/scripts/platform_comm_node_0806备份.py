#!/usr/bin/env python3
"""AutoScene 平台通信节点。

本节点是实物运动平台和 AutoScene world/map 坐标之间的唯一边界：
1. 订阅 runtime 发布的 AutoScene world 轨迹；
2. 用固定 G->A 标定的逆变换，把轨迹转成经纬度解算后的大地投影坐标；
3. 按新平台轨迹 V2 协议编码 UDP payload，必要时下发给平台；
4. 接收平台上游 176 字节底盘状态 UDP，按“上游信息协议.xlsx”解码；
5. 用经纬度解算出的 G 坐标和固定 G->A 标定，发布 AutoScene world 状态反馈。

这里的 UDP socket 收发的都是应用层 payload，不包含 Ethernet/IP/UDP 头。
"""
import copy
import json
import math
import os
import select
import socket
import struct
import sys
import threading
import time

# catkin 的 devel/lib 包装器会 exec() 本源脚本，但不会保证本 scripts
# 目录位于 sys.path。geo_projection.py 与本文件同目录，因此显式加入
# 源脚本目录；若文件缺失，后续 import 会明确失败，不做兜底。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rospy
from std_msgs.msg import String, UInt8MultiArray

from autoscene_msgs.msg import (
    PlatformCommStatus,
    PlatformState,
    PlatformStateArray,
    TrajectoryArray,
    VehicleState,
    VehicleStateArray,
)
from geo_projection import (
    EarthToAutoSceneTransform,
    earth_yaw_to_ins_heading_deg,
    ins_heading_deg_to_earth_yaw,
    latlon_to_xy,
)

ADS_PAYLOAD_SIZE = 176

TRAJECTORY_POINT_NUM = 50
TRAJECTORY_APP_HEADER_SIZE = 8
TRAJECTORY_POINT_SIZE = 24
TRAJECTORY_PAYLOAD_SIZE = TRAJECTORY_APP_HEADER_SIZE + TRAJECTORY_POINT_NUM * TRAJECTORY_POINT_SIZE

TRAJECTORY_SENDER = 10
TRAJECTORY_VERSION = 0xF1
TRAJECTORY_MESSAGE_ID = 12

FEEDBACK_MESSAGE_ID = 1

def valid_platform_id(value):
    """判断配置中的 platform_id 是否表示受控平台。"""
    if value is None:
        return False

    return str(value).strip().lower() not in (
        "",
        "-1",
        "none",
        "null",
    )

def quantize(value, factor, low, high):
    """把物理量按协议 factor 转为整数；超范围直接报错，禁止静默饱和。"""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("cannot quantize non-finite value")
    raw = int(round(value / factor))
    if raw < low or raw > high:
        raise ValueError(
            "quantized value out of range: value=%r factor=%r raw=%d range=[%d,%d]"
            % (value, factor, raw, low, high)
        )
    return raw


def yaw_to_quat(yaw):
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q

def read_u8(data, offset):
    return struct.unpack_from("<B", data, offset)[0]


def read_u16(data, offset):
    return struct.unpack_from("<H", data, offset)[0]


def read_i16(data, offset):
    return struct.unpack_from("<h", data, offset)[0]


def read_u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def read_i32(data, offset):
    return struct.unpack_from("<i", data, offset)[0]


def read_f32(data, offset):
    return struct.unpack_from("<f", data, offset)[0]


def read_f64(data, offset):
    return struct.unpack_from("<d", data, offset)[0]


# 按“上游信息协议.xlsx”解码 176 字节 UDP 应用层 payload。
# 字段偏移、大小端和 factor 与 C++ ads_udp_protocol.hpp 的 decodePayload176() 保持一致。
# 这里收到的是 UDP payload，不含 Ethernet/IP/UDP 头。
def decode_ads_payload176(data):
    """Decode 上游信息协议.xlsx / ads_udp_decoder_node.cpp compatible payload."""
    if len(data) != ADS_PAYLOAD_SIZE:
        raise ValueError("ADS UDP payload must be exactly 176 bytes, got %d" % len(data))
    p = {
        "latitude_deg": read_f64(data, 0),
        "longitude_deg": read_f64(data, 8),
        "ads_id": read_u32(data, 16),
        "gps_time_raw_ms": read_i32(data, 20),
        "gps_week": read_i32(data, 24),
        "fs_message1": read_u32(data, 28),
        "fs_message2": read_u32(data, 32),
        "bat_state1": read_u32(data, 36),
        "bat_state2": read_u32(data, 40),
        "bat_state3": read_u32(data, 44),
        "bat_state4": read_u32(data, 48),
        "mpc_flags": read_u32(data, 52),
        "posi_rms_m": read_f32(data, 56),
        "lat_dev_m": read_f32(data, 60),
        "dist_x_m": read_f32(data, 64),
        "dist_y_m": read_f32(data, 68),
        "x_rel_m": read_f32(data, 72),
        "y_rel_m": read_f32(data, 76),
        "x_rel_vut_m": read_f32(data, 80),
        "y_rel_vut_m": read_f32(data, 84),
        "ttc_s": read_f32(data, 88),
        "heading_deg": read_u16(data, 92) * 0.01,
        "speed_kmh": read_i16(data, 94) * 0.01,
        "speed_e_kmh": read_i16(data, 96) * 0.01,
        "speed_n_kmh": read_i16(data, 98) * 0.01,
        "ax_mps2": read_i16(data, 100) * 0.01,
        "ay_mps2": read_i16(data, 102) * 0.01,
        "dyaw_dps": read_i16(data, 104) * 0.01,
        "bvolt1_v": read_u16(data, 106) * 0.1,
        "bvolt2_v": read_u16(data, 108) * 0.1,
        "bvolt3_v": read_u16(data, 110) * 0.1,
        "bvolt4_v": read_u16(data, 112) * 0.1,
        "btemp1_c": read_i16(data, 114) * 0.1,
        "btemp2_c": read_i16(data, 116) * 0.1,
        "btemp3_c": read_i16(data, 118) * 0.1,
        "btemp4_c": read_i16(data, 120) * 0.1,
        "bcurr1_a": read_i16(data, 122) * 0.1,
        "bcurr2_a": read_i16(data, 124) * 0.1,
        "bcurr3_a": read_i16(data, 126) * 0.1,
        "bcurr4_a": read_i16(data, 128) * 0.1,
        "mtemp1_c": read_i16(data, 130) * 0.1,
        "mtemp2_c": read_i16(data, 132) * 0.1,
        "mtemp3_c": read_i16(data, 134) * 0.1,
        "mtemp4_c": read_i16(data, 136) * 0.1,
        "cpu_temp_c": read_i16(data, 138) * 0.1,
        "steering_angle_deg": read_i16(data, 140) * 0.1,
        "steering_pwm": read_i16(data, 142),
        "mov_pwm1": read_i16(data, 144),
        "mov_pwm2": read_i16(data, 146),
        "mov_pwm3": read_i16(data, 148),
        "mov_pwm4": read_i16(data, 150),
        "brake_pwm": read_i16(data, 152),
        "sr_sw1": read_u16(data, 154),
        "sr_sw2": read_u16(data, 156),
        "sr_sw3": read_u16(data, 158),
        "sr_sw4": read_u16(data, 160),
        "reserved": read_i16(data, 162),
        "sender": read_u8(data, 164),
        "version": read_u8(data, 165),
        "message_id": read_u8(data, 166),
        "counter": read_u8(data, 167),
        "gnss_status": read_u8(data, 168),
        "ins_status": read_u8(data, 169),
        "soc1": read_u8(data, 170),
        "soc2": read_u8(data, 171),
        "soc3": read_u8(data, 172),
        "soc4": read_u8(data, 173),
        "operation_mode": read_u8(data, 174),
        "test_state": read_u8(data, 175),
    }
    p["gps_time_s"] = p["gps_time_raw_ms"] * 1e-3
    return p


# 与 C++ ads_udp_protocol.hpp 基本一致的协议合法性检查。
def validate_ads_packet(
        p,
        expected_message_id=FEEDBACK_MESSAGE_ID,
):
    errors = []
    if not -90.0 <= p["latitude_deg"] <= 90.0:
        errors.append("latitude_out_of_wgs84_range")
    if not -180.0 <= p["longitude_deg"] <= 180.0:
        errors.append("longitude_out_of_wgs84_range")
    if not (p["sender"] in (1, 2) or 100 <= p["sender"] <= 120 or 121 <= p["sender"] <= 135 or 136 <= p["sender"] <= 149):
        errors.append("sender_not_defined_by_protocol")
    if p["version"] != 0xF0:
        errors.append("version_not_0xF0")
    if p["message_id"] not in (
            1,
            2,
            3,
            4,
            11,
    ):
        errors.append(
            "message_id_not_defined_by_protocol"
        )
    elif p["message_id"] != int(
            expected_message_id
    ):
        errors.append(
            "unexpected_message_id:%d"
            % p["message_id"]
        )

    finite_fields = (
        "latitude_deg",
        "longitude_deg",
        "posi_rms_m",
        "lat_dev_m",
        "dist_x_m",
        "dist_y_m",
        "x_rel_m",
        "y_rel_m",
        "x_rel_vut_m",
        "y_rel_vut_m",
        "ttc_s",
        "heading_deg",
        "speed_kmh",
        "speed_e_kmh",
        "speed_n_kmh",
        "ax_mps2",
        "ay_mps2",
        "dyaw_dps",
    )

    for key in finite_fields:
        if not math.isfinite(
                float(p[key])
        ):
            errors.append(
                "%s_not_finite" % key
            )

    if not 0.0 <= p["heading_deg"] <= 360.0:
        errors.append("heading_out_of_0_360_deg")
    if p["gnss_status"] > 9:
        errors.append("gnss_status_out_of_range")
    if p["ins_status"] > 3:
        errors.append("ins_status_out_of_range")
    if p["soc1"] > 100 or p["soc2"] > 100 or p["soc3"] > 100 or p["soc4"] > 100:
        errors.append("soc_out_of_0_100_percent")
    if (p["fs_message1"] & 0xFF) > 2:
        errors.append("fs_message1_temperature_out_of_range")
    if ((p["fs_message1"] >> 8) & 0xF8) != 0:
        errors.append("fs_message1_sensor_reserved_bits_set")
    if ((p["fs_message1"] >> 16) & 0xFF) > 2:
        errors.append("fs_message1_battery_out_of_range")
    if ((p["fs_message1"] >> 24) & 0xF8) != 0:
        errors.append("fs_message1_communication_reserved_bits_set")
    if (p["fs_message2"] & 0xFF) > 1:
        errors.append("fs_message2_traj_following_out_of_range")
    if ((p["fs_message2"] >> 8) & 0xFF) > 1:
        errors.append("fs_message2_overrun_out_of_range")
    if ((p["fs_message2"] >> 16) & 0xFF) > 3:
        errors.append("fs_message2_ins_state_out_of_range")
    if ((p["fs_message2"] >> 24) & 0xFF) > 1:
        errors.append("fs_message2_emergency_stop_out_of_range")
    if p["operation_mode"] not in (0x00, 0x0A, 0x0B, 0x0C):
        errors.append("operation_mode_not_defined_by_protocol")
    if p["reserved"] != 0:
        errors.append("reserved_field_nonzero")
    if (p["test_state"] & 0xF0) != 0:
        errors.append("test_state_reserved_bits_set")
    return errors


# 单个真实运动平台的 UDP 端点和坐标标定信息。
class PlatformEndpoint:
    def __init__(self, cfg, global_cfg):
        # 每个 endpoint 描述一个远端平台；所有 UDP 收发共用
        # platform.local_ip:local_port 这个本机 socket。
        self.platform_id = str(cfg.get("platform_id", "")).strip()
        self.vehicle_id = str(cfg.get("vehicle_id", "")).strip()
        self.ip = str(cfg.get("ip", "127.0.0.1")).strip()
        self.send_port = int(cfg.get("send_port", 31000))
        self.expected_remote_ip = cfg.get("expected_remote_ip", cfg.get("ip", ""))
        self.expected_remote_port = int(cfg.get("expected_remote_port", -1))
        self.earth_to_autoscene = EarthToAutoSceneTransform.from_config(
            global_cfg.get("earth_to_autoscene", {})
        )

        self.tx_count = 0
        self.rx_count = 0
        self.packet_counter = 0
        self.last_tx = 0.0
        self.last_rx = 0.0
        self.latest_state = None
        self.detail = "idle"
        self.last_protocol_key = None

    def vector_earth_to_world(
            self,
            x_earth,
            y_earth,
    ):
        """把 G 坐标速度/加速度向量旋转到 AutoScene A 坐标。"""
        return self.earth_to_autoscene.rotate_g_to_a(
            x_earth,
            y_earth,
        )

    def protocol_sample_order_error(self, decoded):
        """检查重复/乱序，但不修改状态；协议校验成功后再 commit。"""
        key = (
            int(decoded.get("gps_week", 0)),
            int(decoded.get("gps_time_raw_ms", 0)),
            int(decoded.get("counter", 0)),
        )
        previous = self.last_protocol_key
        if previous is not None:
            if key == previous:
                return "duplicate_feedback"
            previous_time = previous[:2]
            current_time = key[:2]
            if all(current_time) and all(previous_time) and current_time < previous_time:
                return "out_of_order_feedback"
        return None

    def commit_protocol_sample(self, decoded):
        self.last_protocol_key = (
            int(decoded.get("gps_week", 0)),
            int(decoded.get("gps_time_raw_ms", 0)),
            int(decoded.get("counter", 0)),
        )

    def remote_matches(self, addr):
        """按配置过滤 UDP 来源，防止误收其他设备广播/调试包。"""
        ip, port = addr[0], int(addr[1])
        if self.expected_remote_ip and ip != self.expected_remote_ip:
            return False
        if self.expected_remote_port >= 0 and port != self.expected_remote_port:
            return False
        return True

    def has_fresh_feedback(self, now_wall, timeout_sec):
        """判断真实平台反馈是否足够新鲜，可供 runtime 闭环继续推进。"""
        return (
                self.latest_state is not None
                and self.latest_state.valid
                and self.last_rx > 0.0
                and now_wall - self.last_rx <= timeout_sec
        )

    def earth_to_world_pose(self, earth_x, earth_y, earth_yaw):
        """把平台上游 G 坐标 pose 转成 AutoScene world/map pose。"""
        world_x, world_y = self.earth_to_autoscene.point_g_to_a(
            earth_x,
            earth_y,
        )
        world_yaw = self.earth_to_autoscene.yaw_g_to_a(
            earth_yaw,
        )
        return world_x, world_y, world_yaw

    def world_to_earth_pose(self, world_x, world_y, world_yaw):
        """把 AutoScene world/map pose 转成平台 V2 协议使用的 G 坐标 pose。"""
        earth_x, earth_y = self.earth_to_autoscene.point_a_to_g(
            world_x,
            world_y,
        )
        earth_yaw = self.earth_to_autoscene.yaw_a_to_g(
            world_yaw,
        )
        return earth_x, earth_y, earth_yaw


class PlatformCommNode:
    def __init__(self):
        rospy.init_node("platform_comm_node")
        # 轨迹下发使用新平台 V2 应用层协议；固定项可由配置覆盖，
        # 但不再复用旧 808 字节轨迹包格式。
        self.platform_cfg = rospy.get_param("/platform", {})
        self.frame_id = rospy.get_param("/runtime/frame_id", "map")

        self.vehicles_cfg = rospy.get_param("/vehicles", {})
        self.vehicle_cfg_by_id = {
            str(agent.get("id", "")): agent
            for agent in self.vehicles_cfg.get("agents", [])
            if isinstance(agent, dict) and agent.get("id")
        }

        self.receive_udp = bool(
            self.platform_cfg.get("receive_udp", False)
        )

        # send_udp=false 时仍编码完整 UDP payload 并发布 trajectory_udp_payload；
        # 这用于无实物 hybrid/dry-run 审核，不会触发真实 sock.sendto。
        self.send_udp = bool(
            self.platform_cfg.get("send_udp", False)
        )

        # 所有平台共用本机一个 UDP endpoint。
        # 该 socket 同时用于：
        # - 从 local_ip:local_port 向不同平台 sendto；
        # - 在 local_ip:local_port 接收不同平台反馈。
        # 因此真实发送的源端口固定为 local_port。
        self.local_ip = str(
            self.platform_cfg.get("local_ip", "0.0.0.0")
        ).strip()
        self.local_port = int(
            self.platform_cfg.get("local_port", 31100)
        )
        self.udp_sock = None

        # emulate=true 用 MetaDrive 状态模拟平台反馈，只能用于无实物混合调试。
        # 真接硬件时必须 false，否则平台反馈不会进入闭环。
        self.emulate_controlled_feedback = bool(
            self.platform_cfg.get(
                "emulate_controlled_feedback",
                False,
            )
        )

        # /agents/state 是 MetaDrive 原始状态，只用于虚拟车辆和 dry-run。
        # 该超时不能拿来判断真实平台 UDP 是否有效。
        self.sim_state_timeout_sec = float(
            self.platform_cfg.get(
                "sim_state_timeout_sec",
                1.0,
            )
        )

        # 真实平台 UDP 状态保鲜时间。
        self.feedback_timeout_sec = float(
            self.platform_cfg.get(
                "feedback_timeout_sec",
                1.0,
            )
        )
        self.target_timeout_sec = float(
            self.platform_cfg.get(
                "target_timeout_sec",
                1.0,
            )
        )

        if self.receive_udp and self.emulate_controlled_feedback:
            raise ValueError(
                "platform.receive_udp and "
                "platform.emulate_controlled_feedback "
                "cannot both be true"
            )

        # 模拟反馈只能用于不接实物的 dry-run。
        # send_udp=true 表示会触发真实网络下发，不能同时使用模拟反馈。
        if self.send_udp and self.emulate_controlled_feedback:
            raise ValueError(
                "platform.send_udp and "
                "platform.emulate_controlled_feedback "
                "cannot both be true"
            )

        self.send_rate_hz = float(self.platform_cfg.get("send_rate_hz", 10.0))
        self.point_num = int(
            self.platform_cfg.get(
                "packet_point_num",
                TRAJECTORY_POINT_NUM,
            )
        )
        if self.point_num != TRAJECTORY_POINT_NUM:
            raise ValueError(
                "platform.packet_point_num must be %d for the fixed UDP protocol, got %d"
                % (TRAJECTORY_POINT_NUM, self.point_num)
            )

        self.sender = int(
            self.platform_cfg.get(
                "sender",
                TRAJECTORY_SENDER,
            )
        )

        self.version = int(
            self.platform_cfg.get(
                "version",
                TRAJECTORY_VERSION,
            )
        )

        self.message_id = int(
            self.platform_cfg.get(
                "message_id",
                TRAJECTORY_MESSAGE_ID,
            )
        )

        self.expected_feedback_message_id = int(
            self.platform_cfg.get(
                "expected_feedback_message_id",
                FEEDBACK_MESSAGE_ID,
            )
        )

        self.latest_targets = {}
        self.latest_targets_stamp = None
        self.lock = threading.RLock()

        # 多个平台共享一个本机 socket；每个平台分别保存远端地址、标定和状态。
        self.controlled_platform_by_vehicle = {}
        controlled_vehicle_by_platform = {}

        '''从 /vehicles/agents 找出受平台控制的车辆'''
        for index, agent in enumerate(
                self.vehicles_cfg.get("agents", []),
                start=1,
        ):
            platform_id = agent.get("platform_id")
            if not valid_platform_id(platform_id):
                continue

            vehicle_id = str(agent.get("id", "")).strip()
            platform_id = str(platform_id).strip()

            if not vehicle_id:
                raise ValueError(
                    "controlled vehicles.agents[%d] has empty id" % index
                )
            if vehicle_id in self.controlled_platform_by_vehicle:
                raise ValueError(
                    "duplicate controlled vehicle id: %s" % vehicle_id
                )
            if platform_id in controlled_vehicle_by_platform:
                raise ValueError(
                    "platform_id=%s is assigned to both %s and %s"
                    % (
                        platform_id,
                        controlled_vehicle_by_platform[platform_id],
                        vehicle_id,
                    )
                )

            self.controlled_platform_by_vehicle[vehicle_id] = platform_id
            controlled_vehicle_by_platform[platform_id] = vehicle_id

        # 先筛选并完整校验配置，再构造远端 endpoint。
        selected_endpoint_cfgs = []
        seen_endpoint_vehicle_ids = set()
        seen_endpoint_platform_ids = set()

        '''筛选本轮真正使用的 endpoint'''
        for cfg_index, cfg in enumerate(
                self.platform_cfg.get("platforms", []),
                start=1,
        ):
            if not isinstance(cfg, dict):
                continue

            vehicle_id = str(cfg.get("vehicle_id", "")).strip()
            platform_id = str(cfg.get("platform_id", "")).strip()

            if self.controlled_platform_by_vehicle.get(vehicle_id) != platform_id:
                continue

            if vehicle_id in seen_endpoint_vehicle_ids:
                raise ValueError(
                    "duplicate selected endpoint vehicle_id=%s" % vehicle_id
                )
            if platform_id in seen_endpoint_platform_ids:
                raise ValueError(
                    "duplicate selected endpoint platform_id=%s" % platform_id
                )

            selected_endpoint_cfgs.append(cfg)
            seen_endpoint_vehicle_ids.add(vehicle_id)
            seen_endpoint_platform_ids.add(platform_id)

        '''创建 PlatformEndpoint'''
        self.endpoints = [
            PlatformEndpoint(cfg, self.platform_cfg)
            for cfg in selected_endpoint_cfgs
        ]

        '''建立车辆索引'''
        self.endpoint_by_vehicle = {
            endpoint.vehicle_id: endpoint
            for endpoint in self.endpoints
        }

        if self.receive_udp:
            seen_remote_filters = set()
            for endpoint in self.endpoints:
                if not endpoint.expected_remote_ip:
                    raise ValueError(
                        "platform %s expected_remote_ip is required"
                        % endpoint.platform_id
                    )
                remote_filter = (
                    endpoint.expected_remote_ip,
                    endpoint.expected_remote_port,
                )
                if remote_filter in seen_remote_filters:
                    raise ValueError(
                        "duplicate platform feedback filter %s:%s"
                        % remote_filter
                    )
                seen_remote_filters.add(remote_filter)

        '''确保每辆受控 BV 都有 endpoint'''
        for vehicle_id, platform_id in self.controlled_platform_by_vehicle.items():
            endpoint = self.endpoint_by_vehicle.get(vehicle_id)

            if endpoint is None:
                raise ValueError(
                    "controlled BV %s has platform_id=%s, "
                    "but platform.platforms has no matching vehicle_id"
                    % (vehicle_id, platform_id)
                )

            if str(endpoint.platform_id) != str(platform_id):
                raise ValueError(
                    "platform mapping mismatch for %s: "
                    "vehicle config=%s endpoint config=%s"
                    % (
                        vehicle_id,
                        platform_id,
                        endpoint.platform_id,
                    )
                )

        '''创建共享 UDP socket'''
        # 真实模式使用一个共享 UDP socket，避免多个 endpoint 重复绑定本机
        # local_ip:local_port；同一 socket 也保证发送源地址一致。
        if self.receive_udp or self.send_udp:
            self.udp_sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM,
            )
            self.udp_sock.bind(
                (
                    self.local_ip,
                    self.local_port,
                )
            )
            if self.receive_udp:
                self.udp_sock.setblocking(False)

            rospy.on_shutdown(
                self.close_udp_socket
            )

        '''检查受控 BV 是否有状态来源'''
        if (
                self.controlled_platform_by_vehicle
                and not self.receive_udp
                and not self.emulate_controlled_feedback
        ):
            raise ValueError(
                "controlled BV exists, but neither receive_udp "
                "nor emulate_controlled_feedback is enabled"
            )

        self.state_pub = rospy.Publisher("platform/state", PlatformStateArray, queue_size=1) # 用于评估
        self.agent_feedback_pub = rospy.Publisher(
            "agents/state_feedback",
            VehicleStateArray,
            queue_size=1,
        )
        self.status_pub = rospy.Publisher("platform/comm_status", PlatformCommStatus, queue_size=10) # 通信状态
        # 下发轨迹 UDP 的完整字节镜像：send_udp=false 时也发布，方便 dry-run 审核。
        self.payload_pub = rospy.Publisher("platform/trajectory_udp_payload", UInt8MultiArray, queue_size=10) # 轨迹 UDP

        # 接收到底盘状态 UDP 后的解码 JSON，用于协议核对和现场调试。
        self.decoded_pub = rospy.Publisher("platform/ads_udp_decoded", String, queue_size=10) # 上游信息解码后发出来，用于检查
        rospy.Subscriber("platform/target_trajectory", TrajectoryArray, self.on_targets, queue_size=1)

        self.latest_agents_state = None
        self.latest_agents_state_stamp = None

        # 只有 dry-run 仿真反馈需要订阅 MetaDrive 状态；硬件模式只接收 UDP。
        # 所有模式都订阅：/agents/state，在硬件反馈缺失时错误回退仿真
        if self.emulate_controlled_feedback:
            rospy.Subscriber(
                "agents/state",
                VehicleStateArray,
                self.on_agents_state,
                queue_size=1,
            )

        '''启动 UDP 接收线程，把平台 GNSS/G 坐标状态转成 AutoScene 坐标。'''
        if self.receive_udp:
            self.recv_thread = threading.Thread(
                target=self.recv_loop
            )
            self.recv_thread.daemon = True
            self.recv_thread.start()

        rospy.Timer(rospy.Duration(1.0 / max(self.send_rate_hz, 1.0)), self.on_timer) # 处理PLUTO轨迹并转UDP发出去
        rospy.loginfo(
            "platform_comm_node started: endpoints=%d, "
            "receive_udp=%s, send_udp=%s, local_udp=%s:%d, "
            "trajectory_protocol=v2, payload_bytes=%d, emulate_controlled_feedback=%s, "
            "subscriptions=[platform/target_trajectory%s], "
            "publishers=[agents/state_feedback, platform/state]",
            len(self.endpoints),
            self.receive_udp,
            self.send_udp,
            self.local_ip,
            self.local_port,
            TRAJECTORY_PAYLOAD_SIZE,
            self.emulate_controlled_feedback,
            ", agents/state" if self.emulate_controlled_feedback else "",
        )

    def close_udp_socket(self):
        """关闭共享 UDP socket。"""
        sock = self.udp_sock
        self.udp_sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def on_targets(self, msg):
        """缓存 runtime 发布的平台目标轨迹。

        key 优先使用 platform_id，兼容按 vehicle_id 查找。
        只有 platform.platforms 中声明的 endpoint 会参与编码/下发；配置里已取消
        enabled 字段，是否真正发 UDP 由全局 send_udp 控制。
        """
        new_targets = {}

        for traj in msg.trajectories:
            key = traj.platform_id or traj.vehicle_id

            if traj.valid and traj.points and key:
                new_targets[key] = copy.deepcopy(traj)

        with self.lock:
            self.latest_targets = new_targets
            self.latest_targets_stamp = rospy.Time.now()

    def on_agents_state(self, msg):
        """缓存 runtime 发布的全部 MetaDrive BV 原始状态。"""
        with self.lock:
            self.latest_agents_state = copy.deepcopy(msg)
            self.latest_agents_state_stamp = rospy.Time.now()

    def vehicle_state_to_platform_state(
            self,
            vehicle_state,
            endpoint,
    ):
        """把 MetaDrive BV 状态包装成模拟 PlatformState。

        只在 emulate_controlled_feedback=true 时使用，目的是验证 hybrid ROS
        链路；实物模式必须依赖 parse_feedback() 解析真实 UDP。
        """
        state = PlatformState()
        state.header.stamp = rospy.Time.now()
        state.header.frame_id = (
                vehicle_state.header.frame_id
                or self.frame_id
        )

        state.platform_id = endpoint.platform_id
        state.vehicle_id = endpoint.vehicle_id
        state.state = "metadrive_emulated_platform_feedback"

        state.pose = copy.deepcopy(vehicle_state.pose)
        state.twist = copy.deepcopy(vehicle_state.twist)
        state.heading = vehicle_state.heading

        state.tracking_error = 0.0
        state.feedback_age = 0.0
        state.valid = vehicle_state.valid

        return state

    def make_invalid_platform_vehicle(
            self,
            endpoint,
            sim_template,
            now_ros,
    ):
        """构造无 MetaDrive 动态状态泄漏的 invalid 平台反馈。

        sim_template 只用于保留 vehicle_id、length、width 等静态信息；
        pose、twist、accel、heading 不允许来自 MetaDrive。
        """
        if sim_template is not None:
            state = copy.deepcopy(sim_template)
        else:
            state = VehicleState()

        state.header.stamp = now_ros
        state.header.frame_id = self.frame_id
        state.vehicle_id = endpoint.vehicle_id
        state.entity_type = "physical_platform"
        state.valid = False

        # 无真实平台状态时，不能把 MetaDrive pose 冒充平台状态。
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 0.0
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
        state.pose.orientation.w = 1.0

        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        state.accel.linear.x = 0.0
        state.accel.linear.y = 0.0
        state.accel.linear.z = 0.0
        state.accel.angular.x = 0.0
        state.accel.angular.y = 0.0
        state.accel.angular.z = 0.0

        state.heading = 0.0

        return state

    def static_vehicle_template(self, endpoint):
        """构造只包含车辆 ID 和尺寸的静态模板。"""
        cfg = self.vehicle_cfg_by_id.get(endpoint.vehicle_id, {})
        state = VehicleState()
        state.vehicle_id = endpoint.vehicle_id
        state.length = float(cfg.get("length", 4.5))
        state.width = float(cfg.get("width", 2.0))
        state.pose.orientation.w = 1.0
        return state

    def publish_agent_feedback(self):
        """只发布受控平台 BV 的反馈。

        virtual BV 由 runtime 直接使用 MetaDrive 状态；dry-run 显式使用
        ``agents/state`` 模拟平台反馈；硬件模式只接受新鲜、有效的 UDP 状态。
        """
        now_wall = time.time()
        now_ros = rospy.Time.now()

        with self.lock:
            raw_msg = copy.deepcopy(self.latest_agents_state)
            raw_stamp = self.latest_agents_state_stamp
            endpoint_snapshots = {
                endpoint.vehicle_id: (
                    copy.deepcopy(endpoint.latest_state),
                    float(endpoint.last_rx),
                )
                for endpoint in self.endpoints
            }

        raw_fresh = False
        sim_by_vehicle = {}
        if raw_msg is not None and raw_stamp is not None:
            raw_age = max((now_ros - raw_stamp).to_sec(), 0.0)
            raw_fresh = raw_age <= self.sim_state_timeout_sec
            sim_by_vehicle = {
                state.vehicle_id: state
                for state in raw_msg.vehicles
                if state.vehicle_id
            }

        feedback = VehicleStateArray()
        feedback.header.stamp = now_ros
        feedback.header.frame_id = self.frame_id

        for endpoint in self.endpoints:
            sim_template = sim_by_vehicle.get(endpoint.vehicle_id)
            template = sim_template or self.static_vehicle_template(endpoint)
            actual_state = None

            if self.emulate_controlled_feedback:
                if raw_fresh and sim_template is not None and sim_template.valid:
                    actual_state = self.vehicle_state_to_platform_state(
                        sim_template,
                        endpoint,
                    )
                    with self.lock:
                        endpoint.latest_state = copy.deepcopy(actual_state)
                        endpoint.last_rx = now_wall
                        endpoint.detail = "metadrive_emulation_ok"
            else:
                candidate, last_rx = endpoint_snapshots[endpoint.vehicle_id]
                if (
                    candidate is not None
                    and candidate.valid
                    and last_rx > 0.0
                    and now_wall - last_rx <= self.feedback_timeout_sec
                ):
                    actual_state = candidate

            if actual_state is None:
                reason = (
                    "waiting_emulated_platform_state"
                    if self.emulate_controlled_feedback
                    else "waiting_actual_platform_state"
                )
                invalid_state = self.make_invalid_platform_vehicle(
                    endpoint,
                    template,
                    now_ros,
                )
                with self.lock:
                    endpoint.detail = reason
                feedback.vehicles.append(invalid_state)
                continue

            merged = copy.deepcopy(template)
            merged.header.stamp = actual_state.header.stamp
            merged.header.frame_id = self.frame_id
            merged.vehicle_id = endpoint.vehicle_id
            merged.entity_type = "physical_platform"
            merged.pose = copy.deepcopy(actual_state.pose)
            merged.twist = copy.deepcopy(actual_state.twist)
            merged.heading = float(actual_state.heading)
            # PlatformState has no acceleration field. Never retain simulated
            # acceleration in a hardware feedback message.
            merged.accel.linear.x = 0.0
            merged.accel.linear.y = 0.0
            merged.accel.linear.z = 0.0
            merged.accel.angular.x = 0.0
            merged.accel.angular.y = 0.0
            merged.accel.angular.z = 0.0
            merged.valid = True
            feedback.vehicles.append(merged)

        self.agent_feedback_pub.publish(feedback)

    def encode_packet(self, traj, endpoint):
        """按新平台轨迹 V2 协议编码 50 点 UDP 应用层 payload。

        V2 不带 42 字节链路头，不带 flags。每点 24 字节：
        `float64 earth_x_m, float64 earth_y_m, uint16 heading_deg_x100,
        int16 speed_kmh_x100, int16 accel_mps2_x100, uint16 time_s_x100`。
        """
        if len(traj.points) < self.point_num:
            raise ValueError(
                "trajectory requires at least %d points, got %d"
                % (
                    self.point_num,
                    len(traj.points),
                )
            )

        # Intel 小端应用头：magic/sender/version/message_id/counter/point_num。
        payload = bytearray(
            struct.pack(
                "<HBBBBH",
                0x7E7E,
                self.sender & 0xFF,
                self.version & 0xFF,
                self.message_id & 0xFF,
                endpoint.packet_counter & 0xFF,
                self.point_num,
            )
        )

        earth_points = self.trajectory_to_earth_points(
            traj,
            endpoint,
            self.point_num,
        )

        for point in earth_points:
            # Intel 小端轨迹点。x/y 是 double 米，不做毫米量化。
            payload.extend(
                struct.pack(
                    "<ddHhhH",
                    float(point["earth_x_m"]),
                    float(point["earth_y_m"]),
                    quantize(
                        point["heading_deg"],
                        0.01,
                        0,
                        36000,
                    ),
                    quantize(
                        point["speed_kmh"],
                        0.01,
                        -32768,
                        32767,
                    ),
                    quantize(
                        point["accel_mps2"],
                        0.01,
                        -32768,
                        32767,
                    ),
                    quantize(
                        point["time"],
                        0.01,
                        0,
                        65535,
                    ),
                )
            )

        expected_size = (
                TRAJECTORY_APP_HEADER_SIZE
                + TRAJECTORY_POINT_SIZE
                * self.point_num
        )

        if len(payload) != expected_size:
            raise RuntimeError(
                "trajectory payload size mismatch: "
                "expected=%d actual=%d"
                % (
                    expected_size,
                    len(payload),
                )
            )

        return bytes(payload)

    def trajectory_to_earth_points(self, traj, endpoint, point_count):
        """把 AutoScene world 轨迹转换成平台 V2 协议的大地投影轨迹。

        平台接收的 `x/y` 是经纬度解算后的高斯投影绝对坐标，不是车体局部坐标。
        速度使用标量 km/h；加速度使用沿轨迹航向的切向加速度，单位 m/s^2。
        """
        earth_points = []
        previous_time = -1.0
        for index in range(point_count):
            point = traj.points[index]
            x = float(point.pose.position.x)
            y = float(point.pose.position.y)
            yaw = float(point.heading)
            vx = float(point.velocity.linear.x)
            vy = float(point.velocity.linear.y)
            ax_world = float(point.acceleration.linear.x)
            ay_world = float(point.acceleration.linear.y)
            point_time = float(point.time_from_start)
            values = (x, y, yaw, vx, vy, ax_world, ay_world, point_time)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("trajectory point %d contains non-finite values" % index)
            if point_time < previous_time:
                raise ValueError("trajectory time is not monotonic at point %d" % index)

            earth_x, earth_y, earth_yaw = endpoint.world_to_earth_pose(
                x,
                y,
                yaw,
            )
            speed_kmh = math.hypot(vx, vy) * 3.6
            tangential_accel = (
                ax_world * math.cos(yaw)
                + ay_world * math.sin(yaw)
            )
            earth_points.append(
                {
                    "earth_x_m": earth_x,
                    "earth_y_m": earth_y,
                    "heading_deg": earth_yaw_to_ins_heading_deg(earth_yaw),
                    "speed_kmh": speed_kmh,
                    "accel_mps2": tangential_accel,
                    "time": point_time,
                }
            )
            previous_time = point_time
        return earth_points

    def parse_feedback(self, data, endpoint, addr):
        """解析平台上游 UDP，并转换成 world-frame PlatformState。"""
        if not endpoint.remote_matches(addr):
            return None
        # socket 收到的是 176 字节应用层 payload；Excel 中 218 字节总长包含
        # 42 字节链路/网络头，在线 UDP 接收时不应再出现在 data 中。
        decoded = decode_ads_payload176(data)
        # print("2222222")
        errors = validate_ads_packet(
            decoded,
            expected_message_id=(
                self.expected_feedback_message_id
            ),
        )
        # if errors:
        #     detail = "rx_ads_protocol_invalid:%s" % ",".join(errors)
        #     with self.lock:
        #         endpoint.detail = detail
        #     self.decoded_pub.publish(
        #         String(
        #             data=json.dumps(
        #                 {
        #                     "remote_ip": addr[0],
        #                     "remote_port": addr[1],
        #                     "udp_payload_bytes": len(data),
        #                     "protocol_valid": False,
        #                     "protocol_errors": errors,
        #                     "platform_id": endpoint.platform_id,
        #                     "vehicle_id": endpoint.vehicle_id,
        #                     "decoded": decoded,
        #                 },
        #                 ensure_ascii=False,
        #             )
        #         )
        #     )
        #     return None
        # if errors:
        #     rospy.logwarn(
        #         "[DEBUG] ignore ADS protocol errors=%s, continue parsing",
        #         errors
        #     )
        #     # return None

        order_error = endpoint.protocol_sample_order_error(decoded)
        if order_error:
            with self.lock:
                endpoint.detail = order_error
            self.decoded_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "remote_ip": addr[0],
                            "remote_port": addr[1],
                            "udp_payload_bytes": len(data),
                            "protocol_valid": False,
                            "protocol_errors": [order_error],
                            "platform_id": endpoint.platform_id,
                            "vehicle_id": endpoint.vehicle_id,
                            "decoded": decoded,
                        },
                        ensure_ascii=False,
                    )
                )
            )
            return None
        endpoint.commit_protocol_sample(decoded)

        earth_x, earth_y = latlon_to_xy(
            decoded["latitude_deg"],
            decoded["longitude_deg"],
        )
        earth_yaw = ins_heading_deg_to_earth_yaw(
            decoded["heading_deg"]
        )
        world_x, world_y, world_yaw = endpoint.earth_to_world_pose(
            earth_x,
            earth_y,
            earth_yaw,
        )

        state = PlatformState()
        state.header.stamp = rospy.Time.now()
        state.header.frame_id = self.frame_id
        state.platform_id = endpoint.platform_id
        state.vehicle_id = endpoint.vehicle_id
        state.state = "ads_udp_feedback"
        state.valid = True
        state.pose.position.x = world_x
        state.pose.position.y = world_y
        state.pose.orientation = yaw_to_quat(world_yaw)

        speed_e_mps = decoded["speed_e_kmh"] / 3.6
        speed_n_mps = decoded["speed_n_kmh"] / 3.6
        world_vx, world_vy = endpoint.vector_earth_to_world(
            speed_e_mps,
            speed_n_mps,
        )

        state.twist.linear.x = world_vx
        state.twist.linear.y = world_vy

        state.heading = world_yaw
        state.feedback_age = 0.0
        state.tracking_error = float(decoded.get("lat_dev_m", 0.0))

        decoded_msg = {
            # 保留完整 decoded JSON，方便用 rostopic echo 与 Wireshark/Excel 协议逐字段对比。
            "remote_ip": addr[0],
            "remote_port": addr[1],
            "udp_payload_bytes": len(data),
            "protocol_valid": True,
            "protocol_errors": [],
            "platform_id": endpoint.platform_id,
            "vehicle_id": endpoint.vehicle_id,
            "earth_pose": {
                "x": earth_x,
                "y": earth_y,
                "yaw": earth_yaw,
            },
            "autoscene_pose": {
                "x": world_x,
                "y": world_y,
                "yaw": world_yaw,
            },
            "decoded": decoded,
        }
        self.decoded_pub.publish(String(data=json.dumps(decoded_msg, ensure_ascii=False)))
        return state

    def recv_loop(self):
        """后台线程：从共享 socket 接收并按来源地址分发到平台 endpoint。"""
        while not rospy.is_shutdown():
            sock_obj = self.udp_sock
            if sock_obj is None:
                time.sleep(0.1)
                continue

            try:
                readable, _, _ = select.select(
                    [sock_obj],
                    [],
                    [],
                    0.1,
                )
            except (OSError, ValueError):
                if rospy.is_shutdown():
                    break
                time.sleep(0.1)
                continue

            if not readable:
                continue

            try:
                data, addr = sock_obj.recvfrom(4096) # 收UDP数据
                # print("1111111")

                matches = [
                    endpoint
                    for endpoint in self.endpoints
                    if endpoint.remote_matches(addr)
                ]

                if len(matches) != 1:
                    rospy.logwarn_throttle(
                        1.0,
                        "drop UDP feedback from %s:%d: matched endpoints=%d",
                        addr[0],
                        int(addr[1]),
                        len(matches),
                    )
                    continue

                endpoint = matches[0]
                state = self.parse_feedback(
                    data,
                    endpoint,
                    addr,
                ) # 将 176 字节 GNSS/底盘 UDP 解码，并从 G 坐标转为 AutoScene 坐标。
                if state is None:
                    continue

                with self.lock:
                    endpoint.latest_state = copy.deepcopy(
                        state
                    )
                    endpoint.rx_count += 1
                    endpoint.last_rx = time.time()
                    endpoint.detail = (
                        "rx_ads_ok"
                        if state.valid
                        else "rx_ads_protocol_invalid"
                    )

            except Exception as exc:
                rospy.logwarn_throttle(
                    1.0,
                    "UDP receive error: %s",
                    exc,
                )

    def publish_states_and_status(self):
        """发布平台状态和通信健康信息。"""
        stamp = rospy.Time.now()
        now_wall = time.time()
        states = PlatformStateArray()
        states.header.stamp = stamp
        states.header.frame_id = self.frame_id

        with self.lock:
            snapshots = [
                {
                    "endpoint": endpoint,
                    "state": copy.deepcopy(endpoint.latest_state),
                    "last_rx": float(endpoint.last_rx),
                    "last_tx": float(endpoint.last_tx),
                    "tx_count": int(endpoint.tx_count),
                    "rx_count": int(endpoint.rx_count),
                    "detail": str(endpoint.detail),
                }
                for endpoint in self.endpoints
            ]

        for snapshot in snapshots:
            endpoint = snapshot["endpoint"]
            state = snapshot["state"]
            last_rx = snapshot["last_rx"]
            last_tx = snapshot["last_tx"]
            if state is not None:
                state.feedback_age = max(now_wall - last_rx, 0.0)
                if state.feedback_age > self.feedback_timeout_sec:
                    state.valid = False
                states.states.append(state)

            status = PlatformCommStatus()
            status.header.stamp = stamp
            status.platform_id = endpoint.platform_id
            status.vehicle_id = endpoint.vehicle_id
            status.enabled = True
            # connected 专指真实 UDP 链路；dry-run 由 detail 标明模拟状态。
            status.connected = bool(
                self.receive_udp
                and last_rx > 0.0
                and now_wall - last_rx <= self.feedback_timeout_sec
            )
            status.tx_count = snapshot["tx_count"]
            status.rx_count = snapshot["rx_count"]
            status.last_tx_age = now_wall - last_tx if last_tx else -1.0
            status.last_rx_age = now_wall - last_rx if last_rx else -1.0
            status.detail = snapshot["detail"]
            self.status_pub.publish(status)

        self.state_pub.publish(states)

    def on_timer(self, _event):
        """按 send_rate_hz 周期发布平台反馈并编码最新 V2 轨迹包。"""
        # 先发布平台反馈，供 runtime 判断实物/模拟平台状态是否新鲜。
        # 下发轨迹使用固定 A->G 标定。
        self.publish_agent_feedback() #发布状态反馈

        with self.lock:
            targets = dict(self.latest_targets)
            targets_stamp = self.latest_targets_stamp

        now_ros = rospy.Time.now()

        targets_fresh = (
                targets_stamp is not None
                and max(
            (now_ros - targets_stamp).to_sec(),
            0.0,
        ) <= self.target_timeout_sec
        )
        now_wall = time.time()

        for endpoint in self.endpoints:
            if not targets_fresh:
                with self.lock:
                    endpoint.detail = "target_timeout"
                continue

            traj = targets.get(endpoint.platform_id) or targets.get(endpoint.vehicle_id)
            if traj is None:
                with self.lock:
                    endpoint.detail = "waiting_target"
                continue

            try:
                packet = self.encode_packet(traj, endpoint)

                if self.send_udp:
                    # 只有 send_udp=true 才触发真实 UDP 下发；dry-run 路径不会碰硬件。
                    if self.udp_sock is None:
                        raise RuntimeError(
                            "shared UDP socket is not initialized"
                        )
                    self.udp_sock.sendto(
                        packet,
                        (endpoint.ip, endpoint.send_port),
                    )
                    detail = "tx_udp_ok bytes=%d frame=earth_projection_v2" % len(packet)
                else:
                    detail = "tx_dry_run_ok bytes=%d frame=earth_projection_v2" % len(packet)

                with self.lock:
                    endpoint.detail = detail
                    endpoint.packet_counter = (endpoint.packet_counter + 1) & 0xFF
                    if self.send_udp:
                        endpoint.tx_count += 1
                        endpoint.last_tx = now_wall

                msg = UInt8MultiArray()
                msg.data = list(packet)
                self.payload_pub.publish(msg)
            except Exception as exc:
                rospy.logerr_throttle(1.0, "platform trajectory TX error: %s", exc)
                with self.lock:
                    endpoint.detail = "tx_error: %s" % exc
        self.publish_states_and_status()


if __name__ == "__main__":
    try:
        PlatformCommNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass