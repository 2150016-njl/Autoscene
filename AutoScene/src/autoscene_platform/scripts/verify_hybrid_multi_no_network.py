#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify hybrid YAML and protocol helpers without binding sockets.

Run from a sourced AutoScene/ROS environment:
  python3 verify_hybrid_multi_no_network.py \
    --config src/autoscene_bringup/config/hybrid.yaml \
    --project-root .

This script checks dry-run flags, endpoint uniqueness, projected-earth G
coordinates, 176-byte ADS payload decoding, G<->A transform round-trip, and
ego external-origin conversion. It does not send or receive UDP.
"""
import argparse
import importlib.util
import math
import os
import struct
import sys

import yaml


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load module: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap_rad(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def close(a, b, eps=1e-6):
    return abs(float(a) - float(b)) <= eps


def yaw_close(a, b, eps=1e-6):
    return abs(wrap_rad(float(a) - float(b))) <= eps


def build_ads_payload(latitude_deg, longitude_deg, heading_deg, speed_mps):
    """Build one valid 176-byte upstream ADS payload for decoder checks."""
    data = bytearray(176)
    struct.pack_into("<d", data, 0, float(latitude_deg))
    struct.pack_into("<d", data, 8, float(longitude_deg))
    struct.pack_into("<I", data, 16, 1)
    struct.pack_into("<i", data, 20, 123456)
    struct.pack_into("<i", data, 24, 2200)
    struct.pack_into("<f", data, 56, 0.02)
    struct.pack_into("<f", data, 60, 0.01)
    struct.pack_into("<f", data, 64, 1.0)
    struct.pack_into("<f", data, 68, 2.0)
    struct.pack_into("<f", data, 72, 3.0)
    struct.pack_into("<f", data, 76, 4.0)
    struct.pack_into("<f", data, 80, 5.0)
    struct.pack_into("<f", data, 84, 6.0)
    struct.pack_into("<f", data, 88, 9.9)
    struct.pack_into("<H", data, 92, int(round(float(heading_deg) * 100.0)) % 36000)
    struct.pack_into("<h", data, 94, int(round(float(speed_mps) * 3.6 * 100.0)))
    struct.pack_into("<h", data, 96, int(round(float(speed_mps) * 3.6 * 100.0)))
    struct.pack_into("<h", data, 98, 0)

    data[164] = 100
    data[165] = 0xF0
    data[166] = 11
    data[167] = 1
    data[168] = 1
    data[169] = 1
    data[170:174] = bytes([80, 80, 80, 80])
    data[174] = 0
    data[175] = 0
    return bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    root = os.path.abspath(args.project_root)
    config_path = os.path.abspath(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    platform_cfg = cfg.get("platform", {})
    endpoints_cfg = platform_cfg.get("platforms", [])

    assert cfg.get("runtime", {}).get("mode") == "hybrid"
    assert platform_cfg.get("receive_udp") is False
    assert platform_cfg.get("send_udp") is False
    assert platform_cfg.get("emulate_controlled_feedback") is True
    assert len(endpoints_cfg) >= 1

    platform_ids = [str(x.get("platform_id", "")) for x in endpoints_cfg]
    vehicle_ids = [str(x.get("vehicle_id", "")) for x in endpoints_cfg]
    local_endpoint = (
        str(platform_cfg.get("local_ip", "")),
        int(platform_cfg.get("local_port", 0)),
    )

    assert len(platform_ids) == len(set(platform_ids)), platform_ids
    assert len(vehicle_ids) == len(set(vehicle_ids)), vehicle_ids
    assert local_endpoint[0], local_endpoint
    assert local_endpoint[1] > 0, local_endpoint
    assert isinstance(platform_cfg.get("earth_to_autoscene"), dict)

    platform_path = os.path.join(
        root, "src", "autoscene_platform", "scripts", "platform_comm_node.py"
    )
    ego_path = os.path.join(
        root, "src", "autoscene_ego", "scripts", "ego_adapter_node.py"
    )

    platform_mod = load_module("verify_platform_comm_node", platform_path)
    ego_mod = load_module("verify_ego_adapter_node", ego_path)

    print("Shared local UDP endpoint: %s:%d" % local_endpoint)
    print("Configured remote endpoints:")
    for idx, ep_cfg in enumerate(endpoints_cfg, 1):
        print(
            "  %d: %s/%s remote=%s:%s expected=%s:%s"
            % (
                idx,
                ep_cfg.get("platform_id"),
                ep_cfg.get("vehicle_id"),
                ep_cfg.get("ip"),
                ep_cfg.get("send_port"),
                ep_cfg.get("expected_remote_ip"),
                ep_cfg.get("expected_remote_port"),
            )
        )

    endpoint = platform_mod.PlatformEndpoint(endpoints_cfg[0], platform_cfg)
    earth_pose = {
        "x": 500123.4,
        "y": 3420456.7,
        "yaw": math.radians(15.0),
    }
    wx, wy, wyaw = endpoint.earth_to_world_pose(
        earth_pose["x"], earth_pose["y"], earth_pose["yaw"]
    )
    gx, gy, gyaw = endpoint.world_to_earth_pose(wx, wy, wyaw)
    assert close(gx, earth_pose["x"])
    assert close(gy, earth_pose["y"])
    assert yaw_close(gyaw, earth_pose["yaw"])

    payload = build_ads_payload(
        latitude_deg=30.0,
        longitude_deg=120.0,
        heading_deg=90.0,
        speed_mps=5.0,
    )
    decoded = platform_mod.decode_ads_payload176(payload)
    errors = platform_mod.validate_ads_packet(decoded)
    assert not errors, errors
    earth_x, earth_y = platform_mod.latlon_to_xy(
        decoded["latitude_deg"], decoded["longitude_deg"]
    )
    assert math.isfinite(earth_x) and math.isfinite(earth_y)
    earth_yaw = platform_mod.ins_heading_deg_to_earth_yaw(decoded["heading_deg"])
    assert yaw_close(earth_yaw, 0.0)

    print(
        "PASS platform G<->A: earth=(%.3f, %.3f, %.6f) world=(%.3f, %.3f, %.6f)"
        % (gx, gy, gyaw, wx, wy, wyaw)
    )
    print(
        "PASS ADS decode: lat=%.8f lon=%.8f earth=(%.3f, %.3f) heading=%.2f"
        % (decoded["latitude_deg"], decoded["longitude_deg"], earth_x, earth_y, decoded["heading_deg"])
    )

    from autoscene_msgs.msg import VehicleState

    ego_origin = cfg.get("ego", {}).get("origin", {})
    ego_node = ego_mod.EgoAdapterNode.__new__(ego_mod.EgoAdapterNode)
    ego_node.origin_x = float(ego_origin.get("x", 0.0))
    ego_node.origin_y = float(ego_origin.get("y", 0.0))
    ego_node.origin_yaw = float(ego_origin.get("yaw", 0.0))
    ego_node.frame_id = cfg.get("runtime", {}).get("frame_id", "map")
    ego_node.ego_id = cfg.get("ego", {}).get("id", "ego_001")
    ego_node.length = 4.8
    ego_node.width = 2.0

    msg = VehicleState()
    msg.pose.position.x = 2.0
    msg.pose.position.y = -1.0
    msg.heading = math.radians(-15.0)
    msg.twist.linear.x = 3.0 * math.cos(msg.heading)
    msg.twist.linear.y = 3.0 * math.sin(msg.heading)
    msg.accel.linear.x = 0.5
    msg.accel.linear.y = -0.2
    msg.valid = True

    converted = ego_node.external_state_to_autoscene(msg)
    c = math.cos(ego_node.origin_yaw)
    s = math.sin(ego_node.origin_yaw)
    expected_x = ego_node.origin_x + c * 2.0 - s * (-1.0)
    expected_y = ego_node.origin_y + s * 2.0 + c * (-1.0)
    expected_yaw = wrap_rad(ego_node.origin_yaw + msg.heading)

    assert close(converted.pose.position.x, expected_x)
    assert close(converted.pose.position.y, expected_y)
    assert yaw_close(converted.heading, expected_yaw)

    print(
        "PASS ego external transform: world=(%.6f, %.6f, %.6f rad)"
        % (
            converted.pose.position.x,
            converted.pose.position.y,
            converted.heading,
        )
    )
    print("PASS: hybrid YAML, endpoint config, ADS decode, G<->A transform, ego transform")


if __name__ == "__main__":
    main()
