"""Validated planner choices shared by factory and per-vehicle policy."""
import json
import os


PLANNERS = {"pluto": "Pluto", "vbd": "VBD", "pluto_hybrid": "pluto_hybrid"}


def planner_name(value):
    name = PLANNERS.get(str(value).strip().lower())
    if name is None:
        raise ValueError("planner_method must be Pluto, VBD or pluto_hybrid, got %r" % value)
    return name


def apply_planner_overrides(cfg):
    payload = json.loads(os.environ.get("UNITRAJ_VEHICLE_PLANNERS", "{}"))
    if not isinstance(payload, dict):
        raise ValueError("UNITRAJ_VEHICLE_PLANNERS must be a JSON object")
    for role in ("ego_method", "adv_method"):
        if role in payload:
            cfg[role] = planner_name(payload[role])
    if "vehicle_planners" in payload:
        mapping = payload["vehicle_planners"]
        if not isinstance(mapping, dict):
            raise ValueError("vehicle_planners must be a track-id mapping")
        cfg["vehicle_planners"] = {str(key): planner_name(value) for key, value in mapping.items()}
    if payload:
        cfg["per_vehicle_dispatch"] = True
    return cfg
