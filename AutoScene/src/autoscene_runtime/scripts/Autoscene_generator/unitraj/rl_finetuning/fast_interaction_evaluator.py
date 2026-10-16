from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class VehicleCircleSpec:
    length: float = 4.8
    width: float = 2.0
    fallback_radius: float = 2.5


def _as_xy(value):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 2 or not np.isfinite(arr[:2]).all():
        return None
    return arr[:2]


def make_vehicle_state(
        position,
        velocity=None,
        heading: float = 0.0,
        speed: Optional[float] = None,
        length: Optional[float] = None,
        width: Optional[float] = None,
        fallback_radius: float = 2.5,
) -> Optional[Dict]:
    pos = _as_xy(position)
    if pos is None:
        return None

    vel = _as_xy(velocity)
    heading = float(heading) if np.isfinite(heading) else 0.0
    if vel is None:
        speed = 0.0 if speed is None else float(speed)
        vel = speed * np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)

    try:
        length = float(length)
        width = float(width)
    except Exception:
        length = 2.0 * float(fallback_radius)
        width = float(fallback_radius)
    if not np.isfinite(length) or length <= 0.0:
        length = 2.0 * float(fallback_radius)
    if not np.isfinite(width) or width <= 0.0:
        width = float(fallback_radius)

    return {
        "position": pos,
        "velocity": vel,
        "heading": heading,
        "length": length,
        "width": width,
        "fallback_radius": float(fallback_radius),
    }


def _three_circle_centers(state: Dict):
    pos = np.asarray(state["position"], dtype=np.float64)
    heading = float(state.get("heading", 0.0))
    length = float(state.get("length", 4.8))
    width = float(state.get("width", 2.0))
    radius = max(0.5 * width, float(state.get("fallback_radius", 2.5)) * 0.45)
    offset = max(0.0, 0.5 * length - radius)
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    offsets = np.array([-offset, 0.0, offset], dtype=np.float64)
    centers = pos[None, :] + offsets[:, None] * direction[None, :]
    return centers, radius


def pair_ttc_three_circle(source: Optional[Dict], target: Optional[Dict]) -> float:
    if source is None or target is None:
        return float("inf")

    source_centers, source_radius = _three_circle_centers(source)
    target_centers, target_radius = _three_circle_centers(target)
    rel_pos = target_centers[None, :, :] - source_centers[:, None, :]
    rel_vel = np.asarray(target["velocity"], dtype=np.float64) - np.asarray(source["velocity"], dtype=np.float64)
    radius_sum = source_radius + target_radius

    a = float(np.dot(rel_vel, rel_vel))
    c = np.sum(rel_pos * rel_pos, axis=-1) - radius_sum ** 2
    if np.any(c <= 0.0):
        return 0.0
    if a < 1e-8:
        return float("inf")

    b = 2.0 * np.sum(rel_pos * rel_vel[None, None, :], axis=-1)
    approaching = b < 0.0
    disc = b * b - 4.0 * a * c
    valid = approaching & (disc >= 0.0)
    if not np.any(valid):
        return float("inf")

    ttc = (-b[valid] - np.sqrt(disc[valid])) / (2.0 * a)
    ttc = ttc[ttc >= 0.0]
    return float(np.min(ttc)) if len(ttc) > 0 else float("inf")


def evaluate_three_vehicle_interaction(
        ego_state: Optional[Dict],
        av1_state: Optional[Dict],
        av2_state: Optional[Dict],
) -> Dict[str, float]:
    ttc_av2_to_av1 = pair_ttc_three_circle(av2_state, av1_state)
    ttc_av1_to_ego = pair_ttc_three_circle(av1_state, ego_state)
    ttc_av2_to_ego = pair_ttc_three_circle(av2_state, ego_state)
    return {
        "ttc_av2_to_av1": ttc_av2_to_av1,
        "ttc_av1_to_ego": ttc_av1_to_ego,
        "ttc_av2_to_ego": ttc_av2_to_ego,
        "urgency_u": min(ttc_av1_to_ego, ttc_av2_to_ego),
    }
