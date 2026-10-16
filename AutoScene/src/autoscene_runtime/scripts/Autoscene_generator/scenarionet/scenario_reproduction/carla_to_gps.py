"""
Convert Carla scenario coordinates back to GPS (lat, lon).

Usage:
    python carla_to_gps.py           # demo with a few test points
    python carla_to_gps.py --x 10.0 --y -20.0   # single point
    python carla_to_gps.py --scenario /path/to/sd_tjwest260528obj_info_1.pkl  # dump all vehicle start positions
"""
import argparse
import math
import pickle as pkl
import sys
from pathlib import Path

import numpy as np

# ============ WGS84 ellipsoid ============
A = 6378137.0
F = 1.0 / 298.257223563
E2 = 2 * F - F ** 2  # first eccentricity squared
E4 = E2 * E2
E6 = E4 * E2
EP2 = E2 / (1.0 - E2)  # second eccentricity squared

# ============ Gauss-Kruger offsets (from GPS_to_Global) ============
GK_OFFSET_X = 615000.0
GK_OFFSET_Y = 3463000.0
FALSE_EASTING = 500000.0
# Central meridian: 3-degree zone for Tongji area (~121.5°E) → 120°E
L0_DEG = 120.0
L0 = math.radians(L0_DEG)

# ============ Carla West calibration params (from GNSS_Transform) ============
DX_WEST = 169.2974
DY_WEST = 846.6133
THETA_G2C_WEST_DEG = 0.8 + 180.0  # 180.8°
# theta_c2g_west = -theta_g2c_west
THETA_C2G_WEST_DEG = -THETA_G2C_WEST_DEG
THETA_C2G = math.radians(THETA_C2G_WEST_DEG)


# ---------- helper: meridian arc length from latitude ----------
def _meridian_arc(lat_rad: float) -> float:
    """
    Meridian distance from equator to latitude lat_rad.
    Formula from Snyder p.101 eq.3-21.
    """
    a0 = A * (1.0 - E2 / 4.0 - 3.0 * E4 / 64.0 - 5.0 * E6 / 256.0)
    a2 = A * (3.0 * E2 / 8.0 + 3.0 * E4 / 32.0 + 45.0 * E6 / 1024.0) * (-1.0)
    a4 = A * (15.0 * E4 / 256.0 + 45.0 * E6 / 1024.0)
    a6 = A * (35.0 * E6 / 3072.0) * (-1.0)
    cos2l = math.cos(2.0 * lat_rad)
    cos4l = math.cos(4.0 * lat_rad)
    cos6l = math.cos(6.0 * lat_rad)
    return a0 * lat_rad + a2 * math.sin(2.0 * lat_rad) + a4 * math.sin(4.0 * lat_rad) + a6 * math.sin(6.0 * lat_rad)


def _footpoint_latitude(northing: float) -> float:
    """
    Footpoint latitude φ₁ — latitude of the point on the central meridian
    with the same northing (Snyder p.83 eq.3-26, p.94).
    """
    e1 = (1.0 - math.sqrt(1.0 - E2)) / (1.0 + math.sqrt(1.0 - E2))
    # M = northing  (k0 = 1 for Gauss-Kruger, FN = 0 for northern hemisphere)
    M = northing
    mu = M / (A * (1.0 - E2 / 4.0 - 3.0 * E4 / 64.0 - 5.0 * E6 / 256.0))
    phi1 = mu + (3.0 * e1 / 2.0 - 27.0 * e1 ** 3 / 32.0) * math.sin(2.0 * mu) \
           + (21.0 * e1 ** 2 / 16.0 - 55.0 * e1 ** 4 / 32.0) * math.sin(4.0 * mu) \
           + (151.0 * e1 ** 3 / 96.0) * math.sin(6.0 * mu)
    return phi1


def inverse_gauss_kruger(easting: float, northing: float, cm_rad: float):
    """
    Inverse Gauss-Kruger (Transverse Mercator) projection.
    Uses WGS84 ellipsoid, k0=1, false_easting=500000, false_northing=0.
    """
    # Relative easting (remove false easting)
    E_rel = easting - FALSE_EASTING

    # Footpoint latitude
    phi1 = _footpoint_latitude(northing)
    sin_p1 = math.sin(phi1)
    cos_p1 = math.cos(phi1)
    tan_p1 = math.tan(phi1)

    # N1 - radius of curvature in prime vertical
    N1 = A / math.sqrt(1.0 - E2 * sin_p1 * sin_p1)
    # C1
    C1 = EP2 * cos_p1 * cos_p1
    # D
    D = E_rel / (N1 * cos_p1)  # k0 = 1.0

    D2 = D * D
    D4 = D2 * D2
    D6 = D4 * D2

    # Latitude  (Snyder p.83 eq.5-5)
    phi = phi1 - (N1 * tan_p1) ** (-1) * (
        D2 / 2.0
        - (5.0 + 3.0 * tan_p1 ** 2 + 10.0 * C1 - 4.0 * C1 ** 2 - 9.0 * EP2) * D4 / 24.0
        + (61.0 + 90.0 * tan_p1 ** 2 + 298.0 * C1 + 45.0 * tan_p1 ** 4 - 252.0 * EP2 - 3.0 * C1 ** 2) * D6 / 720.0
    )
    # Alternative: use N1 as in Snyder
    # phi = phi1 - (tan_p1 / (k0 * N1)) ...
    # Since k0=1:
    phi = phi1 - (tan_p1 / N1) * (
        E_rel * D / 2.0
        - (5.0 + 3.0 * tan_p1 ** 2 + 10.0 * C1 - 4.0 * C1 ** 2 - 9.0 * EP2) * E_rel * D4 / 24.0
        + (61.0 + 90.0 * tan_p1 ** 2 + 298.0 * C1 + 45.0 * tan_p1 ** 4 - 252.0 * EP2 - 3.0 * C1 ** 2) * E_rel * D6 / 720.0
    ) / (N1 * cos_p1)

    # Longitude (Snyder p.84 eq.5-6)
    lam = cm_rad + (
        D
        - (1.0 + 2.0 * tan_p1 ** 2 + C1) * D4 / 6.0
        + (5.0 - 2.0 * C1 + 28.0 * tan_p1 ** 2 - 3.0 * C1 ** 2 + 8.0 * EP2 + 24.0 * tan_p1 ** 4) * D6 / 120.0
    ) / cos_p1

    return math.degrees(phi), math.degrees(lam)


def carla_to_gps(carla_x: float, carla_y: float, z: float = 0.0):
    """
    Convert a Carla-scenario coordinate to GPS (lat, lon).
    Follows the reverse of the pipeline in dongfeng.py / GNSS_Transform.py / GPS_to_Global.
    """
    # Step 1: Carla coordinate → Global coordinate (inverse of GNSS_Global_to_CarlaWest)
    x_temp = carla_x - DX_WEST
    y_temp = carla_y - DY_WEST
    global_x = x_temp * math.cos(THETA_C2G) - y_temp * math.sin(THETA_C2G)
    global_y = x_temp * math.sin(THETA_C2G) + y_temp * math.cos(THETA_C2G)
    global_x = -global_x  # match the negation in GNSS_Global_to_CarlaWest

    # Step 2: Global → Gauss-Kruger easting/northing
    gk_x = global_x + GK_OFFSET_X
    gk_y = global_y + GK_OFFSET_Y

    # Step 3: Inverse Gauss-Kruger → lat/lon
    lat, lon = inverse_gauss_kruger(gk_x, gk_y, L0)
    return lat, lon


def stored_heading_to_geographic(stored_heading_rad: float):
    """
    Convert the stored heading (from scenario tracks) to a geographic
    heading (degrees clockwise from north).

    Transformation in dongfeng.py:
        angle = 2π - psi
        stored = (angle + π) % (2π) - π   → normalize to [-π, π)

    Reverse:
        psi = (-stored) % (2π)             → original Carla heading in [0, 2π)
        geo = π/2 - psi                    → geographic (clockwise from north)
             = π/2 - (-stored % 2π)

    Carla heading convention: 0=east, π/2=north, CCW positive.
    Geographic convention:    0=north, π/2=east, CW positive.
    """
    # Reverse back to original Carla heading (actor_psi)
    psi_rad = (-stored_heading_rad) % (2 * math.pi)
    # Convert to geographic: π/2 - psi, normalized to [0, 2π)
    geo_rad = (math.pi / 2 - psi_rad) % (2 * math.pi)
    return math.degrees(geo_rad), math.degrees(psi_rad)


def demo():
    """Demo: compute GPS of Carla origin and a few offsets."""
    print("=== Carla coordinate → GPS (lat, lon) ===")
    print(f"Central meridian: {L0_DEG}°E, WGS84 ellipsoid")
    print(f"Carla→Global: dx={DX_WEST}, dy={DY_WEST}, theta={THETA_G2C_WEST_DEG:.2f}°\n")

    print(f"{'Carla X':>10} {'Carla Y':>10} → {'Latitude':>12} {'Longitude':>12}")
    print("-" * 48)
    for cx, cy in [(0, 0), (10, 0), (0, 10), (-10, 0), (100, 50)]:
        lat, lon = carla_to_gps(cx, cy)
        print(f"{cx:10.1f} {cy:10.1f} → {lat:12.6f} {lon:12.6f}")

    print()
    print("=== Carla heading → Geographic heading ===")
    print(f"{'Stored(rad)':>14} {'Carla(deg)':>12} {'Geo(deg)':>10}")
    print("-" * 40)
    for h in [0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2]:
        geo, carla = stored_heading_to_geographic(h)
        print(f"{h:14.4f} {carla:12.1f} {geo:10.1f}")


def from_scenario_pkl(scenario_path: str):
    """Read a scenario pkl and dump each vehicle's start position as GPS."""
    path = Path(scenario_path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("rb") as f:
        scenario = pkl.load(f)

    tracks = scenario.get("tracks", {})
    print(f"Scenario: {scenario.get('id', path.stem)}")
    print(f"Vehicles / objects: {len(tracks)}\n")

    print(f"{'Object ID':>12} {'Type':>12} → {'Latitude':>12} {'Longitude':>12} {'GeoHead(°)':>10}")
    print("-" * 60)
    for obj_id, track in tracks.items():
        pos = track["state"]["position"]
        heading = track["state"]["heading"]
        start_x, start_y, _ = pos[0]
        start_h = heading[0]
        # obj_info stored (x, -y, 0), so undo y-negate for Carla coordinate
        carla_x, carla_y = float(start_x), -float(start_y)
        lat, lon = carla_to_gps(carla_x, carla_y)
        geo_deg, _ = stored_heading_to_geographic(start_h)
        print(f"{obj_id:>12} {track['type']:>12} → {lat:12.6f} {lon:12.6f} {geo_deg:10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Convert Carla scenario coordinates to GPS (lat, lon)")
    parser.add_argument("--x", type=float, help="Carla X coordinate")
    parser.add_argument("--y", type=float, help="Carla Y coordinate")
    parser.add_argument(
        "--scenario", type=str,
        help="Path to scenario .pkl file (e.g. sd_tjwest260528obj_info_1.pkl). "
             "Dumps start position of every object as GPS."
    )
    args = parser.parse_args()

    if args.scenario:
        from_scenario_pkl(args.scenario)
    elif args.x is not None and args.y is not None:
        lat, lon = carla_to_gps(args.x, args.y)
        print(f"({args.x:.3f}, {args.y:.3f}) → ({lat:.6f}, {lon:.6f})")
    else:
        demo()


if __name__ == "__main__":
    main()
