#!/usr/bin/env python3
"""
Infer lanelet left/right boundary relations from OSM geometry.

The script is intentionally independent from the MetaDrive runtime. It reads an
OSM file, finds centerline ways and boundary ways, matches the nearest
left/right boundary for each centerline, then writes a copy of the OSM with
lanelet relations appended.
"""

import argparse
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EARTH_RADIUS_M = 6378137.0
EPS = 1e-9
BOUNDARY_SUBTYPES = {"solid", "dashed", "curbstone"}


@dataclass
class WayGeom:
    way_id: str
    element: ET.Element
    tags: Dict[str, str]
    node_refs: List[str]
    xy: np.ndarray
    bbox: Tuple[float, float, float, float]
    seg_start: np.ndarray
    seg_vec: np.ndarray
    seg_len2: np.ndarray


@dataclass
class BoundaryMatch:
    boundary_id: str
    side: str
    score: float
    mean_distance: float
    p90_distance: float
    side_ratio: float
    coverage: float


def extract_tags(element: ET.Element) -> Dict[str, str]:
    return {
        tag.get("k"): tag.get("v")
        for tag in element.findall("./tag")
        if tag.get("k") is not None and tag.get("v") is not None
    }


def get_tag(tags: Dict[str, str], *keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        if key in tags:
            return tags[key]
    lower_tags = {key.lower(): value for key, value in tags.items()}
    for key in keys:
        value = lower_tags.get(key.lower())
        if value is not None:
            return value
    return default


def set_tag(element: ET.Element, tags: Dict[str, str], key: str, value: str, overwrite: bool = False) -> None:
    if key in tags and not overwrite:
        return
    for tag in element.findall("./tag"):
        if tag.get("k") == key:
            if overwrite:
                tag.set("v", value)
                tags[key] = value
            return
    ET.SubElement(element, "tag", attrib={"k": key, "v": value})
    tags[key] = value


def is_deleted(element: ET.Element) -> bool:
    return element.get("action") == "delete"


def is_center_way(tags: Dict[str, str]) -> bool:
    type_tag = get_tag(tags, "Type", "type", default="")
    subtype = get_tag(tags, "subtype", default="")
    if type_tag == "Lane":
        return True
    if subtype == "centerlane":
        return True
    return False


def is_boundary_way(tags: Dict[str, str]) -> bool:
    type_tag = get_tag(tags, "Type", "type", default="")
    subtype = get_tag(tags, "subtype", default="")
    if type_tag == "LaneBoundary":
        return True
    if subtype in BOUNDARY_SUBTYPES:
        return True
    return False


def bbox_for(points: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def inflate_bbox(bbox: Tuple[float, float, float, float], margin: float) -> Tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = bbox
    return min_x - margin, min_y - margin, max_x + margin, max_y + margin


def bboxes_intersect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def filter_repeated_points(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = [True]
    diffs = np.linalg.norm(points[1:] - points[:-1], axis=1)
    keep.extend(diff > EPS for diff in diffs)
    return points[np.asarray(keep, dtype=bool)]


def build_way_geom(way: ET.Element, node_xy: Dict[str, np.ndarray]) -> Optional[WayGeom]:
    node_refs = [nd.get("ref") for nd in way.findall("./nd") if nd.get("ref") in node_xy]
    if len(node_refs) < 2:
        return None

    xy = np.asarray([node_xy[ref] for ref in node_refs], dtype=float)
    xy = filter_repeated_points(xy)
    if len(xy) < 2:
        return None

    seg_start = xy[:-1]
    seg_vec = xy[1:] - xy[:-1]
    seg_len2 = np.sum(seg_vec * seg_vec, axis=1)
    valid = seg_len2 > EPS
    if not np.any(valid):
        return None

    return WayGeom(
        way_id=way.get("id"),
        element=way,
        tags=extract_tags(way),
        node_refs=node_refs,
        xy=xy,
        bbox=bbox_for(xy),
        seg_start=seg_start[valid],
        seg_vec=seg_vec[valid],
        seg_len2=seg_len2[valid],
    )


def parse_node_xy(root: ET.Element) -> Dict[str, np.ndarray]:
    raw_geo: List[Tuple[str, float, float]] = []
    node_xy: Dict[str, np.ndarray] = {}
    anchor: Optional[Tuple[float, float]] = None

    for node in root.findall("./node"):
        node_id = node.get("id")
        if node_id is None:
            continue

        tags = extract_tags(node)
        if "local_x" in tags and "local_y" in tags:
            node_xy[node_id] = np.asarray([float(tags["local_x"]), float(tags["local_y"])], dtype=float)
            continue

        lat = node.get("lat")
        lon = node.get("lon")
        if lat is None or lon is None:
            continue
        lat_f = float(lat)
        lon_f = float(lon)
        if anchor is None:
            anchor = (lat_f, lon_f)
        raw_geo.append((node_id, lat_f, lon_f))

    if raw_geo:
        lat0, lon0 = anchor
        lat0_rad = math.radians(lat0)
        lon0_rad = math.radians(lon0)
        cos_lat0 = math.cos(lat0_rad)
        for node_id, lat, lon in raw_geo:
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)
            x = EARTH_RADIUS_M * cos_lat0 * (lon_rad - lon0_rad)
            y = EARTH_RADIUS_M * (lat_rad - lat0_rad)
            node_xy[node_id] = np.asarray([x, y], dtype=float)

    return node_xy


def sample_polyline(points: np.ndarray, sample_count: int) -> Tuple[np.ndarray, np.ndarray]:
    points = filter_repeated_points(points)
    seg_vec = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg_vec, axis=1)
    valid = seg_len > EPS
    seg_vec = seg_vec[valid]
    seg_len = seg_len[valid]
    seg_start = points[:-1][valid]
    if len(seg_len) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)

    total = float(np.sum(seg_len))
    count = max(2, int(sample_count))
    targets = np.linspace(0.0, total, count)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    samples = []
    tangents = []
    for target in targets:
        seg_idx = int(np.searchsorted(cum, target, side="right") - 1)
        seg_idx = min(max(seg_idx, 0), len(seg_len) - 1)
        local = (target - cum[seg_idx]) / max(seg_len[seg_idx], EPS)
        local = min(max(local, 0.0), 1.0)
        tangent = seg_vec[seg_idx] / max(seg_len[seg_idx], EPS)
        samples.append(seg_start[seg_idx] + local * seg_vec[seg_idx])
        tangents.append(tangent)

    return np.asarray(samples, dtype=float), np.asarray(tangents, dtype=float)


def closest_point_on_way(point: np.ndarray, boundary: WayGeom) -> Tuple[float, np.ndarray]:
    rel = point - boundary.seg_start
    t = np.sum(rel * boundary.seg_vec, axis=1) / boundary.seg_len2
    t = np.clip(t, 0.0, 1.0)
    closest = boundary.seg_start + t[:, None] * boundary.seg_vec
    dist2 = np.sum((closest - point) ** 2, axis=1)
    best_idx = int(np.argmin(dist2))
    return math.sqrt(float(dist2[best_idx])), closest[best_idx]


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def evaluate_boundary(
    center_samples: np.ndarray,
    center_tangents: np.ndarray,
    boundary: WayGeom,
    max_distance: float,
    min_lateral: float,
    min_side_ratio: float,
    min_coverage: float,
) -> Optional[BoundaryMatch]:
    distances = []
    signed_lateral = []

    for point, tangent in zip(center_samples, center_tangents):
        distance, closest = closest_point_on_way(point, boundary)
        distances.append(distance)
        signed_lateral.append(cross_2d(tangent, closest - point))

    distances_np = np.asarray(distances, dtype=float)
    signed_np = np.asarray(signed_lateral, dtype=float)
    left_ratio = float(np.mean(signed_np > min_lateral))
    right_ratio = float(np.mean(signed_np < -min_lateral))

    if max(left_ratio, right_ratio) < min_side_ratio:
        return None

    side = "left" if left_ratio >= right_ratio else "right"
    side_ratio = left_ratio if side == "left" else right_ratio
    coverage = float(np.mean(distances_np <= max_distance))
    if coverage < min_coverage:
        return None

    mean_distance = float(np.mean(distances_np))
    p90_distance = float(np.percentile(distances_np, 90))
    if p90_distance > max_distance * 1.5:
        return None

    score = mean_distance + 0.25 * p90_distance + 2.0 * (1.0 - side_ratio) + 2.0 * (1.0 - coverage)
    return BoundaryMatch(
        boundary_id=boundary.way_id,
        side=side,
        score=float(score),
        mean_distance=mean_distance,
        p90_distance=p90_distance,
        side_ratio=side_ratio,
        coverage=coverage,
    )


def choose_boundaries(
    center: WayGeom,
    boundaries: Sequence[WayGeom],
    sample_count: int,
    max_distance: float,
    min_lateral: float,
    min_side_ratio: float,
    min_coverage: float,
) -> Tuple[Optional[BoundaryMatch], Optional[BoundaryMatch]]:
    center_samples, center_tangents = sample_polyline(center.xy, sample_count)
    if len(center_samples) == 0:
        return None, None

    search_bbox = inflate_bbox(center.bbox, max_distance)
    left_matches: List[BoundaryMatch] = []
    right_matches: List[BoundaryMatch] = []

    for boundary in boundaries:
        if not bboxes_intersect(search_bbox, boundary.bbox):
            continue
        match = evaluate_boundary(
            center_samples=center_samples,
            center_tangents=center_tangents,
            boundary=boundary,
            max_distance=max_distance,
            min_lateral=min_lateral,
            min_side_ratio=min_side_ratio,
            min_coverage=min_coverage,
        )
        if match is None:
            continue
        if match.side == "left":
            left_matches.append(match)
        else:
            right_matches.append(match)

    left_matches.sort(key=lambda item: item.score)
    right_matches.sort(key=lambda item: item.score)
    left = left_matches[0] if left_matches else None
    right = right_matches[0] if right_matches else None

    if left is not None and right is not None and left.boundary_id == right.boundary_id:
        if left.score <= right.score:
            right = None
        else:
            left = None
    return left, right


def is_lanelet_relation(relation: ET.Element) -> bool:
    if is_deleted(relation):
        return False
    tags = extract_tags(relation)
    return get_tag(tags, "type", default="") == "lanelet"


def get_relation_center_ref(relation: ET.Element) -> Optional[str]:
    for member in relation.findall("./member"):
        if member.get("type") == "way" and member.get("role") == "center":
            return member.get("ref")
    return None


def next_relation_id(root: ET.Element, requested_start: Optional[int]) -> int:
    if requested_start is not None:
        return requested_start

    relation_ids = []
    for relation in root.findall("./relation"):
        try:
            relation_ids.append(int(relation.get("id")))
        except (TypeError, ValueError):
            pass

    min_existing = min(relation_ids) if relation_ids else 0
    return min(0, min_existing) - 1


def append_lanelet_relation(
    root: ET.Element,
    relation_id: int,
    center: WayGeom,
    left: Optional[BoundaryMatch],
    right: Optional[BoundaryMatch],
    relation_version: Optional[str],
    default_turn_direction: str,
) -> None:
    attrs = {"id": str(relation_id), "action": "modify", "visible": "true"}
    if relation_version:
        attrs["version"] = relation_version

    relation = ET.SubElement(root, "relation", attrib=attrs)
    ET.SubElement(relation, "member", attrib={"type": "way", "ref": center.way_id, "role": "center"})
    if right is not None:
        ET.SubElement(relation, "member", attrib={"type": "way", "ref": right.boundary_id, "role": "right"})
    if left is not None:
        ET.SubElement(relation, "member", attrib={"type": "way", "ref": left.boundary_id, "role": "left"})

    road_level = get_tag(center.tags, "road_level")
    if road_level is not None:
        ET.SubElement(relation, "tag", attrib={"k": "road_level", "v": road_level})

    speed_limit = get_tag(center.tags, "speed_limit", "SpeedLimit", "maxspeed")
    if speed_limit is not None:
        ET.SubElement(relation, "tag", attrib={"k": "speed_limit", "v": speed_limit})

    turn_direction = get_tag(center.tags, "turn_direction", "TurnRelation", default=default_turn_direction)
    if turn_direction:
        ET.SubElement(relation, "tag", attrib={"k": "turn_direction", "v": turn_direction})

    junction = get_tag(center.tags, "junction")
    if junction is not None:
        ET.SubElement(relation, "tag", attrib={"k": "junction", "v": junction})

    ET.SubElement(relation, "tag", attrib={"k": "type", "v": "lanelet"})


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(element):
        child_indent = "\n" + (level + 1) * "  "
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in element:
            indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_indent
        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = indent
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def write_report(report_path: str, rows: Iterable[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    fieldnames = [
        "center_id",
        "status",
        "left_boundary_id",
        "right_boundary_id",
        "left_score",
        "right_score",
        "left_mean_distance",
        "right_mean_distance",
        "left_coverage",
        "right_coverage",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_score(match: Optional[BoundaryMatch]) -> str:
    return "" if match is None else f"{match.score:.6f}"


def format_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def collect_existing_centers(root: ET.Element, replace_existing: bool) -> Tuple[set, int]:
    existing_centers = set()
    removed = 0
    for relation in list(root.findall("./relation")):
        if not is_lanelet_relation(relation):
            continue
        center_ref = get_relation_center_ref(relation)
        if center_ref is None:
            continue
        if replace_existing:
            root.remove(relation)
            removed += 1
        else:
            existing_centers.add(center_ref)
    return existing_centers, removed


def maybe_add_metadrive_type_tags(centers: Sequence[WayGeom], boundaries: Sequence[WayGeom]) -> None:
    for center in centers:
        if "type" not in center.tags:
            set_tag(center.element, center.tags, "type", "LANE_SURFACE_STREET")
    for boundary in boundaries:
        if "type" not in boundary.tags:
            set_tag(boundary.element, boundary.tags, "type", "ROAD_EDGE_BOUNDARY")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer lanelet relations from centerline/boundary geometry in an OSM file."
    )
    parser.add_argument("input_osm", help="Input OSM file.")
    parser.add_argument("output_osm", help="Output OSM file with appended relation elements.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing output_osm if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only compute matches and print/report results.")
    parser.add_argument("--report-path", help="Optional CSV report path.")
    parser.add_argument("--replace-existing", action="store_true", help="Remove existing lanelet relations with centers.")
    parser.add_argument("--allow-partial", action="store_true", help="Write relation if only one side is matched.")
    parser.add_argument(
        "--add-metadrive-type-tags",
        action="store_true",
        help="Add lowercase MetaDrive type tags to Type=Lane/LaneBoundary ways when missing.",
    )
    parser.add_argument("--relation-id-start", type=int, help="First relation id. Defaults to a new negative id.")
    parser.add_argument("--relation-version", help="Optional version attribute to add to new relations.")
    parser.add_argument("--default-turn-direction", default="straight", help="Relation turn_direction fallback.")
    parser.add_argument("--sample-count", type=int, default=17, help="Samples per centerline for matching.")
    parser.add_argument("--max-distance", type=float, default=6.0, help="Max boundary search distance in meters.")
    parser.add_argument("--min-lateral", type=float, default=0.25, help="Minimum lateral offset to count a side.")
    parser.add_argument("--min-side-ratio", type=float, default=0.65, help="Required same-side sample ratio.")
    parser.add_argument("--min-coverage", type=float, default=0.55, help="Required samples within max-distance.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = os.path.abspath(args.input_osm)
    output_path = os.path.abspath(args.output_osm)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input OSM not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Refusing to use the same input and output path. Write to a copy instead.")
    if os.path.exists(output_path) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output file already exists: {output_path}")

    tree = ET.parse(input_path)
    root = tree.getroot()
    node_xy = parse_node_xy(root)

    centers: List[WayGeom] = []
    boundaries: List[WayGeom] = []
    for way in root.findall("./way"):
        if is_deleted(way):
            continue
        geom = build_way_geom(way, node_xy)
        if geom is None:
            continue
        if is_center_way(geom.tags):
            centers.append(geom)
        elif is_boundary_way(geom.tags):
            boundaries.append(geom)

    existing_centers, removed_relations = collect_existing_centers(root, args.replace_existing)
    if args.add_metadrive_type_tags:
        maybe_add_metadrive_type_tags(centers, boundaries)

    relation_id = next_relation_id(root, args.relation_id_start)
    relation_step = -1 if relation_id < 0 else 1

    rows: List[Dict[str, object]] = []
    matched = 0
    partial = 0
    skipped_existing = 0
    unmatched = 0

    for center in centers:
        if center.way_id in existing_centers:
            skipped_existing += 1
            rows.append({"center_id": center.way_id, "status": "skip_existing"})
            continue

        left, right = choose_boundaries(
            center=center,
            boundaries=boundaries,
            sample_count=args.sample_count,
            max_distance=args.max_distance,
            min_lateral=args.min_lateral,
            min_side_ratio=args.min_side_ratio,
            min_coverage=args.min_coverage,
        )

        has_both = left is not None and right is not None
        has_any = left is not None or right is not None
        if not has_both and not (args.allow_partial and has_any):
            unmatched += 1
            status = "unmatched"
        else:
            status = "matched" if has_both else "partial"
            if has_both:
                matched += 1
            else:
                partial += 1
            if not args.dry_run:
                append_lanelet_relation(
                    root=root,
                    relation_id=relation_id,
                    center=center,
                    left=left,
                    right=right,
                    relation_version=args.relation_version,
                    default_turn_direction=args.default_turn_direction,
                )
                relation_id += relation_step

        rows.append(
            {
                "center_id": center.way_id,
                "status": status,
                "left_boundary_id": "" if left is None else left.boundary_id,
                "right_boundary_id": "" if right is None else right.boundary_id,
                "left_score": format_score(left),
                "right_score": format_score(right),
                "left_mean_distance": format_float(None if left is None else left.mean_distance),
                "right_mean_distance": format_float(None if right is None else right.mean_distance),
                "left_coverage": format_float(None if left is None else left.coverage),
                "right_coverage": format_float(None if right is None else right.coverage),
            }
        )

    if args.report_path:
        write_report(args.report_path, rows)

    if not args.dry_run:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        indent_xml(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(
        "centers={centers} boundaries={boundaries} matched={matched} partial={partial} "
        "unmatched={unmatched} skipped_existing={skipped_existing} removed_existing={removed} output={output}".format(
            centers=len(centers),
            boundaries=len(boundaries),
            matched=matched,
            partial=partial,
            unmatched=unmatched,
            skipped_existing=skipped_existing,
            removed=removed_relations,
            output="(dry-run)" if args.dry_run else output_path,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
