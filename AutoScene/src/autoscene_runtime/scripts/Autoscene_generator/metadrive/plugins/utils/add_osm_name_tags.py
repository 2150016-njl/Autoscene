#!/usr/bin/env python3
"""
Add or normalize OSM name tags for nodes and ways.

By default, each node/way gets a ``name`` tag whose value is the XML element
``id`` attribute. The name source can also be switched to another tag key, such
as ``Id``, when needed.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from typing import Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
DEFAULT_INPUT_OSM = os.path.join(
    REPO_ROOT,
    "metadrive",
    "CHT",
    "map_files",
    "TesT_Field_Tongji_West_with_relations_direction.osm",
)


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


def find_name_tag(element: ET.Element) -> Optional[ET.Element]:
    for child in element.findall("./tag"):
        if child.get("k") == "name":
            return child
    return None


def find_tag(element: ET.Element, key: str) -> Optional[ET.Element]:
    for child in element.findall("./tag"):
        if child.get("k") == key:
            return child
    return None


def resolve_name_value(element: ET.Element, source_key: str) -> Optional[str]:
    if source_key == "element-id":
        return element.get("id")

    source_tag = find_tag(element, source_key)
    if source_tag is None:
        return None
    return source_tag.get("v")


def ensure_node_name(node: ET.Element, source_key: str) -> bool:
    name_value = resolve_name_value(node, source_key)
    if name_value is None:
        return False

    name_tag = find_name_tag(node)
    if name_tag is None:
        ET.SubElement(node, "tag", attrib={"k": "name", "v": name_value})
        return True

    if name_tag.get("v") != name_value:
        name_tag.set("v", name_value)
        return True
    return False


def ensure_way_name(way: ET.Element, source_key: str) -> bool:
    name_value = resolve_name_value(way, source_key)
    if name_value is None:
        return False

    name_tag = find_name_tag(way)
    if name_tag is not None:
        changed = False
        if name_tag.get("v") != name_value:
            name_tag.set("v", name_value)
            changed = True

        children = list(way)
        desired_index = len(way.findall("./nd"))
        current_index = children.index(name_tag)
        if current_index != desired_index:
            way.remove(name_tag)
            way.insert(desired_index, name_tag)
            changed = True
        return changed

    insert_index = len(way.findall("./nd"))
    new_tag = ET.Element("tag", attrib={"k": "name", "v": name_value})
    way.insert(insert_index, new_tag)
    return True


def process_osm(input_path: str, output_path: str, name_source: str) -> Tuple[int, int]:
    tree = ET.parse(input_path)
    root = tree.getroot()

    node_changes = 0
    way_changes = 0

    for node in root.findall("./node"):
        if ensure_node_name(node, name_source):
            node_changes += 1

    for way in root.findall("./way"):
        if ensure_way_name(way, name_source):
            way_changes += 1

    indent_xml(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return node_changes, way_changes


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add OSM name tags for nodes and ways.")
    parser.add_argument(
        "input_osm",
        nargs="?",
        default=DEFAULT_INPUT_OSM,
        help=f"Input OSM path. Defaults to {DEFAULT_INPUT_OSM}.",
    )
    parser.add_argument("output_osm", nargs="?", help="Output OSM path. Defaults to in-place overwrite.")
    parser.add_argument(
        "--name-source",
        default="element-id",
        help="Source for the `name` value. Use `element-id` or a tag key such as `Id`.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = os.path.abspath(args.input_osm)
    output_path = os.path.abspath(args.output_osm or args.input_osm)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input OSM not found: {input_path}")

    in_place = input_path == output_path
    if not in_place and os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    node_changes, way_changes = process_osm(input_path, output_path, args.name_source)
    print(
        f"updated_nodes={node_changes} updated_ways={way_changes} "
        f"name_source={args.name_source} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
