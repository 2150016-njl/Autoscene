from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_reproduction.dongfeng import CURRENT_DIR, OSMMapProcessor


def main():
    default_osm = CURRENT_DIR / "TesT_Field_Tongji_West_with_relations_name.osm"
    processor = OSMMapProcessor(default_osm, include_polygons=False)
    processor.run(Path(CURRENT_DIR) / "map_features_dongfeng.pkl")
    print(f"map features saved: {CURRENT_DIR / 'map_features_dongfeng.pkl'}")


if __name__ == "__main__":
    main()
