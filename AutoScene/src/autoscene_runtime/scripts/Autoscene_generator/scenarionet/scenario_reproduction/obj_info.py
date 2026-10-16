from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_reproduction.dongfeng import CURRENT_DIR, ObjectInfoProcessor


def main():
    processor = ObjectInfoProcessor(CURRENT_DIR / "Obj_info_intersection.csv")
    processor.run(Path(CURRENT_DIR) / "Obj_info_intersection.pkl")
    print(f"object info saved: {CURRENT_DIR / 'Obj_info_intersection.pkl'}")


if __name__ == "__main__":
    main()
