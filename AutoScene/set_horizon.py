#!/usr/bin/env python3
from pathlib import Path
import re, shutil, sys
from datetime import datetime

if len(sys.argv) not in (2, 3):
    raise SystemExit("usage: set_horizon.py /path/to/unitraj/configs/config.yaml [steps]")
path = Path(sys.argv[1]).expanduser().resolve()
steps = int(sys.argv[2]) if len(sys.argv) == 3 else 500
if steps <= 0:
    raise SystemExit("steps must be > 0")
text = path.read_text(encoding="utf-8")
backup = path.with_suffix(path.suffix + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
shutil.copy2(path, backup)
pattern = re.compile(r"(?m)^horizon\s*:\s*[^#\n]*(?P<comment>\s*#.*)?$")
if pattern.search(text):
    text = pattern.sub(lambda m: f"horizon: {steps}" + (m.group("comment") or ""), text, count=1)
else:
    text = f"# MetaDrive/AutoScene closed-loop experiment horizon in 0.1 s steps\nhorizon: {steps}\n" + text
path.write_text(text, encoding="utf-8")
print("updated:", path)
print("backup :", backup)
print("horizon:", steps)
