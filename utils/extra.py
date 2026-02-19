import site, os
from pathlib import Path

candidates = []
for sp in site.getsitepackages():
    sp = Path(sp)
    for p in sp.rglob("node.exe"):
        candidates.append(p)

print("Found node.exe candidates:")
for p in candidates:
    print(" -", p)
