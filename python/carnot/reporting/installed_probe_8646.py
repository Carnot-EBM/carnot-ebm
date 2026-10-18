"""REQ-VERIFY-8646: record actual installed modules after public contract execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import sys


def probe(mode: str) -> dict[str, object]:
    """Hash resolved module bytes so cached imports cannot conceal checkout shadowing."""
    original = Path(__file__).with_name("installed_probe_8634.py")
    result: dict[str, object] = runpy.run_path(str(original))["probe"](mode)
    modules, dependencies = [], []
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename or not name.startswith(("carnot", "numpy", "jax", "flax")):
            continue
        path = Path(filename).resolve()
        row = dict(
            name=name,
            path=str(path),
            sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        if name == "carnot" or name.startswith("carnot."):
            assert path.is_relative_to(Path(sys.prefix).resolve()), str(path)
            modules.append(row)
        else:
            dependencies.append(row)
    result.update(modules=modules, dependencies=dependencies)
    return result


if __name__ == "__main__":
    print(json.dumps(probe(sys.argv[1]), sort_keys=True), flush=True)
