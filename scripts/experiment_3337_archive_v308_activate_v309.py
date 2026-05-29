#!/usr/bin/env python3
"""Run Exp 3337 archive .308 and .309 activation aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.archive_v308_activate_v309_3337 import (  # noqa: E402
    ACTIVE_ROADMAP_REL_PATH,
    OUTPUT_REL_PATH,
    STAGED_ROADMAP_REL_PATH,
    write_artifact,
)


def main() -> int:
    output_path = write_artifact(REPO_ROOT)
    artifact = json.loads((REPO_ROOT / OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    if not output_path.is_file():
        raise RuntimeError(f"deliverable was not written: {output_path}")

    json.loads(output_path.read_text(encoding="utf-8"))
    try:
        import yaml

        yaml.safe_load((REPO_ROOT / ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8"))
        staged_path = REPO_ROOT / STAGED_ROADMAP_REL_PATH
        if staged_path.is_file():
            yaml.safe_load(staged_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("roadmap YAML validation failed") from exc

    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["archive_v308_activate_v309_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
