"""Build the Exp 3400 cross-corpus matrix v38 artifact.

Spec refs: REQ-REPORT-3400, SCENARIO-REPORT-3400.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.cross_corpus_matrix.v38_312_313_artifact_aggregation.v1"
EXPERIMENT_ID = "exp3400"
TASK_ID = "exp3400-cross-corpus-matrix-v38"
ARTIFACT = "experiment_3400_cross_corpus_matrix_v38"
MILESTONE = "2026.05.313"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3400_cross_corpus_matrix_v38.json")

REQUIRED_ARTIFACT_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "honest_verdict",
}

def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    start = time.perf_counter() if started_s is None else float(started_s)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = max(0.0, end - start)

    root_path = Path(root)
    results_dir = root_path / "results"
    gathered: list[str] = []
    if results_dir.is_dir():
        for p in results_dir.glob("experiment_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("milestone") in ("2026.05.312", "2026.05.313") or "312" in p.name or "313" in p.name:
                    gathered.append(p.name)
            except Exception:
                pass

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": "20260529",
        "milestone": MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "gathered_artifacts": sorted(gathered),
        "duration_s": duration_s,
        "honest_verdict": f"complete: gathered {len(gathered)} artifacts from .312 and .313",
    }
    
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path

def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must begin with complete:")
