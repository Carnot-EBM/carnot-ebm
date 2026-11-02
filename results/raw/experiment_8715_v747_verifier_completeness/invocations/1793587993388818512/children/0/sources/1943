"""Experiment 5702: real-world pass-rate survey of the live pipeline's
`min_heldout_accuracy=1.0` dynamics gate (task 11 -- follow-up to task 8's
REQ-ARC-WMTE-5593-3 finding that "in practice, on real first-shot LLM induction,
the already-strict min_heldout_accuracy=1.0 dynamics gate is frequently the
dominant blocker").

Aggregates every real `heldout_accuracy` value recorded across the checked-in
corpus of `inference_substrate == "live_llm_inference"` result artifacts (real
GPU-backed induction rounds, not synthetic fixtures) to estimate: how often does
a real induction round actually clear the live call site's own threshold of 1.0?

This is NOT a new live episode -- it is an honest aggregation over already-real,
already-adversarially-verified upstream artifacts (each of which independently
paid its own real GPU cost when it was produced). `exp5700`'s own rows are
excluded: that experiment deliberately set `min_heldout_accuracy=0.0` to bypass
the gate for an unrelated isolation test, so its rows are not representative of
real-world pass-rate at the live threshold.

Spec refs: REQ-ARC-WMTE-5593-3 (extends, task-8 dynamics-gate-dominance finding).
"""

from __future__ import annotations

import glob
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5702_dynamics_gate_pass_rate_survey"
RESULT_RELATIVE_PATH = "results/experiment_5702_dynamics_gate_pass_rate_survey.json"
SCHEMA = "carnot.exp5702.dynamics_gate_pass_rate_survey.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 5702
LIVE_THRESHOLD = 1.0
EXCLUDED_EXPERIMENT_SUBSTRINGS = ("5700",)  # deliberately-bypassed-threshold isolation test

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "live_threshold",
    "n_rows",
    "n_source_files",
    "mean",
    "median",
    "pass_rate_at_live_threshold",
    "exact_zero_rate",
    "threshold_sweep",
    "cited_upstream_artifacts",
    "excluded_artifacts",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a low pass rate is a real, citable finding about gate calibration, not a failure of this survey"
    },
    "pass_rate_at_live_threshold": {
        "principle": "the direct answer to task 11's question -- how often a real induction round clears the exact threshold (1.0) the live call site enforces"
    },
    "excluded_artifacts": {
        "principle": "exp5700 deliberately set min_heldout_accuracy=0.0 to isolate an unrelated veto test; including its rows would understate how strict the REAL live threshold is by mixing in rows collected under a different, lower bar"
    },
    "cited_upstream_artifacts": {
        "principle": "CLAUDE.md Inference-Substrate Declaration Discipline -- this is an aggregation_from_upstream_artifacts run; the audit trail must trace every row back to the real artifact that measured it"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    results_dir = root / "results"
    checks["results_dir_present"] = results_dir.is_dir()
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _walk_heldout_accuracy_rows(obj: Any, out: list[float]) -> None:
    if isinstance(obj, dict):
        val = obj.get("heldout_accuracy")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out.append(float(val))
        for v in obj.values():
            _walk_heldout_accuracy_rows(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_heldout_accuracy_rows(v, out)


def collect_rows(root: Path = REPO_ROOT) -> tuple[list[float], list[str], list[str]]:
    """Returns (values, cited_files, excluded_files)."""

    values: list[float] = []
    cited: list[str] = []
    excluded: list[str] = []
    for path_str in sorted(glob.glob(str(root / "results" / "*.json"))):
        rel = str(Path(path_str).relative_to(root))
        try:
            data = json.loads(Path(path_str).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("inference_substrate") != "live_llm_inference":
            continue
        if any(token in rel for token in EXCLUDED_EXPERIMENT_SUBSTRINGS):
            excluded.append(rel)
            continue
        rows: list[float] = []
        _walk_heldout_accuracy_rows(data, rows)
        if rows:
            values.extend(rows)
            cited.append(rel)
    return values, cited, excluded


def build_artifact(*, root: Path = REPO_ROOT) -> JsonDict:
    started = time.monotonic()
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "field_principles": FIELD_PRINCIPLES,
            "live_threshold": LIVE_THRESHOLD,
            "n_rows": 0,
            "n_source_files": 0,
            "mean": None,
            "median": None,
            "pass_rate_at_live_threshold": None,
            "exact_zero_rate": None,
            "threshold_sweep": {},
            "cited_upstream_artifacts": [],
            "excluded_artifacts": [],
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.monotonic() - started, 6),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    values, cited, excluded = collect_rows(root)
    n = len(values)

    if n == 0:
        verdict = "complete: no_real_induction_rows_found"
        mean = median = pass_rate = exact_zero_rate = None
        sweep: JsonDict = {}
    else:
        mean = round(statistics.mean(values), 4)
        median = round(statistics.median(values), 4)
        passed = sum(1 for v in values if v >= LIVE_THRESHOLD)
        pass_rate = round(passed / n, 4)
        exact_zero_rate = round(sum(1 for v in values if v == 0.0) / n, 4)
        sweep = {
            str(t): round(sum(1 for v in values if v >= t) / n, 4)
            for t in (1.0, 0.9, 0.8, 0.75, 0.7, 0.5, 0.3)
        }
        if pass_rate < 0.25:
            verdict = f"complete: dynamics_gate_strict_low_pass_rate_{int(pass_rate * 100)}pct"
        elif pass_rate < 0.5:
            verdict = (
                f"complete: dynamics_gate_moderately_strict_pass_rate_{int(pass_rate * 100)}pct"
            )
        else:
            verdict = f"complete: dynamics_gate_pass_rate_{int(pass_rate * 100)}pct_not_clearly_over_strict"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "live_threshold": LIVE_THRESHOLD,
        "n_rows": n,
        "n_source_files": len(cited),
        "mean": mean,
        "median": median,
        "pass_rate_at_live_threshold": pass_rate,
        "exact_zero_rate": exact_zero_rate,
        "threshold_sweep": sweep,
        "cited_upstream_artifacts": cited,
        "excluded_artifacts": excluded,
        "methodology_note": (
            "Corpus is round-level heldout_accuracy snapshots from real live_llm_inference "
            "artifacts. This measures the PER-ROW pass rate at the live threshold, not the "
            "bounded 3-round retry loop's eventual within-budget success rate -- the checked-in "
            "corpus does not contain enough same-attempt multi-round traces to reconstruct that "
            "distinct statistic. The per-row rate is still the right first-order answer to "
            "'how strict is 1.0 in practice': a low per-row rate means most individual real "
            "induction attempts miss the bar, which is the direct evidence for whether the gate "
            "is calibrated tightly or loosely against real model output quality."
        ),
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.monotonic() - started, 6),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
