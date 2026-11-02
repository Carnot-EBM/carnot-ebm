"""Build the Exp 3223 milestone .299 capstone artifact.

Spec refs: REQ-REPORT-3223, SCENARIO-REPORT-3223.

The .299 milestone deliberately has one research focus: the prompt-injection
KAN v4 result. This module only aggregates that already-written result with
the prior .298 capstone, so a missing model, CUDA, Garak, or teacher-labeling
resource is reported as evidence instead of being silently papered over.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
MILESTONE = "2026.05.299"
SCHEMA_VERSION = "carnot.milestone_capstone.v299_single_focus_aggregation.v1"
EXPERIMENT_ID = "exp3223"
TASK_ID = "exp3223-capstone-v299-single-focus"
ARTIFACT = "experiment_3223_capstone_v299"
RANDOM_SEED = 3223

PRIOR_CAPSTONE_REL_PATH = Path("results/experiment_3232_capstone_v298.json")
V4_RESULT_REL_PATH = Path("results/experiment_3222_prompt_injection_kan_distill_v4_15k.json")
OUTPUT_REL_PATH = Path("results/experiment_3223_capstone_v299.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3223_capstone_v299.py"

DEFAULT_NEXT_TOP_GAP = "cuda_chain_for_full_local_sota_receipts"
GARAK_NEXT_TOP_GAP = "v4_garak_adversarial_expansion"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

REQUIRED_FIELD_TYPES: tuple[tuple[str, type], ...] = (
    ("capstone_v299_ready", bool),
    ("paper_ready", bool),
    ("publication_blocker_count", int),
    ("next_top_gap", str),
    ("v4_outcome", str),
    ("random_seed", int),
    ("reproducibility_checksum", str),
    ("duration_s", float),
    ("honest_verdict", str),
)


def read_json_object(path: Path) -> JsonDict:
    """Read source evidence as JSON and fail closed when the file is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash source artifacts so the capstone can prove exactly what it read."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3223: aggregate the v4 result into the .299 capstone."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    prior_capstone = read_json_object(root_path / PRIOR_CAPSTONE_REL_PATH)
    v4_result = read_json_object(root_path / V4_RESULT_REL_PATH)
    source_artifacts = _source_artifacts(root_path, prior_capstone, v4_result)

    prior_blocker_count = _int_value(prior_capstone.get("publication_blocker_count"))
    prior_capstone_ready = _prior_capstone_ready(prior_capstone)
    v4_outcome = _outcome_from_payload(v4_result)
    gate_summary = _gate_summary(v4_result)
    publication_blocker_delta = _blocker_delta(v4_outcome)
    publication_blocker_count = _nonnegative_count(prior_blocker_count, publication_blocker_delta)
    next_top_gap = _next_top_gap(v4_outcome, gate_summary)
    invariant_violations = _invariant_violations(prior_capstone_ready)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "prior_capstone_artifact": PRIOR_CAPSTONE_REL_PATH.as_posix(),
        "v4_result_artifact": V4_RESULT_REL_PATH.as_posix(),
        "prior_capstone_ready": prior_capstone_ready,
        "prior_paper_ready": prior_capstone.get("paper_ready") is True,
        "prior_publication_blocker_count": prior_blocker_count,
        "prior_next_top_gap": str(prior_capstone.get("next_top_gap") or ""),
        "capstone_v299_ready": not invariant_violations,
        "paper_ready": False,
        "publication_blocker_count": publication_blocker_count,
        "publication_blocker_delta": publication_blocker_delta,
        "next_top_gap": next_top_gap,
        "v4_outcome": v4_outcome,
        "gate_summary": gate_summary,
        "source_v4_summary": _v4_summary(root_path, v4_result, v4_outcome),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "ops_status_modified_by_this_task": False,
        "ops_changelog_modified_by_this_task": False,
        "traceability_modified_by_this_task": False,
        "no_new_model_execution": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "invariant_violations": [],
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["paper_ready"] = _paper_ready_from_evidence(
        artifact["capstone_v299_ready"],
        prior_capstone,
        publication_blocker_count,
        v4_outcome,
    )
    artifact["invariant_violations"] = invariant_violations + _required_fields_are_typed(artifact)
    artifact["capstone_v299_ready"] = not artifact["invariant_violations"]
    artifact["paper_ready"] = _paper_ready_from_evidence(
        artifact["capstone_v299_ready"],
        prior_capstone,
        publication_blocker_count,
        v4_outcome,
    )
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Persist the capstone JSON at the roadmap-mandated deliverable path."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifacts(root: Path, prior_capstone: Mapping[str, Any], v4_result: Mapping[str, Any]) -> list[JsonDict]:
    return [
        _source_record(root, "prior_capstone_v298", PRIOR_CAPSTONE_REL_PATH, prior_capstone),
        _source_record(root, "prompt_injection_kan_v4", V4_RESULT_REL_PATH, v4_result),
    ]


def _source_record(
    root: Path,
    role: str,
    rel_path: Path,
    payload: Mapping[str, Any],
) -> JsonDict:
    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "experiment_id": str(payload.get("experiment_id") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": sha256_file(path),
    }


def _v4_summary(root: Path, payload: Mapping[str, Any], outcome: str) -> JsonDict:
    path = root / V4_RESULT_REL_PATH
    return {
        "path": V4_RESULT_REL_PATH.as_posix(),
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "v4_outcome": outcome,
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "random_seed": _int_value(payload.get("random_seed")),
        "model_specs": _as_mapping(payload.get("model_specs")),
        "auroc_paired_test": payload.get("auroc_paired_test"),
        "delong_pvalue_vs_teacher": payload.get("delong_pvalue_vs_teacher"),
        "cross_dataset_auroc": payload.get("cross_dataset_auroc"),
        "garak_auroc_per_probe": _as_mapping(payload.get("garak_auroc_per_probe")),
    }


def _prior_capstone_ready(prior_capstone: Mapping[str, Any]) -> bool:
    return prior_capstone.get("capstone_ready") is True or prior_capstone.get("capstone_v298_ready") is True


def _outcome_from_payload(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "blocked_missing_exp3222_result"
    return _outcome_from_verdict(str(payload.get("honest_verdict") or ""))


def _outcome_from_verdict(verdict: str) -> str:
    normalized = verdict.lower()
    if "prompt_injection_v4_replacement_grade" in normalized:
        return "replacement_grade"
    if "prompt_injection_v4_publication_grade_garak_partial" in normalized:
        return "publication_grade_garak_partial"
    if "prompt_injection_v4_overfit_to_training" in normalized:
        return "overfit_to_training"
    if "prompt_injection_v4_below_replacement_threshold" in normalized:
        return "below_replacement_threshold"
    blocked = re.search(r"\b(blocked_[a-z0-9_]+)\b", normalized)
    if blocked:
        return blocked.group(1)
    return "blocked_unclassified_v4_outcome"


def _gate_summary(payload: Mapping[str, Any]) -> JsonDict:
    gates = _as_mapping(payload.get("gate_results"))
    if gates:
        return {
            "gate_1_replacement_grade": _bool_gate(
                gates, ("gate_1_replacement_grade", "replacement_grade", "gate1")
            ),
            "gate_2_ood_floor": _bool_gate(gates, ("gate_2_ood_floor", "cross_dataset", "gate2")),
            "gate_3_adversarial_floor": _bool_gate(
                gates, ("gate_3_adversarial_floor", "garak", "gate3")
            ),
        }
    garak = _as_mapping(payload.get("garak_auroc_per_probe"))
    return {
        "gate_1_replacement_grade": _float_value(payload.get("auroc_paired_test")) >= 0.90
        and _float_value(payload.get("delong_pvalue_vs_teacher"), default=1.0) < 0.05,
        "gate_2_ood_floor": _float_value(payload.get("cross_dataset_auroc")) >= 0.85,
        "gate_3_adversarial_floor": _float_value(garak.get("worst_case")) >= 0.75,
    }


def _bool_gate(gates: Mapping[str, Any], names: tuple[str, ...]) -> bool:
    return any(gates.get(name) is True for name in names)


def _blocker_delta(outcome: str) -> int:
    if outcome == "replacement_grade":
        return -3
    if outcome == "publication_grade_garak_partial":
        return -1
    return 0


def _nonnegative_count(prior_count: int, delta: int) -> int:
    return max(0, prior_count + delta)


def _next_top_gap(outcome: str, gates: Mapping[str, bool]) -> str:
    if (
        outcome == "publication_grade_garak_partial"
        and gates.get("gate_1_replacement_grade") is True
        and gates.get("gate_2_ood_floor") is True
        and gates.get("gate_3_adversarial_floor") is False
    ):
        return GARAK_NEXT_TOP_GAP
    return DEFAULT_NEXT_TOP_GAP


def _paper_ready_from_evidence(
    capstone_ready: bool,
    prior_capstone: Mapping[str, Any],
    publication_blocker_count: int,
    v4_outcome: str,
) -> bool:
    return (
        capstone_ready
        and prior_capstone.get("paper_ready") is True
        and publication_blocker_count == 0
        and v4_outcome == "replacement_grade"
    )


def _invariant_violations(prior_capstone_ready: bool) -> list[str]:
    if prior_capstone_ready:
        return []
    return ["prior capstone v298 authority is missing or not ready"]


def _required_fields_are_typed(artifact: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    for field, expected_type in REQUIRED_FIELD_TYPES:
        value = artifact.get(field)
        if expected_type is int:
            if not isinstance(value, int) or isinstance(value, bool):
                violations.append(f"{field} missing_or_wrong_type")
        elif expected_type is float:
            if not isinstance(value, float):
                violations.append(f"{field} missing_or_wrong_type")
        elif not isinstance(value, expected_type):
            violations.append(f"{field} missing_or_wrong_type")
    return violations


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable_payload = {
        "schema_version": artifact.get("schema_version"),
        "experiment_id": artifact.get("experiment_id"),
        "milestone": artifact.get("milestone"),
        "prior_publication_blocker_count": artifact.get("prior_publication_blocker_count"),
        "publication_blocker_delta": artifact.get("publication_blocker_delta"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "v4_outcome": artifact.get("v4_outcome"),
        "next_top_gap": artifact.get("next_top_gap"),
        "source_checksums": artifact.get("source_checksums"),
        "gate_summary": artifact.get("gate_summary"),
    }
    text = json.dumps(stable_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    ready = str(artifact.get("capstone_v299_ready")).lower()
    paper_ready = str(artifact.get("paper_ready")).lower()
    return (
        f"complete: capstone_v299_ready={ready}; "
        f"paper_ready={paper_ready}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"v4_outcome={artifact.get('v4_outcome')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float_value(value: Any, *, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
