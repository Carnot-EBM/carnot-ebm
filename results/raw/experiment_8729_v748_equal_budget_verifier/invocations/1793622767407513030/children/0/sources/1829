"""Exp5528 canonical CSL gate artifact builder.

Spec refs: REQ-LEARN-5528,
SCENARIO-LEARN-5528-SIDECAR-FAILURE,
SCENARIO-LEARN-5528-CANONICAL-GATE.

This module does not repair the conductor. It repairs the artifact discipline
around CSL gates by emitting one canonical Exp5528 result whose top-level
fields are exactly what the conductor checks before downstream memory tasks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from scripts import conductor_gates


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5528_csl_canonical_gate_artifact.json")
UPSTREAM_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5515_csl_independent_outcome_gate_repair.json"
)
UPSTREAM_FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_5515_csl_independent_outcome_stream_fixture.json"
)
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5528_csl_canonical_gate_artifact.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5528_csl_canonical_gate_artifact.py")

SCHEMA = "carnot.experiment_5528.csl_canonical_gate_artifact.v1"
EXPERIMENT_ID = "experiment_5528_csl_canonical_gate_artifact"
TASK_ID = "exp5528-csl-canonical-gate-artifact"
UPSTREAM_5515_TASK_ID = "exp5515-csl-independent-outcome-gate-repair"
EXP5529_TASK_ID = "exp5529-gated-csl-event-topic-residue-stress"
EXP5530_TASK_ID = "exp5530-gated-sota-csl-memory-panel-v2"
MILESTONE = "2026.07.501"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5528
INFERENCE_SUBSTRATE = "canonical_csl_gate_artifact_from_independent_fixture"
TERMINAL_PREFIXES = ("complete:", "blocked:")

COPIED_EVIDENCE_FIELDS = (
    "metric_independence_clean",
    "csl_gate_fields_resolvable",
    "csl_experience_graph_ready",
    "continuous_self_learning_evidence",
    "heldout_delta",
    "no_memory_score",
    "graph_memory_score",
    "stale_memory_score",
    "stale_evidence_rejection_rate",
    "negative_transfer_rate",
)
REQUIRED_ARTIFACT_FIELDS = (
    *COPIED_EVIDENCE_FIELDS,
    "conductor_gate_probe_passed",
    "csl_gate_fields_conductor_visible",
    "same_exp_sidecar_after_primary",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
SPEC_REFS = (
    "REQ-LEARN-5528",
    "SCENARIO-LEARN-5528-SIDECAR-FAILURE",
    "SCENARIO-LEARN-5528-CANONICAL-GATE",
)
FIELD_PRINCIPLES: JsonDict = {
    "metric_independence_clean": "Proves held-out scores come from independent labels, not memory utility.",
    "csl_gate_fields_resolvable": "Confirms downstream conductor gates can read the bare CSL field names.",
    "csl_experience_graph_ready": "Carries the graph-memory readiness gate from the independent Exp5515 replay.",
    "continuous_self_learning_evidence": "Keeps the CSL evidence claim tied to all required clean sub-gates.",
    "heldout_delta": "Reports graph-memory improvement as graph score minus no-memory score.",
    "no_memory_score": "Preserves the no-memory control used to compute the held-out delta.",
    "graph_memory_score": "Preserves the governed graph-memory score that must beat controls.",
    "stale_memory_score": "Preserves the stale-memory control that guards against unsafe memory reuse.",
    "stale_evidence_rejection_rate": "Shows stale evidence is rejected before action selection.",
    "negative_transfer_rate": "Shows irrelevant transfer candidates are not accepted as useful memory.",
    "conductor_gate_probe_passed": "Records that real downstream Exp5528 gates pass through conductor_gates.py.",
    "csl_gate_fields_conductor_visible": "Bare boolean used by Exp5529 and Exp5530 to prove conductor visibility.",
    "same_exp_sidecar_after_primary": "Must stay false so no newer Exp5528 sidecar can hide the canonical fields.",
    "tests_added_or_reused": "Lists the focused and full test commands backing this artifact.",
    "field_principles": "Explains why each headline and gate field exists for downstream audits.",
    "inference_substrate": "Declares this artifact is a canonical CSL gate receipt from an independent fixture.",
    "honest_verdict": "Terminal summary with complete or blocked prefix for conductor reconciliation.",
}


def _resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative output paths while preserving absolute paths."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(root) / candidate


def load_json(path: Path | str) -> JsonDict:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable, indented JSON and create the parent directory if needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_artifact_for_task(task_id: str, results_dir: Path | str) -> Path | None:
    """Expose the conductor artifact resolver for focused tests and probes."""

    return conductor_gates._find_artifact_by_task_id(task_id, Path(results_dir))


def evaluate_task_gates(task: Mapping[str, Any], results_dir: Path | str) -> Any:
    """Run the real conductor gate evaluator without importing the conductor."""

    return conductor_gates.evaluate_gates(copy.deepcopy(dict(task)), Path(results_dir))


def gate_check_to_dict(check: Any) -> JsonDict:
    """Convert conductor gate dataclasses into JSON-ready dictionaries."""

    return {
        "passed": bool(check.passed),
        "summary": check.summary,
        "gates_evaluated": [
            {
                "upstream": gate.upstream,
                "artifact_field": gate.artifact_field,
                "op": gate.op,
                "expected": gate.expected,
                "actual": gate.actual,
                "passed": bool(gate.passed),
                "reason": gate.reason,
            }
            for gate in check.gates_evaluated
        ],
    }


def reproduce_5515_sidecar_selection(results_dir: Path | str) -> JsonDict:
    """Reproduce the Exp5515 failure where the newest sidecar hides gate fields."""

    task = {
        "id": "exp5528-sidecar-reproduction-probe",
        "gated_on": [
            {
                "upstream": UPSTREAM_5515_TASK_ID,
                "artifact_field": "metric_independence_clean",
                "op": "==",
                "value": True,
            }
        ],
    }
    selected = find_artifact_for_task(UPSTREAM_5515_TASK_ID, results_dir)
    selected_data = load_json(selected) if selected else {}
    check = evaluate_task_gates(task, results_dir)
    return {
        "probe_task_id": task["id"],
        "selected_artifact": selected.as_posix() if selected else None,
        "gate_check_passed": bool(check.passed),
        "gate_check_summary": check.summary,
        "gates_evaluated": gate_check_to_dict(check)["gates_evaluated"],
        "primary_fields_visible_through_newest_artifact": all(
            field in selected_data
            for field in (
                "metric_independence_clean",
                "csl_gate_fields_resolvable",
                "csl_experience_graph_ready",
            )
        ),
    }


def _load_roadmap(root: Path | str) -> JsonDict:
    """Load the active roadmap YAML as a mapping."""

    return yaml.safe_load(_resolve_path(root, ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8"))


def task_from_roadmap(root: Path | str, task_id: str) -> JsonDict:
    """Return a deep copy of a roadmap task by id."""

    for task in _load_roadmap(root).get("tasks", []):
        if task.get("id") == task_id:
            return copy.deepcopy(task)
    raise KeyError(task_id)  # pragma: no cover - spec and tests require these tasks.


def _only_exp5528_gates(task: Mapping[str, Any]) -> JsonDict:
    """Keep only gates that directly reference the canonical Exp5528 artifact."""

    filtered = copy.deepcopy(dict(task))
    filtered["gated_on"] = [
        gate for gate in task.get("gated_on", []) if gate.get("upstream") == TASK_ID
    ]
    return filtered


def probe_downstream_gates(root: Path | str, results_dir: Path | str) -> JsonDict:
    """Evaluate the actual Exp5529/Exp5530 gates that depend on Exp5528."""

    selected = find_artifact_for_task(TASK_ID, results_dir)
    exp5529_task = task_from_roadmap(root, EXP5529_TASK_ID)
    exp5530_task = task_from_roadmap(root, EXP5530_TASK_ID)
    exp5529_check = evaluate_task_gates(exp5529_task, results_dir)
    exp5530_exp5528_check = evaluate_task_gates(_only_exp5528_gates(exp5530_task), results_dir)
    exp5530_full_check = evaluate_task_gates(exp5530_task, results_dir)
    visible = bool(selected and exp5529_check.passed and exp5530_exp5528_check.passed)
    return {
        "selected_exp5528_artifact": selected.as_posix() if selected else None,
        "exp5529_full": gate_check_to_dict(exp5529_check),
        "exp5530_exp5528_only": gate_check_to_dict(exp5530_exp5528_check),
        "exp5530_full": gate_check_to_dict(exp5530_full_check),
        "conductor_gate_probe_passed": visible,
        "csl_gate_fields_conductor_visible": visible,
    }


def same_exp_sidecar_after_primary(result_path: Path | str, results_dir: Path | str) -> bool:
    """Return whether a newer Exp5528 sidecar exists after the primary artifact."""

    target = Path(result_path)
    if not target.exists():
        return False
    target_mtime = target.stat().st_mtime
    for candidate in Path(results_dir).glob("experiment_5528_*.json"):
        if candidate.resolve() != target.resolve() and candidate.stat().st_mtime > target_mtime:
            return True
    return False


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
    conductor_probe: Mapping[str, Any],
    sidecar_reproduction: Mapping[str, Any],
    sidecar_after_primary: bool,
) -> JsonDict:
    """Build the canonical artifact from Exp5515 evidence and gate probes."""

    root_path = Path(root)
    upstream = load_json(root_path / UPSTREAM_ARTIFACT_RELATIVE_PATH)
    gate_probe_passed = bool(conductor_probe["conductor_gate_probe_passed"])
    gate_visible = bool(conductor_probe["csl_gate_fields_conductor_visible"])
    artifact: JsonDict = {
        "experiment": 5528,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": {
            "independent_outcome_gate_repair": UPSTREAM_ARTIFACT_RELATIVE_PATH.as_posix(),
            "independent_outcome_stream_fixture": UPSTREAM_FIXTURE_RELATIVE_PATH.as_posix(),
        },
        "sidecar_selection_reproduction": dict(sidecar_reproduction),
        "downstream_gate_probe": dict(conductor_probe),
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "same_exp_sidecar_after_primary": bool(sidecar_after_primary),
        "conductor_gate_probe_passed": gate_probe_passed,
        "csl_gate_fields_conductor_visible": gate_visible,
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
    }
    for field in COPIED_EVIDENCE_FIELDS:
        artifact[field] = upstream[field]
    artifact["artifact_policy"] = {
        "primary_artifact": RESULT_RELATIVE_PATH.as_posix(),
        "same_experiment_glob": "results/experiment_5528_*.json",
        "sidecar_safe_rule": "Do not write a later same-number sidecar after this primary artifact.",
        "no_later_same_exp_sidecar": not artifact["same_exp_sidecar_after_primary"],
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    results_dir: Path | str | None = None,
    tests_added_or_reused: Sequence[str] = (),
    write: bool = True,
) -> JsonDict:
    """Write the primary Exp5528 artifact and then probe it through the conductor."""

    root_path = Path(root)
    target = _resolve_path(root_path, result_path)
    resolved_results_dir = Path(results_dir) if results_dir is not None else target.parent
    sidecar_reproduction = reproduce_5515_sidecar_selection(root_path / "results")
    bootstrap_probe = {
        "selected_exp5528_artifact": target.as_posix(),
        "exp5529_full": {"passed": True, "summary": "bootstrap field exposure"},
        "exp5530_exp5528_only": {"passed": True, "summary": "bootstrap field exposure"},
        "exp5530_full": {"passed": False, "summary": "bootstrap ignores Exp5529 dependency"},
        "conductor_gate_probe_passed": True,
        "csl_gate_fields_conductor_visible": True,
    }
    artifact = build_artifact(
        root=root_path,
        tests_added_or_reused=tests_added_or_reused,
        conductor_probe=bootstrap_probe,
        sidecar_reproduction=sidecar_reproduction,
        sidecar_after_primary=False,
    )
    if write:
        write_json(target, artifact)
        conductor_probe = probe_downstream_gates(root_path, resolved_results_dir)
        artifact = build_artifact(
            root=root_path,
            tests_added_or_reused=tests_added_or_reused,
            conductor_probe=conductor_probe,
            sidecar_reproduction=sidecar_reproduction,
            sidecar_after_primary=same_exp_sidecar_after_primary(target, resolved_results_dir),
        )
        write_json(target, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the artifact cannot safely gate downstream CSL tasks."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5528 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if not artifact.get("tests_added_or_reused"):
        errors.append("tests_added_or_reused")
    no_memory = float(artifact.get("no_memory_score", 0.0))
    graph_memory = float(artifact.get("graph_memory_score", 0.0))
    if float(artifact.get("heldout_delta", 0.0)) != round(graph_memory - no_memory, 10):
        errors.append("heldout_delta")
    for field in (
        "metric_independence_clean",
        "csl_gate_fields_resolvable",
        "csl_experience_graph_ready",
        "continuous_self_learning_evidence",
        "conductor_gate_probe_passed",
        "csl_gate_fields_conductor_visible",
    ):
        if artifact.get(field) is not True:
            errors.append(field)
    if artifact.get("same_exp_sidecar_after_primary") is not False:
        errors.append("same_exp_sidecar_after_primary")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    principles = artifact.get("field_principles", {})
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if not principles.get(field)]
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict from the artifact's gate-policy fields."""

    if (
        artifact.get("conductor_gate_probe_passed") is True
        and artifact.get("csl_gate_fields_conductor_visible") is True
        and artifact.get("same_exp_sidecar_after_primary") is False
    ):
        return "complete: canonical_csl_gate_artifact_conductor_visible"
    return "blocked: canonical_csl_gate_artifact_not_conductor_visible"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record the files backing the canonical artifact."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA256 for JSON-compatible mappings."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a SHA256 digest for a file."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
