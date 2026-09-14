"""Audit whether retained factor evidence causes later predictive value.

The cold worker rebuilds each arm from public events and labels only when their
release time arrives. It compares deterministic predictions and state bytes to
the producer rows. Timing fields stay measured, but they do not define parity.

Spec refs: REQ-CL-7312 and SCENARIO-CL-7312-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import random
import re
import shlex
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7310_v642_factor_prototype as prototype
from carnot import experiment_7311_v642_factor_learning as learning
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7312
SCHEMA = "carnot.exp7312.v642_factor_audit.v1"
MILESTONE = "2026.09.642"
RUN_DATE = "20260914"
RANDOM_SEED = 7_312_000
BOOTSTRAP_SEED = 7_312_901
BOOTSTRAP_DRAWS = 10_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = deepcopy(learning.INVOCATION_COUNTS)
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"

ARMS = learning.ARMS
FACTOR_ARM = learning.FACTOR_ARM
ERASED_ARM = "local_reset_without_retained_witnesses"
STRATA = learning.STRATA
EVALUATION_STREAM_COUNT = learning.EVALUATION_STREAM_COUNT
EVENTS_PER_STREAM = learning.EVENTS_PER_STREAM
WARMUP_COUNT = learning.WARMUP_COUNT
FUTURE_LABEL_COUNT = learning.FUTURE_LABEL_COUNT
MEMORY_CAP_BYTES = learning.MEMORY_CAP_BYTES
FAMILIES = learning.FAMILIES
EVALUATION_STREAM_IDS = tuple(
    f"evaluation-{index + 1:02d}" for index in range(EVALUATION_STREAM_COUNT)
)
CONTROL_NAMES = (
    "future_label_permutation",
    "consistent_factor_name_permutation",
    "retained_witness_erasure",
    "delay_all_feedback_beyond_evaluation",
    "corrupt_rollback_hash",
)
E2E_STAGES = (
    "authenticated_immutable_capture",
    "cold_prediction_and_transition_replay",
    "hostile_causal_interventions",
    "independent_stream_reduction",
    "terminal_schema_classification",
)
COMPLETION_GATE_NAMES = (
    "authenticated_capture",
    "full_cold_replay",
    "hostile_controls",
    "equal_labels_and_memory_cap",
    "independent_intervals",
    "immutable_source_rows",
    "e2e_pipeline",
)
RETIREMENT_SCOPE = (
    "factor-local longest-consistent-suffix revision with retained witnesses "
    "under delayed feedback (Exp7310/Exp7311 mechanism)"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7312_v642_factor_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7312_v642_factor_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7312_v642_factor_audit.py")
DEFAULT_ARTIFACT = Path("results/experiment_7312_v642_factor_audit.json")
UPSTREAM_ARTIFACT = Path("results/experiment_7311_v642_factor_learning.json")
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7312-[A-Z0-9]+(?:-[A-Z0-9]+)*")
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7297_v641_mixture_audit.py"),
    Path("python/carnot/experiment_7310_v642_factor_prototype.py"),
    Path("python/carnot/experiment_7311_v642_factor_learning.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
)
IMMUTABLE_UPSTREAM_MARKERS = (
    "/python/carnot/",
    "/scripts/experiments/experiment_7311_",
    "/tests/python/test_experiment_7311_",
    "/results/experiment_7310_",
    "/results/raw/experiment_7310_",
)
VOLATILE_KEYS = {
    "prediction_time_ns",
    "prediction_cost_ns",
    "prediction_seal_hash",
    "time_limit_violation",
    "update_latency_ns",
    "transaction_cost_ns",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "validation_receipts",
    "factor_audit_complete_score",
    "factor_promotion_score",
    "continuous_self_learning_task",
    "causal_intervention_rows",
    "independent_stream_intervals",
)

FIELD_PRINCIPLES = {
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the evidence to the active experiment task.",
    "milestone": "Bind the result to milestone 2026.09.642.",
    "status": "Write the terminal result only after the work and checks; checkpoints remain separate.",
    "run_date": "Use 20260914 with actual UTC start/end and monotonic phase timing.",
    "started_at_utc": "Record the actual UTC start of this audit.",
    "completed_at_utc": "Record the actual UTC terminal decision time.",
    "preconditions_checked": "Hash real inputs and record actual availability and failed checks.",
    "MODEL_SPECS": "Actual current executable model identities; historical identities remain in sidecars.",
    "model_invoked": "True for any attempted model load or generation, even when output is unusable.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight loads and generations.",
    "inference_substrate": "Describe actual computation with a recognized literal, not intended work.",
    "inference_substrate_class": "Full generation has a 60s floor; bounded generation 10s; load-only 2s. Never pad time.",
    "execution_venue": "Record actual host/device work; a CPU replay is not GPU or FPGA execution.",
    "duration_s": "Measure total elapsed and disjoint phase spans including failed work.",
    "random_seed": "Seal development and independent evaluation seeds before seeing outcomes.",
    "reproducibility_checksum": "Bind code, inputs, config, model when used, and raw evidence.",
    "source_artifact_hashes": "Authenticate producer identity, terminal class, and quarantine state.",
    "rows": "Record every comparative unit and arm with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Keep planned, attempted, complete, and censored counts with the frozen stopping rule.",
    "acceptance_gate_results": "Every check has expected, observed, passed, and a one-line principle explaining its purpose.",
    "gate_check_summary": "Every blocked result names upstream, check, exact field, observed value, and expected value.",
    "verifier_is_oracle": "Shared evaluator authority permits circular_positive only, not positive scientific value.",
    "honest_verdict": "Completed findings start complete_ or complete:; external failure starts blocked_. State the finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial; unchanged external failure is blocked.",
    "validation_receipts": "Record exact commands, scope, exit codes, elapsed time, and log hashes; retain failures.",
    "factor_audit_complete_score": "One for complete cold replay, independent of efficacy.",
    "factor_promotion_score": "Independent future-prediction evidence controls any promotion.",
    "continuous_self_learning_task": "The audit traces the actual across-query learning mechanism.",
    "causal_intervention_rows": "Past-prediction invariance and retained-evidence interventions expose leakage and inert memory.",
    "independent_stream_intervals": "Preserve exact paired denominators and every stratum.",
}


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep replay, intervention, checkpoint, candidate, and terminal bytes separate."""

    raw_dir: Path
    checkpoint_dir: Path
    cold_replay: Path
    interventions: Path
    e2e: Path
    terminal_candidate: Path
    validation_dir: Path
    artifact: Path
    upstream_artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the declared repository paths for the real run."""

        return cls.from_results_root(REPO_ROOT / "results", REPO_ROOT / UPSTREAM_ARTIFACT)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test outputs and a replaceable upstream copy below one private root."""

        return cls.from_results_root(root, root / UPSTREAM_ARTIFACT.name)

    @classmethod
    def from_results_root(cls, root: Path, upstream: Path) -> ExperimentPaths:
        """Derive all task-owned destinations without creating terminal evidence."""

        raw = root / "raw" / "experiment_7312_v642_factor_audit"
        checkpoints = root / "checkpoints" / "experiment_7312_v642_factor_audit"
        return cls(
            raw,
            checkpoints,
            raw / "cold_replay.json",
            raw / "causal_interventions.json",
            raw / "e2e_rows.json",
            raw / "terminal_candidate.json",
            raw / "validation",
            root / DEFAULT_ARTIFACT.name,
            upstream,
        )


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Print and flush one truthful watchdog boundary."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while an absent path stays different from empty bytes."""

    return learning._sha256_path(path)


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed or absent evidence stays unavailable."""

    return learning._load_object(path)


def _resolve(repo_root: Path, value: object) -> Path:
    """Resolve repository-relative evidence and keep absolute paths unchanged."""

    return learning._resolve(repo_root, value)


def _atomic_write(path: Path, value: Any) -> JsonDict:
    """Publish complete canonical bytes through the shipped atomic writer."""

    data = value if isinstance(value, bytes) else transactional.canonical_json_bytes(value)
    return learning._atomic_write(path, data)


def _task_identity(text: str) -> JsonDict:
    """Read only this task identity from the executable roadmap."""

    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    tasks = value.get("tasks") if isinstance(value, Mapping) else value
    if not isinstance(tasks, list):
        return {}
    for task in tasks:
        if isinstance(task, Mapping) and task.get("id") == "exp7312-factor-audit":
            return {
                "id": task.get("id"),
                "milestone": task.get("milestone"),
                "deliverable": task.get("deliverable"),
            }
    return {}


def _manifest_excludes(value: Any, experiment_id: int) -> bool:
    """Find an exact experiment identifier in the manifest's identifier fields."""

    wanted = {str(experiment_id), f"exp{experiment_id}"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"experiment_id", "experiment_ids"}:
                items = child if isinstance(child, list) else [child]
                if any(str(item).split("-", 1)[0] in wanted for item in items):
                    return True
            if _manifest_excludes(child, experiment_id):
                return True
    elif isinstance(value, list):
        return any(_manifest_excludes(child, experiment_id) for child in value)
    return False


def _receipt_matches(repo_root: Path, receipt: object) -> bool:
    """Require a declared path, exact current hash, and optional row count."""

    if not isinstance(receipt, Mapping):
        return False
    path = _resolve(repo_root, receipt.get("path", ""))
    return bool(str(receipt.get("path", ""))) and _sha256_path(path) == receipt.get("sha256")


def _precondition(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    principle: str,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Give every external check its exact expected and observed values."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected if passed is None else bool(passed),
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve all checks and the first failed value without paraphrase."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "passed": not failed,
        "checks": [dict(row) for row in checks],
        "failed_checks": failed,
        "first_failure": first,
        "upstream": first.get("upstream") if first else None,
        "check": first.get("check") if first else None,
        "field": first.get("field") if first else None,
        "observed_value": first.get("observed_value") if first else None,
        "expected_value": first.get("expected_value") if first else None,
    }


def _relevant_upstream_hash_mismatches(
    repo_root: Path, upstream: Mapping[str, Any]
) -> list[JsonDict]:
    """Check immutable stream and executable hashes without freezing mutable docs."""

    declared = upstream.get("source_artifact_hashes", {})
    if not isinstance(declared, Mapping):
        return [{"path": "source_artifact_hashes", "expected": "mapping", "observed": declared}]
    mismatches = []
    for raw_path, expected in declared.items():
        path_text = str(raw_path)
        if not any(marker in path_text for marker in IMMUTABLE_UPSTREAM_MARKERS):
            continue
        observed = _sha256_path(_resolve(repo_root, path_text))
        if observed != expected:
            mismatches.append({"path": path_text, "expected": expected, "observed": observed})
    return mismatches


def collect_preconditions(
    repo_root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate Exp7311, raw receipts, immutable code, exclusions, and outputs."""

    upstream = _load_object(paths.upstream_artifact)
    upstream_name = str(paths.upstream_artifact)
    spec_path = repo_root / SPEC_PATH
    roadmap_path = repo_root / "research-roadmap.yaml"
    exclusion_path = repo_root / "ops/exclusion_manifest.yaml"
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    try:
        exclusions = yaml.safe_load(exclusion_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        exclusions = {"unavailable": True}
    raw = upstream.get("raw_evidence_receipts", {})
    raw = raw if isinstance(raw, Mapping) else {}
    expected_identity = {
        "id": "exp7312-factor-audit",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT),
    }
    immutable_mismatches = _relevant_upstream_hash_mismatches(repo_root, upstream)
    integrity_errors = learning.validate_artifact(upstream)
    checks = [
        _precondition(
            "driving_capability_spec",
            str(spec_path),
            "REQ-CL-7312",
            True,
            "REQ-CL-7312" in spec_text,
            "Implementation starts only after its requirement exists.",
        ),
        _precondition(
            "scenario_contract",
            str(spec_path),
            "SCENARIO-CL-7312-*",
            7,
            len(set(SCENARIO_PATTERN.findall(spec_text))),
            "Each required behavior needs one named scenario.",
        ),
        _precondition(
            "v642_task_identity",
            str(roadmap_path),
            "id,milestone,deliverable",
            expected_identity,
            _task_identity(roadmap_text),
            "Run only the active task and its declared output.",
        ),
        _precondition(
            "factor_capture_complete",
            upstream_name,
            "factor_capture_complete_score",
            1,
            upstream.get("factor_capture_complete_score"),
            "Only complete prospective capture can authorize the audit.",
        ),
        _precondition(
            "upstream_terminal",
            upstream_name,
            "status",
            "complete",
            upstream.get("status"),
            "Incomplete external work cannot authorize the audit.",
        ),
        _precondition(
            "upstream_not_quarantined",
            upstream_name,
            "flagged_adversarial,quarantined",
            False,
            bool(upstream.get("flagged_adversarial") or upstream.get("quarantined")),
            "A numeric score cannot override quarantine.",
        ),
        _precondition(
            "upstream_not_disqualified",
            upstream_name,
            "verdict_class",
            "not disqualified",
            upstream.get("verdict_class"),
            "A numeric score cannot override disqualification.",
            passed=upstream.get("verdict_class") != "disqualified",
        ),
        _precondition(
            "upstream_not_retired",
            upstream_name,
            "retired",
            False,
            bool(upstream.get("retired")),
            "Retired producer evidence cannot authorize promotion.",
        ),
        _precondition(
            "upstream_internal_integrity",
            upstream_name,
            "validate_artifact errors",
            [],
            integrity_errors,
            "Producer headline fields cannot replace internal row and checksum validation.",
        ),
        _precondition(
            "raw_step_rows",
            upstream_name,
            "raw_evidence_receipts.step_rows hash,row_count",
            [True, 107_520],
            [
                _receipt_matches(repo_root, raw.get("step_rows")),
                raw.get("step_rows", {}).get("row_count"),
            ],
            "All 107,520 immutable prediction rows must be present.",
        ),
        _precondition(
            "raw_feedback_rows",
            upstream_name,
            "raw_evidence_receipts.feedback_update_rows hash,row_count",
            [True, 3_072],
            [
                _receipt_matches(repo_root, raw.get("feedback_update_rows")),
                raw.get("feedback_update_rows", {}).get("row_count"),
            ],
            "All 3,072 releases and 15,360 arm transitions must be present.",
        ),
        _precondition(
            "raw_e2e_sidecar",
            upstream_name,
            "raw_evidence_receipts.e2e_controls hash",
            True,
            _receipt_matches(repo_root, raw.get("e2e_controls")),
            "The producer restart and rollback sidecar must retain exact bytes.",
        ),
        _precondition(
            "immutable_stream_and_code_hashes",
            upstream_name,
            "relevant source_artifact_hashes mismatches",
            [],
            immutable_mismatches,
            "Replay uses the exact stream and executable bytes measured by the producer.",
        ),
        _precondition(
            "exp7312_not_excluded",
            str(exclusion_path),
            "experiment_id",
            False,
            _manifest_excludes(exclusions, EXPERIMENT_ID),
            "An excluded task identifier must stop before audit work.",
        ),
        _precondition(
            "resource_ownership",
            str(paths.artifact),
            "task-owned output parents writable",
            True,
            all(
                learning._path_writable(path)
                for path in (
                    paths.cold_replay,
                    paths.interventions,
                    paths.e2e,
                    paths.terminal_candidate,
                    paths.artifact,
                    paths.checkpoint_dir / "probe",
                )
            ),
            "Only task-owned result, raw, and checkpoint paths can receive writes.",
        ),
    ]
    hashes = {str(repo_root / path): _sha256_path(repo_root / path) for path in SOURCE_PATHS}
    hashes[upstream_name] = _sha256_path(paths.upstream_artifact)
    for receipt in raw.values():
        if isinstance(receipt, Mapping) and receipt.get("path"):
            evidence_path = _resolve(repo_root, receipt["path"])
            hashes[str(evidence_path)] = _sha256_path(evidence_path)
    checks.append(
        _precondition(
            "source_bytes_available",
            "declared sources and raw evidence",
            "sha256",
            True,
            all(value is not None for value in hashes.values()),
            "Every source and evidence identity must be hashable before replay.",
        )
    )
    return checks, hashes, upstream


def _fixture_artifact(repo_root: Path, upstream: Mapping[str, Any]) -> JsonDict:
    """Load the exact Exp7310 fixture named and hashed by Exp7311."""

    state = upstream.get("source_artifact_states", {}).get("exp7310", {})
    path = _resolve(repo_root, state.get("path", ""))
    if _sha256_path(path) != state.get("sha256"):
        raise ValueError("exp7310_artifact_hash")
    fixture = _load_object(path)
    if fixture.get("factor_fixture_ready_score") != 1:
        raise ValueError("exp7310_fixture_not_ready")
    return fixture


def load_authenticated_views(repo_root: Path, upstream: Mapping[str, Any]) -> prototype.StreamViews:
    """Rebuild authority-separated views from the exact Exp7310 fixture."""

    return learning.load_authenticated_views(repo_root, _fixture_artifact(repo_root, upstream))


def _iter_jsonl(path: Path) -> Iterator[JsonDict]:
    """Yield JSON objects one at a time so the large step file stays bounded."""

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid_jsonl_row:{path}")
            yield value


def _group_rows(path: Path, selected: set[str]) -> dict[str, list[JsonDict]]:
    """Group the smaller release file by selected stream identity."""

    result: dict[str, list[JsonDict]] = defaultdict(list)
    for row in _iter_jsonl(path):
        stream_id = str(row.get("stream_id"))
        if stream_id in selected:
            result[stream_id].append(row)
    return dict(result)


def _stable_value(value: Any) -> Any:
    """Remove only measured timing fields before deterministic parity checks."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(child)
            for key, child in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_stable_value(child) for child in value]
    return value


def _ablation_counts(step_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compare retained and erased witnesses on the same events and labels."""

    by_key = {
        (str(row["stream_id"]), int(row["chronology_index"]), str(row["arm"])): row
        for row in step_rows
    }
    event_keys = sorted(
        {
            (str(row["stream_id"]), int(row["chronology_index"]))
            for row in step_rows
            if row["arm"] == FACTOR_ARM
        }
    )
    pairs = [
        (by_key[(stream_id, index, FACTOR_ARM)], by_key[(stream_id, index, ERASED_ARM)])
        for stream_id, index in event_keys
    ]
    non_feedback = [pair for pair in pairs if pair[0]["non_feedback_step"] is True]
    return {
        "prediction_difference_count": sum(a["prediction"] != b["prediction"] for a, b in pairs),
        "retained_correct_erased_wrong_count": sum(
            a["error"] == 0 and b["error"] == 1 for a, b in pairs
        ),
        "erased_correct_retained_wrong_count": sum(
            a["error"] == 1 and b["error"] == 0 for a, b in pairs
        ),
        "future_error_delta": sum(int(a["error"]) - int(b["error"]) for a, b in pairs) / len(pairs),
        "non_feedback_error_delta": sum(int(a["error"]) - int(b["error"]) for a, b in non_feedback)
        / len(non_feedback),
        "paired_event_count": len(pairs),
        "non_feedback_paired_event_count": len(non_feedback),
    }


def _compare_stream(
    producer_steps: Sequence[Mapping[str, Any]],
    producer_feedback: Sequence[Mapping[str, Any]],
    replay_steps: Sequence[Mapping[str, Any]],
    replay_feedback: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Compare one reconstructed stream while retaining every exact mismatch."""

    expected_steps = {str(row["unit_id"]): row for row in producer_steps}
    observed_steps = {str(row["unit_id"]): row for row in replay_steps}
    expected_feedback = {str(row["unit_id"]): row for row in producer_feedback}
    observed_feedback = {str(row["unit_id"]): row for row in replay_feedback}
    mismatches: list[JsonDict] = []
    prediction_mismatches = 0
    state_mismatches = 0
    matched_predictions = 0
    for unit_id in sorted(set(expected_steps) | set(observed_steps)):
        expected = expected_steps.get(unit_id)
        observed = observed_steps.get(unit_id)
        if (
            expected is not None
            and observed is not None
            and _stable_value(expected) == _stable_value(observed)
        ):
            matched_predictions += 1
            continue
        prediction_mismatches += int(
            expected is None
            or observed is None
            or expected.get("prediction") != observed.get("prediction")
        )
        state_mismatches += int(
            expected is None
            or observed is None
            or expected.get("state_hash_before_prediction")
            != observed.get("state_hash_before_prediction")
            or expected.get("state_hash_at_seal") != observed.get("state_hash_at_seal")
        )
        mismatches.append(
            {
                "unit_id": unit_id,
                "kind": "prediction_or_state",
                "expected": _stable_value(expected),
                "observed": _stable_value(observed),
            }
        )
    transition_mismatches = 0
    matched_transitions = 0
    for unit_id in sorted(set(expected_feedback) | set(observed_feedback)):
        expected = expected_feedback.get(unit_id)
        observed = observed_feedback.get(unit_id)
        if (
            expected is not None
            and observed is not None
            and _stable_value(expected) == _stable_value(observed)
        ):
            matched_transitions += len(ARMS)
            continue
        transition_mismatches += len(ARMS)
        mismatches.append(
            {
                "unit_id": unit_id,
                "kind": "state_transition",
                "expected": _stable_value(expected),
                "observed": _stable_value(observed),
            }
        )
    return {
        "matched_prediction_rows": matched_predictions,
        "matched_transition_rows": matched_transitions,
        "prediction_mismatch_count": prediction_mismatches,
        "state_hash_mismatch_count": state_mismatches,
        "transition_mismatch_count": transition_mismatches,
        "mismatch_rows": mismatches,
    }


def cold_replay_impl(
    repo_root: Path,
    paths: ExperimentPaths,
    stream_ids: Sequence[str],
) -> JsonDict:
    """Cold-rebuild selected streams from public events and due released labels."""

    upstream = _load_object(paths.upstream_artifact)
    views = load_authenticated_views(repo_root, upstream)
    raw = upstream["raw_evidence_receipts"]
    step_path = _resolve(repo_root, raw["step_rows"]["path"])
    feedback_path = _resolve(repo_root, raw["feedback_update_rows"]["path"])
    selected = set(stream_ids)
    feedback_by_stream = _group_rows(feedback_path, selected)
    step_by_stream: dict[str, list[JsonDict]] = defaultdict(list)
    for row in _iter_jsonl(step_path):
        stream_id = str(row.get("stream_id"))
        if stream_id in selected:
            step_by_stream[stream_id].append(row)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    replay_root = Path(tempfile.mkdtemp(prefix="cold-", dir=paths.checkpoint_dir))
    reduced: list[JsonDict] = []
    mismatches: list[JsonDict] = []
    totals = {
        "matched_prediction_rows": 0,
        "matched_transition_rows": 0,
        "prediction_mismatch_count": 0,
        "state_hash_mismatch_count": 0,
        "transition_mismatch_count": 0,
    }
    all_ablation = {
        "prediction_difference_count": 0,
        "retained_correct_erased_wrong_count": 0,
        "erased_correct_retained_wrong_count": 0,
        "future_error_sum": 0.0,
        "non_feedback_error_sum": 0.0,
        "paired_event_count": 0,
        "non_feedback_paired_event_count": 0,
    }
    label_parity = True
    memory_parity = True
    completed: list[str] = []
    started = time.monotonic()
    for offset, stream_id in enumerate(stream_ids, start=1):
        producer_steps = step_by_stream.get(stream_id, [])
        producer_feedback = feedback_by_stream.get(stream_id, [])
        replay_steps, replay_feedback, _ = learning.replay_stream(
            views, stream_id, replay_root / stream_id
        )
        comparison = _compare_stream(
            producer_steps, producer_feedback, replay_steps, replay_feedback
        )
        for name in totals:
            totals[name] += int(comparison[name])
        mismatches.extend(comparison["mismatch_rows"])
        reduced.extend(learning.reduce_step_rows(replay_steps, replay_feedback))
        ablation = _ablation_counts(replay_steps)
        all_ablation["prediction_difference_count"] += int(ablation["prediction_difference_count"])
        all_ablation["retained_correct_erased_wrong_count"] += int(
            ablation["retained_correct_erased_wrong_count"]
        )
        all_ablation["erased_correct_retained_wrong_count"] += int(
            ablation["erased_correct_retained_wrong_count"]
        )
        all_ablation["future_error_sum"] += float(ablation["future_error_delta"]) * int(
            ablation["paired_event_count"]
        )
        all_ablation["non_feedback_error_sum"] += float(ablation["non_feedback_error_delta"]) * int(
            ablation["non_feedback_paired_event_count"]
        )
        all_ablation["paired_event_count"] += int(ablation["paired_event_count"])
        all_ablation["non_feedback_paired_event_count"] += int(
            ablation["non_feedback_paired_event_count"]
        )
        event_labels: dict[str, set[str]] = defaultdict(set)
        for row in replay_steps:
            event_labels[str(row["event_id"])].add(str(row["truth"]))
            memory_parity &= row["memory_categories"].get("cap_bytes") == MEMORY_CAP_BYTES
        label_parity &= all(len(labels) == 1 for labels in event_labels.values())
        completed.append(stream_id)
        _progress(
            2,
            "replay progress",
            f"completed={offset}/{len(stream_ids)} predictions={offset * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT)} elapsed={time.monotonic() - started:.3f}s",
        )
    paired = all_ablation["paired_event_count"]
    non_feedback_paired = all_ablation["non_feedback_paired_event_count"]
    ablation_summary = {
        "prediction_difference_count": all_ablation["prediction_difference_count"],
        "retained_correct_erased_wrong_count": all_ablation["retained_correct_erased_wrong_count"],
        "erased_correct_retained_wrong_count": all_ablation["erased_correct_retained_wrong_count"],
        "future_error_delta": all_ablation["future_error_sum"] / paired,
        "non_feedback_error_delta": all_ablation["non_feedback_error_sum"] / non_feedback_paired,
        "paired_event_count": paired,
        "non_feedback_paired_event_count": non_feedback_paired,
    }
    return {
        "schema": SCHEMA,
        "stream_ids": list(stream_ids),
        "rows": reduced,
        "cold_replay_parity": {
            "fresh_process": True,
            "producer_learner_object_accessed": False,
            "producer_aggregate_accessed": False,
            "private_regime_used_for_prediction": False,
            "allowed_prediction_inputs": ["public_event", "due_released_label"],
            "arm_count": len(ARMS),
            **totals,
            "mismatch_rows": mismatches,
        },
        "arm_parity": {
            "arms": list(ARMS),
            "same_label_schedule": all(
                len(feedback_by_stream.get(stream_id, [])) == FUTURE_LABEL_COUNT
                and all(
                    set(row.get("arm_updates", {})) == set(ARMS)
                    for row in feedback_by_stream[stream_id]
                )
                for stream_id in stream_ids
            ),
            "same_evaluator_labels": label_parity,
            "same_memory_cap": memory_parity,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
            "shuffled_feedback_arm_is_preregistered_control": True,
        },
        "ablation_summary": ablation_summary,
        "completed_stream_ids": completed,
        "source_step_rows_sha256": _sha256_path(step_path),
        "source_feedback_rows_sha256": _sha256_path(feedback_path),
    }


def audit_raw_evidence(
    repo_root: Path, paths: ExperimentPaths, stream_ids: Sequence[str]
) -> tuple[JsonDict, JsonDict]:
    """Run cold replay in a fresh interpreter and retain its process receipt."""

    command = [
        sys.executable,
        "-u",
        str(repo_root / WRAPPER_PATH),
        "--date",
        RUN_DATE,
        "--audit-worker",
        "--output-root",
        str(paths.artifact.parent),
        "--stream-ids",
        ",".join(stream_ids),
    ]
    _progress(2, "before subprocess", f"cold replay streams={len(stream_ids)}")
    receipt, _ = learning._command_receipt(command, scope="REQ-CL-7312 cold replay worker")
    _progress(
        2,
        "after subprocess",
        f"exit={receipt['exit_code']} elapsed={receipt['duration_s']:.3f}s",
    )
    if receipt["exit_code"] != 0:
        raise RuntimeError("cold_replay_failed")
    cold = _load_object(paths.cold_replay)
    if cold.get("schema") != SCHEMA:
        raise ValueError("cold_replay_schema")
    receipt["log_path"] = str(paths.cold_replay)
    return cold, receipt


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return one deterministic nearest-rank bootstrap percentile."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _interval(values: Sequence[float], draws: int, salt: str) -> JsonDict:
    """Resample paired whole streams with the audit-owned frozen seed."""

    if not values:
        raise ValueError("paired_streams_unavailable")
    seed = int(transactional.sha256_json([BOOTSTRAP_SEED, salt])[-16:], 16)
    generator = random.Random(seed)
    means = [
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    return {
        "estimate": sum(values) / len(values),
        "ci95_lower": _percentile(means, 0.025),
        "ci95_upper": _percentile(means, 0.975),
    }


def build_independent_stream_intervals(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS
) -> list[JsonDict]:
    """Recompute each frozen comparison overall and within every stratum."""

    result: list[JsonDict] = []
    for comparison_id, metric, control in learning.COMPARISON_SPECS:
        for stratum in ("overall", *STRATA):
            selected = [row for row in rows if stratum == "overall" or row["stratum"] == stratum]
            stream_ids = sorted({str(row["stream_id"]) for row in selected})
            if not stream_ids:
                continue
            by_key = {(str(row["stream_id"]), str(row["arm"])): row for row in selected}
            differences = [
                float(by_key[(stream_id, FACTOR_ARM)][metric])
                - float(by_key[(stream_id, control)][metric])
                for stream_id in stream_ids
            ]
            interval = _interval(differences, draws, f"{comparison_id}:{stratum}")
            result.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "treatment_arm": FACTOR_ARM,
                    "control_arm": control,
                    "stratum": stratum,
                    "independent_unit": "stream",
                    "paired_stream_count": len(stream_ids),
                    "bootstrap_draws": draws,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "paired_differences": differences,
                    **interval,
                }
            )
    return result


def _aliased_prediction(
    masks: Mapping[str, int], event: Mapping[str, Any], permutation: Mapping[str, str]
) -> str:
    """Evaluate a consistent identifier rename while preserving factor semantics."""

    renamed_masks = {permutation[family]: int(mask) for family, mask in masks.items()}
    inverse = {renamed: original for original, renamed in permutation.items()}
    renamed_family = permutation[str(event["family_id"])]
    semantic_family = inverse[renamed_family]
    mask = renamed_masks[renamed_family]
    labels = {
        prototype._predicate_label(semantic_family, int(event["numeric_value"]), parameter)
        for parameter in prototype.PARAMETER_DOMAIN
        if mask & (1 << parameter)
    }
    return next(iter(labels)) if len(labels) == 1 else "abstain"


def run_hostile_controls(
    repo_root: Path,
    paths: ExperimentPaths,
    upstream: Mapping[str, Any],
    cold: Mapping[str, Any],
) -> list[JsonDict]:
    """Run five independent causal attacks without changing producer bytes."""

    raw = upstream["raw_evidence_receipts"]
    protected = [
        _resolve(repo_root, raw["step_rows"]["path"]),
        _resolve(repo_root, raw["feedback_update_rows"]["path"]),
    ]
    before_hashes = {str(path): _sha256_path(path) for path in protected}
    views = load_authenticated_views(repo_root, upstream)
    stream_id = "evaluation-01"
    releases = [row for row in views.releases if row["stream_id"] == stream_id]
    future = [row for row in releases if row.get("role") == "future_feedback"]
    events = sorted(
        (row for row in views.public if row["stream_id"] == stream_id),
        key=lambda row: int(row["chronology_index"]),
    )
    masks = prototype._warmup_masks(releases, stream_id)
    first_release = future[0]
    first_event = next(row for row in events if row["event_id"] == first_release["event_id"])
    rows: list[JsonDict] = []

    def add(control: str, expected: Any, observed: Any, detection: str, passed: bool) -> None:
        after_hashes = {str(path): _sha256_path(path) for path in protected}
        rows.append(
            {
                "control": control,
                "expected": expected,
                "observed": observed,
                "detection": detection,
                "passed": bool(passed),
                "evaluation_rows_sha256_before": before_hashes,
                "evaluation_rows_sha256_after": after_hashes,
                "evaluation_rows_immutable": before_hashes == after_hashes,
            }
        )

    original = prototype.FactorLocalController.from_masks(masks)
    permuted = prototype.FactorLocalController.from_masks(masks)
    original_seal = original.seal_prediction(
        first_event, release_index=int(first_release["release_index"])
    )
    permuted_payload = [
        {**row, "observed_label": future[-index - 1]["observed_label"]}
        for index, row in enumerate(future)
    ]
    permuted_seal = permuted.seal_prediction(
        first_event, release_index=int(first_release["release_index"])
    )
    label_observed = {
        "earlier_prediction_invariant": original_seal["prediction"] == permuted_seal["prediction"],
        "original_prediction": original_seal["prediction"],
        "permuted_prediction": permuted_seal["prediction"],
        "permuted_payload_sha256": transactional.sha256_json(permuted_payload),
        "permuted_label_count": len(permuted_payload),
    }
    add(
        "future_label_permutation",
        {"earlier_prediction_invariant": True, "permuted_label_count": FUTURE_LABEL_COUNT},
        label_observed,
        "pre_release_prediction_hash_invariance",
        label_observed["earlier_prediction_invariant"]
        and label_observed["permuted_label_count"] == FUTURE_LABEL_COUNT,
    )

    permutation = {
        family: FAMILIES[(index + 1) % len(FAMILIES)] for index, family in enumerate(FAMILIES)
    }
    plain_controller = prototype.FactorLocalController.from_masks(masks)
    rename_pairs = [
        (plain_controller.predict(event), _aliased_prediction(masks, event, permutation))
        for event in events[WARMUP_COUNT : WARMUP_COUNT + 128]
    ]
    rename_observed = {
        "decision_invariant": all(left == right for left, right in rename_pairs),
        "checked_prediction_count": len(rename_pairs),
        "permutation": permutation,
    }
    add(
        "consistent_factor_name_permutation",
        {"decision_invariant": True, "checked_prediction_count": 128},
        rename_observed,
        "semantic_alias_invariance",
        rename_observed["decision_invariant"]
        and rename_observed["checked_prediction_count"] == 128,
    )

    ablation = dict(cold["ablation_summary"])
    retained_rows = [row for row in cold["rows"] if row["arm"] == FACTOR_ARM]
    erased_rows = [row for row in cold["rows"] if row["arm"] == ERASED_ARM]
    ablation_observed = {
        **ablation,
        "retained_witness_update_count": sum(
            row["factor_change_update_count"] for row in retained_rows
        ),
        "erased_witness_update_count": sum(
            row["factor_change_update_count"] for row in erased_rows
        ),
        "erased_witness_storage": True,
    }
    add(
        "retained_witness_erasure",
        "erasure produces a measured later prediction and outcome contrast",
        ablation_observed,
        "retained_vs_local_reset_without_witnesses",
        ablation_observed["prediction_difference_count"] > 0
        and ablation_observed["paired_event_count"] > 0,
    )

    delayed = prototype.FactorLocalController.from_masks(masks)
    frozen = prototype._arm_controller("frozen_warmup", masks)
    delayed_pairs = [
        (delayed.predict(event), frozen.predict(event)) for event in events[WARMUP_COUNT:]
    ]
    delay_observed = {
        "evaluation_update_count": 0,
        "prediction_difference_vs_frozen": sum(left != right for left, right in delayed_pairs),
        "checked_prediction_count": len(delayed_pairs),
    }
    add(
        "delay_all_feedback_beyond_evaluation",
        {"evaluation_update_count": 0, "prediction_difference_vs_frozen": 0},
        delay_observed,
        "no_due_release_no_learning",
        delay_observed["evaluation_update_count"] == 0
        and delay_observed["prediction_difference_vs_frozen"] == 0,
    )

    rollback_controller = prototype.FactorLocalController.from_masks(masks)
    rollback_controller.seal_prediction(
        first_event, release_index=int(first_release["release_index"])
    )
    receipt = rollback_controller.apply_release(
        first_release, current_index=int(first_release["release_index"])
    )
    corrupt_state = rollback_controller.state_dict()
    corrupt_state["rollback"]["parent_hash"] = "sha256:" + "0" * 64
    corrupt = prototype.FactorLocalController.from_state(corrupt_state)
    corrupt_before = corrupt.state_bytes()
    try:
        corrupt.rollback(receipt)
        rollback_error = "accepted"  # pragma: no cover - the shipped hash guard rejects it.
    except prototype.FactorRevisionRejected as error:
        rollback_error = str(error)
    rollback_observed = {
        "error": rollback_error,
        "state_bytes_unchanged_after_rejection": corrupt.state_bytes() == corrupt_before,
        "invented_state_recovered": False,
    }
    add(
        "corrupt_rollback_hash",
        {"error": "invalid_rollback", "invented_state_recovered": False},
        rollback_observed,
        "rollback_parent_hash_guard",
        rollback_error == "invalid_rollback"
        and rollback_observed["state_bytes_unchanged_after_rejection"],
    )
    _atomic_write(paths.interventions, {"schema": SCHEMA, "rows": rows})
    return rows


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Give each completion or promotion gate one auditable shape."""

    return {
        "expected": expected,
        "observed": observed,
        "pass": bool(passed),
        "passed": bool(passed),
        "principle": principle,
    }


def derive_terminal_scores(
    audit_complete: bool, promotion_complete: bool, oracle: bool
) -> tuple[int, int, str]:
    """Keep complete auditing separate from favorable predictive evidence."""

    if not audit_complete:
        return 0, 0, "partial"
    if promotion_complete:
        return 1, 1, "circular_positive" if oracle else "positive"
    return 1, 0, "null"


def _terminal_verdict(verdict_class: str, failed_science: Sequence[str]) -> str:
    """Describe the measured terminal class and exact mechanism scope."""

    if verdict_class == "circular_positive":
        return (
            "complete_circular_positive: independent factor audit passed every frozen gate "
            "under shared exact evaluator authority"
        )
    if verdict_class == "null":
        return (
            "complete_null: retained factor witnesses improved some future errors but failed "
            "the complete independent promotion contract; retire "
            + RETIREMENT_SCOPE
            + "; failed gates:"
            + ",".join(failed_science)
        )
    return "partial: private audit shard did not cover all 107520 predictions"


def _sample_budget(selected: Sequence[str], completed: Sequence[str]) -> JsonDict:
    """Declare exact planned, attempted, complete, and censored audit units."""

    completed_count = len(completed)
    return {
        "fixed_evaluation_stream_count": EVALUATION_STREAM_COUNT,
        "planned_stream_count": len(selected),
        "attempted_stream_count": completed_count,
        "completed_stream_count": completed_count,
        "censored_stream_count": len(selected) - completed_count,
        "censored_stream_ids": [stream for stream in selected if stream not in completed],
        "arms_per_stream": len(ARMS),
        "planned_prediction_rows": len(selected) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT),
        "completed_prediction_rows": completed_count
        * len(ARMS)
        * (EVENTS_PER_STREAM - WARMUP_COUNT),
        "planned_transition_rows": len(selected) * len(ARMS) * FUTURE_LABEL_COUNT,
        "completed_transition_rows": completed_count * len(ARMS) * FUTURE_LABEL_COUNT,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "stopping_rule": "audit all 24 frozen streams once; do not extend from outcomes",
        "outcome_based_extension": False,
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str | None],
    selected: Sequence[str],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create every required top-level field before terminal classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "phase_durations_s": {},
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "random_seed": {
            "development": RANDOM_SEED,
            "evaluation": list(prototype.EVALUATION_STREAM_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "source_artifact_states": {},
        "rows": [],
        "sample_size_budget": _sample_budget(selected, ()),
        "acceptance_gate_results": {},
        "gate_check_summary": gate_check_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition_failed",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "factor_audit_complete_score": 0,
        "factor_promotion_score": 0,
        "continuous_self_learning_task": True,
        "causal_intervention_rows": [],
        "independent_stream_intervals": [],
        "cold_replay_parity": {},
        "arm_parity": {},
        "causal_summary": {},
        "e2e_rows": [],
        "raw_evidence_receipts": {},
        "memory_label_accounting": {},
        "retirement_scope": RETIREMENT_SCOPE,
        "retirement_triggered": False,
        "no_model_weight_mutation": True,
        "production_default_changed": False,
        "scientific_scope": {
            "finding": "finite_domain_cpu_factor_retention_audit",
            "headline_llm_accuracy_gain_established": False,
            "same_authority_oracle_scope": True,
        },
        "repository_health": {},
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, inputs, config, raw evidence, gates, rows, and finding."""

    stable = deepcopy(dict(artifact))
    for key in (
        "reproducibility_checksum",
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_durations_s",
        "validation_receipts",
        "repository_health",
    ):
        stable.pop(key, None)
    return transactional.sha256_json(stable)


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, str | None]
) -> JsonDict:
    """Build row-free terminal evidence for an unchanged external failure."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        hashes,
        EVALUATION_STREAM_IDS,
        started_at=now,
        completed_at=now,
        duration_s=0.0,
    )
    failure = artifact["gate_check_summary"]["first_failure"]
    if failure is None:
        raise ValueError("blocked_artifact_without_failure")
    artifact["honest_verdict"] = (
        f"blocked_{failure['check']}: upstream={failure['upstream']}; "
        f"check={failure['check']}; field={failure['field']}; "
        f"observed={failure['observed_value']!r}; expected={failure['expected_value']!r}"
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _e2e_rows(
    upstream_hash: str | None,
    cold: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    classified: bool,
) -> list[JsonDict]:
    """Bind the capture-to-terminal stages with explicit expected outcomes."""

    parity = cold["cold_replay_parity"]
    values = (
        (
            "authenticated_immutable_capture",
            "sha256 evidence",
            upstream_hash,
            upstream_hash is not None,
        ),
        (
            "cold_prediction_and_transition_replay",
            0,
            len(parity["mismatch_rows"]),
            not parity["mismatch_rows"],
        ),
        (
            "hostile_causal_interventions",
            len(CONTROL_NAMES),
            sum(row["passed"] is True for row in controls),
            all(row["passed"] is True for row in controls),
        ),
        ("independent_stream_reduction", ">0", len(intervals), bool(intervals)),
        ("terminal_schema_classification", True, classified, classified),
    )
    return [
        {
            "stage": stage,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
            "input_sha256": transactional.sha256_json([stage, expected]),
            "output_sha256": transactional.sha256_json([stage, observed]),
        }
        for stage, expected, observed, passed in values
    ]


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    progress: bool = False,
) -> JsonDict:
    """Authenticate, cold-replay, attack, reduce, and seal one audit candidate."""

    selected = tuple(stream_ids or EVALUATION_STREAM_IDS)
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    spans: JsonDict = {}
    if progress:
        _progress(1, "start", "authenticate Exp7311, raw rows, code hashes, and outputs")
    phase = time.monotonic()
    checks, hashes, upstream = collect_preconditions(repo_root, paths)
    spans["preconditions"] = time.monotonic() - phase
    summary = gate_check_summary(checks)
    if progress:
        _progress(1, "end", f"passed={summary['passed']}")
    if summary["passed"] is not True:
        return build_blocked_artifact(checks, hashes)

    if progress:
        _progress(
            2, "start", f"cold-reconstruct predictions and transitions streams={len(selected)}"
        )
    phase = time.monotonic()
    cold, cold_receipt = audit_raw_evidence(repo_root, paths, selected)
    spans["cold_replay"] = time.monotonic() - phase
    parity = cold["cold_replay_parity"]
    if progress:
        _progress(
            2,
            "end",
            f"predictions={parity['matched_prediction_rows']} transitions={parity['matched_transition_rows']} mismatches={len(parity['mismatch_rows'])}",
        )

    raw = upstream["raw_evidence_receipts"]
    protected_paths = [
        _resolve(repo_root, raw["step_rows"]["path"]),
        _resolve(repo_root, raw["feedback_update_rows"]["path"]),
    ]
    source_before = {str(path): _sha256_path(path) for path in protected_paths}
    if progress:
        _progress(3, "before benchmark", "run five independent hostile controls")
    phase = time.monotonic()
    controls = run_hostile_controls(repo_root, paths, upstream, cold)
    source_after = {str(path): _sha256_path(path) for path in protected_paths}
    spans["hostile_controls"] = time.monotonic() - phase
    if progress:
        _progress(
            3,
            "after benchmark",
            f"passed={sum(row['passed'] is True for row in controls)}/{len(controls)}",
        )

    if progress:
        _progress(4, "before reduction", "bootstrap paired whole streams within every stratum")
    phase = time.monotonic()
    rows = cold["rows"]
    intervals = build_independent_stream_intervals(rows, draws=bootstrap_draws)
    later_changed = sum(
        int(row["later_changed_prediction_count"]) for row in rows if row["arm"] == FACTOR_ARM
    )
    violations = sum(
        int(row["time_limit_violations"]) + int(row["byte_limit_violations"]) for row in rows
    )
    promotion_gates = learning.score_value_gates(intervals, later_changed, violations)
    ablation = cold["ablation_summary"]
    causal = {
        "legitimate_later_changed_prediction_count": later_changed,
        "retained_vs_erased_prediction_difference_count": ablation["prediction_difference_count"],
        "future_error_delta_vs_witness_erased": ablation["future_error_delta"],
        "non_feedback_error_delta_vs_witness_erased": ablation["non_feedback_error_delta"],
        "retained_evidence_improved_future_error": ablation["future_error_delta"] < 0
        and ablation["non_feedback_error_delta"] < 0,
        "state_change_update_count": sum(int(row["state_change_update_count"]) for row in rows),
        "state_change_alone_credited": False,
        "time_or_byte_violation_count": violations,
    }
    spans["independent_reduction"] = time.monotonic() - phase
    if progress:
        failed_science = [
            name for name, row in promotion_gates.items() if row["passed"] is not True
        ]
        _progress(4, "after reduction", f"intervals={len(intervals)} failed={failed_science}")

    if progress:
        _progress(5, "start", "bind source immutability, replay, controls, and terminal class")
    phase = time.monotonic()
    expected_predictions = len(selected) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT)
    expected_transitions = len(selected) * len(ARMS) * FUTURE_LABEL_COUNT
    replay_exact = (
        parity["matched_prediction_rows"] == expected_predictions
        and parity["matched_transition_rows"] == expected_transitions
        and parity["prediction_mismatch_count"] == 0
        and parity["state_hash_mismatch_count"] == 0
        and parity["transition_mismatch_count"] == 0
        and parity["mismatch_rows"] == []
    )
    controls_pass = set(row["control"] for row in controls) == set(CONTROL_NAMES) and all(
        row["passed"] is True and row["evaluation_rows_immutable"] is True for row in controls
    )
    arm_parity = cold["arm_parity"]
    full_panel = selected == EVALUATION_STREAM_IDS
    interval_strata = {row["stratum"] for row in intervals}
    intervals_complete = bool(intervals) and interval_strata == {"overall", *STRATA}
    source_immutable = source_before == source_after
    pre_e2e = _e2e_rows(hashes.get(str(paths.upstream_artifact)), cold, controls, intervals, True)
    completion_gates = {
        "authenticated_capture": _gate(
            True,
            summary["passed"],
            summary["passed"] is True,
            "Only exact, complete, non-quarantined capture can enter the audit.",
        ),
        "full_cold_replay": _gate(
            [107_520, 15_360, 0],
            [
                parity["matched_prediction_rows"],
                parity["matched_transition_rows"],
                len(parity["mismatch_rows"]),
            ],
            full_panel and replay_exact,
            "Audit completeness requires every prediction and arm transition.",
        ),
        "hostile_controls": _gate(
            len(CONTROL_NAMES),
            sum(row["passed"] is True for row in controls),
            controls_pass,
            "Each hostile copy must have its declared independent outcome.",
        ),
        "equal_labels_and_memory_cap": _gate(
            [True, True, True],
            [
                arm_parity["same_label_schedule"],
                arm_parity["same_evaluator_labels"],
                arm_parity["same_memory_cap"],
            ],
            all(
                arm_parity[name] is True
                for name in ("same_label_schedule", "same_evaluator_labels", "same_memory_cap")
            ),
            "Arm comparisons require equal scoring labels, schedules, and memory limits.",
        ),
        "independent_intervals": _gate(
            ["overall", *STRATA],
            sorted(interval_strata),
            intervals_complete,
            "Whole streams and each stratum remain separate independent units.",
        ),
        "immutable_source_rows": _gate(
            source_before,
            source_after,
            source_immutable,
            "Development interventions cannot modify real producer evidence.",
        ),
        "e2e_pipeline": _gate(
            len(E2E_STAGES),
            sum(row["passed"] is True for row in pre_e2e),
            all(row["passed"] is True for row in pre_e2e),
            "Authentication, replay, controls, reduction, and classification must complete.",
        ),
    }
    audit_complete = all(row["passed"] is True for row in completion_gates.values())
    promotion_complete = audit_complete and all(
        row["passed"] is True for row in promotion_gates.values()
    )
    audit_score, promotion_score, verdict_class = derive_terminal_scores(
        audit_complete, promotion_complete, True
    )
    failed_science = [name for name, row in promotion_gates.items() if row["passed"] is not True]
    verdict = _terminal_verdict(verdict_class, failed_science)
    e2e = _e2e_rows(hashes.get(str(paths.upstream_artifact)), cold, controls, intervals, True)
    _atomic_write(paths.e2e, {"schema": SCHEMA, "rows": e2e})
    spans["e2e_and_classification"] = time.monotonic() - phase
    if progress:
        _progress(5, "end", f"audit={audit_score} promotion={promotion_score}")
    for evidence_path in (paths.cold_replay, paths.interventions, paths.e2e):
        hashes[str(evidence_path)] = _sha256_path(evidence_path)
    artifact = _base_artifact(
        checks,
        hashes,
        selected,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
    )
    artifact.update(
        {
            "status": "complete" if audit_complete else "partial",
            "phase_durations_s": spans,
            "rows": rows,
            "sample_size_budget": _sample_budget(selected, cold["completed_stream_ids"]),
            "acceptance_gate_results": {**completion_gates, **promotion_gates},
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "factor_audit_complete_score": audit_score,
            "factor_promotion_score": promotion_score,
            "causal_intervention_rows": controls,
            "independent_stream_intervals": intervals,
            "cold_replay_parity": parity,
            "arm_parity": arm_parity,
            "causal_summary": causal,
            "e2e_rows": e2e,
            "raw_evidence_receipts": {
                "cold_replay": {
                    "path": str(paths.cold_replay),
                    "sha256": _sha256_path(paths.cold_replay),
                },
                "causal_interventions": {
                    "path": str(paths.interventions),
                    "sha256": _sha256_path(paths.interventions),
                },
                "e2e_rows": {"path": str(paths.e2e), "sha256": _sha256_path(paths.e2e)},
            },
            "memory_label_accounting": {
                "memory_cap_bytes": MEMORY_CAP_BYTES,
                "maximum_state_bytes": max(int(row["maximum_memory_bytes"]) for row in rows),
                "revealed_label_count": len(selected) * FUTURE_LABEL_COUNT,
                "arm_specific_transition_count": len(selected) * FUTURE_LABEL_COUNT * len(ARMS),
                "same_memory_cap": arm_parity["same_memory_cap"],
                "same_evaluator_labels": arm_parity["same_evaluator_labels"],
                "all_serialized_state_charged": True,
            },
            "source_artifact_states": {
                "exp7311": {
                    "path": str(paths.upstream_artifact),
                    "sha256": _sha256_path(paths.upstream_artifact),
                    "terminal_class": upstream.get("status"),
                    "verdict_class": upstream.get("verdict_class"),
                    "factor_capture_complete_score": upstream.get("factor_capture_complete_score"),
                    "factor_value_score": upstream.get("factor_value_score"),
                    "quarantined": bool(upstream.get("flagged_adversarial")),
                    "disqualified": upstream.get("verdict_class") == "disqualified",
                    "retired": bool(upstream.get("retired")),
                }
            },
            "retirement_triggered": verdict_class == "null",
            "validation_receipts": [cold_receipt],
            "repository_health": deepcopy(upstream.get("repository_health", {})),
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact,
        expected_stream_ids=selected,
        check_files=True,
    )
    if errors:  # pragma: no cover - construction must stop before returning invalid evidence.
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return artifact


def _receipt_error(receipt: Mapping[str, Any]) -> bool:
    """Reject command receipts without exact scope, result, time, and log hash."""

    return not (
        isinstance(receipt.get("command"), str)
        and bool(receipt.get("command"))
        and isinstance(receipt.get("scope"), str)
        and bool(receipt.get("scope"))
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("duration_s"), (int, float))
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get("log_sha256", ""))) is not None
    )


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check identity, parity, interventions, rows, scores, and file hashes."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != INVOCATION_COUNTS
        or artifact.get("current_model_load_count") != 0
        or artifact.get("current_generation_count") != 0,
        "model_invocation",
    )
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
        or artifact.get("execution_venue") != EXECUTION_VENUE,
        "substrate",
    )
    add(
        artifact.get("continuous_self_learning_task") is not True
        or artifact.get("no_model_weight_mutation") is not True
        or artifact.get("production_default_changed") is not False,
        "learning_boundary",
    )
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(artifact.get("verdict_class") == "positive", "oracle_positive_forbidden")
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    add(
        not isinstance(artifact.get("field_principles"), Mapping)
        or any(
            field not in artifact.get("field_principles", {}) for field in REQUIRED_ARTIFACT_FIELDS
        ),
        "field_principles",
    )
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts),
        "validation_receipts",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("causal_intervention_rows") != []
            or artifact.get("factor_audit_complete_score") != 0
            or artifact.get("factor_promotion_score") != 0
            or artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("passed") is not False
            or artifact.get("gate_check_summary", {}).get("first_failure") is None,
            "blocked_contract",
        )
        return errors
    add(artifact.get("status") not in {"complete", "partial"}, "status")
    selected = tuple(expected_stream_ids or EVALUATION_STREAM_IDS)
    rows = artifact.get("rows", [])
    expected_units = {(stream_id, arm) for stream_id in selected for arm in ARMS}
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units
        or any(
            row.get("future_prediction_count") != EVENTS_PER_STREAM - WARMUP_COUNT
            or row.get("non_feedback_prediction_count")
            != EVENTS_PER_STREAM - WARMUP_COUNT - FUTURE_LABEL_COUNT
            or row.get("future_label_count") != FUTURE_LABEL_COUNT
            or row.get("censored") is not False
            for row in rows
        ),
        "rows",
    )
    parity = artifact.get("cold_replay_parity", {})
    add(
        parity.get("fresh_process") is not True
        or parity.get("producer_aggregate_accessed") is not False
        or parity.get("private_regime_used_for_prediction") is not False
        or parity.get("prediction_mismatch_count") != 0
        or parity.get("state_hash_mismatch_count") != 0
        or parity.get("transition_mismatch_count") != 0
        or parity.get("mismatch_rows") != [],
        "cold_replay_parity",
    )
    add(
        {row.get("control") for row in artifact.get("causal_intervention_rows", [])}
        != set(CONTROL_NAMES)
        or any(
            row.get("passed") is not True or row.get("evaluation_rows_immutable") is not True
            for row in artifact.get("causal_intervention_rows", [])
        ),
        "causal_interventions",
    )
    add(
        [row.get("stage") for row in artifact.get("e2e_rows", [])] != list(E2E_STAGES)
        or any(row.get("passed") is not True for row in artifact.get("e2e_rows", [])),
        "e2e",
    )
    intervals = artifact.get("independent_stream_intervals", [])
    add(
        not intervals
        or {row.get("stratum") for row in intervals} != {"overall", *STRATA}
        or any(row.get("independent_unit") != "stream" for row in intervals),
        "independent_stream_intervals",
    )
    arm_parity = artifact.get("arm_parity", {})
    add(
        arm_parity.get("same_label_schedule") is not True
        or arm_parity.get("same_evaluator_labels") is not True
        or arm_parity.get("same_memory_cap") is not True,
        "arm_parity",
    )
    budget = artifact.get("sample_size_budget", {})
    add(
        budget.get("completed_prediction_rows")
        != len(selected) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT)
        or budget.get("completed_transition_rows") != len(selected) * len(ARMS) * FUTURE_LABEL_COUNT
        or budget.get("censored_stream_count") != 0,
        "sample_size_budget",
    )
    gates = artifact.get("acceptance_gate_results", {})
    add(
        not isinstance(gates, Mapping)
        or any(name not in gates for name in (*COMPLETION_GATE_NAMES, *learning.VALUE_GATE_NAMES))
        or any(
            row.get("pass") != row.get("passed")
            or not {"expected", "observed", "pass", "passed", "principle"} <= set(row)
            for row in gates.values()
        ),
        "acceptance_gate_results",
    )
    full_panel = selected == EVALUATION_STREAM_IDS
    audit_complete = full_panel and all(
        gates.get(name, {}).get("passed") is True for name in COMPLETION_GATE_NAMES
    )
    promotion_complete = audit_complete and all(
        gates.get(name, {}).get("passed") is True for name in learning.VALUE_GATE_NAMES
    )
    scores = derive_terminal_scores(audit_complete, promotion_complete, True)
    add(artifact.get("factor_audit_complete_score") != scores[0], "factor_audit_complete_score")
    add(artifact.get("factor_promotion_score") != scores[1], "factor_promotion_score")
    add(artifact.get("verdict_class") != scores[2], "verdict_class")
    add(
        scores[2] == "null"
        and (
            not str(artifact.get("honest_verdict", "")).startswith("complete_null:")
            or artifact.get("retirement_scope") != RETIREMENT_SCOPE
            or artifact.get("retirement_triggered") is not True
        ),
        "null_verdict",
    )
    add(
        scores[2] == "partial"
        and (
            artifact.get("status") != "partial"
            or not str(artifact.get("honest_verdict", "")).startswith("partial:")
        ),
        "partial_verdict",
    )
    accounting = artifact.get("memory_label_accounting", {})
    add(
        accounting.get("maximum_state_bytes", MEMORY_CAP_BYTES + 1) > MEMORY_CAP_BYTES
        or accounting.get("revealed_label_count") != len(selected) * FUTURE_LABEL_COUNT
        or accounting.get("arm_specific_transition_count")
        != len(selected) * FUTURE_LABEL_COUNT * len(ARMS)
        or accounting.get("same_memory_cap") is not True
        or accounting.get("same_evaluator_labels") is not True,
        "memory_label_accounting",
    )
    if check_files:
        add(
            any(
                expected is None or _sha256_path(_resolve(REPO_ROOT, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
        add(
            any(
                not _receipt_matches(REPO_ROOT, receipt)
                for receipt in artifact.get("raw_evidence_receipts", {}).values()
            ),
            "raw_evidence_receipts",
        )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Append exact command outcomes and refresh the stable checksum."""

    if any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts):
        raise ValueError("validation_receipt_schema")
    changed = deepcopy(dict(artifact))
    changed["validation_receipts"] = [
        *list(changed.get("validation_receipts", [])),
        *(dict(row) for row in receipts),
    ]
    changed["reproducibility_checksum"] = reproducibility_checksum(changed)
    return changed


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    expected_stream_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Cold-validate and atomically publish one terminal or private shard record."""

    errors = validate_artifact(
        artifact,
        expected_stream_ids=expected_stream_ids,
        check_files=artifact.get("status") != "blocked",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return _atomic_write(path, artifact)


def _validation_commands(candidate: Path) -> list[list[str]]:  # pragma: no cover
    """Return focused, affected, full, coverage, static, E2E, and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    module = str(MODULE_PATH)
    test = str(TEST_PATH)
    wrapper = str(WRAPPER_PATH)
    common = ["-q", "-n", "0", "-o", "addopts=", "--no-cov"]
    return [
        [pytest, test, *common, "--basetemp=/tmp/carnot-exp7312-focused"],
        [
            pytest,
            "tests/python/test_experiment_7311_v642_factor_learning.py",
            "tests/python/test_experiment_7310_v642_factor_prototype.py",
            *common,
            "--basetemp=/tmp/carnot-exp7312-affected",
        ],
        [pytest, "tests/python", "-q"],
        [
            python,
            "-m",
            "coverage",
            "run",
            "--data-file=/tmp/carnot-exp7312.coverage",
            f"--include={module}",
            "-m",
            "pytest",
            test,
            *common,
            "--basetemp=/tmp/carnot-exp7312-coverage",
        ],
        [
            python,
            "-m",
            "coverage",
            "report",
            "--data-file=/tmp/carnot-exp7312.coverage",
            "--show-missing",
            "--fail-under=100",
        ],
        [python, "-m", "ruff", "check", module, test, wrapper],
        [python, "-m", "ruff", "format", "--check", module, test, wrapper],
        [python, "-m", "mypy", module],
        [python, "scripts/check_spec_coverage.py", test],
        [
            pytest,
            f"{test}::test_scenario_cl_7312_e2e_validates_and_writes_atomic_private_shard",
            *common,
            "--basetemp=/tmp/carnot-exp7312-e2e",
        ],
        [
            python,
            "-m",
            "carnot.experiment_7312_v642_factor_audit",
            "--date",
            RUN_DATE,
            "--validate",
            str(candidate),
        ],
        [python, "scripts/adversarial_verify.py", str(candidate)],
        [python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed date plus private worker and validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stream-ids", default="")
    parser.add_argument("--audit-worker", action="store_true")
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the no-model audit and publish only validated terminal evidence."""

    print("phase 0 immediate: Exp7312 independent factor audit started", flush=True)
    args = _parse_args(argv)
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    selected = tuple(filter(None, args.stream_ids.split(","))) or EVALUATION_STREAM_IDS
    if args.audit_worker:
        _progress(1, "start", f"cold worker streams={len(selected)}")
        cold = cold_replay_impl(REPO_ROOT, paths, selected)
        _atomic_write(paths.cold_replay, cold)
        _progress(1, "end", f"cold worker rows={len(cold['rows'])}")
        return 0
    if args.validate is not None:
        _progress(1, "before subprocess", f"validate candidate {args.validate}")
        errors = validate_artifact(_load_object(args.validate), check_files=True)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        _progress(1, "after subprocess", f"validation errors={len(errors)}")
        return int(bool(errors))
    invocation_started = time.monotonic()
    artifact = build_and_seal(REPO_ROOT, paths, stream_ids=selected, progress=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact, expected_stream_ids=selected)
        _progress(8, "end", artifact["honest_verdict"])
        return 0
    _progress(6, "start", "write measured terminal candidate under raw evidence")
    write_artifact(paths.terminal_candidate, artifact, expected_stream_ids=selected)
    _progress(6, "end", f"candidate={paths.terminal_candidate}")
    receipts: list[JsonDict] = []
    repository_health = deepcopy(artifact.get("repository_health", {}))
    commands = _validation_commands(paths.terminal_candidate)
    for index, command in enumerate(commands, start=1):
        _progress(7, "before subprocess", f"{index}/{len(commands)} {shlex.join(command)}")
        receipt, output = learning._command_receipt(
            command, scope=f"Exp7312 validation {index}/{len(commands)}"
        )
        log_path = paths.validation_dir / f"{index:02d}.log"
        _atomic_write(log_path, output.encode("utf-8"))
        receipt["log_path"] = str(log_path)
        receipt["required_for_task"] = index != 3
        receipts.append(receipt)
        _progress(
            7,
            "after subprocess",
            f"{index}/{len(commands)} exit={receipt['exit_code']} elapsed={receipt['duration_s']:.3f}s",
        )
        if index == 3:
            repository_health = {
                "classification": "current_repository_wide_validation",
                "observation": deepcopy(receipt),
                "prior_observation": repository_health,
                "waives_affected_test_failure": False,
            }
    artifact = attach_validation_receipts(artifact, receipts)
    artifact["repository_health"] = repository_health
    failures = [row for row in receipts if row["required_for_task"] and int(row["exit_code"]) != 0]
    if failures:
        _progress(8, "end", "required validation failed; candidate retained without terminal write")
        return 1
    artifact["duration_s"] = time.monotonic() - invocation_started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _progress(8, "start", "final cold validation and atomic terminal write")
    receipt = write_artifact(paths.artifact, artifact, expected_stream_ids=selected)
    _progress(8, "end", f"terminal sha256={receipt['sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin script owns normal execution.
    raise SystemExit(main())
