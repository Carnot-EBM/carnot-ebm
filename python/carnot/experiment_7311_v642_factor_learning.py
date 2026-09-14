"""Measure prospective factor-local learning on frozen delayed-label streams.

The learner sees only public events and labels whose release time has arrived.
Evaluator truth is joined after every arm seals its current prediction. This
means that only a changed prediction on a later event can count as FR-11
progress. A state change caused by fitting the current label is not enough.

Spec refs: REQ-CL-7311 and SCENARIO-CL-7311-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import random
import re
import selectors
import shlex
import subprocess
import time
from typing import Any

import yaml

from carnot import experiment_7310_v642_factor_prototype as prototype
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7311
SCHEMA = "carnot.exp7311.v642_factor_learning.v1"
MILESTONE = "2026.09.642"
RUN_DATE = "20260914"
RANDOM_SEED = 7_311_000
BOOTSTRAP_SEED = 7_311_901
BOOTSTRAP_DRAWS = 10_000
MEASUREMENT_LIMIT_S = 1_800.0
PER_CALL_LIMIT_NS = 1_000_000_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = deepcopy(prototype.INVOCATION_COUNTS)
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

ARMS = prototype.ARMS
FACTOR_ARM = "factor_local_retained_witnesses"
STRATA = (
    "isolated_factor_changes",
    "overlapping_factor_changes",
    "stationary_controls",
)
EVALUATION_STREAM_COUNT = prototype.EVALUATION_STREAM_COUNT
EVENTS_PER_STREAM = prototype.EVENTS_PER_STREAM
WARMUP_COUNT = prototype.WARMUP_COUNT
FUTURE_LABEL_COUNT = prototype.FUTURE_LABEL_COUNT
FUTURE_LABEL_POSITIONS = prototype.FUTURE_LABEL_POSITIONS
FEEDBACK_DELAY = prototype.FEEDBACK_DELAY
MEMORY_CAP_BYTES = prototype.MEMORY_CAP_BYTES
FAMILIES = prototype.FAMILIES

CAPTURE_GATE_NAMES = (
    "authenticated_inputs",
    "complete_chronological_capture",
    "complete_feedback_capture",
    "cold_reducer_parity",
    "pipeline_restart_and_rollback",
    "exact_memory_label_accounting",
)
VALUE_GATE_NAMES = (
    "future_error_vs_global_reset",
    "future_error_vs_local_reset",
    "non_feedback_error_vs_global_reset",
    "non_feedback_error_vs_local_reset",
    "recurrence_error_vs_frozen",
    "false_accept_vs_global_reset",
    "false_accept_vs_local_reset",
    "coverage_vs_global_reset",
    "coverage_vs_local_reset",
    "legitimate_later_changed_predictions",
    "time_and_byte_violations",
)
COMPARISON_SPECS = (
    ("future_error_vs_global_reset", "future_error_rate", "global_reset_on_contradiction"),
    (
        "future_error_vs_local_reset",
        "future_error_rate",
        "local_reset_without_retained_witnesses",
    ),
    (
        "non_feedback_error_vs_global_reset",
        "non_feedback_error_rate",
        "global_reset_on_contradiction",
    ),
    (
        "non_feedback_error_vs_local_reset",
        "non_feedback_error_rate",
        "local_reset_without_retained_witnesses",
    ),
    ("recurrence_error_vs_frozen", "recurrence_error_rate", "frozen_warmup"),
    (
        "false_accept_vs_global_reset",
        "false_accept_rate",
        "global_reset_on_contradiction",
    ),
    (
        "false_accept_vs_local_reset",
        "false_accept_rate",
        "local_reset_without_retained_witnesses",
    ),
    ("coverage_vs_global_reset", "coverage_rate", "global_reset_on_contradiction"),
    (
        "coverage_vs_local_reset",
        "coverage_rate",
        "local_reset_without_retained_witnesses",
    ),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7311_v642_factor_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7311_v642_factor_learning.py")
MODULE_PATH = Path("python/carnot/experiment_7311_v642_factor_learning.py")
DEFAULT_ARTIFACT = Path("results/experiment_7311_v642_factor_learning.json")
UPSTREAM_ARTIFACT = Path("results/experiment_7310_v642_factor_prototype.json")
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7311-[A-Z0-9]+(?:-[A-Z0-9]+)*")
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7296_v641_mixture_learning.py"),
    Path("python/carnot/experiment_7297_v641_mixture_audit.py"),
    Path("python/carnot/experiment_7310_v642_factor_prototype.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/pipeline/verify_repair.py"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "status",
    "run_date",
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
    "factor_capture_complete_score",
    "factor_value_score",
    "continuous_self_learning_task",
    "per_stream_results",
    "feedback_update_rows",
    "memory_label_accounting",
    "no_model_weight_mutation",
)

FIELD_PRINCIPLES = {
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "status": "Write the terminal result only after the work and checks; checkpoints remain separate.",
    "run_date": "Use 20260914 with actual UTC start/end and monotonic phase timing.",
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
    "rows": "Record every comparative unit/arm with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Keep planned, attempted, complete, and censored counts with the frozen stopping rule.",
    "acceptance_gate_results": "Every check has expected, observed, passed, and a one-line principle explaining its purpose.",
    "gate_check_summary": "Every blocked_* names upstream/check, exact field, observed value, and expected value.",
    "verifier_is_oracle": "Shared evaluator authority permits circular_positive only, not positive scientific value.",
    "honest_verdict": "Completed findings start complete_ or complete:; external failure starts blocked_. State the finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial; unchanged external failure is blocked.",
    "validation_receipts": "Record exact commands, scope, exit codes, elapsed time, and log hashes; retain failures.",
    "factor_capture_complete_score": "One for full causal measurement, including complete nulls.",
    "factor_value_score": "One only when every frozen value and safety gate passes.",
    "continuous_self_learning_task": "True marks learning between queries on actual pipeline calls.",
    "per_stream_results": "Independent stream/arm metrics retain recurrence and stationary strata.",
    "feedback_update_rows": "Before/after hashes, release IDs, and changed factors establish causal learning.",
    "memory_label_accounting": "Count every retained byte and revealed label, including pending and rollback state.",
    "no_model_weight_mutation": "The experiment changes constraint memory only.",
}


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw rows, checkpoints, validation logs, and terminal bytes separate."""

    raw_dir: Path
    checkpoint_dir: Path
    step_rows: Path
    feedback_rows: Path
    e2e_rows: Path
    terminal_candidate: Path
    validation_dir: Path
    artifact: Path
    upstream_artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the declared repository paths for the production run."""

        return cls.from_results_root(REPO_ROOT / "results", REPO_ROOT / UPSTREAM_ARTIFACT)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put all test writes and the replaceable upstream copy under one private root."""

        return cls.from_results_root(root, root / UPSTREAM_ARTIFACT.name)

    @classmethod
    def from_results_root(cls, root: Path, upstream: Path) -> ExperimentPaths:
        """Derive task-owned destinations without creating an early terminal file."""

        raw = root / "raw" / "experiment_7311_v642_factor_learning"
        checkpoints = root / "checkpoints" / "experiment_7311_v642_factor_learning"
        return cls(
            raw,
            checkpoints,
            raw / "step_rows.jsonl",
            raw / "feedback_update_rows.jsonl",
            raw / "e2e_controls.json",
            raw / "terminal_candidate.json",
            raw / "validation",
            root / DEFAULT_ARTIFACT.name,
            upstream,
        )


@dataclass(frozen=True)
class EvaluationPanel:
    """Keep chronological rows and independent reductions together for validation."""

    step_rows: list[JsonDict]
    feedback_update_rows: list[JsonDict]
    per_stream_results: list[JsonDict]
    controls: list[JsonDict]
    completed_stream_ids: list[str]
    censored_stream_ids: list[str]


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Print one flushed boundary so the task watchdog sees truthful progress."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _sha256_path(path: Path) -> str | None:
    """Hash exact file bytes while absence stays distinct from an empty file."""

    try:
        return transactional.sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed or missing evidence stays unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _resolve(repo_root: Path, value: object) -> Path:
    """Resolve repository-relative evidence and preserve declared absolute paths."""

    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Load nonempty JSON objects while malformed evidence fails closed."""

    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"invalid_jsonl_row:{path}")
        rows.append(value)
    return rows


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence before authentication."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _atomic_write(path: Path, value: Any) -> JsonDict:
    """Write exact complete bytes through the shipped durable atomic writer."""

    data = value if isinstance(value, bytes) else transactional.canonical_json_bytes(value)
    return prototype._atomic_write(path, data)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode chronological raw rows with one canonical object per line."""

    return b"".join(transactional.canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def precondition_gate(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    principle: str,
) -> JsonDict:
    """Record an exact expected and observed external precondition."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
        "principle": principle,
    }


def _precondition_result(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Record a precondition whose pass rule is not simple equality."""

    row = precondition_gate(check, upstream, field, expected, observed, principle)
    row["passed"] = bool(passed)
    return row


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve every check and expose the first exact failure without paraphrase."""

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


def _task_identity(text: str) -> JsonDict:
    """Read only the active Exp7311 identity from the executable roadmap."""

    try:
        roadmap = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(roadmap, dict) or roadmap.get("milestone") != MILESTONE:
        return {}
    for task in roadmap.get("tasks", []):
        if isinstance(task, dict) and task.get("id") == "exp7311-factor-learning":
            return {
                "id": task.get("id"),
                "milestone": task.get("milestone"),
                "deliverable": task.get("deliverable"),
            }
    return {}


def _receipt_authenticates(repo_root: Path, receipt: object) -> bool:
    """Require a declared evidence file whose current bytes match its sealed hash."""

    if not isinstance(receipt, Mapping):
        return False
    path = _resolve(repo_root, receipt.get("path", ""))
    return bool(receipt.get("sha256")) and _sha256_path(path) == receipt.get("sha256")


def collect_preconditions(
    repo_root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate Exp7310, sealed stream bytes, exclusions, sources, and outputs."""

    upstream = _load_object(paths.upstream_artifact)
    upstream_path = str(paths.upstream_artifact)
    spec_text = (repo_root / SPEC_PATH).read_text(encoding="utf-8")
    roadmap_path = repo_root / "research-roadmap.yaml"
    roadmap_text = roadmap_path.read_text(encoding="utf-8")
    exclusion_path = repo_root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8")
    expected_identity = {
        "id": "exp7311-factor-learning",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT),
    }
    manifest_receipt = upstream.get("raw_evidence_receipts", {}).get("stream_manifest", {})
    manifest_path = _resolve(repo_root, manifest_receipt.get("path", ""))
    sealed_manifest = _load_object(manifest_path)
    stream_receipts = upstream.get("stream_manifest", {}).get("receipts", {})
    evaluation_receipts = (
        [
            stream_receipts.get(name)
            for name in (
                "evaluation_public",
                "evaluation_authority",
                "evaluation_releases",
            )
        ]
        if isinstance(stream_receipts, Mapping)
        else []
    )
    scenario_count = len(set(SCENARIO_PATTERN.findall(spec_text)))
    excluded = bool(re.search(r"experiment_id:\s*(?:exp)?7311\b", exclusion_text))
    checks = [
        precondition_gate(
            "driving_capability_spec",
            str(repo_root / SPEC_PATH),
            "REQ-CL-7311",
            True,
            "REQ-CL-7311" in spec_text,
            "Implementation starts only after its requirement exists.",
        ),
        precondition_gate(
            "scenario_contract",
            str(repo_root / SPEC_PATH),
            "SCENARIO-CL-7311-*",
            7,
            scenario_count,
            "Every required behavior has a named scenario.",
        ),
        precondition_gate(
            "v642_task_identity",
            str(roadmap_path),
            "id,milestone,deliverable",
            expected_identity,
            _task_identity(roadmap_text),
            "Run only the active task and its declared terminal output.",
        ),
        precondition_gate(
            "upstream_schema",
            upstream_path,
            "schema",
            prototype.SCHEMA,
            upstream.get("schema"),
            "Only the declared factor fixture schema can authorize evaluation.",
        ),
        precondition_gate(
            "upstream_terminal",
            upstream_path,
            "status",
            "complete",
            upstream.get("status"),
            "Incomplete external work cannot authorize a prospective claim.",
        ),
        precondition_gate(
            "factor_fixture_ready",
            upstream_path,
            "factor_fixture_ready_score",
            1,
            upstream.get("factor_fixture_ready_score"),
            "Only the complete bounded factor fixture can authorize evaluation.",
        ),
        precondition_gate(
            "upstream_not_quarantined",
            upstream_path,
            "flagged_adversarial",
            False,
            bool(upstream.get("flagged_adversarial")),
            "Quarantined evidence cannot authorize a later experiment.",
        ),
        _precondition_result(
            "upstream_not_disqualified",
            upstream_path,
            "verdict_class",
            "not disqualified",
            upstream.get("verdict_class"),
            upstream.get("verdict_class") != "disqualified",
            "A numeric score cannot override a disqualified terminal class.",
        ),
        precondition_gate(
            "stream_manifest_receipt",
            str(manifest_path),
            "sha256",
            manifest_receipt.get("sha256"),
            _sha256_path(manifest_path),
            "The exact sealed manifest owns the evaluation stream identities.",
        ),
        precondition_gate(
            "stream_manifest_identity",
            str(manifest_path),
            "schema,frozen",
            ["carnot.exp7310.stream_manifest.v1", True],
            [sealed_manifest.get("schema"), sealed_manifest.get("frozen")],
            "Only the frozen Exp7310 manifest can define this panel.",
        ),
        precondition_gate(
            "evaluation_stream_receipts",
            upstream_path,
            "evaluation_public,evaluation_authority,evaluation_releases",
            True,
            len(evaluation_receipts) == 3
            and all(_receipt_authenticates(repo_root, row) for row in evaluation_receipts),
            "Every authority-separated evaluation sidecar must retain its exact bytes.",
        ),
        precondition_gate(
            "exp7311_not_excluded",
            str(exclusion_path),
            "experiment_id",
            False,
            excluded,
            "A retired or excluded task identifier must stop before measurement.",
        ),
        precondition_gate(
            "resource_ownership",
            str(paths.artifact),
            "task_owned_outputs_writable",
            True,
            _path_writable(paths.artifact)
            and _path_writable(paths.raw_dir / "probe")
            and _path_writable(paths.checkpoint_dir / "probe"),
            "Only task-owned result, raw, and checkpoint paths may receive writes.",
        ),
    ]
    hashes: dict[str, str | None] = {
        str(repo_root / path): _sha256_path(repo_root / path) for path in SOURCE_PATHS
    }
    hashes[upstream_path] = _sha256_path(paths.upstream_artifact)
    hashes[str(manifest_path)] = _sha256_path(manifest_path)
    for receipt in evaluation_receipts:
        if isinstance(receipt, Mapping):
            path = _resolve(repo_root, receipt.get("path", ""))
            hashes[str(path)] = _sha256_path(path)
    checks.append(
        precondition_gate(
            "source_bytes_available",
            "declared sources and sealed evaluation sidecars",
            "sha256",
            True,
            all(value is not None for value in hashes.values()),
            "Every executable and evidence identity must be hashable before measurement.",
        )
    )
    return checks, hashes, upstream


def load_authenticated_views(repo_root: Path, upstream: Mapping[str, Any]) -> prototype.StreamViews:
    """Load only the evaluation bytes authenticated by the Exp7310 manifest."""

    receipts = upstream.get("stream_manifest", {}).get("receipts", {})
    if not isinstance(receipts, Mapping):
        raise ValueError("evaluation_receipts")
    loaded: dict[str, list[JsonDict]] = {}
    for key, name in (
        ("evaluation_public", "public"),
        ("evaluation_authority", "authority"),
        ("evaluation_releases", "releases"),
    ):
        receipt = receipts.get(key)
        if not _receipt_authenticates(repo_root, receipt):
            raise ValueError(f"view_hash:{key}")
        assert isinstance(receipt, Mapping)
        loaded[name] = _read_jsonl(_resolve(repo_root, receipt["path"]))
    frozen = prototype.build_stream_views("evaluation")
    if (
        loaded["public"] != frozen.public
        or loaded["authority"] != frozen.authority
        or loaded["releases"] != frozen.releases
    ):
        raise ValueError("sealed_view_content")
    views = prototype.StreamViews(
        loaded["public"],
        loaded["authority"],
        loaded["releases"],
        frozen.shuffled_labels,
        frozen.manifest,
    )
    errors = prototype.stream_conformance_errors(views, "evaluation")
    if errors:
        raise ValueError("stream_conformance:" + ",".join(errors))
    expected_shuffle = (
        upstream.get("stream_manifest", {}).get("shuffled_label_hashes", {}).get("evaluation")
    )
    if transactional.sha256_json(views.shuffled_labels) != expected_shuffle:
        raise ValueError("shuffled_label_hash")
    return views


def _controller(hook: prototype.FactorPipelineHook) -> prototype.FactorLocalController:
    """Return the enabled hook controller for measurement-only state receipts."""

    controller = hook._controller
    if controller is None:
        raise prototype.FactorRevisionRejected("factor_hook_disabled")
    return controller


def _prediction_seal(row: Mapping[str, Any]) -> str:
    """Bind only information available when the pre-label prediction is sealed."""

    return transactional.sha256_json(
        {
            "stream_id": row["stream_id"],
            "arm": row["arm"],
            "event_id": row["event_id"],
            "chronology_index": row["chronology_index"],
            "prediction": row["prediction"],
            "state_hash_at_seal": row["state_hash_at_seal"],
            "prediction_time_ns": row["prediction_time_ns"],
            "abstention": row["abstention"],
        }
    )


def _family_hashes(controller: prototype.FactorLocalController) -> dict[str, str]:
    """Hash each owned factor separately so a reset cannot hide its scope."""

    return {
        family: transactional.sha256_bytes(controller.family_bytes(family)) for family in FAMILIES
    }


def _memory_categories(controller: prototype.FactorLocalController) -> JsonDict:
    """Return every charged category from the controller's canonical state."""

    return dict(controller.memory_usage())


def replay_stream(
    views: prototype.StreamViews,
    stream_id: str,
    state_root: Path,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    """Replay one stream through five durable hooks with prediction-first chronology."""

    events = sorted(
        (row for row in views.public if row.get("stream_id") == stream_id),
        key=lambda row: int(row["chronology_index"]),
    )
    if len(events) != EVENTS_PER_STREAM:
        raise ValueError(f"incomplete_stream:{stream_id}")
    authority = {
        str(row["event_id"]): row for row in views.authority if row.get("stream_id") == stream_id
    }
    releases = [row for row in views.releases if row.get("stream_id") == stream_id]
    warmup_masks = prototype._warmup_masks(releases, stream_id)
    future_releases = [row for row in releases if row.get("role") == "future_feedback"]
    due_by_index = {int(row["release_index"]): row for row in future_releases}
    selected_positions = set(FUTURE_LABEL_POSITIONS)
    hooks: dict[str, prototype.FactorPipelineHook] = {}
    for arm in ARMS:
        controller = prototype._arm_controller(arm, warmup_masks)
        hooks[arm] = prototype.FactorPipelineHook(
            state_root / arm,
            enabled=True,
            controller=controller,
        )

    seed = int(authority[str(events[0]["event_id"])]["stream_seed"])
    stratum = str(authority[str(events[0]["event_id"])]["stratum"])
    step_rows: list[JsonDict] = []
    feedback_rows: list[JsonDict] = []
    revealed_count = 0
    poison_counts = dict.fromkeys(ARMS, 0)
    latest_receipts: dict[str, JsonDict] = {}
    for event in events[WARMUP_COUNT:]:
        index = int(event["chronology_index"])
        due = due_by_index.get(index)
        prediction_order = index * 4
        sealed: dict[str, JsonDict] = {}
        for arm in ARMS:
            hook = hooks[arm]
            controller = _controller(hook)
            before_hash = hook.transactional_memory.state_hash()
            started_ns = time.perf_counter_ns()
            if index in selected_positions:
                prediction = str(
                    hook.pre_label(event, release_index=index + FEEDBACK_DELAY)["prediction"]
                )
            else:
                prediction = hook.predict(event)
            cost_ns = max(1, time.perf_counter_ns() - started_ns)
            usage = _memory_categories(controller)
            row = {
                "unit_id": f"{stream_id}:{arm}:{index:04d}",
                "stream_id": stream_id,
                "seed": seed,
                "stratum": stratum,
                "arm": arm,
                "event_id": str(event["event_id"]),
                "chronology_index": index,
                "feedback_selected_step": index in selected_positions,
                "non_feedback_step": index not in selected_positions,
                "recurrence_step": index >= 768,
                "stationary_step": stratum == "stationary_controls",
                "poison_exposed_before_prediction": poison_counts[arm] > 0,
                "prediction": prediction,
                "predicted_class": prediction,
                "prediction_order": prediction_order,
                "evaluator_order": prediction_order + 1,
                "release_order": prediction_order + 2 if due is not None else None,
                "update_order": prediction_order + 3 if due is not None else None,
                "prediction_time_ns": time.monotonic_ns(),
                "prediction_cost_ns": cost_ns,
                "state_hash_before_prediction": before_hash,
                "state_hash_at_seal": hook.transactional_memory.state_hash(),
                "memory_bytes": int(usage["serialized_state_bytes"]),
                "memory_categories": usage,
                "abstention": int(prediction == "abstain"),
                "covered": int(prediction != "abstain"),
                "prediction_frozen_before_release": True,
                "actual_pipeline_hook": True,
                "feedback_count_before_prediction": revealed_count,
                "learner_read_unreleased_label": False,
                "learner_read_private_authority": False,
                "learner_read_change_point": False,
                "censored": False,
            }
            row["prediction_seal_hash"] = _prediction_seal(row)
            sealed[arm] = row

        truth = authority[str(event["event_id"])]
        frozen_prediction = str(sealed["frozen_warmup"]["prediction"])
        for arm in ARMS:
            row = sealed[arm]
            prediction = str(row["prediction"])
            exact_label = str(truth["exact_label"])
            row.update(
                {
                    "evaluator_exact_label": exact_label,
                    "truth": exact_label,
                    "regime_id": str(truth["regime_id"]),
                    "error": int(prediction != exact_label),
                    "false_accept": int(prediction == "accept" and exact_label == "reject"),
                    "changed_from_frozen_after_feedback": int(
                        arm == FACTOR_ARM and revealed_count > 0 and prediction != frozen_prediction
                    ),
                    "time_limit_violation": int(row["prediction_cost_ns"] > PER_CALL_LIMIT_NS),
                    "byte_limit_violation": int(row["memory_bytes"] > MEMORY_CAP_BYTES),
                }
            )
            step_rows.append(row)

        if due is None:
            continue
        arm_updates: dict[str, JsonDict] = {}
        for arm in ARMS:
            hook = hooks[arm]
            controller = _controller(hook)
            before_usage = _memory_categories(controller)
            before_family_hashes = _family_hashes(controller)
            before_family_bytes = {
                family: len(controller.family_bytes(family)) for family in FAMILIES
            }
            observed = dict(due)
            poison = False
            if arm == "label_shuffled_factor_local_revision":
                observed["observed_label"] = views.shuffled_labels[str(due["event_id"])]
                poison = observed["observed_label"] != due["observed_label"]
            started_ns = time.perf_counter_ns()
            receipt = hook.release(observed, current_index=index)
            transaction_cost = max(1, time.perf_counter_ns() - started_ns)
            latest_receipts[arm] = dict(receipt)
            after_usage = _memory_categories(controller)
            after_family_hashes = _family_hashes(controller)
            after_family_bytes = {
                family: len(controller.family_bytes(family)) for family in FAMILIES
            }
            changed_factors = [
                family
                for family in FAMILIES
                if before_family_hashes[family] != after_family_hashes[family]
            ]
            if poison:
                poison_counts[arm] += 1
            factor_byte_changes = {
                family: after_family_bytes[family] - before_family_bytes[family]
                for family in FAMILIES
            }
            arm_updates[arm] = {
                "state_hash_before": receipt["state_hash_before"],
                "state_hash_after": receipt["state_hash_after"],
                "memory_bytes_before": int(before_usage["serialized_state_bytes"]),
                "memory_bytes_after": int(after_usage["serialized_state_bytes"]),
                "state_byte_delta": int(after_usage["serialized_state_bytes"])
                - int(before_usage["serialized_state_bytes"]),
                "factor_hashes_before": before_family_hashes,
                "factor_hashes_after": after_family_hashes,
                "factor_local_byte_changes": factor_byte_changes,
                "changed_factors": changed_factors,
                "evidence_consumed": 1,
                "evidence_id": transactional.sha256_json(
                    [due["event_id"], due["source_index"], due["release_index"]]
                ),
                "applied_label": observed["observed_label"],
                "poison_label": poison,
                "rejected": False,
                "ambiguous_after_update": any(
                    int(controller.family_state(family)["survivor_mask"]).bit_count() > 1
                    for family in FAMILIES
                ),
                "contradiction": bool(receipt["contradiction"]),
                "reset_scope": (
                    "all_factors"
                    if arm == "global_reset_on_contradiction" and receipt["contradiction"]
                    else str(receipt["rebuilt_family"])
                    if receipt["rebuilt_family"] is not None
                    else "metadata_only"
                ),
                "update_latency_ns": int(receipt["update_latency_ns"]),
                "transaction_cost_ns": transaction_cost,
                "predicate_evaluations": int(receipt["predicate_evaluations"]),
                "memory_categories_after": after_usage,
                "first_later_changed_prediction_event_id": None,
                "first_later_changed_prediction_index": None,
            }
        revealed_count += 1
        feedback_rows.append(
            {
                "unit_id": f"{stream_id}:release:{revealed_count:03d}",
                "stream_id": stream_id,
                "seed": seed,
                "stratum": stratum,
                "revealed_label_id": str(due["event_id"]),
                "source_index": int(due["source_index"]),
                "release_index": int(due["release_index"]),
                "prediction_order": prediction_order,
                "release_order": prediction_order + 2,
                "update_order": prediction_order + 3,
                "prediction_persisted_before_release": True,
                "observed_label": str(due["observed_label"]),
                "feedback_count_after_update": revealed_count,
                "arm_updates": arm_updates,
                "first_later_changed_prediction_event_id": None,
                "first_later_changed_prediction_index": None,
                "later_prediction_changed": 0,
                "learner_read_unreleased_label": False,
                "learner_read_private_authority": False,
                "learner_read_change_point": False,
                "censored": False,
            }
        )

    factor_changes = [
        row
        for row in step_rows
        if row["arm"] == FACTOR_ARM and row["changed_from_frozen_after_feedback"] == 1
    ]
    for offset, update in enumerate(feedback_rows):
        next_release = (
            int(feedback_rows[offset + 1]["release_index"])
            if offset + 1 < len(feedback_rows)
            else EVENTS_PER_STREAM
        )
        later = next(
            (
                row
                for row in factor_changes
                if int(update["release_index"]) < int(row["chronology_index"]) <= next_release
            ),
            None,
        )
        if later is not None:
            update["first_later_changed_prediction_event_id"] = later["event_id"]
            update["first_later_changed_prediction_index"] = later["chronology_index"]
            update["later_prediction_changed"] = 1
            factor_update = update["arm_updates"][FACTOR_ARM]
            factor_update["first_later_changed_prediction_event_id"] = later["event_id"]
            factor_update["first_later_changed_prediction_index"] = later["chronology_index"]

    final_event = events[-1]
    restart_checks = {}
    for arm, hook in hooks.items():
        expected = hook.predict(final_event)
        restarted = prototype.FactorPipelineHook(hook.state_dir, enabled=True)
        restart_checks[arm] = restarted.predict(final_event) == expected
    factor_hook = prototype.FactorPipelineHook(hooks[FACTOR_ARM].state_dir, enabled=True)
    rollback_result = _controller(factor_hook).rollback(latest_receipts[FACTOR_ARM])
    _controller(factor_hook).save(factor_hook.state_path)
    controls = {
        "cold_restart": {
            "passed": all(restart_checks.values()),
            "arm_parity": restart_checks,
            "public_sequence_event_id": final_event["event_id"],
        },
        "rollback": {
            "passed": rollback_result["passed"] is True,
            "byte_identical": rollback_result["byte_identical"],
            "public_sequence_event_id": final_event["event_id"],
            "restored_hash": rollback_result["restored_hash"],
        },
    }
    return step_rows, feedback_rows, controls


def step_row_errors(rows: Sequence[Mapping[str, Any]], stream_ids: Sequence[str]) -> list[str]:
    """Check the complete arm matrix, seals, chronology, authority, costs, and bytes."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    expected = len(stream_ids) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT)
    add(len(rows) != expected, "step_row_count")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("stream_id")), str(row.get("arm")))].append(row)
    add(
        set(groups) != {(stream_id, arm) for stream_id in stream_ids for arm in ARMS},
        "stream_arm_matrix",
    )
    add(
        any(
            sorted(int(row.get("chronology_index", -1)) for row in group)
            != list(range(WARMUP_COUNT, EVENTS_PER_STREAM))
            for group in groups.values()
        ),
        "chronology",
    )
    add(
        any(
            row.get("prediction") not in {"accept", "reject", "abstain"}
            or row.get("truth") not in {"accept", "reject"}
            or row.get("prediction_frozen_before_release") is not True
            or row.get("actual_pipeline_hook") is not True
            or row.get("learner_read_unreleased_label") is not False
            or row.get("learner_read_private_authority") is not False
            or row.get("learner_read_change_point") is not False
            or row.get("prediction_seal_hash") != _prediction_seal(row)
            for row in rows
        ),
        "prediction_seal_or_authority",
    )
    add(
        any(
            row.get("release_order") is not None
            and not (
                int(row["prediction_order"])
                < int(row["evaluator_order"])
                < int(row["release_order"])
                < int(row["update_order"])
            )
            for row in rows
        ),
        "release_order",
    )
    add(
        any(
            int(row.get("error", -1)) != int(row.get("prediction") != row.get("truth"))
            or int(row.get("false_accept", -1))
            != int(row.get("prediction") == "accept" and row.get("truth") == "reject")
            or int(row.get("abstention", -1)) != int(row.get("prediction") == "abstain")
            or int(row.get("covered", -1)) != int(row.get("prediction") != "abstain")
            for row in rows
        ),
        "evaluator_metrics",
    )
    add(
        any(
            int(row.get("prediction_cost_ns", 0)) <= 0
            or int(row.get("memory_bytes", 0)) <= 0
            or int(row.get("memory_bytes", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
            or row.get("censored") is not False
            for row in rows
        ),
        "cost_byte_or_censoring",
    )
    return errors


def feedback_row_errors(rows: Sequence[Mapping[str, Any]], stream_ids: Sequence[str]) -> list[str]:
    """Check every delayed release, arm update, factor hash, byte count, and later link."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(len(rows) != len(stream_ids) * FUTURE_LABEL_COUNT, "feedback_row_count")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("stream_id"))].append(row)
    add(set(groups) != set(stream_ids), "feedback_streams")
    for stream_id, group in groups.items():
        ordered = sorted(group, key=lambda row: int(row.get("source_index", -1)))
        add(
            [int(row.get("source_index", -1)) for row in ordered] != list(FUTURE_LABEL_POSITIONS),
            f"feedback_schedule:{stream_id}",
        )
        add(
            [int(row.get("feedback_count_after_update", -1)) for row in ordered]
            != list(range(1, FUTURE_LABEL_COUNT + 1)),
            f"feedback_count:{stream_id}",
        )
    add(
        any(
            int(row.get("release_index", -1)) != int(row.get("source_index", -1)) + FEEDBACK_DELAY
            or int(row.get("prediction_order", 1)) >= int(row.get("release_order", 0))
            or int(row.get("release_order", 1)) >= int(row.get("update_order", 0))
            or row.get("prediction_persisted_before_release") is not True
            or row.get("learner_read_unreleased_label") is not False
            or row.get("learner_read_private_authority") is not False
            or row.get("learner_read_change_point") is not False
            or row.get("censored") is not False
            or (
                row.get("first_later_changed_prediction_index") is not None
                and int(row["first_later_changed_prediction_index"])
                <= int(row.get("release_index", -1))
            )
            for row in rows
        ),
        "feedback_chronology",
    )
    add(
        any(
            not isinstance(row.get("arm_updates"), Mapping)
            or set(row["arm_updates"]) != set(ARMS)
            or any(
                not update.get("state_hash_before")
                or not update.get("state_hash_after")
                or int(update.get("memory_bytes_before", 0)) <= 0
                or int(update.get("memory_bytes_after", 0)) <= 0
                or int(update.get("memory_bytes_after", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
                or set(update.get("factor_hashes_before", {})) != set(FAMILIES)
                or set(update.get("factor_hashes_after", {})) != set(FAMILIES)
                or set(update.get("factor_local_byte_changes", {})) != set(FAMILIES)
                or int(update.get("evidence_consumed", 0)) != 1
                or int(update.get("transaction_cost_ns", 0)) <= 0
                for update in row["arm_updates"].values()
            )
            for row in rows
        ),
        "arm_update_receipts",
    )
    return errors


def reduce_step_rows(
    step_rows: Sequence[Mapping[str, Any]], feedback_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Reduce every stream and arm without dropping safety or stationary outcomes."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    updates_by_stream: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    for row in feedback_rows:
        updates_by_stream[str(row["stream_id"])].append(row)
    reduced: list[JsonDict] = []
    for stream_id, arm in sorted(groups, key=lambda key: (key[0], ARMS.index(key[1]))):
        events = sorted(groups[(stream_id, arm)], key=lambda row: int(row["chronology_index"]))
        non_feedback = [row for row in events if row["non_feedback_step"] is True]
        recurrence = [row for row in events if row["recurrence_step"] is True]
        poison = [row for row in events if row["poison_exposed_before_prediction"] is True]
        stationary = [row for row in events if row["stationary_step"] is True]
        updates = [row["arm_updates"][arm] for row in updates_by_stream[stream_id]]

        def rate(selected: Sequence[Mapping[str, Any]], field: str) -> float:
            return sum(int(row[field]) for row in selected) / len(selected)

        reduced.append(
            {
                "unit_id": f"{stream_id}:{arm}",
                "stream_id": stream_id,
                "seed": int(events[0]["seed"]),
                "stratum": str(events[0]["stratum"]),
                "arm": arm,
                "metric": "prospective_post_warmup_full_denominator_error",
                "future_prediction_count": len(events),
                "future_error_count": sum(int(row["error"]) for row in events),
                "future_error_rate": rate(events, "error"),
                "non_feedback_prediction_count": len(non_feedback),
                "non_feedback_error_count": sum(int(row["error"]) for row in non_feedback),
                "non_feedback_error_rate": rate(non_feedback, "error"),
                "recurrence_prediction_count": len(recurrence),
                "recurrence_error_count": sum(int(row["error"]) for row in recurrence),
                "recurrence_error_rate": rate(recurrence, "error"),
                "poison_exposed_prediction_count": len(poison),
                "poison_error_rate": rate(poison, "error") if poison else None,
                "stationary_prediction_count": len(stationary),
                "stationary_error_rate": rate(stationary, "error") if stationary else None,
                "false_accept_count": sum(int(row["false_accept"]) for row in events),
                "false_accept_rate": rate(events, "false_accept"),
                "abstention_count": sum(int(row["abstention"]) for row in events),
                "abstention_rate": rate(events, "abstention"),
                "coverage_rate": rate(events, "covered"),
                "warmup_label_count": WARMUP_COUNT,
                "future_label_count": len(updates),
                "state_change_update_count": sum(
                    row["state_hash_before"] != row["state_hash_after"] for row in updates
                ),
                "factor_change_update_count": sum(bool(row["changed_factors"]) for row in updates),
                "rejected_update_count": sum(bool(row["rejected"]) for row in updates),
                "ambiguous_update_count": sum(
                    bool(row["ambiguous_after_update"]) for row in updates
                ),
                "poison_update_count": sum(bool(row["poison_label"]) for row in updates),
                "later_changed_prediction_count": (
                    sum(
                        int(row["later_prediction_changed"]) for row in updates_by_stream[stream_id]
                    )
                    if arm == FACTOR_ARM
                    else 0
                ),
                "maximum_memory_bytes": max(int(row["memory_bytes"]) for row in events),
                "prediction_latency_ns": sum(int(row["prediction_cost_ns"]) for row in events),
                "update_latency_ns": sum(int(row["transaction_cost_ns"]) for row in updates),
                "predicate_evaluation_count": sum(
                    int(row["predicate_evaluations"]) for row in updates
                ),
                "time_limit_violations": sum(int(row["time_limit_violation"]) for row in events)
                + sum(int(row["transaction_cost_ns"] > PER_CALL_LIMIT_NS) for row in updates),
                "byte_limit_violations": sum(int(row["byte_limit_violation"]) for row in events),
                "censored": False,
            }
        )
    return reduced


def independent_reduce(step_path: Path, feedback_path: Path) -> list[JsonDict]:
    """Reload immutable rows and reduce them without producer aggregates."""

    steps = _read_jsonl(step_path)
    feedback = _read_jsonl(feedback_path)
    stream_ids = tuple(sorted({str(row.get("stream_id")) for row in steps}))
    errors = step_row_errors(steps, stream_ids) + feedback_row_errors(feedback, stream_ids)
    if errors:
        raise ValueError("raw_row_conformance:" + ",".join(errors))
    return reduce_step_rows(steps, feedback)


def run_learning_panel(
    views: prototype.StreamViews,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
    measurement_limit_s: float = MEASUREMENT_LIMIT_S,
) -> EvaluationPanel:
    """Run each frozen stream once, checkpoint it, and retain any bounded censoring."""

    selected = tuple(
        stream_ids or (f"evaluation-{index + 1:02d}" for index in range(EVALUATION_STREAM_COUNT))
    )
    all_steps: list[JsonDict] = []
    all_feedback: list[JsonDict] = []
    controls: list[JsonDict] = []
    completed: list[str] = []
    censored: list[str] = []
    started = time.monotonic()
    last_heartbeat = started
    for offset, stream_id in enumerate(selected):
        if time.monotonic() - started > measurement_limit_s:
            censored.extend(selected[offset:])
            break
        checkpoint_path = paths.checkpoint_dir / f"{stream_id}.json"
        checkpoint = _load_object(checkpoint_path)
        if checkpoint.get("schema") == SCHEMA and checkpoint.get("status") == "complete":
            steps = checkpoint.get("step_rows", [])
            feedback = checkpoint.get("feedback_update_rows", [])
            control = checkpoint.get("controls", {})
        else:
            steps, feedback, control = replay_stream(
                views,
                stream_id,
                paths.checkpoint_dir / stream_id / "state",
            )
            _atomic_write(
                checkpoint_path,
                {
                    "schema": SCHEMA,
                    "status": "complete",
                    "stream_id": stream_id,
                    "step_rows": steps,
                    "feedback_update_rows": feedback,
                    "controls": control,
                },
            )
        errors = step_row_errors(steps, (stream_id,)) + feedback_row_errors(feedback, (stream_id,))
        if errors:
            raise ValueError(f"stream_conformance:{stream_id}:" + ",".join(errors))
        all_steps.extend(steps)
        all_feedback.extend(feedback)
        controls.append({"stream_id": stream_id, **dict(control)})
        completed.append(stream_id)
        now = time.monotonic()
        if progress:
            _progress(
                3,
                "benchmark progress",
                f"completed={offset + 1}/{len(selected)} rows={len(all_steps)} elapsed={now - started:.3f}s",
            )
        if now - last_heartbeat >= 60.0:
            _progress(
                3,
                "benchmark heartbeat",
                f"completed={len(completed)} outstanding={len(selected) - len(completed)} elapsed={now - started:.1f}s",
            )
            last_heartbeat = now
    _atomic_write(paths.step_rows, _jsonl_bytes(all_steps))
    _atomic_write(paths.feedback_rows, _jsonl_bytes(all_feedback))
    reduced = reduce_step_rows(all_steps, all_feedback) if all_steps else []
    return EvaluationPanel(all_steps, all_feedback, reduced, controls, completed, censored)


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return the deterministic nearest-rank bootstrap percentile."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _bootstrap_interval(values: Sequence[float], salt: str) -> JsonDict:
    """Resample paired whole-stream differences with the frozen evaluation seed."""

    if not values:
        raise ValueError("paired_streams_unavailable")
    seed = int(transactional.sha256_json([BOOTSTRAP_SEED, salt])[-16:], 16)
    generator = random.Random(seed)
    means = [
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "estimate": sum(values) / len(values),
        "ci95_lower": _percentile(means, 0.025),
        "ci95_upper": _percentile(means, 0.975),
    }


def build_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build paired intervals overall and inside every independent stream stratum."""

    result: list[JsonDict] = []
    for comparison_id, metric, control in COMPARISON_SPECS:
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
            interval = _bootstrap_interval(differences, f"{comparison_id}:{stratum}")
            result.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "treatment_arm": FACTOR_ARM,
                    "control_arm": control,
                    "stratum": stratum,
                    "independent_unit": "stream",
                    "paired_stream_count": len(stream_ids),
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "paired_differences": differences,
                    **interval,
                }
            )
    return result


def _comparison(
    rows: Sequence[Mapping[str, Any]], comparison_id: str, stratum: str = "overall"
) -> Mapping[str, Any]:
    """Select one frozen comparison and fail when its stream units are absent."""

    return next(
        row
        for row in rows
        if row.get("comparison_id") == comparison_id and row.get("stratum") == stratum
    )


def gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Give every capture or scientific check the same auditable shape."""

    return {
        "expected": expected,
        "observed": observed,
        "pass": bool(passed),
        "passed": bool(passed),
        "principle": principle,
    }


def score_value_gates(
    comparisons: Sequence[Mapping[str, Any]],
    later_changed: int,
    violations: int,
) -> dict[str, JsonDict]:
    """Apply every frozen Exp7310 efficacy, safety, coverage, causal, and cost gate."""

    result: dict[str, JsonDict] = {}
    for name in (
        "future_error_vs_global_reset",
        "future_error_vs_local_reset",
        "non_feedback_error_vs_global_reset",
        "non_feedback_error_vs_local_reset",
    ):
        observed = float(_comparison(comparisons, name)["ci95_upper"])
        result[name] = gate(
            "ci95_upper<0",
            observed,
            observed < 0,
            "Released labels must improve later full-denominator error over both reset controls.",
        )
    recurrence_rows = [
        _comparison(comparisons, "recurrence_error_vs_frozen", stratum) for stratum in STRATA[:2]
    ]
    recurrence = {str(row["stratum"]): row["ci95_upper"] for row in recurrence_rows}
    result["recurrence_error_vs_frozen"] = gate(
        "each drift-stratum ci95_upper<=0.02",
        recurrence,
        all(float(value) <= 0.02 for value in recurrence.values()),
        "A pooled result cannot hide harm after either form of recurrence.",
    )
    for name in ("false_accept_vs_global_reset", "false_accept_vs_local_reset"):
        observed = float(_comparison(comparisons, name)["ci95_upper"])
        result[name] = gate(
            "ci95_upper<=0",
            observed,
            observed <= 0,
            "Lower error cannot come from more unsafe accept decisions.",
        )
    for name in ("coverage_vs_global_reset", "coverage_vs_local_reset"):
        observed = float(_comparison(comparisons, name)["ci95_lower"])
        result[name] = gate(
            "ci95_lower>=-0.02",
            observed,
            observed >= -0.02,
            "Lower error cannot come from excessive new abstention.",
        )
    result["legitimate_later_changed_predictions"] = gate(
        ">=24",
        later_changed,
        later_changed >= 24,
        "Only changed predictions after release count as learning progress.",
    )
    result["time_and_byte_violations"] = gate(
        0,
        violations,
        violations == 0,
        "A favorable result cannot exceed the frozen time or state bounds.",
    )
    return result


def derive_terminal_scores(gates: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Keep complete causal capture separate from favorable scientific value."""

    capture = int(all(gates.get(name, {}).get("passed") is True for name in CAPTURE_GATE_NAMES))
    value = int(
        capture == 1 and all(gates.get(name, {}).get("passed") is True for name in VALUE_GATE_NAMES)
    )
    if capture == 0:
        return {
            "factor_capture_complete_score": 0,
            "factor_value_score": 0,
            "verdict_class": "partial",
            "honest_verdict": "partial: prospective factor capture did not complete",
        }
    if value == 1:
        return {
            "factor_capture_complete_score": 1,
            "factor_value_score": 1,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive: prospective factor-local learning passed every frozen gate under finite exact evaluator authority",
        }
    failed = [name for name in VALUE_GATE_NAMES if gates.get(name, {}).get("passed") is not True]
    return {
        "factor_capture_complete_score": 1,
        "factor_value_score": 0,
        "verdict_class": "null",
        "honest_verdict": "complete_null: prospective factor-local learning completed but frozen value gates failed: "
        + ",".join(failed),
    }


def run_e2e_controls(views: prototype.StreamViews, root: Path) -> JsonDict:
    """Exercise pre-label prediction, due update, later query, restart, and rollback."""

    stream_id = "evaluation-01"
    releases = [row for row in views.releases if row.get("stream_id") == stream_id]
    masks = prototype._warmup_masks(releases, stream_id)
    release = next(row for row in releases if row.get("role") == "future_feedback")
    public = {str(row["event_id"]): row for row in views.public if row["stream_id"] == stream_id}
    event = public[str(release["event_id"])]
    hook = prototype.FactorPipelineHook(
        root / "pipeline",
        enabled=True,
        controller=prototype._arm_controller(FACTOR_ARM, masks),
    )
    prediction = str(
        hook.pre_label(event, release_index=int(release["release_index"]))["prediction"]
    )
    before_hash = hook.transactional_memory.state_hash()
    receipt = hook.release(release, current_index=int(release["release_index"]))
    after_hash = hook.transactional_memory.state_hash()
    later_index = int(release["release_index"]) + 1
    later_event = next(
        row
        for row in views.public
        if row["stream_id"] == stream_id and row["chronology_index"] == later_index
    )
    later_prediction = hook.predict(later_event)
    restarted = prototype.FactorPipelineHook(hook.state_dir, enabled=True)
    parity = restarted.predict(later_event) == later_prediction
    rollback = _controller(restarted).rollback(receipt)
    _controller(restarted).save(restarted.state_path)
    return {
        "prediction_before_release": prediction,
        "release_index": int(release["release_index"]),
        "release_state_hash_changed": before_hash != after_hash,
        "later_prediction_index": later_index,
        "later_prediction": later_prediction,
        "later_prediction_was_pre_label": True,
        "cold_restart_parity": parity,
        "rollback_byte_identical": rollback["byte_identical"],
        "actual_pipeline_hook": True,
    }


def _sample_budget(
    selected: Sequence[str], completed: Sequence[str], censored: Sequence[str]
) -> JsonDict:
    """Declare fixed plans, attempts, completion, censoring, draws, and stopping rule."""

    return {
        "fixed_evaluation_stream_count": EVALUATION_STREAM_COUNT,
        "planned_stream_count": len(selected),
        "attempted_stream_count": len(completed),
        "completed_stream_count": len(completed),
        "censored_stream_count": len(censored),
        "censored_stream_ids": list(censored),
        "arms_per_stream": len(ARMS),
        "events_per_stream": EVENTS_PER_STREAM,
        "warmup_labels_per_stream_arm": WARMUP_COUNT,
        "future_labels_per_stream_arm": FUTURE_LABEL_COUNT,
        "post_warmup_predictions_per_stream_arm": EVENTS_PER_STREAM - WARMUP_COUNT,
        "planned_post_warmup_predictions": len(selected)
        * len(ARMS)
        * (EVENTS_PER_STREAM - WARMUP_COUNT),
        "completed_post_warmup_predictions": len(completed)
        * len(ARMS)
        * (EVENTS_PER_STREAM - WARMUP_COUNT),
        "planned_released_labels": len(selected) * FUTURE_LABEL_COUNT,
        "completed_released_labels": len(completed) * FUTURE_LABEL_COUNT,
        "planned_arm_specific_updates": len(selected) * FUTURE_LABEL_COUNT * len(ARMS),
        "completed_arm_specific_updates": len(completed) * FUTURE_LABEL_COUNT * len(ARMS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "measurement_limit_s": MEASUREMENT_LIMIT_S,
        "stopping_rule": "run each of the 24 frozen evaluation streams once; do not extend from outcomes",
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
    """Create all required fields before the terminal class is known."""

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
            "master": RANDOM_SEED,
            "evaluation": list(prototype.EVALUATION_STREAM_SEEDS),
            "label_shuffle": prototype.RANDOM_SEED + 900,
            "bootstrap": BOOTSTRAP_SEED,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "source_artifact_states": {},
        "rows": [],
        "sample_size_budget": _sample_budget(selected, (), ()),
        "acceptance_gate_results": {},
        "gate_check_summary": gate_check_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition_failed",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "factor_capture_complete_score": 0,
        "factor_value_score": 0,
        "continuous_self_learning_task": True,
        "per_stream_results": [],
        "feedback_update_rows": [],
        "memory_label_accounting": {
            "memory_cap_bytes": MEMORY_CAP_BYTES,
            "maximum_state_bytes": 0,
            "revealed_label_count": 0,
            "arm_specific_update_count": 0,
        },
        "no_model_weight_mutation": True,
        "production_default_changed": False,
        "chronology_violations": {"count": 0, "error_classes": [], "offending_row_ids": []},
        "comparison_rows": [],
        "e2e_controls": {},
        "raw_evidence_receipts": {},
        "repository_health": {},
        "scientific_scope": {
            "finding": "finite_domain_cpu_factor_learning",
            "headline_llm_accuracy_gain_established": False,
            "model_weight_learning_established": False,
        },
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, inputs, config, raw evidence, rows, gates, and terminal finding."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "honest_verdict",
        "verdict_class",
        "factor_capture_complete_score",
        "factor_value_score",
        "feedback_update_rows",
        "memory_label_accounting",
        "comparison_rows",
        "raw_evidence_receipts",
        "e2e_controls",
        "scientific_scope",
    )
    return transactional.sha256_json({key: artifact.get(key) for key in keys})


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, str | None]
) -> JsonDict:
    """Return a row-free terminal block for an unchanged external failure."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        hashes,
        tuple(f"evaluation-{index + 1:02d}" for index in range(EVALUATION_STREAM_COUNT)),
        started_at=now,
        completed_at=now,
        duration_s=0.0,
    )
    failure = artifact["gate_check_summary"]["first_failure"]
    if failure is None:
        raise ValueError("blocked_artifact_without_failure")
    artifact["honest_verdict"] = (
        f"blocked_{failure['check']}: upstream={failure['upstream']}; check={failure['check']}; "
        f"field={failure['field']}; observed={failure['observed_value']!r}; "
        f"expected={failure['expected_value']!r}"
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _receipt(path: Path, row_count: int | None = None) -> JsonDict:
    """Describe exact raw bytes and their optional complete row count."""

    result: JsonDict = {"path": str(path), "sha256": _sha256_path(path)}
    if row_count is not None:
        result["row_count"] = row_count
    return result


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
) -> JsonDict:
    """Authenticate, replay, cold-reduce, score, and seal one measured candidate."""

    selected = tuple(
        stream_ids or (f"evaluation-{index + 1:02d}" for index in range(EVALUATION_STREAM_COUNT))
    )
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: dict[str, float] = {}
    phase = time.monotonic()
    if progress:
        _progress(1, "start", "authenticate Exp7310, sealed streams, sources, and outputs")
    checks, hashes, upstream = collect_preconditions(repo_root, paths)
    spans["preconditions"] = time.monotonic() - phase
    if progress:
        _progress(1, "end", f"checks={len(checks)} passed={gate_check_summary(checks)['passed']}")
    if gate_check_summary(checks)["passed"] is not True:
        artifact = build_blocked_artifact(checks, hashes)
        artifact["started_at_utc"] = started_at
        artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
        artifact["duration_s"] = time.monotonic() - started
        artifact["phase_durations_s"] = spans
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    phase = time.monotonic()
    if progress:
        _progress(2, "start", "load the three authenticated evaluation views")
    views = load_authenticated_views(repo_root, upstream)
    spans["authenticated_view_load"] = time.monotonic() - phase
    if progress:
        _progress(2, "end", f"public_events={len(views.public)} releases={len(views.releases)}")
        _progress(3, "before benchmark", f"streams={len(selected)} arms={len(ARMS)}")
    phase = time.monotonic()
    panel = run_learning_panel(views, paths, stream_ids=selected, progress=progress)
    spans["prospective_benchmark"] = time.monotonic() - phase
    if panel.censored_stream_ids:
        raise RuntimeError("measurement_limit_censored:" + ",".join(panel.censored_stream_ids))
    if progress:
        _progress(
            3,
            "after benchmark",
            f"completed={len(panel.completed_stream_ids)} predictions={len(panel.step_rows)}",
        )
        _progress(4, "before reduction", "reload raw evidence and bootstrap paired streams")
    phase = time.monotonic()
    reduced = independent_reduce(paths.step_rows, paths.feedback_rows)
    if reduced != panel.per_stream_results:
        raise ValueError("independent_reducer_mismatch")
    comparisons = build_comparison_rows(reduced)
    later_changed = sum(
        int(row["later_changed_prediction_count"]) for row in reduced if row["arm"] == FACTOR_ARM
    )
    violations = sum(
        int(row["time_limit_violations"]) + int(row["byte_limit_violations"]) for row in reduced
    )
    value_gates = score_value_gates(comparisons, later_changed, violations)
    spans["cold_reduction_and_bootstrap"] = time.monotonic() - phase
    if progress:
        _progress(
            4, "after reduction", f"comparisons={len(comparisons)} later_changes={later_changed}"
        )
        _progress(5, "start", "run public-sequence restart and rollback E2E controls")
    phase = time.monotonic()
    e2e = run_e2e_controls(views, paths.raw_dir / "e2e")
    _atomic_write(paths.e2e_rows, e2e)
    step_errors = step_row_errors(panel.step_rows, selected)
    feedback_errors = feedback_row_errors(panel.feedback_update_rows, selected)
    exact_full_panel = len(selected) == EVALUATION_STREAM_COUNT
    expected_predictions = EVALUATION_STREAM_COUNT * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT)
    expected_updates = EVALUATION_STREAM_COUNT * FUTURE_LABEL_COUNT * len(ARMS)
    maximum_bytes = max((int(row["memory_bytes"]) for row in panel.step_rows), default=0)
    category_maxima = {
        name: max(
            (int(row["memory_categories"].get(name, 0)) for row in panel.step_rows),
            default=0,
        )
        for name in (
            "witness_count",
            "pending_release_count",
            "deduplication_id_count",
            "rollback_bytes",
        )
    }
    controls_passed = all(
        row.get("cold_restart", {}).get("passed") is True
        and row.get("rollback", {}).get("passed") is True
        for row in panel.controls
    )
    capture_gates = {
        "authenticated_inputs": gate(
            True,
            gate_check_summary(checks)["passed"],
            gate_check_summary(checks)["passed"] is True,
            "Only exact non-quarantined fixture and stream bytes may start evaluation.",
        ),
        "complete_chronological_capture": gate(
            [expected_predictions, []],
            [len(panel.step_rows), step_errors],
            exact_full_panel and len(panel.step_rows) == expected_predictions and not step_errors,
            "Every planned prediction needs a pre-label seal and full-denominator outcome.",
        ),
        "complete_feedback_capture": gate(
            [expected_updates, []],
            [len(panel.feedback_update_rows) * len(ARMS), feedback_errors],
            exact_full_panel
            and len(panel.feedback_update_rows) * len(ARMS) == expected_updates
            and not feedback_errors,
            "Every scheduled label needs five arm-specific state and factor receipts.",
        ),
        "cold_reducer_parity": gate(
            transactional.sha256_json(panel.per_stream_results),
            transactional.sha256_json(reduced),
            reduced == panel.per_stream_results,
            "The producer cannot supply its own unchecked aggregate rows.",
        ),
        "pipeline_restart_and_rollback": gate(
            True,
            controls_passed and all(bool(value) for value in e2e.values()),
            controls_passed
            and e2e["cold_restart_parity"] is True
            and e2e["rollback_byte_identical"] is True
            and e2e["actual_pipeline_hook"] is True,
            "The measured hook must survive restart and restore only authentic parent bytes.",
        ),
        "exact_memory_label_accounting": gate(
            [MEMORY_CAP_BYTES, EVALUATION_STREAM_COUNT * FUTURE_LABEL_COUNT, expected_updates],
            [
                maximum_bytes,
                len(panel.feedback_update_rows),
                len(panel.feedback_update_rows) * len(ARMS),
            ],
            exact_full_panel
            and maximum_bytes <= MEMORY_CAP_BYTES
            and len(panel.feedback_update_rows) == EVALUATION_STREAM_COUNT * FUTURE_LABEL_COUNT,
            "All retained state categories and every revealed label remain charged.",
        ),
    }
    all_gates = {**capture_gates, **value_gates}
    scores = derive_terminal_scores(all_gates)
    spans["e2e_and_terminal_scoring"] = time.monotonic() - phase
    if progress:
        _progress(
            5,
            "end",
            f"capture={scores['factor_capture_complete_score']} value={scores['factor_value_score']}",
        )
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
            "status": "complete",
            "phase_durations_s": spans,
            "rows": reduced,
            "per_stream_results": reduced,
            "feedback_update_rows": panel.feedback_update_rows,
            "sample_size_budget": _sample_budget(
                selected, panel.completed_stream_ids, panel.censored_stream_ids
            ),
            "acceptance_gate_results": all_gates,
            **scores,
            "chronology_violations": {
                "count": len(step_errors) + len(feedback_errors),
                "error_classes": [*step_errors, *feedback_errors],
                "offending_row_ids": [],
            },
            "memory_label_accounting": {
                "memory_cap_bytes": MEMORY_CAP_BYTES,
                "maximum_state_bytes": maximum_bytes,
                "category_maxima": category_maxima,
                "warmup_label_count": len(selected) * WARMUP_COUNT * len(ARMS),
                "revealed_label_count": len(panel.feedback_update_rows),
                "arm_specific_update_count": len(panel.feedback_update_rows) * len(ARMS),
                "pending_labels_never_entered_early": True,
                "rejected_case_count": sum(int(row["rejected_update_count"]) for row in reduced),
                "ambiguous_case_count": sum(int(row["abstention_count"]) for row in reduced),
                "rollback_state_charged": True,
                "all_serialized_state_charged": True,
            },
            "comparison_rows": comparisons,
            "e2e_controls": e2e,
            "raw_evidence_receipts": {
                "step_rows": _receipt(paths.step_rows, len(panel.step_rows)),
                "feedback_update_rows": _receipt(
                    paths.feedback_rows, len(panel.feedback_update_rows)
                ),
                "e2e_controls": _receipt(paths.e2e_rows),
            },
            "source_artifact_states": {
                "exp7310": {
                    "path": str(paths.upstream_artifact),
                    "sha256": _sha256_path(paths.upstream_artifact),
                    "producer_schema": upstream.get("schema"),
                    "terminal_class": upstream.get("status"),
                    "verdict_class": upstream.get("verdict_class"),
                    "quarantined": bool(upstream.get("flagged_adversarial")),
                    "disqualified": upstream.get("verdict_class") == "disqualified",
                    "factor_fixture_ready_score": upstream.get("factor_fixture_ready_score"),
                }
            },
            "repository_health": deepcopy(upstream.get("repository_health", {})),
            "validation_receipts": [
                {
                    "command": f"independent_reduce {paths.step_rows} {paths.feedback_rows}",
                    "scope": "REQ-CL-7311 immutable raw reduction",
                    "exit_code": 0,
                    "duration_s": spans["cold_reduction_and_bootstrap"],
                    "log_sha256": transactional.sha256_json(reduced),
                },
                {
                    "command": "E2E-007 adapted factor hook restart and rollback",
                    "scope": "SCENARIO-CL-7311-E2E",
                    "exit_code": 0 if e2e["cold_restart_parity"] else 1,
                    "duration_s": spans["e2e_and_terminal_scoring"],
                    "log_sha256": transactional.sha256_json(e2e),
                },
            ],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, expected_stream_ids=selected, check_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return artifact


def _receipt_error(receipt: Mapping[str, Any]) -> bool:
    """Reject validation receipts without exact command, scope, result, time, and log hash."""

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
    """Cold-check identity, boundaries, exact counts, gates, raw bytes, and classification."""

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
        "model_boundary",
    )
    add(
        artifact.get("continuous_self_learning_task") is not True
        or artifact.get("model_invoked") is not False
        or artifact.get("no_model_weight_mutation") is not True
        or artifact.get("production_default_changed") is not False
        or artifact.get("scientific_scope", {}).get("headline_llm_accuracy_gain_established")
        is not False,
        "learning_boundary",
    )
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
        or artifact.get("execution_venue") != EXECUTION_VENUE,
        "substrate",
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
            or artifact.get("per_stream_results") != []
            or artifact.get("feedback_update_rows") != []
            or artifact.get("factor_capture_complete_score") != 0
            or artifact.get("factor_value_score") != 0
            or artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("passed") is not False
            or artifact.get("gate_check_summary", {}).get("first_failure") is None,
            "blocked_contract",
        )
        return errors
    add(artifact.get("status") != "complete", "status")
    if artifact.get("status") != "complete":
        return errors
    selected = tuple(
        expected_stream_ids
        or (f"evaluation-{index + 1:02d}" for index in range(EVALUATION_STREAM_COUNT))
    )
    rows = artifact.get("rows", [])
    expected_units = {(stream_id, arm) for stream_id in selected for arm in ARMS}
    add(
        not isinstance(rows, list)
        or rows != artifact.get("per_stream_results")
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units
        or any(
            row.get("future_prediction_count") != EVENTS_PER_STREAM - WARMUP_COUNT
            or row.get("non_feedback_prediction_count")
            != EVENTS_PER_STREAM - WARMUP_COUNT - FUTURE_LABEL_COUNT
            or row.get("recurrence_prediction_count") != 256
            or row.get("future_label_count") != FUTURE_LABEL_COUNT
            or row.get("censored") is not False
            for row in rows
        ),
        "rows",
    )
    feedback = artifact.get("feedback_update_rows", [])
    add(
        not isinstance(feedback, list)
        or len(feedback) != len(selected) * FUTURE_LABEL_COUNT
        or feedback_row_errors(feedback, selected) != [],
        "feedback_update_rows",
    )
    gates = artifact.get("acceptance_gate_results", {})
    add(
        not isinstance(gates, Mapping)
        or any(name not in gates for name in (*CAPTURE_GATE_NAMES, *VALUE_GATE_NAMES))
        or any(
            gate_row.get("pass") != gate_row.get("passed")
            or not {"expected", "observed", "pass", "passed", "principle"} <= set(gate_row)
            for gate_row in gates.values()
        ),
        "acceptance_gate_results",
    )
    scores = derive_terminal_scores(gates)
    add(
        artifact.get("factor_capture_complete_score") != scores["factor_capture_complete_score"],
        "factor_capture_complete_score",
    )
    add(artifact.get("factor_value_score") != scores["factor_value_score"], "factor_value_score")
    add(
        artifact.get("verdict_class") != scores["verdict_class"]
        or artifact.get("honest_verdict") != scores["honest_verdict"],
        "terminal_classification",
    )
    accounting = artifact.get("memory_label_accounting", {})
    add(
        not isinstance(accounting, Mapping)
        or int(accounting.get("maximum_state_bytes", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
        or accounting.get("revealed_label_count") != len(selected) * FUTURE_LABEL_COUNT
        or accounting.get("arm_specific_update_count")
        != len(selected) * FUTURE_LABEL_COUNT * len(ARMS)
        or accounting.get("pending_labels_never_entered_early") is not True
        or accounting.get("rollback_state_charged") is not True,
        "memory_label_accounting",
    )
    budget = artifact.get("sample_size_budget", {})
    add(
        budget.get("completed_post_warmup_predictions")
        != len(selected) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT)
        or budget.get("completed_arm_specific_updates")
        != len(selected) * FUTURE_LABEL_COUNT * len(ARMS)
        or budget.get("censored_stream_count") != 0,
        "sample_size_budget",
    )
    comparisons = artifact.get("comparison_rows", [])
    add(
        not isinstance(comparisons, list)
        or any(
            row.get("bootstrap_draws") != BOOTSTRAP_DRAWS or row.get("independent_unit") != "stream"
            for row in comparisons
        ),
        "comparison_rows",
    )
    if check_files:
        raw = artifact.get("raw_evidence_receipts", {})
        add(
            not isinstance(raw, Mapping)
            or any(
                _sha256_path(_resolve(REPO_ROOT, receipt.get("path", ""))) != receipt.get("sha256")
                for receipt in raw.values()
            ),
            "raw_evidence_receipts",
        )
        try:
            cold = independent_reduce(
                _resolve(REPO_ROOT, raw["step_rows"]["path"]),
                _resolve(REPO_ROOT, raw["feedback_update_rows"]["path"]),
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            cold = []
        add(cold != rows, "cold_reducer")
        add(
            any(
                expected is None or _sha256_path(_resolve(REPO_ROOT, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach exact command outcomes and refresh the stable checksum."""

    if any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts):
        raise ValueError("validation_receipt_schema")
    changed = deepcopy(dict(artifact))
    changed["validation_receipts"] = [
        *list(changed.get("validation_receipts", [])),
        *(dict(row) for row in receipts),
    ]
    changed["reproducibility_checksum"] = reproducibility_checksum(changed)
    return changed


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Cold-validate and atomically publish one terminal JSON object."""

    errors = validate_artifact(artifact, check_files=artifact.get("status") == "complete")
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return _atomic_write(path, dict(artifact))


def _command_receipt(
    command: Sequence[str], *, scope: str
) -> tuple[JsonDict, str]:  # pragma: no cover
    """Stream a child process and print an outstanding-call heartbeat each minute."""

    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output: list[str] = []
    while process.poll() is None:
        events = selector.select(timeout=60.0)
        if not events:
            _progress(
                7,
                "subprocess heartbeat",
                f"outstanding={shlex.join(command)} elapsed={time.monotonic() - started:.1f}s",
            )
            continue
        line = process.stdout.readline()
        if line:
            print(line, end="", flush=True)
            output.append(line)
    remainder = process.stdout.read()
    if remainder:
        print(remainder, end="", flush=True)
        output.append(remainder)
    selector.close()
    exit_code = process.wait()
    log = "".join(output)
    return (
        {
            "command": shlex.join(command),
            "scope": scope,
            "exit_code": exit_code,
            "duration_s": time.monotonic() - started,
            "log_sha256": transactional.sha256_bytes(log.encode("utf-8")),
        },
        log,
    )


def _validation_commands(candidate: Path) -> list[tuple[list[str], str, bool]]:  # pragma: no cover
    """Return focused, affected, full, coverage, static, E2E, and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    module = str(MODULE_PATH)
    test = str(TEST_PATH)
    wrapper = str(WRAPPER_PATH)
    common = ["-q", "-n", "0", "-o", "addopts=", "--no-cov"]
    return [
        (
            [pytest, test, *common, "--basetemp=/tmp/carnot-exp7311-focused"],
            "focused REQ-CL-7311 tests",
            True,
        ),
        (
            [
                pytest,
                "tests/python/test_experiment_7310_v642_factor_prototype.py",
                *common,
                "--basetemp=/tmp/carnot-exp7311-affected",
            ],
            "affected factor prototype and pipeline hook suite",
            True,
        ),
        (
            [pytest, "tests/python", "-q"],
            "repository-wide pytest observation required by task",
            False,
        ),
        (
            [
                python,
                "-m",
                "coverage",
                "run",
                "--data-file=/tmp/carnot-exp7311.coverage",
                f"--include={module}",
                "-m",
                "pytest",
                test,
                *common,
                "--basetemp=/tmp/carnot-exp7311-coverage",
            ],
            "changed-module coverage execution",
            True,
        ),
        (
            [
                python,
                "-m",
                "coverage",
                "report",
                "--data-file=/tmp/carnot-exp7311.coverage",
                "--show-missing",
                "--fail-under=100",
            ],
            "100 percent changed-module coverage",
            True,
        ),
        ([python, "-m", "ruff", "check", module, test, wrapper], "scoped Ruff check", True),
        (
            [python, "-m", "ruff", "format", "--check", module, test, wrapper],
            "scoped Ruff format",
            True,
        ),
        ([python, "-m", "mypy", module], "changed-module mypy", True),
        ([python, "scripts/check_spec_coverage.py", test], "REQ-CL-7311 spec coverage", True),
        (
            [
                pytest,
                f"{test}::test_scenario_cl_7311_e2e_credits_only_later_prediction",
                *common,
                "--basetemp=/tmp/carnot-exp7311-e2e",
            ],
            "E2E-007 adapted prospective factor learning path",
            True,
        ),
        (
            [
                python,
                "-m",
                "carnot.experiment_7311_v642_factor_learning",
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ],
            "cold terminal candidate validation",
            True,
        ),
        (
            [python, "scripts/adversarial_verify.py", str(candidate)],
            "adversarial artifact verification",
            True,
        ),
        (
            [python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            "strict verdict-row consistency",
            True,
        ),
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed execution date and private cold-validation mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Measure, validate, and atomically publish only terminal prospective evidence."""

    print("phase 0 immediate: Exp7311 prospective factor learning started", flush=True)
    args = _parse_args(argv)
    if args.validate is not None:
        _progress(1, "before subprocess", f"cold validate {args.validate}")
        artifact = _load_object(args.validate)
        errors = validate_artifact(artifact, check_files=True)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        _progress(1, "after subprocess", f"validation errors={len(errors)}")
        return int(bool(errors))
    paths = ExperimentPaths.defaults()
    invocation_started = time.monotonic()
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact)
        _progress(8, "end", artifact["honest_verdict"])
        return 0
    _progress(6, "start", "write measured terminal candidate")
    write_artifact(paths.terminal_candidate, artifact)
    _progress(6, "end", f"candidate={paths.terminal_candidate}")
    receipts: list[JsonDict] = []
    repository_health = deepcopy(artifact.get("repository_health", {}))
    for index, (command, scope, required) in enumerate(
        _validation_commands(paths.terminal_candidate), start=1
    ):
        _progress(7, "before subprocess", f"{index} {shlex.join(command)}")
        receipt, output = _command_receipt(command, scope=scope)
        log_path = paths.validation_dir / f"{index:02d}.log"
        _atomic_write(log_path, output.encode("utf-8"))
        receipt["log_path"] = str(log_path)
        receipt["required_for_task"] = required
        receipts.append(receipt)
        _progress(
            7,
            "after subprocess",
            f"{index} exit={receipt['exit_code']} elapsed={receipt['duration_s']:.3f}s",
        )
        if scope == "repository-wide pytest observation required by task":
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
    receipt = write_artifact(paths.artifact, artifact)
    _progress(8, "end", f"terminal sha256={receipt['sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin script owns normal execution.
    raise SystemExit(main())
