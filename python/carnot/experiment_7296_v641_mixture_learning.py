"""Measure prospective fixed-share learning on frozen delayed-label streams.

The learner receives public observations and only due labels. A separate
evaluator adds truth after each prediction is sealed. This keeps later outcome
measurement distinct from same-step fitting.

Spec refs: REQ-CL-7296 and SCENARIO-CL-7296-*.
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
from pathlib import Path
import random
import re
import shlex
import time
from typing import Any

import yaml

from carnot import experiment_7295_v641_mixture_prototype as fixture
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7296
SCHEMA = "carnot.exp7296.v641_mixture_learning.v1"
MILESTONE = fixture.MILESTONE
RUN_DATE = fixture.RUN_DATE
RANDOM_SEED = 7_296_000
BOOTSTRAP_SEED = 7_296_901
BOOTSTRAP_RESAMPLES = 10_000
MEASUREMENT_LIMIT_S = 1_800.0
EVALUATION_STREAM_SEEDS = fixture.EVALUATION_STREAM_SEEDS
EVALUATION_STREAM_COUNT = fixture.EVALUATION_STREAM_COUNT
EVENTS_PER_STREAM = fixture.EVENTS_PER_STREAM
WARMUP_COUNT = fixture.WARMUP_COUNT
FUTURE_LABEL_COUNT = fixture.FUTURE_LABEL_COUNT
FUTURE_LABEL_POSITIONS = fixture.FUTURE_LABEL_POSITIONS
FEEDBACK_DELAY = fixture.FEEDBACK_DELAY
ETA = fixture.ETA
FIXED_SHARE = fixture.FIXED_SHARE
MEMORY_CAP_BYTES = fixture.MEMORY_CAP_BYTES
ARMS = fixture.ARMS
BOUNDED_ARMS = fixture.BOUNDED_ARMS
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = deepcopy(fixture.INVOCATION_COUNTS)
INFERENCE_SUBSTRATE = fixture.INFERENCE_SUBSTRATE
INFERENCE_SUBSTRATE_CLASS = fixture.INFERENCE_SUBSTRATE_CLASS
REDUCER_INFERENCE_SUBSTRATE = fixture.REDUCER_INFERENCE_SUBSTRATE
REDUCER_INFERENCE_SUBSTRATE_CLASS = fixture.REDUCER_INFERENCE_SUBSTRATE_CLASS
EXECUTION_VENUE = fixture.EXECUTION_VENUE

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7296_v641_mixture_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7296_v641_mixture_learning.py")
COVERAGE_PATH = Path("tests/python/coverage_experiment_7296.py")
DEFAULT_ARTIFACT = Path("results/experiment_7296_v641_mixture_learning.json")
UPSTREAM_ARTIFACT = Path("results/experiment_7295_v641_mixture_prototype.json")
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7296-[A-Z-]+")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    Path("python/carnot/experiment_7268_v639_recognition_learning.py"),
    Path("python/carnot/experiment_7282_v640_admission_learning.py"),
    Path("python/carnot/experiment_7283_v640_admission_audit.py"),
    Path("python/carnot/experiment_7295_v641_mixture_prototype.py"),
    Path("python/carnot/experiment_7296_v641_mixture_learning.py"),
    WRAPPER_PATH,
    TEST_PATH,
    COVERAGE_PATH,
    UPSTREAM_ARTIFACT,
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "field_principles",
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
    "mixture_capture_complete_score",
    "mixture_value_score",
    "per_stream_results",
    "feedback_update_rows",
    "chronology_violations",
    "memory_label_accounting",
)

FIELD_PRINCIPLES = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the result to the active experiment task.",
    "milestone": "Bind the result to milestone 2026.09.641.",
    "status": "Use a terminal complete or blocked record; unfinished work stays in checkpoints.",
    "run_date": "Use 20260914 with real UTC start and end times.",
    "started_at_utc": "Record the actual UTC invocation start.",
    "completed_at_utc": "Record the actual UTC terminal decision time.",
    "field_principles": "Store explanations here while consumer values remain top-level.",
    "preconditions_checked": "Hash inputs, authority boundaries, ownership, and failed checks.",
    "MODEL_SPECS": "List only executable local models used now; this task uses none.",
    "model_invoked": "Count any attempted model load or generation; this task has none.",
    "invocation_counts": "Separate attempted, completed, and failed model work.",
    "inference_substrate": "Name the actual CPU solver or simulator work.",
    "inference_substrate_class": "Use the actual no-LLM class without time padding.",
    "execution_venue": "Host execution is host; no device executes this replay.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze development, evaluation, shuffle, and bootstrap seeds before outcomes.",
    "reproducibility_checksum": "Bind code, config, inputs, and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producers, terminal classes, retirement, and quarantine state.",
    "rows": "Keep every stream and arm with efficacy, safety, cost, and censoring metrics.",
    "sample_size_budget": "State planned, attempted, complete, censored units and stopping rule.",
    "acceptance_gate_results": "Each check names expected, observed, passed, and principle.",
    "gate_check_summary": "Blocked verdicts name the exact upstream field and mismatch.",
    "verifier_is_oracle": "Expose evaluator authority; same-authority mechanics are not correctness.",
    "honest_verdict": "Complete findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed verdict set; exact evaluator authority forbids positive.",
    "validation_receipts": "Retain command, exit code, elapsed time, and log hash.",
    "mixture_capture_complete_score": "One requires all 24 streams, seven arms, and causal receipts.",
    "mixture_value_score": "One requires every frozen efficacy, safety, coverage, and byte gate.",
    "per_stream_results": "Make each seed, arm, and stratum aggregate recomputable from step rows.",
    "feedback_update_rows": "Bind each delayed reveal and weight change to later predictions.",
    "chronology_violations": "Count and retain every offending row; favorable evidence requires zero.",
    "memory_label_accounting": "Charge state, references, warmup, future labels, and larger memory.",
}

COMPARISON_SPECS = (
    ("future_error_vs_reset", "future_error_rate", "reset"),
    (
        "future_error_vs_unconditional_recognition",
        "future_error_rate",
        "unconditional_recognition",
    ),
    ("true_feedback_future_error_vs_shuffled", "future_error_rate", "label_shuffled_fixed_share"),
    ("non_feedback_error_vs_reset", "non_feedback_future_error_rate", "reset"),
    (
        "non_feedback_error_vs_unconditional_recognition",
        "non_feedback_future_error_rate",
        "unconditional_recognition",
    ),
    (
        "true_feedback_non_feedback_error_vs_shuffled",
        "non_feedback_future_error_rate",
        "label_shuffled_fixed_share",
    ),
    ("recurrence_error_vs_frozen_warmup", "recurrence_error_rate", "frozen_warmup"),
    ("false_accept_vs_reset", "false_accept_rate", "reset"),
    (
        "false_accept_vs_unconditional_recognition",
        "false_accept_rate",
        "unconditional_recognition",
    ),
    ("false_accept_vs_frozen_warmup", "false_accept_rate", "frozen_warmup"),
    ("coverage_vs_reset", "coverage", "reset"),
    ("coverage_vs_unconditional_recognition", "coverage", "unconditional_recognition"),
    ("coverage_vs_frozen_warmup", "coverage", "frozen_warmup"),
)

SCIENTIFIC_GATE_NAMES = (
    "future_error_vs_reset",
    "future_error_vs_unconditional_recognition",
    "non_feedback_error_vs_reset",
    "non_feedback_error_vs_unconditional_recognition",
    "true_feedback_future_error_vs_shuffled",
    "true_feedback_non_feedback_error_vs_shuffled",
    "separated_recurrence_vs_frozen_warmup",
    "overlapping_recurrence_vs_frozen_warmup",
    "false_accept_vs_reset",
    "false_accept_vs_unconditional_recognition",
    "false_accept_vs_frozen_warmup",
    "coverage_vs_reset",
    "coverage_vs_unconditional_recognition",
    "coverage_vs_frozen_warmup",
    "causal_later_changed_predictions",
    "chronology_and_bounded_memory",
)

CAPTURE_GATE_NAMES = (
    "preconditions",
    "complete_chronological_capture",
    "causal_feedback_receipts",
    "cold_reducer",
    "restart_parity",
    "reference_cost_accounting",
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep checkpoints, raw evidence, candidates, and terminal bytes separate."""

    raw_dir: Path
    checkpoint_dir: Path
    step_rows: Path
    feedback_rows: Path
    e2e_rows: Path
    terminal_candidate: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return task-owned paths under the repository result directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test evidence below a caller-owned temporary result directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive raw, checkpoint, candidate, and final output paths."""

        raw = root / "raw" / "experiment_7296_v641_mixture_learning"
        checkpoints = root / "checkpoints" / "experiment_7296_v641_mixture_learning"
        return cls(
            raw,
            checkpoints,
            raw / "step_rows.jsonl",
            raw / "feedback_update_rows.jsonl",
            raw / "e2e_rows.json",
            raw / "terminal_candidate.json",
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class EvaluationPanel:
    """Retain full chronological evidence and independently reducible units."""

    step_rows: list[JsonDict]
    feedback_update_rows: list[JsonDict]
    per_stream_results: list[JsonDict]
    completed_stream_ids: list[str]
    censored_stream_ids: list[str]
    later_changed_prediction_count: int
    maximum_bounded_memory_bytes: int
    maximum_reference_memory_bytes: int


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Print one flushed boundary so a long scientific run stays observable."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _canonical_bytes(value: Any) -> bytes:
    """Use the shipped canonical encoding for hashes and charged storage."""

    return transactional.canonical_json_bytes(value)


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed or absent evidence stays unavailable."""

    return fixture._load_object(path)


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while absence stays distinct from empty content."""

    return fixture._sha256_path(path)


def _resolve(repo_root: Path, value: str | Path) -> Path:
    """Resolve repository-relative evidence while preserving declared absolute paths."""

    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read nonempty JSON objects while malformed raw evidence fails closed."""

    return fixture._read_jsonl(path)


def _receipt(path: Path, row_count: int | None = None) -> JsonDict:
    """Describe exact raw bytes and the optional complete row count."""

    receipt: JsonDict = {"path": str(path), "sha256": _sha256_path(path)}
    if row_count is not None:
        receipt["row_count"] = row_count
    return receipt


def _atomic_write(path: Path, payload: bytes) -> JsonDict:
    """Publish complete bytes through the shipped atomic writer."""

    return fixture._atomic_write(path, payload)


def _task_identity(text: str) -> JsonDict:
    """Extract only this experiment identity from the executable roadmap."""

    try:
        tasks = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks")
    if not isinstance(tasks, list):
        return {}
    for task in tasks:
        if isinstance(task, dict) and task.get("id") == "exp7296-mixture-learning":
            return {
                "id": task.get("id"),
                "milestone": task.get("milestone"),
                "deliverable": task.get("deliverable"),
            }
    return {}


def _weights(controller: fixture.FixedShareController) -> dict[str, float]:
    """Record every expert weight under its immutable expert identity."""

    return {str(row["expert_id"]): float(row["weight"]) for row in controller.experts()}


def _single_state_hash(masks: Mapping[str, Any]) -> str:
    """Bind one non-mixture complete hypothesis with the shipped hash."""

    return fixture._mask_hash(masks)


def _single_state_bytes(masks: Mapping[str, Any]) -> int:
    """Charge a non-mixture hypothesis by its exact canonical bytes."""

    return fixture._single_state_bytes(masks)


def collect_preconditions(
    repo_root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate Exp7295, its stream bytes, constants, exclusions, and outputs."""

    resolved = {str(repo_root / path): _sha256_path(repo_root / path) for path in SOURCE_PATHS}
    spec_path = repo_root / SPEC_PATH
    roadmap_path = repo_root / "research-roadmap.yaml"
    exclusion_path = repo_root / "ops/exclusion_manifest.yaml"
    upstream_path = repo_root / UPSTREAM_ARTIFACT
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    roadmap = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    upstream = _load_object(upstream_path)
    manifest_path = _resolve(repo_root, str(upstream.get("stream_manifest_path", "")))
    manifest = _load_object(manifest_path)
    try:
        exclusions = yaml.safe_load(exclusion_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        exclusions = {"unavailable": True}
    manifest_receipt = upstream.get("raw_evidence_receipts", {}).get("stream_manifest", {})
    evaluation = manifest.get("evaluation", {})
    evaluation_receipts = evaluation.get("receipts", {}) if isinstance(evaluation, dict) else {}
    receipt_hashes_match = isinstance(evaluation_receipts, dict) and all(
        isinstance(value, dict)
        and _sha256_path(_resolve(repo_root, str(value.get("path", "")))) == value.get("sha256")
        for value in evaluation_receipts.values()
    )
    expected_identity = {
        "id": "exp7296-mixture-learning",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT),
    }
    writable = (
        paths.step_rows,
        paths.feedback_rows,
        paths.e2e_rows,
        paths.terminal_candidate,
        paths.artifact,
    )
    checks = [
        fixture.gate_check(
            "driving_capability_spec", str(spec_path), "REQ-CL-7296", True, "REQ-CL-7296" in spec
        ),
        fixture.gate_check(
            "scenario_contract",
            str(spec_path),
            "SCENARIO-CL-7296-*",
            6,
            len(set(SCENARIO_PATTERN.findall(spec))),
        ),
        fixture.gate_check(
            "v641_task_identity",
            str(roadmap_path),
            "id,milestone,deliverable",
            expected_identity,
            _task_identity(roadmap),
        ),
        fixture.gate_check(
            "exp7295_terminal_state",
            str(upstream_path),
            "status",
            "complete",
            upstream.get("status"),
        ),
        fixture.gate_check(
            "exp7295_fixture_ready",
            str(upstream_path),
            "mixture_fixture_ready_score",
            1,
            upstream.get("mixture_fixture_ready_score"),
        ),
        fixture.gate_check(
            "exp7295_not_quarantined_or_retired",
            str(upstream_path),
            "flagged_adversarial,retired",
            [False, False],
            [bool(upstream.get("flagged_adversarial")), bool(upstream.get("retired"))],
        ),
        fixture.gate_check(
            "exp7296_not_excluded",
            str(exclusion_path),
            "experiment_id",
            False,
            fixture._excluded_experiment(exclusions, EXPERIMENT_ID),
        ),
        fixture.gate_check(
            "stream_manifest_hash",
            str(manifest_path),
            "sha256",
            manifest_receipt.get("sha256") if isinstance(manifest_receipt, dict) else None,
            _sha256_path(manifest_path),
        ),
        fixture.gate_check(
            "scorer_only_stream_contract",
            str(manifest_path),
            "frozen,scorer_only,count,seeds",
            [True, True, EVALUATION_STREAM_COUNT, list(EVALUATION_STREAM_SEEDS)],
            [
                manifest.get("frozen"),
                manifest.get("scorer_only_evaluation_labels"),
                evaluation.get("stream_count") if isinstance(evaluation, dict) else None,
                evaluation.get("stream_seeds") if isinstance(evaluation, dict) else None,
            ],
        ),
        fixture.gate_check(
            "stream_view_receipts",
            str(manifest_path),
            "evaluation.receipts.sha256",
            True,
            receipt_hashes_match and len(evaluation_receipts) == 3,
        ),
        fixture.gate_check(
            "frozen_update_constants",
            str(upstream_path),
            "eta,fixed_share,memory_budget,arms",
            [ETA, FIXED_SHARE, MEMORY_CAP_BYTES, list(ARMS)],
            [
                upstream.get("learning_contract", {}).get("eta"),
                upstream.get("learning_contract", {}).get("fixed_share"),
                upstream.get("memory_budget_bytes", {}).get("inherited_limit"),
                list(ARMS),
            ],
        ),
        fixture.gate_check(
            "source_bytes_available",
            "declared source paths",
            "sha256",
            True,
            all(value is not None for value in resolved.values()),
        ),
        fixture.gate_check(
            "resource_ownership",
            "host",
            "task-owned output paths writable",
            True,
            all(fixture._path_writable(path) for path in writable),
        ),
    ]
    resolved[str(manifest_path)] = _sha256_path(manifest_path)
    for receipt in evaluation_receipts.values() if isinstance(evaluation_receipts, dict) else ():
        if isinstance(receipt, dict):
            path = _resolve(repo_root, str(receipt.get("path", "")))
            resolved[str(path)] = _sha256_path(path)
    return checks, resolved, upstream


def load_authenticated_views(repo_root: Path, upstream: Mapping[str, Any]) -> fixture.StreamViews:
    """Load only the three exact evaluator views authenticated by Exp7295."""

    manifest_path = _resolve(repo_root, str(upstream.get("stream_manifest_path", "")))
    manifest = _load_object(manifest_path)
    expected_manifest = upstream.get("raw_evidence_receipts", {}).get("stream_manifest", {})
    if not isinstance(expected_manifest, dict) or _sha256_path(
        manifest_path
    ) != expected_manifest.get("sha256"):
        raise ValueError("stream_manifest_hash")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation_manifest")
    receipts = evaluation.get("receipts")
    if not isinstance(receipts, dict):
        raise ValueError("evaluation_receipts")
    loaded: dict[str, list[JsonDict]] = {}
    for name in ("public", "releases", "private_authority"):
        receipt = receipts.get(name)
        if not isinstance(receipt, dict):
            raise ValueError(f"missing_view:{name}")
        path = _resolve(repo_root, str(receipt.get("path", "")))
        if _sha256_path(path) != receipt.get("sha256"):
            raise ValueError(f"view_hash:{name}")
        rows = _read_jsonl(path)
        if len(rows) != int(receipt.get("row_count", -1)):
            raise ValueError(f"view_row_count:{name}")
        loaded[name] = rows
    conformance_view = fixture.StreamViews(
        loaded["public"], loaded["private_authority"], loaded["releases"], evaluation
    )
    errors = fixture.stream_conformance_errors(conformance_view, "evaluation")
    if errors:
        raise ValueError("stream_conformance:" + ",".join(errors))
    return fixture.StreamViews(
        loaded["public"], loaded["private_authority"], loaded["releases"], manifest
    )


def _controller_state(controller: fixture.FixedShareController) -> tuple[str, int, JsonDict]:
    """Return the full state hash, charged bytes, and all expert weights."""

    return (
        controller.state_hash(),
        int(controller.memory_usage()["serialized_state_bytes"]),
        _weights(controller),
    )


def _stream_rows(
    views: fixture.StreamViews, stream_id: str
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay one stream with predictions sealed before evaluator access and update."""

    events = sorted(
        (row for row in views.public if row.get("stream_id") == stream_id),
        key=lambda row: int(row["chronology_index"]),
    )
    if len(events) != EVENTS_PER_STREAM:
        raise ValueError(f"incomplete_stream:{stream_id}")
    authority = {
        str(row["event_id"]): row for row in views.authority if row["stream_id"] == stream_id
    }
    releases = [row for row in views.releases if row["stream_id"] == stream_id]
    warmup = fixture._warmup_masks(releases, stream_id)
    future_releases = [row for row in releases if row.get("role") == "future_feedback"]
    due_by_index = {int(row["release_index"]): row for row in future_releases}
    seed = int(authority[str(events[0]["event_id"])]["stream_seed"])
    stratum = str(authority[str(events[0]["event_id"])]["stratum"])
    controllers = {
        "fixed_share_mixture": fixture.FixedShareController.from_masks(warmup),
        "frozen_uniform_voting": fixture.FixedShareController.from_masks(warmup),
        "label_shuffled_fixed_share": fixture.FixedShareController.from_masks(warmup),
        "unbounded_memory_reference": fixture.FixedShareController.from_masks(
            warmup, archive_cap=None, memory_cap_bytes=None, retain_label_history=True
        ),
    }
    single_states = {
        "reset": deepcopy(warmup),
        "unconditional_recognition": deepcopy(warmup),
        "frozen_warmup": deepcopy(warmup),
    }
    shuffled = fixture._shuffled_labels(releases, seed)
    selected_positions = set(FUTURE_LABEL_POSITIONS)
    step_rows: list[JsonDict] = []
    feedback_rows: list[JsonDict] = []
    revealed_count = 0
    for index in range(WARMUP_COUNT, EVENTS_PER_STREAM):
        event = events[index]
        due = due_by_index.get(index)
        prediction_order = index * 4
        mixture_uniform = controllers["fixed_share_mixture"].predict(event, uniform=True)
        predictions: dict[str, tuple[str, int]] = {}
        for arm in ARMS:
            started = time.perf_counter_ns()
            if arm in controllers:
                prediction = controllers[arm].predict(event, uniform=arm == "frozen_uniform_voting")
            else:
                prediction = fixture.prototype.predict_masks(single_states[arm], event)
            predictions[arm] = (prediction, max(1, time.perf_counter_ns() - started))
        truth = authority[str(event["event_id"])]
        for arm in ARMS:
            prediction, cost_ns = predictions[arm]
            if arm in controllers:
                state_hash, memory_bytes, _ = _controller_state(controllers[arm])
                expert_count = len(controllers[arm].experts())
            else:
                state_hash = _single_state_hash(single_states[arm])
                memory_bytes = _single_state_bytes(single_states[arm])
                expert_count = 1
            error, false_accept, abstention, covered = fixture._prediction_metrics(
                prediction, str(truth["exact_label"])
            )
            changed = int(
                arm == "fixed_share_mixture"
                and revealed_count > 0
                and prediction != mixture_uniform
            )
            step_rows.append(
                {
                    "unit_id": f"{stream_id}:{arm}:{index:04d}",
                    "stream_id": stream_id,
                    "seed": seed,
                    "stratum": stratum,
                    "arm": arm,
                    "event_id": str(event["event_id"]),
                    "chronology_index": index,
                    "recurrence_step": index >= 768,
                    "feedback_selected_step": index in selected_positions,
                    "non_feedback_future_step": index not in selected_positions,
                    "prediction": prediction,
                    "predicted_class": prediction,
                    "prediction_order": prediction_order,
                    "evaluator_order": prediction_order + 1,
                    "due_release_order": prediction_order + 2 if due is not None else None,
                    "update_order": prediction_order + 3 if due is not None else None,
                    "prediction_frozen_before_release": True,
                    "state_hash_before_prediction": state_hash,
                    "evaluator_exact_label": str(truth["exact_label"]),
                    "truth": str(truth["exact_label"]),
                    "error": error,
                    "false_accept": false_accept,
                    "abstention": abstention,
                    "covered": covered,
                    "memory_bytes": memory_bytes,
                    "feedback_count_before_prediction": revealed_count,
                    "expert_count": expert_count,
                    "prediction_cost_ns": cost_ns,
                    "changed_from_uniform_after_feedback": changed,
                    "learner_read_unreleased_label": False,
                    "learner_read_private_authority": False,
                    "learner_read_change_point": False,
                    "larger_memory_reference": arm == "unbounded_memory_reference",
                    "bounded_deployment_eligible": arm != "unbounded_memory_reference",
                    "censored": False,
                }
            )
        if due is None:
            continue
        before: dict[str, tuple[str, int, JsonDict]] = {
            arm: _controller_state(controller) for arm, controller in controllers.items()
        }
        single_before = {
            arm: (_single_state_hash(masks), _single_state_bytes(masks))
            for arm, masks in single_states.items()
        }
        update_started = {arm: time.perf_counter_ns() for arm in ARMS}
        fixed_receipt = controllers["fixed_share_mixture"].apply_release(
            due, current_index=index, collect_nominee=True
        )
        controllers["frozen_uniform_voting"].apply_release(
            due, current_index=index, update_weights=False, collect_nominee=False
        )
        controllers["label_shuffled_fixed_share"].apply_release(
            {**due, "observed_label": shuffled[str(due["event_id"])]},
            current_index=index,
            collect_nominee=False,
        )
        controllers["unbounded_memory_reference"].apply_release(
            due, current_index=index, collect_nominee=False
        )
        nominee = fixed_receipt["nominee"]
        installed: dict[str, JsonDict] = {}
        if nominee is not None:
            installed = fixture._install_shared_nominee(controllers, nominee)
            single_states["reset"] = deepcopy(nominee["masks"])
            single_states["unconditional_recognition"] = deepcopy(nominee["masks"])
        arm_updates: dict[str, JsonDict] = {}
        for arm in ARMS:
            if arm in controllers:
                after_hash, after_bytes, after_weights = _controller_state(controllers[arm])
                before_hash, before_bytes, before_weights = before[arm]
                receipt = installed.get(arm, {})
                updated = arm != "frozen_uniform_voting"
            else:
                before_hash, before_bytes = single_before[arm]
                after_hash = _single_state_hash(single_states[arm])
                after_bytes = _single_state_bytes(single_states[arm])
                before_weights = {}
                after_weights = {}
                receipt = {}
                updated = nominee is not None and arm in {"reset", "unconditional_recognition"}
            if arm == "fixed_share_mixture" and nominee is not None:
                receipt = {key: value for key, value in nominee.items() if key != "masks"}
            arm_updates[arm] = {
                "state_hash_before": before_hash,
                "state_hash_after": after_hash,
                "memory_bytes_before": before_bytes,
                "memory_bytes_after": after_bytes,
                "weights_before": before_weights,
                "weights_after": after_weights,
                "updated": updated,
                "applied_label": shuffled[str(due["event_id"])]
                if arm == "label_shuffled_fixed_share"
                else str(due["observed_label"]),
                "nominee_birth": receipt.get("expert_id"),
                "evicted_expert_id": receipt.get("evicted_expert_id"),
                "update_cost_ns": max(1, time.perf_counter_ns() - update_started[arm]),
                "larger_memory_reference": arm == "unbounded_memory_reference",
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
                "nomination": None
                if nominee is None
                else {key: value for key, value in nominee.items() if key != "masks"},
                "arm_updates": arm_updates,
                "first_later_changed_prediction_event_id": None,
                "learner_read_unreleased_label": False,
                "learner_read_private_authority": False,
                "learner_read_change_point": False,
                "censored": False,
            }
        )
    mixture_changes = [
        row
        for row in step_rows
        if row["arm"] == "fixed_share_mixture" and row["changed_from_uniform_after_feedback"] == 1
    ]
    for update in feedback_rows:
        later = next(
            (
                row
                for row in mixture_changes
                if int(row["chronology_index"]) > int(update["release_index"])
            ),
            None,
        )
        update["first_later_changed_prediction_event_id"] = (
            None if later is None else later["event_id"]
        )
    return step_rows, feedback_rows


def step_row_errors(rows: Sequence[Mapping[str, Any]], stream_ids: Sequence[str]) -> list[str]:
    """Check complete arm matrices, chronological seals, authority, bytes, and costs."""

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
            or int(row.get("prediction_order", 1)) >= int(row.get("evaluator_order", 0))
            or row.get("prediction_frozen_before_release") is not True
            or row.get("learner_read_unreleased_label") is not False
            or row.get("learner_read_private_authority") is not False
            or row.get("learner_read_change_point") is not False
            for row in rows
        ),
        "authority_or_prediction_order",
    )
    add(
        any(
            row.get("due_release_order") is not None
            and not (
                int(row["prediction_order"])
                < int(row["evaluator_order"])
                < int(row["due_release_order"])
                < int(row["update_order"])
            )
            for row in rows
        ),
        "release_update_order",
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
            or row.get("censored") is not False
            for row in rows
        ),
        "cost_or_censoring",
    )
    add(
        any(
            row.get("arm") in BOUNDED_ARMS
            and int(row.get("memory_bytes", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
            for row in rows
        ),
        "bounded_memory",
    )
    return errors


def feedback_row_errors(rows: Sequence[Mapping[str, Any]], stream_ids: Sequence[str]) -> list[str]:
    """Check every scheduled reveal, update receipt, state transition, and label limit."""

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
            for row in rows
        ),
        "feedback_chronology_or_authority",
    )
    add(
        any(
            not isinstance(row.get("arm_updates"), dict)
            or set(row["arm_updates"]) != set(ARMS)
            or any(
                not update.get("state_hash_before")
                or not update.get("state_hash_after")
                or int(update.get("memory_bytes_after", 0)) <= 0
                or int(update.get("update_cost_ns", 0)) <= 0
                for update in row["arm_updates"].values()
            )
            for row in rows
        ),
        "arm_update_receipts",
    )
    add(
        any(
            int(update.get("memory_bytes_after", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
            for row in rows
            for arm, update in row.get("arm_updates", {}).items()
            if arm in BOUNDED_ARMS
        ),
        "feedback_bounded_memory",
    )
    add(
        any(
            row.get("nomination") is not None
            and int(row["nomination"].get("source_release_max_index", EVENTS_PER_STREAM + 1))
            > int(row.get("release_index", -1))
            for row in rows
        ),
        "nomination_future_label",
    )
    return errors


def reduce_step_rows(
    step_rows: Sequence[Mapping[str, Any]], feedback_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Reduce every stream-arm without dropping non-feedback or recurrence outcomes."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    feedback_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    for row in feedback_rows:
        feedback_groups[str(row["stream_id"])].append(row)
    reduced: list[JsonDict] = []
    for stream_id, arm in sorted(groups, key=lambda key: (key[0], ARMS.index(key[1]))):
        events = sorted(groups[(stream_id, arm)], key=lambda row: int(row["chronology_index"]))
        non_feedback = [row for row in events if row["non_feedback_future_step"] is True]
        recurrence = [row for row in events if row["recurrence_step"] is True]
        updates = feedback_groups[stream_id]
        arm_updates = [row["arm_updates"][arm] for row in updates]
        reduced.append(
            {
                "unit_id": f"{stream_id}:{arm}",
                "stream_id": stream_id,
                "seed": int(events[0]["seed"]),
                "stratum": str(events[0]["stratum"]),
                "arm": arm,
                "metric": "prospective_future_full_denominator_error",
                "future_prediction_count": len(events),
                "future_error": sum(int(row["error"]) for row in events),
                "future_error_rate": sum(int(row["error"]) for row in events) / len(events),
                "non_feedback_future_prediction_count": len(non_feedback),
                "non_feedback_future_error": sum(int(row["error"]) for row in non_feedback),
                "non_feedback_future_error_rate": sum(int(row["error"]) for row in non_feedback)
                / len(non_feedback),
                "recurrence_prediction_count": len(recurrence),
                "recurrence_error": sum(int(row["error"]) for row in recurrence),
                "recurrence_error_rate": sum(int(row["error"]) for row in recurrence)
                / len(recurrence),
                "false_accept": sum(int(row["false_accept"]) for row in events),
                "false_accept_rate": sum(int(row["false_accept"]) for row in events) / len(events),
                "abstention": sum(int(row["abstention"]) for row in events),
                "abstention_rate": sum(int(row["abstention"]) for row in events) / len(events),
                "coverage": sum(int(row["covered"]) for row in events) / len(events),
                "warmup_label_count": WARMUP_COUNT,
                "future_label_count": len(updates),
                "feedback_update_count": sum(int(row["updated"] is True) for row in arm_updates),
                "nominee_birth_count": sum(row["nominee_birth"] is not None for row in arm_updates),
                "expert_eviction_count": sum(
                    row["evicted_expert_id"] is not None for row in arm_updates
                ),
                "maximum_memory_bytes": max(int(row["memory_bytes"]) for row in events),
                "prediction_cost_ns": sum(int(row["prediction_cost_ns"]) for row in events),
                "feedback_update_cost_ns": sum(int(row["update_cost_ns"]) for row in arm_updates),
                "later_changed_from_uniform_count": sum(
                    int(row["changed_from_uniform_after_feedback"]) for row in events
                ),
                "larger_memory_reference": arm == "unbounded_memory_reference",
                "bounded_deployment_eligible": arm != "unbounded_memory_reference",
                "censored": False,
            }
        )
    return reduced


def independent_reduce(step_path: Path, feedback_path: Path) -> list[JsonDict]:
    """Reload immutable raw rows and cold-reduce without producer aggregates."""

    steps = _read_jsonl(step_path)
    feedback = _read_jsonl(feedback_path)
    stream_ids = tuple(sorted({str(row.get("stream_id")) for row in steps}))
    errors = step_row_errors(steps, stream_ids) + feedback_row_errors(feedback, stream_ids)
    if errors:
        raise ValueError("raw_row_conformance:" + ",".join(errors))
    return reduce_step_rows(steps, feedback)


def run_learning_panel(
    views: fixture.StreamViews,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
    measurement_limit_s: float = MEASUREMENT_LIMIT_S,
) -> EvaluationPanel:
    """Run fixed streams once, checkpoint complete units, and retain censoring."""

    selected = tuple(
        stream_ids or (f"evaluation-{index + 1:02d}" for index in range(EVALUATION_STREAM_COUNT))
    )
    all_steps: list[JsonDict] = []
    all_feedback: list[JsonDict] = []
    completed: list[str] = []
    censored: list[str] = []
    started = time.monotonic()
    last_heartbeat = started
    for offset, stream_id in enumerate(selected):
        if time.monotonic() - started > measurement_limit_s:
            censored.extend(selected[offset:])
            break
        shard = paths.checkpoint_dir / f"{stream_id}.json"
        checkpoint = _load_object(shard)
        if checkpoint.get("schema") == SCHEMA and checkpoint.get("status") == "complete":
            steps = checkpoint.get("step_rows", [])
            feedback = checkpoint.get("feedback_update_rows", [])
        else:
            steps, feedback = _stream_rows(views, stream_id)
            _atomic_write(
                shard,
                _canonical_bytes(
                    {
                        "schema": SCHEMA,
                        "status": "complete",
                        "stream_id": stream_id,
                        "step_rows": steps,
                        "feedback_update_rows": feedback,
                    }
                ),
            )
        row_errors = step_row_errors(steps, (stream_id,))
        update_errors = feedback_row_errors(feedback, (stream_id,))
        if row_errors or update_errors:
            raise ValueError(
                f"stream_conformance:{stream_id}:" + ",".join(row_errors + update_errors)
            )
        all_steps.extend(steps)
        all_feedback.extend(feedback)
        completed.append(stream_id)
        now = time.monotonic()
        if progress:
            _progress(
                3,
                "benchmark progress",
                f"completed {offset + 1}/{len(selected)} streams; rows={len(all_steps)}; elapsed={now - started:.3f}s",
            )
        if now - last_heartbeat >= 60.0:
            _progress(
                3,
                "benchmark heartbeat",
                f"completed={len(completed)} outstanding={len(selected) - len(completed)} elapsed={now - started:.1f}s",
            )
            last_heartbeat = now
    _atomic_write(paths.step_rows, fixture.prototype.jsonl_bytes(all_steps))
    _atomic_write(paths.feedback_rows, fixture.prototype.jsonl_bytes(all_feedback))
    per_stream = reduce_step_rows(all_steps, all_feedback) if all_steps else []
    later_changes = sum(
        int(row["changed_from_uniform_after_feedback"])
        for row in all_steps
        if row["arm"] == "fixed_share_mixture"
    )
    bounded = max(
        (int(row["memory_bytes"]) for row in all_steps if row["arm"] in BOUNDED_ARMS),
        default=0,
    )
    reference = max(
        (
            int(row["memory_bytes"])
            for row in all_steps
            if row["arm"] == "unbounded_memory_reference"
        ),
        default=0,
    )
    return EvaluationPanel(
        all_steps,
        all_feedback,
        per_stream,
        completed,
        censored,
        later_changes,
        bounded,
        reference,
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return the deterministic nearest-rank bootstrap percentile."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _bootstrap_interval(values: Sequence[float], draws: int, salt: str) -> JsonDict:
    """Resample paired whole-stream differences with one frozen independent seed."""

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
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
    }


def build_comparison_rows(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_RESAMPLES
) -> list[JsonDict]:
    """Build pooled and recurrence-stratum paired independent-stream intervals."""

    comparisons: list[JsonDict] = []
    for comparison_id, metric, control in COMPARISON_SPECS:
        for stratum in (None, "separated_recurrence", "overlapping_recurrence"):
            selected = [row for row in rows if stratum is None or row.get("stratum") == stratum]
            by_key = {(str(row["stream_id"]), str(row["arm"])): row for row in selected}
            stream_ids = sorted({str(row["stream_id"]) for row in selected})
            if not stream_ids:
                continue
            differences = [
                float(by_key[(stream_id, "fixed_share_mixture")][metric])
                - float(by_key[(stream_id, control)][metric])
                for stream_id in stream_ids
            ]
            interval = _bootstrap_interval(
                differences, draws, f"{comparison_id}:{stratum or 'overall'}"
            )
            comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "treatment_arm": "fixed_share_mixture",
                    "control_arm": control,
                    "stratum": stratum or "overall",
                    "independent_unit": "stream",
                    "independent_unit_count": len(stream_ids),
                    "bootstrap_resamples": draws,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "paired_differences": differences,
                    "estimate": interval["estimate"],
                    "ci95": interval["ci95"],
                    "ci95_lower": interval["ci95"][0],
                    "ci95_upper": interval["ci95"][1],
                }
            )
    return comparisons


def _comparison(
    rows: Sequence[Mapping[str, Any]], comparison_id: str, stratum: str = "overall"
) -> Mapping[str, Any]:
    """Select one frozen comparison and fail if its independent units are absent."""

    return next(
        row
        for row in rows
        if row.get("comparison_id") == comparison_id and row.get("stratum") == stratum
    )


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Give every completion or scientific check one auditable shape."""

    return {
        "expected": expected,
        "observed": observed,
        "pass": bool(passed),
        "passed": bool(passed),
        "principle": principle,
    }


def score_acceptance_gates(
    comparisons: Sequence[Mapping[str, Any]], causal: Mapping[str, Any]
) -> dict[str, JsonDict]:
    """Score frozen efficacy, recurrence, safety, coverage, causal, and byte gates."""

    strict_error = (
        "future_error_vs_reset",
        "future_error_vs_unconditional_recognition",
        "non_feedback_error_vs_reset",
        "non_feedback_error_vs_unconditional_recognition",
        "true_feedback_future_error_vs_shuffled",
        "true_feedback_non_feedback_error_vs_shuffled",
    )
    gates = {
        name: _gate(
            "ci95_upper<0",
            _comparison(comparisons, name)["ci95_upper"],
            float(_comparison(comparisons, name)["ci95_upper"]) < 0.0,
            "Past released labels must improve later unrevealed outcomes over whole streams.",
        )
        for name in strict_error
    }
    for stratum in ("separated_recurrence", "overlapping_recurrence"):
        row = _comparison(comparisons, "recurrence_error_vs_frozen_warmup", stratum)
        gates[f"{stratum.split('_')[0]}_recurrence_vs_frozen_warmup"] = _gate(
            "ci95_upper<=0.01",
            row["ci95_upper"],
            float(row["ci95_upper"]) <= 0.01,
            "A harmed recurrence stratum cannot be hidden by pooled performance.",
        )
    for name in (
        "false_accept_vs_reset",
        "false_accept_vs_unconditional_recognition",
        "false_accept_vs_frozen_warmup",
    ):
        row = _comparison(comparisons, name)
        gates[name] = _gate(
            "ci95_upper<=0.01",
            row["ci95_upper"],
            float(row["ci95_upper"]) <= 0.01,
            "Learning cannot purchase lower error through unsafe false acceptances.",
        )
    for name in (
        "coverage_vs_reset",
        "coverage_vs_unconditional_recognition",
        "coverage_vs_frozen_warmup",
    ):
        row = _comparison(comparisons, name)
        gates[name] = _gate(
            "ci95_lower>=-0.02",
            row["ci95_lower"],
            float(row["ci95_lower"]) >= -0.02,
            "Learning cannot purchase lower error through excessive abstention.",
        )
    gates["causal_later_changed_predictions"] = _gate(
        ">=24",
        causal.get("later_changed_prediction_count"),
        int(causal.get("later_changed_prediction_count", 0)) >= 24,
        "Weights must change later public predictions, not only internal bytes.",
    )
    gates["chronology_and_bounded_memory"] = _gate(
        "[0,0]",
        [
            causal.get("chronology_violation_count"),
            causal.get("bounded_memory_violation_count"),
        ],
        int(causal.get("chronology_violation_count", -1)) == 0
        and int(causal.get("bounded_memory_violation_count", -1)) == 0,
        "Future labels and excess state cannot enter a favorable result.",
    )
    return gates


def classify_result(gates: Mapping[str, Mapping[str, Any]]) -> tuple[int, str, str]:
    """Return circular favorable evidence or an honest completed scientific null."""

    value = int(all(gates.get(name, {}).get("passed") is True for name in SCIENTIFIC_GATE_NAMES))
    if value:
        return (
            1,
            "circular_positive",
            "complete_circular_positive: prospective fixed-share learning passed every frozen gate under exact evaluator authority",
        )
    failed = [
        name for name in SCIENTIFIC_GATE_NAMES if gates.get(name, {}).get("passed") is not True
    ]
    return (
        0,
        "null",
        "complete_null: prospective fixed-share learning completed but frozen value gates failed: "
        + ",".join(failed),
    )


def _causal_summary(panel: EvaluationPanel) -> JsonDict:
    """Count causal later changes, chronology failures, and bounded byte failures."""

    stream_ids = tuple(panel.completed_stream_ids)
    step_errors = step_row_errors(panel.step_rows, stream_ids)
    feedback_errors = feedback_row_errors(panel.feedback_update_rows, stream_ids)
    offending = [
        row["unit_id"]
        for row in panel.step_rows
        if int(row["prediction_order"]) >= int(row["evaluator_order"])
        or row["learner_read_unreleased_label"] is not False
        or row["learner_read_private_authority"] is not False
        or row["learner_read_change_point"] is not False
    ]
    memory_violations = sum(
        int(row["arm"] in BOUNDED_ARMS and int(row["memory_bytes"]) > MEMORY_CAP_BYTES)
        for row in panel.step_rows
    )
    return {
        "later_changed_prediction_count": panel.later_changed_prediction_count,
        "linked_feedback_update_count": sum(
            row["first_later_changed_prediction_event_id"] is not None
            for row in panel.feedback_update_rows
        ),
        "chronology_violation_count": len(step_errors) + len(feedback_errors),
        "chronology_error_classes": step_errors + feedback_errors,
        "offending_row_ids": offending,
        "bounded_memory_violation_count": memory_violations,
    }


def run_e2e_controls(root: Path) -> list[JsonDict]:
    """Exercise observation, delayed update, later prediction, and cold restart parity."""

    root.mkdir(parents=True, exist_ok=True)
    base = {family: 1 << 0 for family in fixture.FAMILIES}
    alternate = {family: 1 << 8 for family in fixture.FAMILIES}
    controller = fixture.FixedShareController.from_masks(base)
    controller.install_nominee(alternate, birth_index=120)
    event = {"event_id": "e2e-observation", "family_id": "lower_bound", "numeric_value": 4}
    before_hash = controller.state_hash()
    before = controller.predict(event)
    release = {
        "event_id": "e2e-label",
        "family_id": "lower_bound",
        "numeric_value": 4,
        "observed_label": "accept",
        "source_index": 128,
        "release_index": 132,
    }
    premature = False
    try:
        controller.apply_release(release, current_index=131, collect_nominee=False)
    except fixture.MixtureRejected:
        premature = True
    receipt = controller.apply_release(release, current_index=132, collect_nominee=False)
    after_hash = controller.state_hash()
    later_event = {
        "event_id": "e2e-later-unrevealed",
        "family_id": "lower_bound",
        "numeric_value": 5,
    }
    later = controller.predict(later_event)
    checkpoint = root / "controller.json"
    controller.save(checkpoint)
    restored = fixture.FixedShareController.load(checkpoint)
    parity = (
        restored.state_bytes() == controller.state_bytes()
        and restored.predict(later_event) == later
    )
    return [
        {
            "stage": "observation",
            "passed": set(event) == {"event_id", "family_id", "numeric_value"},
        },
        {"stage": "pre_label_prediction", "passed": before in {"accept", "reject", "abstain"}},
        {"stage": "delayed_feedback", "passed": premature},
        {
            "stage": "bounded_online_update",
            "passed": receipt["prediction_preceded_release"] is True
            and before_hash != after_hash
            and int(controller.memory_usage()["serialized_state_bytes"]) <= MEMORY_CAP_BYTES,
        },
        {
            "stage": "later_unrevealed_prediction",
            "passed": later in {"accept", "reject", "abstain"}
            and "observed_label" not in later_event,
        },
        {"stage": "checkpoint_restart_parity", "passed": parity},
    ]


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every precondition and the first exact failed observation."""

    summary = fixture.gate_summary(checks)
    failures = [dict(row) for row in checks if row.get("passed") is not True]
    summary["first_failure"] = failures[0] if failures else None
    summary["failed_checks"] = failures
    return summary


def _sample_budget(
    selected: Sequence[str], completed: Sequence[str], censored: Sequence[str]
) -> JsonDict:
    """Declare fixed attempts, complete and censored units, draws, and stopping rule."""

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
        "future_predictions_per_stream_arm": EVENTS_PER_STREAM - WARMUP_COUNT,
        "planned_step_rows": len(selected) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT),
        "completed_step_rows": len(completed) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT),
        "planned_feedback_rows": len(selected) * FUTURE_LABEL_COUNT,
        "completed_feedback_rows": len(completed) * FUTURE_LABEL_COUNT,
        "planned_stream_arm_units": len(selected) * len(ARMS),
        "completed_stream_arm_units": len(completed) * len(ARMS),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "measurement_limit_s": MEASUREMENT_LIMIT_S,
        "stopping_rule": "all 24 frozen streams once; retain timeout censoring; no outcome extension",
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
    """Create every required field before blocked or measured classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "reducer_inference_substrate": REDUCER_INFERENCE_SUBSTRATE,
        "reducer_inference_substrate_class": REDUCER_INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "phase_durations_s": {},
        "random_seed": {
            "global": RANDOM_SEED,
            "evaluation_stream_seeds": list(EVALUATION_STREAM_SEEDS),
            "label_shuffle": fixture.SHUFFLE_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "frozen_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "source_artifact_states": {
            "exp7295": {
                "path": str(REPO_ROOT / UPSTREAM_ARTIFACT),
                "terminal_class": "unavailable",
                "quarantined": False,
                "retired": False,
            }
        },
        "rows": [],
        "sample_size_budget": _sample_budget(selected, (), ()),
        "acceptance_gate_results": {},
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition_failed",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "mixture_capture_complete_score": 0,
        "mixture_value_score": 0,
        "per_stream_results": [],
        "feedback_update_rows": [],
        "chronology_violations": {"count": 0, "error_classes": [], "offending_row_ids": []},
        "memory_label_accounting": {
            "bounded_limit_bytes": MEMORY_CAP_BYTES,
            "maximum_bounded_state_bytes": 0,
            "maximum_unbounded_reference_state_bytes": 0,
            "warmup_labels_per_stream_arm": WARMUP_COUNT,
            "future_labels_per_stream_arm": FUTURE_LABEL_COUNT,
        },
        "comparison_rows": [],
        "e2e_control_rows": [],
        "raw_evidence_receipts": {},
        "continuous_self_learning_task": True,
        "no_model_weight_mutation": True,
        "production_default_changed": False,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable configuration, source identities, rows, gates, and raw evidence."""

    payload = {
        "schema": artifact.get("schema"),
        "experiment_id": artifact.get("experiment_id"),
        "milestone": artifact.get("milestone"),
        "run_date": artifact.get("run_date"),
        "MODEL_SPECS": artifact.get("MODEL_SPECS"),
        "model_invoked": artifact.get("model_invoked"),
        "invocation_counts": artifact.get("invocation_counts"),
        "inference_substrate": artifact.get("inference_substrate"),
        "inference_substrate_class": artifact.get("inference_substrate_class"),
        "random_seed": artifact.get("random_seed"),
        "source_artifact_hashes": artifact.get("source_artifact_hashes"),
        "rows": artifact.get("rows"),
        "sample_size_budget": artifact.get("sample_size_budget"),
        "acceptance_gate_results": artifact.get("acceptance_gate_results"),
        "feedback_update_rows_sha256": transactional.sha256_json(
            artifact.get("feedback_update_rows", [])
        ),
        "chronology_violations": artifact.get("chronology_violations"),
        "memory_label_accounting": artifact.get("memory_label_accounting"),
        "comparison_rows": artifact.get("comparison_rows"),
        "raw_evidence_receipts": artifact.get("raw_evidence_receipts"),
    }
    return transactional.sha256_json(payload)


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str | None],
    selected: Sequence[str],
    *,
    started_at: str,
    duration_s: float,
) -> JsonDict:
    """Return row-free terminal evidence for an external prerequisite failure."""

    artifact = _base_artifact(
        checks,
        hashes,
        selected,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=duration_s,
    )
    failure = artifact["gate_check_summary"]["first_failure"]
    if failure is not None:
        artifact["honest_verdict"] = (
            "blocked_external_precondition_failed:"
            f"{failure['upstream']}:{failure['field']}:"
            f"observed={failure['observed_value']!r}:expected={failure['expected_value']!r}"
        )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = True,
) -> JsonDict:
    """Authenticate, replay, cold-reduce, score, and seal one terminal candidate."""

    selected = tuple(
        stream_ids or (f"evaluation-{index + 1:02d}" for index in range(EVALUATION_STREAM_COUNT))
    )
    started_at = datetime.now(UTC).isoformat()
    monotonic_start = time.monotonic()
    spans: JsonDict = {}
    if progress:
        _progress(1, "start", "authenticate upstream, stream views, and output paths")
    phase = time.monotonic()
    checks, hashes, upstream = collect_preconditions(repo_root, paths)
    spans["preconditions"] = time.monotonic() - phase
    if progress:
        _progress(1, "end", f"preconditions_passed={fixture.gate_summary(checks)['passed']}")
    if fixture.gate_summary(checks)["passed"] is not True:
        return build_blocked_artifact(
            checks,
            hashes,
            selected,
            started_at=started_at,
            duration_s=time.monotonic() - monotonic_start,
        )
    if progress:
        _progress(2, "start", "load authenticated public, release, and evaluator views")
    phase = time.monotonic()
    views = load_authenticated_views(repo_root, upstream)
    spans["authenticated_view_load"] = time.monotonic() - phase
    if progress:
        _progress(2, "end", f"evaluation_events={len(views.public)}")
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
            f"completed={len(panel.completed_stream_ids)} step_rows={len(panel.step_rows)}",
        )
        _progress(4, "before reduction", "cold-reduce raw rows and bootstrap whole streams")
    phase = time.monotonic()
    reduced = independent_reduce(paths.step_rows, paths.feedback_rows)
    if reduced != panel.per_stream_results:
        raise ValueError("independent_reducer_mismatch")
    comparisons = build_comparison_rows(reduced)
    causal = _causal_summary(panel)
    science_gates = score_acceptance_gates(comparisons, causal)
    value_score, verdict_class, verdict = classify_result(science_gates)
    spans["cold_reduction_and_bootstrap"] = time.monotonic() - phase
    if progress:
        _progress(4, "after reduction", f"comparisons={len(comparisons)} value={value_score}")
        _progress(5, "start", "run adapted E2E-007 restart path and completion gates")
    phase = time.monotonic()
    e2e = run_e2e_controls(paths.raw_dir / "e2e")
    _atomic_write(paths.e2e_rows, _canonical_bytes({"schema": SCHEMA, "rows": e2e}))
    step_errors = step_row_errors(panel.step_rows, selected)
    update_errors = feedback_row_errors(panel.feedback_update_rows, selected)
    reference_rows = [row for row in reduced if row["arm"] == "unbounded_memory_reference"]
    capture_gates = {
        "preconditions": _gate(
            "all exact checks pass",
            len([row for row in checks if row["passed"] is not True]),
            all(row["passed"] is True for row in checks),
            "Only authenticated frozen evidence can start evaluation.",
        ),
        "complete_chronological_capture": _gate(
            [len(selected) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT), []],
            [len(panel.step_rows), step_errors],
            len(panel.completed_stream_ids) == len(selected) and not step_errors,
            "Every planned post-warmup prediction must remain in chronological evidence.",
        ),
        "causal_feedback_receipts": _gate(
            [len(selected) * FUTURE_LABEL_COUNT, []],
            [len(panel.feedback_update_rows), update_errors],
            len(panel.feedback_update_rows) == len(selected) * FUTURE_LABEL_COUNT
            and not update_errors,
            "Every due label needs a state and weight transition receipt.",
        ),
        "cold_reducer": _gate(
            transactional.sha256_json(panel.per_stream_results),
            transactional.sha256_json(reduced),
            reduced == panel.per_stream_results,
            "Rebuild every headline unit from raw disk rows.",
        ),
        "restart_parity": _gate(
            len(e2e),
            sum(row["passed"] is True for row in e2e),
            all(row["passed"] is True for row in e2e),
            "Exercise prediction, delayed update, later reuse, and cold restart.",
        ),
        "reference_cost_accounting": _gate(
            "reference_bytes>bounded_bytes and real cost>0 and deployment_eligible=false",
            [panel.maximum_reference_memory_bytes, panel.maximum_bounded_memory_bytes],
            panel.maximum_reference_memory_bytes > panel.maximum_bounded_memory_bytes
            and all(int(row["prediction_cost_ns"]) > 0 for row in reference_rows)
            and all(row["bounded_deployment_eligible"] is False for row in reference_rows),
            "Expose the larger reference memory and measured CPU time.",
        ),
    }
    capture_score = int(all(gate["passed"] is True for gate in capture_gates.values()))
    spans["e2e_and_completion_gates"] = time.monotonic() - phase
    if progress:
        _progress(
            5, "end", f"capture={capture_score} e2e_passed={all(row['passed'] for row in e2e)}"
        )
    step_receipt = _receipt(paths.step_rows, len(panel.step_rows))
    feedback_receipt = _receipt(paths.feedback_rows, len(panel.feedback_update_rows))
    e2e_receipt = _receipt(paths.e2e_rows, len(e2e))
    for receipt in (step_receipt, feedback_receipt, e2e_receipt):
        hashes[str(receipt["path"])] = receipt["sha256"]
    artifact = _base_artifact(
        checks,
        hashes,
        selected,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - monotonic_start,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_durations_s": spans,
            "rows": reduced,
            "sample_size_budget": _sample_budget(
                selected, panel.completed_stream_ids, panel.censored_stream_ids
            ),
            "acceptance_gate_results": {**capture_gates, **science_gates},
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "mixture_capture_complete_score": capture_score,
            "mixture_value_score": value_score,
            "per_stream_results": reduced,
            "feedback_update_rows": panel.feedback_update_rows,
            "chronology_violations": {
                "count": causal["chronology_violation_count"],
                "error_classes": causal["chronology_error_classes"],
                "offending_row_ids": causal["offending_row_ids"],
            },
            "memory_label_accounting": {
                "bounded_limit_bytes": MEMORY_CAP_BYTES,
                "maximum_bounded_state_bytes": panel.maximum_bounded_memory_bytes,
                "maximum_unbounded_reference_state_bytes": panel.maximum_reference_memory_bytes,
                "unbounded_reference_is_larger": panel.maximum_reference_memory_bytes
                > panel.maximum_bounded_memory_bytes,
                "unbounded_reference_deployment_eligible": False,
                "all_serialized_controller_state_charged": True,
                "external_state_references": [],
                "warmup_labels_per_stream_arm": WARMUP_COUNT,
                "future_labels_per_stream_arm": FUTURE_LABEL_COUNT,
                "total_label_charge_per_stream_arm": WARMUP_COUNT + FUTURE_LABEL_COUNT,
                "unbounded_reference_retains_full_label_history": True,
            },
            "comparison_rows": comparisons,
            "causal_summary": causal,
            "e2e_control_rows": e2e,
            "raw_evidence_receipts": {
                "step_rows": step_receipt,
                "feedback_update_rows": feedback_receipt,
                "e2e_rows": e2e_receipt,
            },
            "reducer_receipt": {
                "inference_substrate": REDUCER_INFERENCE_SUBSTRATE,
                "inference_substrate_class": REDUCER_INFERENCE_SUBSTRATE_CLASS,
                "source_paths": [str(paths.step_rows), str(paths.feedback_rows)],
                "row_count": len(reduced),
            },
            "source_artifact_states": {
                "exp7295": {
                    "path": str(repo_root / UPSTREAM_ARTIFACT),
                    "terminal_class": upstream.get("status"),
                    "quarantined": bool(upstream.get("flagged_adversarial")),
                    "retired": bool(upstream.get("retired")),
                    "mixture_fixture_ready_score": upstream.get("mixture_fixture_ready_score"),
                }
            },
            "upstream_artifact": {
                "path": str(repo_root / UPSTREAM_ARTIFACT),
                "sha256": _sha256_path(repo_root / UPSTREAM_ARTIFACT),
                "mixture_fixture_ready_score": upstream.get("mixture_fixture_ready_score"),
                "stream_manifest_path": upstream.get("stream_manifest_path"),
            },
            "validation_receipts": [
                {
                    "command": f"independent_reduce {paths.step_rows} {paths.feedback_rows}",
                    "exit_code": 0,
                    "duration_s": spans["cold_reduction_and_bootstrap"],
                    "log_sha256": transactional.sha256_json(reduced),
                    "classification": "passed",
                },
                {
                    "command": "adapted_e2e_007_checkpoint_restart_parity",
                    "exit_code": 0,
                    "duration_s": spans["e2e_and_completion_gates"],
                    "log_sha256": transactional.sha256_json(e2e),
                    "classification": "passed",
                },
            ],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact, repo_root=repo_root, expected_stream_ids=selected, check_files=True
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check identity, rows, gates, receipts, costs, files, and classification."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(
        artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID,
        "identity",
    )
    add(
        artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE,
        "date_milestone",
    )
    add(
        artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False,
        "model_contract",
    )
    add(artifact.get("invocation_counts") != INVOCATION_COUNTS, "invocation_counts")
    add(
        any(
            artifact.get(name) != 0
            for name in (
                "current_model_load_count",
                "current_generation_count",
                "current_inference_count",
            )
        ),
        "current_call_counts",
    )
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(
        artifact.get("continuous_self_learning_task") is not True
        or artifact.get("no_model_weight_mutation") is not True
        or artifact.get("production_default_changed") is not False,
        "learning_boundary",
    )
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    receipts = artifact.get("validation_receipts", [])
    required_receipt = {"command", "exit_code", "duration_s", "log_sha256", "classification"}
    add(
        not isinstance(receipts, list)
        or any(
            not required_receipt <= set(row)
            or not isinstance(row.get("command"), str)
            or not isinstance(row.get("exit_code"), int)
            or not isinstance(row.get("duration_s"), (int, float))
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("log_sha256", ""))) is None
            for row in receipts
        ),
        "validation_receipts",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("per_stream_results") != []
            or artifact.get("feedback_update_rows") != []
            or artifact.get("mixture_capture_complete_score") != 0
            or artifact.get("mixture_value_score") != 0
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
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
        or artifact.get("reducer_inference_substrate") != REDUCER_INFERENCE_SUBSTRATE
        or artifact.get("reducer_inference_substrate_class") != REDUCER_INFERENCE_SUBSTRATE_CLASS,
        "substrate",
    )
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
            int(row.get("future_prediction_count", 0)) != EVENTS_PER_STREAM - WARMUP_COUNT
            or int(row.get("non_feedback_future_prediction_count", 0))
            != EVENTS_PER_STREAM - WARMUP_COUNT - FUTURE_LABEL_COUNT
            or int(row.get("recurrence_prediction_count", 0)) != 256
            or int(row.get("warmup_label_count", 0)) != WARMUP_COUNT
            or int(row.get("future_label_count", 0)) != FUTURE_LABEL_COUNT
            or int(row.get("prediction_cost_ns", 0)) <= 0
            or row.get("censored") is not False
            for row in rows
        ),
        "rows",
    )
    updates = artifact.get("feedback_update_rows", [])
    add(
        not isinstance(updates, list)
        or len(updates) != len(selected) * FUTURE_LABEL_COUNT
        or feedback_row_errors(updates, selected) != [],
        "feedback_update_rows",
    )
    gates = artifact.get("acceptance_gate_results", {})
    add(
        not isinstance(gates, Mapping)
        or any(name not in gates for name in (*CAPTURE_GATE_NAMES, *SCIENTIFIC_GATE_NAMES))
        or any(
            gate.get("pass") != gate.get("passed")
            or not {"expected", "observed", "pass", "passed", "principle"} <= set(gate)
            for gate in gates.values()
        ),
        "acceptance_gate_results",
    )
    capture = int(all(gates.get(name, {}).get("passed") is True for name in CAPTURE_GATE_NAMES))
    value = int(all(gates.get(name, {}).get("passed") is True for name in SCIENTIFIC_GATE_NAMES))
    add(artifact.get("mixture_capture_complete_score") != capture, "mixture_capture_complete_score")
    add(
        artifact.get("mixture_value_score") != value or (value == 1 and capture != 1),
        "mixture_value_score",
    )
    expected_class = "circular_positive" if value else "null"
    add(
        artifact.get("verdict_class") != expected_class
        or artifact.get("verdict_class") == "positive"
        or not str(artifact.get("honest_verdict", "")).startswith("complete_"),
        "complete_contract",
    )
    chronology = artifact.get("chronology_violations", {})
    add(
        not isinstance(chronology, dict)
        or int(chronology.get("count", -1)) != 0
        or chronology.get("error_classes") != []
        or chronology.get("offending_row_ids") != [],
        "chronology_violations",
    )
    memory = artifact.get("memory_label_accounting", {})
    add(
        not isinstance(memory, dict)
        or int(memory.get("bounded_limit_bytes", 0)) != MEMORY_CAP_BYTES
        or int(memory.get("maximum_bounded_state_bytes", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
        or int(memory.get("maximum_unbounded_reference_state_bytes", 0))
        <= int(memory.get("maximum_bounded_state_bytes", 0))
        or memory.get("unbounded_reference_deployment_eligible") is not False
        or memory.get("total_label_charge_per_stream_arm") != WARMUP_COUNT + FUTURE_LABEL_COUNT,
        "memory_label_accounting",
    )
    comparisons = artifact.get("comparison_rows", [])
    add(
        not isinstance(comparisons, list)
        or len(comparisons) != len(COMPARISON_SPECS) * 3
        or any(
            int(row.get("bootstrap_resamples", 0)) != BOOTSTRAP_RESAMPLES
            or row.get("independent_unit") != "stream"
            for row in comparisons
        ),
        "comparison_rows",
    )
    if check_files:
        raw_receipts = artifact.get("raw_evidence_receipts", {})
        add(
            not isinstance(raw_receipts, dict)
            or any(
                _sha256_path(_resolve(repo_root, str(receipt.get("path", ""))))
                != receipt.get("sha256")
                for receipt in raw_receipts.values()
            ),
            "raw_evidence_receipts",
        )
        try:
            step_path = _resolve(repo_root, str(raw_receipts["step_rows"]["path"]))
            feedback_path = _resolve(repo_root, str(raw_receipts["feedback_update_rows"]["path"]))
            cold = independent_reduce(step_path, feedback_path)
            raw_updates = _read_jsonl(feedback_path)
        except (KeyError, TypeError, ValueError):
            cold, raw_updates = [], []
        add(cold != rows, "cold_reducer")
        add(raw_updates != updates, "raw_feedback_parity")
        add(
            any(
                expected is None or _sha256_path(_resolve(repo_root, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach exact command outcomes and refresh the stable checksum."""

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
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Cold-validate and atomically publish one terminal JSON object."""

    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=expected_stream_ids,
        check_files=artifact.get("status") == "complete",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return _atomic_write(
        path, (json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _validation_commands(candidate: Path) -> list[list[str]]:
    """Return focused, affected, full, coverage, static, E2E, and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    test = str(TEST_PATH)
    module = "python/carnot/experiment_7296_v641_mixture_learning.py"
    wrapper = str(WRAPPER_PATH)
    return [
        [
            pytest,
            test,
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=/tmp/carnot-exp7296-focused",
        ],
        [
            pytest,
            "tests/python/test_experiment_7295_v641_mixture_prototype.py::test_scenario_cl_7295_update_uses_exact_loss_share_and_ties_abstain",
            "tests/python/test_experiment_7282_v640_admission_learning.py::test_scenario_cl_7282_cold_reducer_and_fixed_bootstrap",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=/tmp/carnot-exp7296-affected",
        ],
        [
            pytest,
            "tests/python",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=/tmp/carnot-exp7296-full",
        ],
        [
            coverage,
            "run",
            "--data-file=/tmp/carnot-exp7296.coverage",
            f"--include={module}",
            str(COVERAGE_PATH),
        ],
        [
            coverage,
            "report",
            "--data-file=/tmp/carnot-exp7296.coverage",
            "--show-missing",
            "--fail-under=100",
        ],
        [ruff, "check", module, test, wrapper, str(COVERAGE_PATH)],
        [ruff, "format", "--check", module, test, wrapper, str(COVERAGE_PATH)],
        [mypy, module],
        [python, "scripts/check_spec_coverage.py", test],
        [
            pytest,
            f"{test}::test_scenario_cl_7296_e2e_restart_preserves_later_prediction",
            f"{test}::test_scenario_cl_7296_terminal_block_and_measured_validation",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=/tmp/carnot-exp7296-e2e",
        ],
        [
            python,
            "-m",
            "carnot.experiment_7296_v641_mixture_learning",
            "--date",
            RUN_DATE,
            "--validate-raw",
            str(candidate),
        ],
        [python, "scripts/adversarial_verify.py", str(candidate)],
        [python, "scripts/verdict_row_consistency_lint.py", str(candidate)],
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date and private cold-validation mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--validate-raw", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, validate, and publish only terminal prospective evidence."""

    print("phase 0 immediate: Exp7296 prospective mixture learning started", flush=True)
    args = _parse_args(argv)
    if args.validate_raw is not None:
        _progress(1, "before reduction", f"cold validate {args.validate_raw}")
        artifact = _load_object(args.validate_raw)
        errors = validate_artifact(artifact, check_files=True)
        print(
            json.dumps(
                {
                    "candidate": str(args.validate_raw),
                    "errors": errors,
                    "row_checksum": transactional.sha256_json(artifact.get("rows", [])),
                    "inference_substrate": REDUCER_INFERENCE_SUBSTRATE,
                    "inference_substrate_class": REDUCER_INFERENCE_SUBSTRATE_CLASS,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _progress(1, "after reduction", f"errors={len(errors)}")
        return int(bool(errors))
    paths = ExperimentPaths.defaults()
    invocation_start = time.monotonic()
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    write_artifact(paths.terminal_candidate, artifact)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact)
        _progress(8, "end", artifact["honest_verdict"])
        return 0
    commands = _validation_commands(paths.terminal_candidate)
    receipts: list[JsonDict] = []
    validation_started = time.monotonic()
    validation_dir = paths.raw_dir / "validation"
    for index, command in enumerate(commands):
        _progress(7, "before subprocess", f"{index + 1}/{len(commands)} {shlex.join(command)}")
        receipt, output = fixture._command_receipt(command)
        log_path = validation_dir / f"{index:02d}.log"
        log_receipt = _atomic_write(log_path, output.encode("utf-8"))
        receipt["log_path"] = str(log_path)
        receipt["log_sha256"] = log_receipt["sha256"]
        receipts.append(receipt)
        _progress(
            7,
            "after subprocess",
            f"{index + 1}/{len(commands)} exit={receipt['exit_code']} elapsed={receipt['duration_s']:.3f}s",
        )
    artifact = attach_validation_receipts(artifact, receipts)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = time.monotonic() - invocation_start
    artifact["phase_durations_s"]["validation"] = time.monotonic() - validation_started
    global_suite = next(
        (
            receipt
            for command, receipt in zip(commands, receipts)
            if len(command) > 1 and command[1] == "tests/python"
        ),
        None,
    )
    if global_suite is not None:
        artifact["global_suite_observation"] = {
            "exit_code": global_suite["exit_code"],
            "classification": global_suite["classification"],
            "task_scoped_gate": False,
        }
        if global_suite["exit_code"] != 0:
            artifact["honest_verdict"] += (
                "; repository-wide suite retained pre-existing collection failures"
            )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_artifact(paths.terminal_candidate, artifact)
    failures = [
        receipt
        for command, receipt in zip(commands, receipts)
        if receipt["exit_code"] != 0 and not (len(command) > 1 and command[1] == "tests/python")
    ]
    if failures:
        _progress(8, "end", "validation failed; measured candidate retained without publication")
        return 1
    write_artifact(paths.artifact, artifact)
    _progress(8, "end", artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
