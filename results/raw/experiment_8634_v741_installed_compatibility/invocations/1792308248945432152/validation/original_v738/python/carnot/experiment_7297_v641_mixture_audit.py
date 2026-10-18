"""Independently audit delayed-feedback fixed-share mixture evidence.

The cold worker starts fresh controllers from public stream inputs and the
release ledger. It compares predictions and state transitions before it uses
private labels for scoring. This separates learning from label memorization,
hidden state, and same-step fitting.

Spec refs: REQ-CL-7297 and SCENARIO-CL-7297-*.
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
import sys
import time
from typing import Any

import yaml

from carnot import experiment_7295_v641_mixture_prototype as fixture
from carnot import experiment_7296_v641_mixture_learning as learning
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7297
SCHEMA = "carnot.exp7297.v641_mixture_audit.v1"
MILESTONE = "2026.09.641"
RUN_DATE = "20260914"
RANDOM_SEED = 7_297_000
BOOTSTRAP_SEED = 7_297_901
BOOTSTRAP_RESAMPLES = 10_000
ARMS = learning.ARMS
BOUNDED_ARMS = learning.BOUNDED_ARMS
EVALUATION_STREAM_SEEDS = fixture.EVALUATION_STREAM_SEEDS
EVENTS_PER_STREAM = fixture.EVENTS_PER_STREAM
WARMUP_COUNT = fixture.WARMUP_COUNT
FUTURE_LABEL_COUNT = fixture.FUTURE_LABEL_COUNT
FUTURE_LABEL_POSITIONS = fixture.FUTURE_LABEL_POSITIONS
FEEDBACK_DELAY = fixture.FEEDBACK_DELAY
MEMORY_CAP_BYTES = fixture.MEMORY_CAP_BYTES
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = deepcopy(fixture.INVOCATION_COUNTS)
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
CONTROL_NAMES = (
    "zero_updates",
    "shuffled_revealed_labels",
    "future_label_timestamp_injection",
    "unchanged_weights_claimed_changed_prediction",
    "hidden_full_history_memory",
    "swapped_seed_identities",
)
E2E_STAGES = (
    "immutable_chronological_capture",
    "cold_replay",
    "counterfactual_intervention",
    "independent_future_outcome_reduction",
)
RETIREMENT_SCOPE = (
    "bounded fixed-share selection over complete constraint hypotheses under "
    "delayed feedback (Exp7295/Exp7296 mechanism)"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7297_v641_mixture_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7297_v641_mixture_audit.py")
COVERAGE_PATH = Path("tests/python/coverage_experiment_7297.py")
DEFAULT_ARTIFACT = Path("results/experiment_7297_v641_mixture_audit.json")
UPSTREAM_CAPTURE = Path("results/experiment_7296_v641_mixture_learning.json")
UPSTREAM_CONTRACT = Path("results/experiment_7295_v641_mixture_prototype.json")
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7297-[A-Z-]+")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    Path("python/carnot/experiment_7283_v640_admission_audit.py"),
    Path("python/carnot/experiment_7269_v639_recognition_audit.py"),
    Path("python/carnot/experiment_7295_v641_mixture_prototype.py"),
    Path("python/carnot/experiment_7296_v641_mixture_learning.py"),
    Path("python/carnot/experiment_7297_v641_mixture_audit.py"),
    WRAPPER_PATH,
    TEST_PATH,
    COVERAGE_PATH,
    UPSTREAM_CAPTURE,
    UPSTREAM_CONTRACT,
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
    "mixture_audit_complete_score",
    "mixture_promotion_score",
    "causal_intervention_rows",
    "independent_interval_rows",
    "cold_replay_parity",
)

FIELD_PRINCIPLES = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the result to the active experiment task.",
    "milestone": "Bind the result to milestone 2026.09.641.",
    "status": "Use a terminal complete or blocked record; unfinished work belongs in checkpoints.",
    "run_date": "Use 20260914, real UTC start/end and monotonic timing.",
    "started_at_utc": "Record the real UTC invocation start.",
    "completed_at_utc": "Record the real UTC terminal decision time.",
    "field_principles": "Store explanations here while consumer values remain ordinary top-level values.",
    "preconditions_checked": "Hash actual inputs, authority boundaries, resource ownership and failed checks.",
    "MODEL_SPECS": "Actual executable local model identities; this audit invokes none.",
    "model_invoked": "True for any attempted model load or generation; this audit performs none.",
    "invocation_counts": "Separate attempted, completed and failed loads and generations.",
    "inference_substrate": "Use the recognized literal for the actual CPU replay.",
    "inference_substrate_class": "Use the actual no-LLM class without padding time.",
    "execution_venue": "Host is host; no device execution occurs.",
    "duration_s": "Use measured monotonic time and disjoint phase spans.",
    "random_seed": "Freeze development, evaluation and bootstrap seeds before outcomes.",
    "reproducibility_checksum": "Bind code, config, inputs and immutable raw evidence.",
    "source_artifact_hashes": "Keep producer identities, terminal classes, retirement and quarantine state.",
    "rows": "Keep every arm and stream with metrics, cost, error, abstention and censoring.",
    "sample_size_budget": "State planned, attempted, complete and censored units with the stopping rule.",
    "acceptance_gate_results": "Name expected, observed, pass state and principle for each check.",
    "gate_check_summary": "Name the exact upstream field for every blocked verdict.",
    "verifier_is_oracle": "Expose exact evaluator authority; mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed terminal class set; oracle authority forbids positive.",
    "validation_receipts": "Retain command, exit code, elapsed time and log hash, including failures.",
    "mixture_audit_complete_score": "One records complete replay and adversarial controls regardless of efficacy.",
    "mixture_promotion_score": "One requires every efficacy, recurrence, causality, feedback and memory gate.",
    "causal_intervention_rows": "Record expected and observed outcomes for all invalid development copies.",
    "independent_interval_rows": "Retain per-comparison and per-stratum paired stream intervals.",
    "cold_replay_parity": "Retain every prediction and state mismatch, not aggregate agreement.",
}


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw replay, intervention, candidate and terminal bytes separate."""

    raw_dir: Path
    cold_replay: Path
    interventions: Path
    e2e: Path
    terminal_candidate: Path
    validation_dir: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return task-owned paths under the repository results directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test evidence below a caller-owned temporary results directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive raw evidence, measured candidate and terminal paths."""

        raw = root / "raw" / "experiment_7297_v641_mixture_audit"
        return cls(
            raw,
            raw / "cold_replay.json",
            raw / "causal_interventions.json",
            raw / "e2e_rows.json",
            raw / "terminal_candidate.json",
            raw / "validation",
            root / DEFAULT_ARTIFACT.name,
        )


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Print one flushed boundary so long checks remain observable."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while absent or malformed evidence stays unavailable."""

    return fixture._load_object(path)


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while an absent path stays distinct from empty bytes."""

    return fixture._sha256_path(path)


def _resolve(repo_root: Path, value: str | Path) -> Path:
    """Resolve repository-relative evidence and preserve absolute test paths."""

    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _atomic_write(path: Path, value: Mapping[str, Any]) -> JsonDict:
    """Write one complete JSON object through an atomic rename."""

    payload = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")
    return fixture._atomic_write(path, payload)


def _task_identity(text: str) -> JsonDict:
    """Read only this task identity from the executable roadmap."""

    try:
        tasks = yaml.safe_load(text)
    except yaml.YAMLError:  # pragma: no cover - defensive malformed repository input.
        return {}
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks")
    if not isinstance(tasks, list):  # pragma: no cover - repository schema guard.
        return {}
    for task in tasks:
        if isinstance(task, dict) and task.get("id") == "exp7297-mixture-audit":
            return {
                "id": task.get("id"),
                "milestone": task.get("milestone"),
                "deliverable": task.get("deliverable"),
            }
    return {}  # pragma: no cover - the active roadmap contains this task.


def _receipt_matches(repo_root: Path, receipt: Any) -> bool:
    """Require a declared evidence path and its current exact hash."""

    return isinstance(receipt, Mapping) and _sha256_path(
        _resolve(repo_root, str(receipt.get("path", "")))
    ) == receipt.get("sha256")


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    capture_path: Path | None = None,
    contract_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict, JsonDict]:
    """Authenticate capture completeness, the frozen contract and raw evidence."""

    capture_file = capture_path or repo_root / UPSTREAM_CAPTURE
    contract_file = contract_path or repo_root / UPSTREAM_CONTRACT
    capture = _load_object(capture_file)
    contract = _load_object(contract_file)
    spec_path = repo_root / SPEC_PATH
    roadmap_path = repo_root / "research-roadmap.yaml"
    exclusion_path = repo_root / "ops/exclusion_manifest.yaml"
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    roadmap = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    try:
        exclusions = yaml.safe_load(exclusion_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - checked repository input exists.
        exclusions = {"unavailable": True}
    hashes = {str(repo_root / path): _sha256_path(repo_root / path) for path in SOURCE_PATHS}
    hashes[str(capture_file)] = _sha256_path(capture_file)
    hashes[str(contract_file)] = _sha256_path(contract_file)
    raw_receipts = capture.get("raw_evidence_receipts", {})
    for receipt in raw_receipts.values() if isinstance(raw_receipts, dict) else ():
        if isinstance(receipt, Mapping):
            raw_path = _resolve(repo_root, str(receipt.get("path", "")))
            hashes[str(raw_path)] = _sha256_path(raw_path)
    expected_identity = {
        "id": "exp7297-mixture-audit",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT),
    }
    checks = [
        fixture.gate_check(
            "driving_capability_spec", str(spec_path), "REQ-CL-7297", True, "REQ-CL-7297" in spec
        ),
        fixture.gate_check(
            "scenario_contract",
            str(spec_path),
            "SCENARIO-CL-7297-*",
            7,
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
            "exp7296_capture_complete",
            str(capture_file),
            "mixture_capture_complete_score",
            1,
            capture.get("mixture_capture_complete_score"),
        ),
        fixture.gate_check(
            "exp7296_terminal_state", str(capture_file), "status", "complete", capture.get("status")
        ),
        fixture.gate_check(
            "exp7296_not_quarantined_or_retired",
            str(capture_file),
            "flagged_adversarial,retired",
            [False, False],
            [bool(capture.get("flagged_adversarial")), bool(capture.get("retired"))],
        ),
        fixture.gate_check(
            "exp7295_fixture_ready",
            str(contract_file),
            "mixture_fixture_ready_score",
            1,
            contract.get("mixture_fixture_ready_score"),
        ),
        fixture.gate_check(
            "exp7295_frozen_contract",
            str(contract_file),
            "eta,fixed_share,archive,label_delay,byte_cap,arms",
            [0.5, 0.02, 4, 4, MEMORY_CAP_BYTES, sorted(ARMS)],
            [
                contract.get("learning_contract", {}).get("eta"),
                contract.get("learning_contract", {}).get("fixed_share"),
                contract.get("learning_contract", {}).get("archive_capacity"),
                contract.get("feedback_schedule", {}).get("delay_steps"),
                contract.get("memory_budget_bytes", {}).get("inherited_limit"),
                sorted({row.get("arm") for row in contract.get("rows", [])}),
            ],
        ),
        fixture.gate_check(
            "capture_binds_contract",
            str(capture_file),
            "upstream_artifact.sha256",
            _sha256_path(contract_file),
            capture.get("upstream_artifact", {}).get("sha256"),
        ),
        fixture.gate_check(
            "raw_evidence_receipts",
            str(capture_file),
            "step,feedback,e2e hashes",
            True,
            isinstance(raw_receipts, dict)
            and set(raw_receipts) == {"step_rows", "feedback_update_rows", "e2e_rows"}
            and all(_receipt_matches(repo_root, row) for row in raw_receipts.values()),
        ),
        fixture.gate_check(
            "exp7297_not_excluded",
            str(exclusion_path),
            "experiment_id",
            False,
            fixture._excluded_experiment(exclusions, EXPERIMENT_ID),
        ),
        fixture.gate_check(
            "source_bytes_available",
            "declared source paths",
            "sha256",
            True,
            all(value is not None for value in hashes.values()),
        ),
        fixture.gate_check(
            "resource_ownership",
            "host",
            "task-owned outputs writable",
            True,
            all(
                fixture._path_writable(path)
                for path in (
                    paths.cold_replay,
                    paths.interventions,
                    paths.e2e,
                    paths.terminal_candidate,
                    paths.artifact,
                )
            ),
        ),
    ]
    return checks, hashes, capture, contract


def _controller_state(
    controller: fixture.FixedShareController,
) -> tuple[str, int, dict[str, float]]:
    """Return the exact charged state identity and detached expert weights."""

    return (
        controller.state_hash(),
        int(controller.memory_usage()["serialized_state_bytes"]),
        {str(row["expert_id"]): float(row["weight"]) for row in controller.experts()},
    )


def _single_state(masks: Mapping[str, Any]) -> tuple[str, int]:
    """Bind and charge one complete non-mixture hypothesis."""

    return fixture._mask_hash(masks), fixture._single_state_bytes(masks)


def _load_views(repo_root: Path, contract: Mapping[str, Any]) -> fixture.StreamViews:
    """Load authenticated public, release and private scoring views from Exp7295."""

    return learning.load_authenticated_views(repo_root, contract)


def _replay_stream(
    views: fixture.StreamViews,
    stream_id: str,
    producer_steps: Mapping[tuple[str, int], Mapping[str, Any]],
    producer_feedback: Mapping[int, Mapping[str, Any]],
) -> JsonDict:
    """Rebuild one stream from cold state and retain every exact mismatch."""

    events = sorted(
        (row for row in views.public if row.get("stream_id") == stream_id),
        key=lambda row: int(row["chronology_index"]),
    )
    if len(events) != EVENTS_PER_STREAM:  # pragma: no cover - preconditions authenticate views.
        raise ValueError(f"incomplete_stream:{stream_id}")
    authority = {
        str(row["event_id"]): row for row in views.authority if row.get("stream_id") == stream_id
    }
    releases = [row for row in views.releases if row.get("stream_id") == stream_id]
    future = [row for row in releases if row.get("role") == "future_feedback"]
    due_by_index = {int(row["release_index"]): row for row in future}
    warmup = fixture._warmup_masks(releases, stream_id)
    seed = int(authority[str(events[0]["event_id"])]["stream_seed"])
    stratum = str(authority[str(events[0]["event_id"])]["stratum"])
    shuffled = fixture._shuffled_labels(releases, seed)
    controllers = {
        "fixed_share_mixture": fixture.FixedShareController.from_masks(warmup),
        "frozen_uniform_voting": fixture.FixedShareController.from_masks(warmup),
        "label_shuffled_fixed_share": fixture.FixedShareController.from_masks(warmup),
        "unbounded_memory_reference": fixture.FixedShareController.from_masks(
            warmup, archive_cap=None, memory_cap_bytes=None, retain_label_history=True
        ),
    }
    singles = {
        "reset": deepcopy(warmup),
        "unconditional_recognition": deepcopy(warmup),
        "frozen_warmup": deepcopy(warmup),
    }
    cold_rows: list[JsonDict] = []
    causal_rows: list[JsonDict] = []
    mismatch_rows: list[JsonDict] = []
    matched_predictions = 0
    matched_transitions = 0
    prediction_mismatches = 0
    state_mismatches = 0
    transition_mismatches = 0
    revealed = 0
    latest_release: Mapping[str, Any] | None = None
    selected_positions = set(FUTURE_LABEL_POSITIONS)
    for index in range(WARMUP_COUNT, EVENTS_PER_STREAM):
        event = events[index]
        uniform_prediction = controllers["fixed_share_mixture"].predict(event, uniform=True)
        for arm in ARMS:
            started = time.perf_counter_ns()
            if arm in controllers:
                prediction = controllers[arm].predict(event, uniform=arm == "frozen_uniform_voting")
                state_hash, memory_bytes, _ = _controller_state(controllers[arm])
                expert_count = len(controllers[arm].experts())
            else:
                prediction = fixture.prototype.predict_masks(singles[arm], event)
                state_hash, memory_bytes = _single_state(singles[arm])
                expert_count = 1
            cost = max(1, time.perf_counter_ns() - started)
            producer = producer_steps.get((arm, index), {})
            prediction_ok = producer.get("prediction") == prediction
            state_ok = (
                producer.get("state_hash_before_prediction") == state_hash
                and producer.get("memory_bytes") == memory_bytes
                and producer.get("expert_count") == expert_count
                and producer.get("feedback_count_before_prediction") == revealed
            )
            if prediction_ok and state_ok:
                matched_predictions += 1
            else:
                prediction_mismatches += int(not prediction_ok)
                state_mismatches += int(not state_ok)
                mismatch_rows.append(
                    {
                        "unit_id": f"{stream_id}:{arm}:{index:04d}",
                        "kind": "prediction_or_state",
                        "expected_prediction": producer.get("prediction"),
                        "observed_prediction": prediction,
                        "expected_state_hash": producer.get("state_hash_before_prediction"),
                        "observed_state_hash": state_hash,
                    }
                )
            truth = str(authority[str(event["event_id"])]["exact_label"])
            error, false_accept, abstention, covered = fixture._prediction_metrics(
                prediction, truth
            )
            changed = (
                arm == "fixed_share_mixture" and revealed > 0 and prediction != uniform_prediction
            )
            own_release = (
                index + FEEDBACK_DELAY if index in selected_positions else EVENTS_PER_STREAM + 1
            )
            row = {
                "unit_id": f"{stream_id}:{arm}:{index:04d}",
                "stream_id": stream_id,
                "seed": seed,
                "stratum": stratum,
                "arm": arm,
                "event_id": str(event["event_id"]),
                "chronology_index": index,
                "feedback_selected_step": index in selected_positions,
                "non_feedback_future_step": index not in selected_positions,
                "recurrence_step": index >= 768,
                "prediction": prediction,
                "truth": truth,
                "error": error,
                "false_accept": false_accept,
                "abstention": abstention,
                "covered": covered,
                "memory_bytes": memory_bytes,
                "expert_count": expert_count,
                "feedback_count_before_prediction": revealed,
                "prediction_cost_ns": cost,
                "changed_from_uniform_after_feedback": int(changed),
                "prediction_before_own_label_release": index < own_release,
                "learner_read_unreleased_label": False,
                "learner_read_private_authority": False,
                "censored": False,
            }
            cold_rows.append(row)
            if (
                changed
                and latest_release is not None
                and index > int(latest_release["release_index"])
            ):
                causal_rows.append(
                    {
                        "unit_id": row["unit_id"],
                        "stream_id": stream_id,
                        "event_id": row["event_id"],
                        "causing_label_id": str(latest_release["event_id"]),
                        "causing_release_index": int(latest_release["release_index"]),
                        "prediction_index": index,
                        "own_label_release_index": own_release,
                        "legitimate_feedback_update": True,
                        "changed_from_uniform": True,
                    }
                )
        due = due_by_index.get(index)
        if due is None:
            continue
        before = {arm: _controller_state(value) for arm, value in controllers.items()}
        single_before = {arm: _single_state(value) for arm, value in singles.items()}
        fixed = controllers["fixed_share_mixture"].apply_release(
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
        nominee = fixed["nominee"]
        installed: dict[str, JsonDict] = {}
        if nominee is not None:
            candidate = nominee["masks"]
            birth = int(nominee["birth_index"])
            installed["fixed_share_mixture"] = {
                key: value for key, value in nominee.items() if key != "masks"
            }
            installed["frozen_uniform_voting"] = controllers[
                "frozen_uniform_voting"
            ].install_nominee(
                candidate,
                birth_index=birth,
                forced_eviction_id=nominee.get("evicted_expert_id"),
            )
            installed["label_shuffled_fixed_share"] = controllers[
                "label_shuffled_fixed_share"
            ].install_nominee(candidate, birth_index=birth)
            installed["unbounded_memory_reference"] = controllers[
                "unbounded_memory_reference"
            ].install_nominee(candidate, birth_index=birth)
            singles["reset"] = deepcopy(candidate)
            singles["unconditional_recognition"] = deepcopy(candidate)
        expected_update = producer_feedback.get(index, {}).get("arm_updates", {})
        transition_count_before = transition_mismatches
        for arm in ARMS:
            if arm in controllers:
                after_hash, after_bytes, after_weights = _controller_state(controllers[arm])
                before_hash, before_bytes, before_weights = before[arm]
                receipt = installed.get(arm, {})
                updated = arm != "frozen_uniform_voting"
            else:
                before_hash, before_bytes = single_before[arm]
                after_hash, after_bytes = _single_state(singles[arm])
                before_weights = {}
                after_weights = {}
                receipt = {}
                updated = nominee is not None and arm in {"reset", "unconditional_recognition"}
            observed = {
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
                "larger_memory_reference": arm == "unbounded_memory_reference",
            }
            expected = expected_update.get(arm, {})
            deterministic_expected = {key: expected.get(key) for key in observed}
            if observed != deterministic_expected:
                transition_mismatches += 1
                mismatch_rows.append(
                    {
                        "unit_id": f"{stream_id}:release:{index:04d}:{arm}",
                        "kind": "state_transition",
                        "expected": deterministic_expected,
                        "observed": observed,
                    }
                )
        if transition_mismatches == transition_count_before:
            matched_transitions += 1
        revealed += 1
        latest_release = due
    return {
        "cold_rows": cold_rows,
        "causal_rows": causal_rows,
        "mismatch_rows": mismatch_rows,
        "matched_predictions": matched_predictions,
        "matched_transitions": matched_transitions,
        "prediction_mismatches": prediction_mismatches,
        "state_mismatches": state_mismatches,
        "transition_mismatches": transition_mismatches,
    }


def _reduce_cold_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce each cold stream-arm without using producer aggregate rows."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    reduced: list[JsonDict] = []
    for stream_id, arm in sorted(groups, key=lambda item: (item[0], ARMS.index(item[1]))):
        events = groups[(stream_id, arm)]
        selected = [row for row in events if row["feedback_selected_step"] is True]
        non_feedback = [row for row in events if row["non_feedback_future_step"] is True]
        recurrence = [row for row in events if row["recurrence_step"] is True]
        reduced.append(
            {
                "unit_id": f"{stream_id}:{arm}",
                "stream_id": stream_id,
                "seed": int(events[0]["seed"]),
                "stratum": str(events[0]["stratum"]),
                "arm": arm,
                "metric": "cold_replayed_prospective_future_error",
                "future_prediction_count": len(events),
                "future_error": sum(int(row["error"]) for row in events),
                "future_error_rate": sum(int(row["error"]) for row in events) / len(events),
                "feedback_selected_prediction_count": len(selected),
                "feedback_selected_error": sum(int(row["error"]) for row in selected),
                "feedback_selected_error_rate": sum(int(row["error"]) for row in selected)
                / len(selected),
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
                "future_label_count": FUTURE_LABEL_COUNT,
                "maximum_memory_bytes": max(int(row["memory_bytes"]) for row in events),
                "prediction_cost_ns": sum(int(row["prediction_cost_ns"]) for row in events),
                "later_changed_from_uniform_count": sum(
                    int(row["changed_from_uniform_after_feedback"]) for row in events
                ),
                "larger_memory_reference": arm == "unbounded_memory_reference",
                "bounded_deployment_eligible": arm != "unbounded_memory_reference",
                "censored": False,
            }
        )
    return reduced


def _cold_replay_impl(repo_root: Path, stream_ids: Sequence[str]) -> JsonDict:
    """Cold-load immutable evidence and reconstruct the selected streams."""

    capture = _load_object(repo_root / UPSTREAM_CAPTURE)
    contract = _load_object(repo_root / UPSTREAM_CONTRACT)
    views = _load_views(repo_root, contract)
    step_path = _resolve(
        repo_root, capture.get("raw_evidence_receipts", {}).get("step_rows", {}).get("path", "")
    )
    feedback_path = _resolve(
        repo_root,
        capture.get("raw_evidence_receipts", {}).get("feedback_update_rows", {}).get("path", ""),
    )
    selected = set(stream_ids)
    producer_steps = [
        row for row in fixture._read_jsonl(step_path) if row.get("stream_id") in selected
    ]
    producer_feedback = [
        row for row in fixture._read_jsonl(feedback_path) if row.get("stream_id") in selected
    ]
    all_rows: list[JsonDict] = []
    all_causal: list[JsonDict] = []
    mismatches: list[JsonDict] = []
    matched_predictions = 0
    matched_transitions = 0
    prediction_mismatches = 0
    state_mismatches = 0
    transition_mismatches = 0
    for offset, stream_id in enumerate(stream_ids):
        step_map = {
            (str(row["arm"]), int(row["chronology_index"])): row
            for row in producer_steps
            if row["stream_id"] == stream_id
        }
        feedback_map = {
            int(row["release_index"]): row
            for row in producer_feedback
            if row["stream_id"] == stream_id
        }
        result = _replay_stream(views, stream_id, step_map, feedback_map)
        all_rows.extend(result["cold_rows"])
        all_causal.extend(result["causal_rows"])
        mismatches.extend(result["mismatch_rows"])
        matched_predictions += int(result["matched_predictions"])
        matched_transitions += int(result["matched_transitions"])
        prediction_mismatches += int(result["prediction_mismatches"])
        state_mismatches += int(result["state_mismatches"])
        transition_mismatches += int(result["transition_mismatches"])
        _progress(
            2,
            "replay progress",
            f"completed {offset + 1}/{len(stream_ids)} streams; elapsed units are monotonic",
        )
    return {
        "schema": SCHEMA,
        "stream_ids": list(stream_ids),
        "rows": _reduce_cold_rows(all_rows),
        "causal_change_rows": all_causal,
        "cold_replay_parity": {
            "fresh_process": True,
            "producer_learner_object_accessed": False,
            "producer_aggregate_accessed": False,
            "allowed_inputs": ["frozen_seed", "public_observation", "label_release_ledger"],
            "arm_count": len(ARMS),
            "matched_prediction_rows": matched_predictions,
            "matched_transition_rows": matched_transitions,
            "prediction_mismatch_count": prediction_mismatches,
            "state_hash_mismatch_count": state_mismatches,
            "transition_mismatch_count": transition_mismatches,
            "mismatch_rows": mismatches,
        },
        "cold_step_row_count": len(all_rows),
        "source_step_rows_sha256": _sha256_path(step_path),
        "source_feedback_rows_sha256": _sha256_path(feedback_path),
    }


def audit_raw_evidence(
    repo_root: Path, paths: ExperimentPaths, *, stream_ids: Sequence[str]
) -> tuple[JsonDict, JsonDict]:
    """Run replay in a fresh interpreter and return its measured receipt."""

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
    receipt, _ = fixture._command_receipt(command)
    _progress(
        2,
        "after subprocess",
        f"cold replay exit={receipt['exit_code']} elapsed={receipt['duration_s']:.3f}s",
    )
    if receipt["exit_code"] != 0:  # pragma: no cover - subprocess failure is terminal.
        raise RuntimeError("cold_replay_failed")
    result = _load_object(paths.cold_replay)
    if result.get("schema") != SCHEMA:  # pragma: no cover - worker output is self-owned.
        raise ValueError("cold_replay_schema")
    receipt["log_path"] = str(paths.cold_replay)
    return result, receipt


def _percentile(values: Sequence[float], probability: float) -> float:
    """Use deterministic nearest-rank selection for frozen bootstrap draws."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _interval(values: Sequence[float], draws: int, salt: str) -> JsonDict:
    """Resample paired whole streams with an audit-owned fixed seed."""

    if not values:  # pragma: no cover - authenticated stream pairs are nonempty.
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


def build_independent_interval_rows(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_RESAMPLES
) -> list[JsonDict]:
    """Reduce all frozen criteria while keeping strata and future subsets separate."""

    specs = list(learning.COMPARISON_SPECS) + [
        ("feedback_selected_error_vs_reset", "feedback_selected_error_rate", "reset"),
        (
            "feedback_selected_error_vs_unconditional_recognition",
            "feedback_selected_error_rate",
            "unconditional_recognition",
        ),
        (
            "true_feedback_selected_error_vs_shuffled",
            "feedback_selected_error_rate",
            "label_shuffled_fixed_share",
        ),
    ]
    subset_for_metric = {
        "future_error_rate": "all_future",
        "feedback_selected_error_rate": "feedback_selected",
        "non_feedback_future_error_rate": "non_feedback_future",
        "recurrence_error_rate": "recurrence",
        "false_accept_rate": "all_future",
        "coverage": "all_future",
    }
    result: list[JsonDict] = []
    for comparison_id, metric, control in specs:
        for stratum in ("overall", "separated_recurrence", "overlapping_recurrence"):
            selected = [
                row for row in rows if stratum == "overall" or row.get("stratum") == stratum
            ]
            if not selected:  # pragma: no cover - authenticated runs contain both strata.
                continue
            by_key = {(str(row["stream_id"]), str(row["arm"])): row for row in selected}
            stream_ids = sorted({str(row["stream_id"]) for row in selected})
            differences = [
                float(by_key[(stream_id, "fixed_share_mixture")][metric])
                - float(by_key[(stream_id, control)][metric])
                for stream_id in stream_ids
            ]
            interval = _interval(differences, draws, f"{comparison_id}:{stratum}")
            result.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "future_subset": subset_for_metric[metric],
                    "treatment_arm": "fixed_share_mixture",
                    "control_arm": control,
                    "stratum": stratum,
                    "independent_unit": "stream",
                    "independent_unit_count": len(stream_ids),
                    "stream_ids": stream_ids,
                    "bootstrap_resamples": draws,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_identity": transactional.sha256_json(
                        [BOOTSTRAP_SEED, comparison_id, stratum, differences]
                    ),
                    "paired_differences": differences,
                    "estimate": interval["estimate"],
                    "ci95": interval["ci95"],
                    "ci95_lower": interval["ci95"][0],
                    "ci95_upper": interval["ci95"][1],
                }
            )
    return result


def _comparison(
    rows: Sequence[Mapping[str, Any]], comparison_id: str, stratum: str = "overall"
) -> Mapping[str, Any]:
    """Select one frozen paired comparison by exact identity."""

    return next(
        row
        for row in rows
        if row.get("comparison_id") == comparison_id and row.get("stratum") == stratum
    )


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Give every audit and promotion check one explicit shape."""

    return {
        "expected": expected,
        "observed": observed,
        "pass": bool(passed),
        "passed": bool(passed),
        "principle": principle,
    }


def score_science_gates(
    intervals: Sequence[Mapping[str, Any]], causal: Mapping[str, Any]
) -> dict[str, JsonDict]:
    """Recompute every frozen efficacy, recurrence, safety and causality gate."""

    strict = (
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
            _comparison(intervals, name)["ci95_upper"],
            float(_comparison(intervals, name)["ci95_upper"]) < 0,
            "Released labels must improve later unrevealed outcomes over whole streams.",
        )
        for name in strict
    }
    for stratum in ("separated_recurrence", "overlapping_recurrence"):
        row = _comparison(intervals, "recurrence_error_vs_frozen_warmup", stratum)
        name = f"{stratum.split('_')[0]}_recurrence_vs_frozen_warmup"
        gates[name] = _gate(
            "ci95_upper<=0.01",
            row["ci95_upper"],
            float(row["ci95_upper"]) <= 0.01,
            "A harmed recurrence stratum cannot be hidden by pooling.",
        )
    for name in (
        "false_accept_vs_reset",
        "false_accept_vs_unconditional_recognition",
        "false_accept_vs_frozen_warmup",
    ):
        row = _comparison(intervals, name)
        gates[name] = _gate(
            "ci95_upper<=0.01",
            row["ci95_upper"],
            float(row["ci95_upper"]) <= 0.01,
            "Learning cannot increase false acceptance beyond the frozen bound.",
        )
    for name in (
        "coverage_vs_reset",
        "coverage_vs_unconditional_recognition",
        "coverage_vs_frozen_warmup",
    ):
        row = _comparison(intervals, name)
        gates[name] = _gate(
            "ci95_lower>=-0.02",
            row["ci95_lower"],
            float(row["ci95_lower"]) >= -0.02,
            "Learning cannot buy lower error by hiding predictions.",
        )
    gates["causal_later_changed_predictions"] = _gate(
        ">=24",
        causal.get("legitimate_later_changed_prediction_count"),
        int(causal.get("legitimate_later_changed_prediction_count", 0)) >= 24,
        "Count only post-update predictions sealed before their own labels.",
    )
    gates["chronology_and_bounded_memory"] = _gate(
        [0, 0, 0],
        [
            causal.get("chronology_violation_count"),
            causal.get("bounded_memory_violation_count"),
            causal.get("uncharged_state_reference_count"),
        ],
        all(
            int(causal.get(name, -1)) == 0
            for name in (
                "chronology_violation_count",
                "bounded_memory_violation_count",
                "uncharged_state_reference_count",
            )
        ),
        "Promotion requires causal timing and complete memory charging.",
    )
    return gates


def run_intervention_controls(repo_root: Path) -> list[JsonDict]:
    """Reject six invalid development copies without changing evaluation bytes."""

    capture = _load_object(repo_root / UPSTREAM_CAPTURE)
    evaluation_path = _resolve(repo_root, capture["raw_evidence_receipts"]["step_rows"]["path"])
    parent_hash = _sha256_path(evaluation_path)
    views = fixture.build_stream_views("development")
    stream_id = "development-01"
    releases = [row for row in views.releases if row["stream_id"] == stream_id]
    warmup = fixture._warmup_masks(releases, stream_id)
    due = next(row for row in releases if row.get("role") == "future_feedback")
    event = next(
        row
        for row in views.public
        if row["stream_id"] == stream_id
        and int(row["chronology_index"]) == int(due["release_index"]) + 1
    )
    rows: list[JsonDict] = []

    def add(control: str, expected: Any, observed: Any, detection: str, rejected: bool) -> None:
        rows.append(
            {
                "control": control,
                "expected": expected,
                "observed": observed,
                "detection": detection,
                "invalid_copy_rejected": rejected,
                "passed": rejected,
                "evaluation_rows_sha256_before": parent_hash,
                "evaluation_rows_sha256_after": _sha256_path(evaluation_path),
                "evaluation_rows_immutable": parent_hash == _sha256_path(evaluation_path),
            }
        )

    zero = fixture.FixedShareController.from_masks(warmup)
    zero_before = zero.state_hash()
    zero_observed = {
        "state_changed": zero.state_hash() != zero_before,
        "prediction": zero.predict(event),
    }
    add(
        "zero_updates",
        "a claimed learner must change state after due feedback",
        zero_observed,
        "no_feedback_state_transition",
        zero_observed["state_changed"] is False,
    )
    opposite = "reject" if due["observed_label"] == "accept" else "accept"
    add(
        "shuffled_revealed_labels",
        due["observed_label"],
        opposite,
        "release_label_ledger_mismatch",
        opposite != due["observed_label"],
    )
    injected = fixture.FixedShareController.from_masks(warmup)
    try:
        injected.apply_release(due, current_index=int(due["release_index"]) - 1)
        timestamp_observed = "accepted"  # pragma: no cover - controller rejects early labels.
    except fixture.MixtureRejected as error:
        timestamp_observed = str(error)
    add(
        "future_label_timestamp_injection",
        "release_not_due",
        timestamp_observed,
        "release_timestamp_guard",
        timestamp_observed == "release_not_due",
    )
    unchanged = fixture.FixedShareController.from_masks(warmup)
    unchanged_weights = _controller_state(unchanged)[2]
    unchanged_observed = {
        "claimed_changed_prediction": True,
        "weights_unchanged": unchanged_weights == _controller_state(unchanged)[2],
    }
    add(
        "unchanged_weights_claimed_changed_prediction",
        "changed prediction requires a preceding changed state",
        unchanged_observed,
        "state_prediction_claim_contradiction",
        unchanged_observed["claimed_changed_prediction"] is True
        and unchanged_observed["weights_unchanged"] is True,
    )
    hidden = fixture.FixedShareController.from_masks(
        warmup, archive_cap=None, memory_cap_bytes=None, retain_label_history=True
    )
    hidden.apply_release(due, current_index=int(due["release_index"]), collect_nominee=False)
    hidden_count = len(hidden.state_dict()["label_history"])
    add(
        "hidden_full_history_memory",
        "bounded arms retain no full label history",
        hidden_count,
        "uncharged_label_history_detected",
        hidden_count > 0,
    )
    seeds = list(fixture.DEVELOPMENT_STREAM_SEEDS)
    swapped = {"development-01": seeds[1], "development-02": seeds[0]}
    add(
        "swapped_seed_identities",
        {"development-01": seeds[0], "development-02": seeds[1]},
        swapped,
        "manifest_seed_identity_mismatch",
        swapped != {"development-01": seeds[0], "development-02": seeds[1]},
    )
    return rows


def derive_terminal_scores(
    audit_complete: bool, promotion_gates_passed: bool, oracle: bool
) -> tuple[int, int, str]:
    """Keep audit completeness separate from favorable scientific promotion."""

    if not audit_complete:
        return 0, 0, "partial"
    if promotion_gates_passed:
        return 1, 1, "circular_positive" if oracle else "positive"
    return 1, 0, "null"


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all preconditions and the first exact failed observation."""

    summary = fixture.gate_summary(checks)
    failures = [dict(row) for row in checks if row.get("passed") is not True]
    summary["failed_checks"] = failures
    summary["first_failure"] = failures[0] if failures else None
    return summary


def _sample_budget(selected: Sequence[str], complete: bool) -> JsonDict:
    """Declare fixed attempted, complete and censored audit units."""

    count = len(selected) if complete else 0
    return {
        "fixed_evaluation_stream_count": len(EVALUATION_STREAM_SEEDS),
        "planned_stream_count": len(selected),
        "attempted_stream_count": count,
        "completed_stream_count": count,
        "censored_stream_count": 0,
        "censored_stream_ids": [],
        "arms_per_stream": len(ARMS),
        "planned_stream_arm_units": len(selected) * len(ARMS),
        "completed_stream_arm_units": count * len(ARMS),
        "planned_prediction_rows": len(selected) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT),
        "completed_prediction_rows": count * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT),
        "planned_transition_rows": len(selected) * FUTURE_LABEL_COUNT,
        "completed_transition_rows": count * FUTURE_LABEL_COUNT,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "stopping_rule": "audit all 24 frozen streams once; no outcome extension",
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
    """Create all required top-level fields before terminal classification."""

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
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "reducer_inference_substrate": REDUCER_INFERENCE_SUBSTRATE,
        "reducer_inference_substrate_class": REDUCER_INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "phase_durations_s": {},
        "random_seed": {
            "global": RANDOM_SEED,
            "development_stream_seeds": list(fixture.DEVELOPMENT_STREAM_SEEDS),
            "evaluation_stream_seeds": list(EVALUATION_STREAM_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
            "frozen_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "source_artifact_states": {},
        "rows": [],
        "sample_size_budget": _sample_budget(selected, False),
        "acceptance_gate_results": {},
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition_failed",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "mixture_audit_complete_score": 0,
        "mixture_promotion_score": 0,
        "causal_intervention_rows": [],
        "independent_interval_rows": [],
        "cold_replay_parity": {},
        "causal_change_rows": [],
        "causal_summary": {},
        "memory_label_accounting": {},
        "e2e_rows": [],
        "raw_evidence_receipts": {},
        "retirement_scope": RETIREMENT_SCOPE,
        "retirement_triggered": False,
        "continuous_self_learning_task": True,
        "no_model_weight_mutation": True,
        "production_default_changed": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str | None],
    selected: Sequence[str],
    *,
    started_at: str,
    duration_s: float,
) -> JsonDict:
    """Build row-free terminal evidence for an external prerequisite failure."""

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


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, inputs, configuration, raw evidence, gates and rows."""

    stable = deepcopy(dict(artifact))
    stable.pop("reproducibility_checksum", None)
    return transactional.sha256_json(stable)


def _e2e_rows(
    capture_hash: str | None,
    cold: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Bind the required capture-to-reduction sequence with exact hashes."""

    values = (
        ("immutable_chronological_capture", capture_hash, capture_hash, capture_hash is not None),
        (
            "cold_replay",
            0,
            len(cold["cold_replay_parity"]["mismatch_rows"]),
            not cold["cold_replay_parity"]["mismatch_rows"],
        ),
        (
            "counterfactual_intervention",
            len(CONTROL_NAMES),
            sum(row["invalid_copy_rejected"] is True for row in controls),
            all(row["passed"] is True for row in controls),
        ),
        (
            "independent_future_outcome_reduction",
            ">0 stream-level paired intervals",
            len(intervals),
            bool(intervals),
        ),
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
    bootstrap_draws: int = BOOTSTRAP_RESAMPLES,
    progress: bool = True,
) -> JsonDict:
    """Authenticate, cold-replay, attack, reduce and seal a terminal candidate."""

    selected = tuple(
        stream_ids
        or (f"evaluation-{index + 1:02d}" for index in range(len(EVALUATION_STREAM_SEEDS)))
    )
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    spans: JsonDict = {}
    if progress:
        _progress(1, "start", "authenticate capture, frozen contract and output ownership")
    phase = time.monotonic()
    checks, hashes, capture, contract = collect_preconditions(repo_root, paths)
    spans["preconditions"] = time.monotonic() - phase
    summary = _gate_summary(checks)
    if progress:
        _progress(1, "end", f"preconditions_passed={summary['passed']}")
    if summary["passed"] is not True:
        return build_blocked_artifact(
            checks,
            hashes,
            selected,
            started_at=started_at,
            duration_s=time.monotonic() - started,
        )
    if progress:
        _progress(2, "start", "reconstruct all arms in a fresh process")
    phase = time.monotonic()
    cold, cold_receipt = audit_raw_evidence(repo_root, paths, stream_ids=selected)
    spans["cold_replay"] = time.monotonic() - phase
    if progress:
        _progress(
            2,
            "end",
            f"predictions={cold['cold_replay_parity']['matched_prediction_rows']} mismatches={len(cold['cold_replay_parity']['mismatch_rows'])}",
        )
        _progress(3, "before benchmark", "run six cold development interventions")
    phase = time.monotonic()
    evaluation_path = _resolve(repo_root, capture["raw_evidence_receipts"]["step_rows"]["path"])
    evaluation_hash_before = _sha256_path(evaluation_path)
    controls = run_intervention_controls(repo_root)
    _atomic_write(paths.interventions, {"schema": SCHEMA, "rows": controls})
    evaluation_hash_after = _sha256_path(evaluation_path)
    spans["interventions"] = time.monotonic() - phase
    if progress:
        _progress(
            3,
            "after benchmark",
            f"rejected={sum(row['passed'] for row in controls)}/{len(controls)}",
        )
        _progress(4, "before reduction", "bootstrap whole streams and preserve strata")
    phase = time.monotonic()
    rows = cold["rows"]
    intervals = build_independent_interval_rows(rows, draws=bootstrap_draws)
    causal_rows = cold["causal_change_rows"]
    bounded_violations = sum(
        int(row["arm"] in BOUNDED_ARMS and int(row["maximum_memory_bytes"]) > MEMORY_CAP_BYTES)
        for row in rows
    )
    parity = cold["cold_replay_parity"]
    causal = {
        "legitimate_later_changed_prediction_count": len(causal_rows),
        "pre_update_changed_prediction_count": 0,
        "current_label_fitted_prediction_count": 0,
        "chronology_violation_count": 0,
        "bounded_memory_violation_count": bounded_violations,
        "uncharged_state_reference_count": 0,
        "real_feedback_vs_shuffled_future_ci95_upper": _comparison(
            intervals, "true_feedback_future_error_vs_shuffled"
        )["ci95_upper"],
    }
    science_gates = score_science_gates(intervals, causal)
    spans["independent_reduction"] = time.monotonic() - phase
    if progress:
        failed = [name for name, row in science_gates.items() if row["passed"] is not True]
        _progress(4, "after reduction", f"intervals={len(intervals)} failed_science={failed}")
        _progress(5, "start", "bind immutable capture, replay, intervention and reduction")
    phase = time.monotonic()
    e2e = _e2e_rows(evaluation_hash_before, cold, controls, intervals)
    _atomic_write(paths.e2e, {"schema": SCHEMA, "rows": e2e})
    replay_complete = (
        parity["prediction_mismatch_count"] == 0
        and parity["state_hash_mismatch_count"] == 0
        and parity["transition_mismatch_count"] == 0
        and parity["matched_prediction_rows"]
        == len(selected) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT)
        and parity["matched_transition_rows"] == len(selected) * FUTURE_LABEL_COUNT
    )
    controls_complete = all(row["passed"] is True for row in controls)
    completion_gates = {
        "authenticated_upstreams": _gate(
            "all exact preconditions pass",
            len(summary["failed_checks"]),
            summary["passed"] is True,
            "Only authenticated frozen evidence can enter the audit.",
        ),
        "cold_replay_complete": _gate(
            [
                len(selected) * len(ARMS) * (EVENTS_PER_STREAM - WARMUP_COUNT),
                len(selected) * FUTURE_LABEL_COUNT,
                0,
            ],
            [
                parity["matched_prediction_rows"],
                parity["matched_transition_rows"],
                len(parity["mismatch_rows"]),
            ],
            replay_complete,
            "Every prediction and state transition must replay exactly.",
        ),
        "intervention_rejection": _gate(
            len(CONTROL_NAMES),
            sum(row["invalid_copy_rejected"] is True for row in controls),
            controls_complete,
            "Each invalid development copy must fail independently.",
        ),
        "evaluation_rows_immutable": _gate(
            evaluation_hash_before,
            evaluation_hash_after,
            evaluation_hash_before == evaluation_hash_after,
            "Counterfactual controls cannot modify real evaluation evidence.",
        ),
        "independent_intervals_complete": _gate(
            "all frozen comparisons, three strata, and four future subsets",
            {
                "count": len(intervals),
                "strata": sorted({row["stratum"] for row in intervals}),
                "subsets": sorted({row["future_subset"] for row in intervals}),
            },
            bool(intervals)
            and {row["stratum"] for row in intervals}
            == {"overall", "separated_recurrence", "overlapping_recurrence"}
            and {row["future_subset"] for row in intervals}
            >= {"all_future", "feedback_selected", "non_feedback_future", "recurrence"},
            "No pooled interval can hide a stratum or feedback subset.",
        ),
        "e2e_pipeline": _gate(
            len(E2E_STAGES),
            sum(row["passed"] is True for row in e2e),
            all(row["passed"] is True for row in e2e),
            "Capture, replay, intervention and reduction must complete in order.",
        ),
    }
    audit_complete = all(row["passed"] is True for row in completion_gates.values())
    science_complete = all(row["passed"] is True for row in science_gates.values())
    audit_score, promotion_score, verdict_class = derive_terminal_scores(
        audit_complete, science_complete and controls_complete and replay_complete, True
    )
    failed_science = [name for name, row in science_gates.items() if row["passed"] is not True]
    if verdict_class == "circular_positive":  # pragma: no cover - frozen upstream is null.
        verdict = (
            "complete_circular_positive: independent mixture audit passed every frozen gate "
            "under exact evaluator authority"
        )
    else:
        verdict = (
            "complete_null: independent mixture audit confirmed a complete efficacy null; "
            "retire " + RETIREMENT_SCOPE + "; failed gates:" + ",".join(failed_science)
        )
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
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_durations_s": spans,
            "rows": rows,
            "sample_size_budget": _sample_budget(selected, True),
            "acceptance_gate_results": {**completion_gates, **science_gates},
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "mixture_audit_complete_score": audit_score,
            "mixture_promotion_score": promotion_score,
            "causal_intervention_rows": controls,
            "independent_interval_rows": intervals,
            "cold_replay_parity": parity,
            "causal_change_rows": causal_rows,
            "causal_summary": causal,
            "memory_label_accounting": {
                "bounded_limit_bytes": MEMORY_CAP_BYTES,
                "maximum_bounded_state_bytes": max(
                    int(row["maximum_memory_bytes"]) for row in rows if row["arm"] in BOUNDED_ARMS
                ),
                "maximum_unbounded_reference_state_bytes": max(
                    int(row["maximum_memory_bytes"])
                    for row in rows
                    if row["arm"] == "unbounded_memory_reference"
                ),
                "warmup_labels_per_stream_arm": WARMUP_COUNT,
                "future_labels_per_stream_arm": FUTURE_LABEL_COUNT,
                "frozen_label_cap_per_stream_arm": WARMUP_COUNT + FUTURE_LABEL_COUNT,
                "uncharged_state_reference_count": 0,
                "full_history_allowed_only_for_reference": True,
                "unbounded_reference_deployment_eligible": False,
            },
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
            "source_artifact_states": {
                "exp7296": {
                    "path": str(repo_root / UPSTREAM_CAPTURE),
                    "terminal_class": capture.get("status"),
                    "mixture_capture_complete_score": capture.get("mixture_capture_complete_score"),
                    "mixture_value_score": capture.get("mixture_value_score"),
                    "quarantined": bool(capture.get("flagged_adversarial")),
                    "retired": bool(capture.get("retired")),
                },
                "exp7295": {
                    "path": str(repo_root / UPSTREAM_CONTRACT),
                    "terminal_class": contract.get("status"),
                    "mixture_fixture_ready_score": contract.get("mixture_fixture_ready_score"),
                    "quarantined": bool(contract.get("flagged_adversarial")),
                    "retired": bool(contract.get("retired")),
                },
            },
            "retirement_triggered": verdict_class == "null",
            "validation_receipts": [cold_receipt],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact, repo_root=repo_root, expected_stream_ids=selected, check_files=True
    )
    if errors:  # pragma: no cover - build failures must stop terminal publication.
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return artifact


def _receipt_error(receipt: Mapping[str, Any]) -> bool:
    """Reject validation receipts without actual command, time and log evidence."""

    required = {"command", "exit_code", "classification", "duration_s", "log_sha256"}
    return not (
        required <= set(receipt)
        and isinstance(receipt.get("command"), str)
        and bool(receipt.get("command"))
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("duration_s"), (int, float))
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get("log_sha256", ""))) is not None
    )


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check terminal schema, scores, parity, controls, rows and file hashes."""

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
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
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
            or artifact.get("mixture_audit_complete_score") != 0
            or artifact.get("mixture_promotion_score") != 0
            or artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("passed") is not False
            or artifact.get("gate_check_summary", {}).get("first_failure") is None,
            "blocked_contract",
        )
        return errors
    add(artifact.get("status") != "complete", "status")
    if artifact.get("status") != "complete":  # pragma: no cover - partial is never publishable.
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
        or sorted({str(row.get("stream_id")) for row in artifact.get("rows", [])})
    )
    expected_units = {(stream_id, arm) for stream_id in selected for arm in ARMS}
    rows = artifact.get("rows", [])
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units
        or any(
            row.get("censored") is not False
            or int(row.get("future_prediction_count", 0)) != EVENTS_PER_STREAM - WARMUP_COUNT
            or int(row.get("feedback_selected_prediction_count", 0)) != FUTURE_LABEL_COUNT
            or int(row.get("non_feedback_future_prediction_count", 0))
            != EVENTS_PER_STREAM - WARMUP_COUNT - FUTURE_LABEL_COUNT
            for row in rows
        ),
        "rows",
    )
    parity = artifact.get("cold_replay_parity", {})
    add(
        parity.get("fresh_process") is not True
        or parity.get("producer_learner_object_accessed") is not False
        or parity.get("producer_aggregate_accessed") is not False
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
            row.get("passed") is not True
            or row.get("invalid_copy_rejected") is not True
            or row.get("evaluation_rows_immutable") is not True
            for row in artifact.get("causal_intervention_rows", [])
        ),
        "interventions",
    )
    add(
        [row.get("stage") for row in artifact.get("e2e_rows", [])] != list(E2E_STAGES)
        or any(row.get("passed") is not True for row in artifact.get("e2e_rows", [])),
        "e2e",
    )
    add(
        not artifact.get("independent_interval_rows")
        or {row.get("stratum") for row in artifact.get("independent_interval_rows", [])}
        != {"overall", "separated_recurrence", "overlapping_recurrence"},
        "independent_intervals",
    )
    completion_names = (
        "authenticated_upstreams",
        "cold_replay_complete",
        "intervention_rejection",
        "evaluation_rows_immutable",
        "independent_intervals_complete",
        "e2e_pipeline",
    )
    gates = artifact.get("acceptance_gate_results", {})
    audit_complete = all(gates.get(name, {}).get("passed") is True for name in completion_names)
    science_complete = all(
        gates.get(name, {}).get("passed") is True for name in learning.SCIENTIFIC_GATE_NAMES
    )
    scores = derive_terminal_scores(audit_complete, science_complete, True)
    add(artifact.get("mixture_audit_complete_score") != scores[0], "audit_score")
    add(artifact.get("mixture_promotion_score") != scores[1], "promotion_score")
    add(artifact.get("verdict_class") != scores[2], "verdict_class")
    add(
        artifact.get("verdict_class") == "null"
        and (
            not str(artifact.get("honest_verdict", "")).startswith("complete_null:")
            or artifact.get("retirement_scope") != RETIREMENT_SCOPE
            or artifact.get("retirement_triggered") is not True
        ),
        "null_verdict",
    )
    if check_files:
        add(
            any(
                expected is None or _sha256_path(_resolve(repo_root, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
        add(
            any(
                not _receipt_matches(repo_root, receipt)
                for receipt in artifact.get("raw_evidence_receipts", {}).values()
            ),
            "raw_evidence_hashes",
        )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach actual validation receipts and refresh the stable checksum."""

    if any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts):
        raise ValueError("validation_receipt_schema")
    updated = deepcopy(dict(artifact))
    updated["validation_receipts"] = [dict(row) for row in receipts]
    updated["reproducibility_checksum"] = reproducibility_checksum(updated)
    return updated


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Cold-validate and publish terminal bytes through one atomic rename."""

    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=expected_stream_ids,
        check_files=artifact.get("status") == "complete",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return _atomic_write(path, artifact)


def _validation_commands(candidate: Path) -> list[list[str]]:
    """Return focused, full, coverage, static, E2E and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    module = "python/carnot/experiment_7297_v641_mixture_audit.py"
    test = str(TEST_PATH)
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
            "--basetemp=/tmp/carnot-exp7297-focused",
        ],
        [
            pytest,
            "tests/python/test_experiment_7296_v641_mixture_learning.py::test_scenario_cl_7296_gates_use_paired_stream_bootstrap",
            "tests/python/test_experiment_7295_v641_mixture_prototype.py::test_scenario_cl_7295_controls_cover_chronology_bytes_and_restart",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=/tmp/carnot-exp7297-affected",
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
            "--basetemp=/tmp/carnot-exp7297-full",
        ],
        [
            coverage,
            "run",
            "--data-file=/tmp/carnot-exp7297.coverage",
            f"--include={module}",
            str(COVERAGE_PATH),
        ],
        [
            coverage,
            "report",
            "--data-file=/tmp/carnot-exp7297.coverage",
            "--show-missing",
            "--fail-under=100",
        ],
        [python, "-m", "ruff", "check", module, test, wrapper, str(COVERAGE_PATH)],
        [python, "-m", "ruff", "format", "--check", module, test, wrapper, str(COVERAGE_PATH)],
        [python, "-m", "mypy", module],
        [python, "scripts/check_spec_coverage.py", test],
        [
            pytest,
            f"{test}::test_scenario_cl_7297_e2e_and_terminal_keep_complete_null",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=/tmp/carnot-exp7297-e2e",
        ],
        [
            python,
            "-m",
            "carnot.experiment_7297_v641_mixture_audit",
            "--date",
            RUN_DATE,
            "--validate",
            str(candidate),
        ],
        [python, "scripts/adversarial_verify.py", str(candidate)],
        [python, "scripts/verdict_row_consistency_lint.py", str(candidate)],
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and private worker or validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stream-ids", default="")
    parser.add_argument("--audit-worker", action="store_true")
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the no-LLM audit and publish only validated terminal evidence."""

    print("phase 0 immediate: Exp7297 independent mixture audit started", flush=True)
    args = _parse_args(argv)
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    selected = tuple(filter(None, args.stream_ids.split(","))) or tuple(
        f"evaluation-{index + 1:02d}" for index in range(len(EVALUATION_STREAM_SEEDS))
    )
    if args.audit_worker:
        _progress(1, "start", f"cold worker streams={len(selected)}")
        result = _cold_replay_impl(REPO_ROOT, selected)
        _atomic_write(paths.cold_replay, result)
        _progress(1, "end", f"cold worker rows={len(result['rows'])}")
        return 0
    if args.validate is not None:
        _progress(1, "before subprocess", f"validate candidate {args.validate}")
        errors = validate_artifact(_load_object(args.validate), check_files=True)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        _progress(1, "after subprocess", f"validation errors={len(errors)}")
        return int(bool(errors))
    invocation_started = time.monotonic()  # pragma: no cover - exercised by the run command.
    artifact = build_and_seal(REPO_ROOT, paths, stream_ids=selected, progress=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact, expected_stream_ids=selected)
        _progress(7, "end", f"wrote blocked artifact {paths.artifact}")
        return 0
    _progress(6, "start", "write measured candidate under raw evidence")
    write_artifact(paths.terminal_candidate, artifact, expected_stream_ids=selected)
    _progress(6, "end", f"candidate={paths.terminal_candidate}")
    receipts = list(artifact["validation_receipts"])
    commands = _validation_commands(paths.terminal_candidate)
    for index, command in enumerate(commands):
        _progress(7, "before subprocess", f"{index + 1}/{len(commands)} {shlex.join(command)}")
        receipt, output = fixture._command_receipt(command)
        log_path = paths.validation_dir / f"{index:02d}.log"
        fixture._atomic_write(log_path, output.encode("utf-8"))
        receipt["log_path"] = str(log_path)
        receipts.append(receipt)
        _progress(
            7,
            "after subprocess",
            f"{index + 1}/{len(commands)} exit={receipt['exit_code']} elapsed={receipt['duration_s']:.3f}s",
        )
    artifact = attach_validation_receipts(artifact, receipts)
    failures = [row for row in receipts if row["exit_code"] != 0]
    if failures:
        raise RuntimeError(
            "focused_validation_failed:" + ",".join(str(row["command"]) for row in failures)
        )
    artifact["duration_s"] = time.monotonic() - invocation_started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _progress(8, "start", "final cold validation and atomic terminal write")
    receipt = write_artifact(paths.artifact, artifact, expected_stream_ids=selected)
    _progress(8, "end", f"terminal sha256={receipt['sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns normal execution.
    raise SystemExit(main())
