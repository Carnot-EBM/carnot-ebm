"""Measure prospective constraint learning after independent admission.

The module reuses the frozen Exp7281 controller and streams. It adds a full
prequential journal and value gates. Exact evaluation makes favorable evidence
circular, so this task cannot produce an oracle-distinct positive claim.

Spec refs: REQ-CL-7282 and SCENARIO-CL-7282-*.
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
import time
from typing import Any

from carnot import experiment_7281_v640_admission_prototype as fixture
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7282
SCHEMA = "carnot.exp7282.v640_admission_learning.v1"
MILESTONE = "2026.09.640"
RUN_DATE = "20260913"
RANDOM_SEED = 7_282_000
BOOTSTRAP_SEED = 7_282_901
BOOTSTRAP_RESAMPLES = 10_000
MEASUREMENT_LIMIT_S = 900
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = deepcopy(fixture.INVOCATION_COUNTS)
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"

ARMS = fixture.ARMS
STREAM_COUNT = fixture.STREAM_COUNT
EVENTS_PER_STREAM = fixture.EVENTS_PER_STREAM
WARMUP_COUNT = fixture.WARMUP_COUNT
MAX_OPPORTUNITIES = fixture.MAX_OPPORTUNITIES
FRESH_LABELS_PER_OPPORTUNITY = fixture.FRESH_LABELS_PER_OPPORTUNITY
NOMINATION_LABEL_BUDGET = fixture.NOMINATION_LABEL_BUDGET
ADMISSION_LABEL_BUDGET = fixture.ADMISSION_LABEL_BUDGET
TOTAL_ALPHA = fixture.TOTAL_ALPHA
DEFAULT_THRESHOLD = fixture.DEFAULT_THRESHOLD
ARCHIVE_CAP = fixture.ARCHIVE_CAP
MEMORY_CAP_BYTES = fixture.MEMORY_CAP_BYTES
EXPECTED_EVENT_ROWS = STREAM_COUNT * EVENTS_PER_STREAM * len(ARMS)

COMPARISON_SPECS = (
    ("future_error_vs_reset", "future_error_rate", "paired_gated", "reset"),
    (
        "future_error_vs_unconditional_recognition",
        "future_error_rate",
        "paired_gated",
        "unconditional_recognition",
    ),
    ("false_accept_vs_reset", "false_accept_rate", "paired_gated", "reset"),
    (
        "false_accept_vs_unconditional_recognition",
        "false_accept_rate",
        "paired_gated",
        "unconditional_recognition",
    ),
    (
        "recurrence_degradation_vs_frozen_warmup",
        "recurrence_error_rate",
        "paired_gated",
        "frozen_warmup",
    ),
    (
        "recurrence_error_vs_label_shuffled_admission",
        "recurrence_error_rate",
        "paired_gated",
        "label_shuffled_paired",
    ),
    (
        "future_error_vs_full_reference",
        "future_error_rate",
        "paired_gated",
        "full_reference",
    ),
    (
        "future_error_vs_range_gated",
        "future_error_rate",
        "paired_gated",
        "range_gated",
    ),
)
SCIENTIFIC_GATE_NAMES = (
    "future_error_vs_reset",
    "future_error_vs_unconditional_recognition",
    "false_accept_vs_reset",
    "false_accept_vs_unconditional_recognition",
    "recurrence_degradation_vs_frozen_warmup",
    "recurrence_error_vs_label_shuffled_admission",
    "admitted_causal_change",
    "chronology_and_resource_safety",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7282_v640_admission_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7282_v640_admission_learning.py")
DEFAULT_UPSTREAM_ARTIFACT = Path("results/experiment_7281_v640_admission_prototype.json")
DEFAULT_ARTIFACT = Path("results/experiment_7282_v640_admission_learning.json")
EXPECTED_UPSTREAM_SHA256 = "sha256:967854fca40f6488d736be00e339fe47f360bd65626afee3816e09a29cd7d356"
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7268_v639_recognition_learning.py"),
    Path("python/carnot/experiment_7269_v639_recognition_audit.py"),
    Path("python/carnot/experiment_7281_v640_admission_prototype.py"),
    Path("python/carnot/experiment_7282_v640_admission_learning.py"),
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
)
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7282-[A-Z-]+")

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
    "admission_run_complete_score",
    "admission_value_score",
    "continuous_self_learning_task",
    "prequential_rows_path",
    "opportunity_rows_path",
    "comparison_rows",
    "no_model_weight_mutation",
)
FIELD_PRINCIPLES = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind evidence to the active Exp7282 task.",
    "milestone": "Bind evidence to milestone 2026.09.640.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "started_at_utc": "Record the actual UTC invocation start.",
    "completed_at_utc": "Record the actual UTC measurement end.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; historical identities stay in sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual computation.",
    "inference_substrate_class": "Use the correct no-LLM class and never pad duration.",
    "execution_venue": "Host orchestration is host; device execution is separate.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, censored units, and stopping.",
    "acceptance_gate_results": "Record expected, observed, passed, and principle for every criterion.",
    "gate_check_summary": "For blocks, name upstream, exact check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared evaluator authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed class set; oracle authority and failed efficacy forbid positive.",
    "validation_receipts": "Retain command, exit code, timing, and log hash without hiding failures.",
    "admission_run_complete_score": "One means every planned stream-arm and opportunity is accounted for.",
    "admission_value_score": "One requires every frozen value and safety gate plus changed predictions.",
    "continuous_self_learning_task": "True marks released-feedback constraint-state changes.",
    "prequential_rows_path": "Every prediction precedes the release that can update it.",
    "opportunity_rows_path": "Retain candidate, incumbent, harm, gain, rejection, and charged evidence.",
    "comparison_rows": "Keep stream-paired intervals, raw differences, and independent-unit counts.",
    "no_model_weight_mutation": "True only when constraint state changes and model weights do not.",
}

gate_check = fixture.gate_check
gate_summary = fixture.gate_summary
_canonical_bytes = transactional.canonical_json_bytes
_atomic_write = fixture._atomic_write


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw rows, checkpoints, candidates, and terminal bytes separate."""

    prequential_rows: Path
    opportunity_rows: Path
    diagnostic_rows: Path
    e2e_sidecar: Path
    provisional: Path
    stream_shards: Path
    terminal_candidate: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return task-owned paths under the repository results directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put every test or worker output below a caller-owned root."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive raw, checkpoint, candidate, and terminal paths."""

        raw = root / "raw" / "experiment_7282"
        checkpoints = root / "checkpoints" / "experiment_7282"
        return cls(
            raw / "prequential_rows.jsonl",
            raw / "opportunity_rows.jsonl",
            raw / "common_candidate_diagnostic.jsonl",
            checkpoints / "e2e_receipts.json",
            checkpoints / "in_progress.json",
            checkpoints / "streams",
            raw / "terminal_candidate.json",
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class LearningPanel:
    """Retain raw events, opportunities, summaries, and censoring state."""

    event_rows: list[JsonDict]
    opportunity_rows: list[JsonDict]
    rows: list[JsonDict]
    diagnostic: JsonDict
    completed_stream_ids: list[str]
    censored_stream_ids: list[str]
    maximum_memory_bytes: int


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Print one flushed boundary for the external watchdog."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository evidence while preserving absolute test paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed or absent evidence stays unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while absence remains different from empty content."""

    try:
        return transactional.sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence early."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_identity(text: str) -> JsonDict:
    """Extract only the active Exp7282 block from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7282-admission-learning\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7282-admission-learning" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate Exp7281, its sealed views, source bytes, and output owners."""

    upstream_file = _resolve(repo_root, upstream_path or DEFAULT_UPSTREAM_ARTIFACT)
    upstream = _load_object(upstream_file)
    spec = _resolve(repo_root, SPEC_PATH).read_text(encoding="utf-8")
    roadmap = _resolve(repo_root, "research-roadmap.yaml").read_text(encoding="utf-8")
    exclusions = _resolve(repo_root, "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    hashes = {
        str(_resolve(repo_root, path)): _sha256_path(_resolve(repo_root, path))
        for path in SOURCE_PATHS
    }
    hashes[str(upstream_file)] = _sha256_path(upstream_file)
    manifest = upstream.get("stream_manifest", {})
    receipts = [manifest.get("manifest_receipt", {}), *manifest.get("receipts", {}).values()]
    expected_receipts = {
        str(receipt.get("path")): receipt.get("sha256")
        for receipt in receipts
        if isinstance(receipt, Mapping) and receipt.get("path")
    }
    observed_receipts = {
        path: _sha256_path(_resolve(repo_root, path)) for path in expected_receipts
    }
    hashes.update(observed_receipts)
    writable = {
        field: _path_writable(getattr(paths, field))
        for field in (
            "prequential_rows",
            "opportunity_rows",
            "diagnostic_rows",
            "e2e_sidecar",
            "provisional",
            "terminal_candidate",
            "artifact",
        )
    }
    current_uid = os.getuid()
    owners: dict[str, bool] = {}
    for field in writable:
        parent = getattr(paths, field).parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        owners[field] = parent.stat().st_uid == current_uid
    expected_identity = {
        "id": "exp7282-admission-learning",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT),
    }
    expected_contract = [
        list(ARMS),
        STREAM_COUNT,
        EVENTS_PER_STREAM,
        WARMUP_COUNT,
        TOTAL_ALPHA,
        NOMINATION_LABEL_BUDGET,
        ADMISSION_LABEL_BUDGET,
        ARCHIVE_CAP,
        MEMORY_CAP_BYTES,
    ]
    contract = upstream.get("admission_contract", {})
    observed_contract = [
        [str(row.get("arm")) for row in upstream.get("rows", [])[: len(ARMS)]],
        manifest.get("prospective", {}).get("stream_count"),
        manifest.get("prospective", {}).get("events_per_stream"),
        manifest.get("prospective", {}).get("warmup_events"),
        contract.get("total_alpha"),
        contract.get("nomination_label_budget"),
        contract.get("admission_label_budget"),
        contract.get("archive_capacity"),
        contract.get("complete_memory_cap_bytes"),
    ]
    checks = [
        gate_check(
            "driving_capability_spec", str(SPEC_PATH), "REQ-CL-7282", True, "REQ-CL-7282" in spec
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7282-*",
            9,
            len(set(SCENARIO_PATTERN.findall(spec))),
        ),
        gate_check(
            "v640_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            _task_identity(roadmap),
        ),
        gate_check(
            "exp7281_artifact_hash",
            str(upstream_file),
            "sha256",
            EXPECTED_UPSTREAM_SHA256,
            _sha256_path(upstream_file),
        ),
        gate_check(
            "exp7281_terminal_state",
            "exp7281-admission-prototype",
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "exp7281_fixture_ready",
            "exp7281-admission-prototype",
            "admission_fixture_ready_score",
            1,
            upstream.get("admission_fixture_ready_score"),
        ),
        gate_check(
            "exp7281_not_quarantined_or_retired",
            "artifact_and_ops/exclusion_manifest.yaml",
            "flagged_adversarial,retired",
            [False, False],
            [upstream.get("flagged_adversarial", False) is True, "exp7281" in exclusions],
        ),
        gate_check(
            "frozen_admission_contract",
            "exp7281-admission-prototype",
            "arms,streams,events,warmup,alpha,quotas,capacity",
            expected_contract,
            observed_contract,
        ),
        gate_check(
            "sealed_stream_receipts",
            "exp7281.stream_manifest",
            "sha256",
            expected_receipts,
            observed_receipts,
        ),
        gate_check(
            "authority_separation",
            "exp7281.stream_manifest",
            "authority_separated,private_regime_evaluator_only",
            [True, True],
            [manifest.get("authority_separated"), manifest.get("private_regime_evaluator_only")],
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            True,
            all(value is not None for value in hashes.values()),
        ),
        gate_check(
            "writable_output_paths",
            "host_filesystem",
            "task-owned outputs",
            dict.fromkeys(writable, True),
            writable,
        ),
        gate_check(
            "resource_ownership", "host", "output parent uid", dict.fromkeys(owners, True), owners
        ),
    ]
    return checks, hashes, upstream


def load_sealed_views(
    repo_root: Path, upstream: Mapping[str, Any]
) -> fixture.prototype.StreamViews:
    """Load only the authenticated prospective public, release, and authority bytes."""

    receipts = upstream.get("stream_manifest", {}).get("receipts", {})
    public = fixture.prototype._read_jsonl(
        _resolve(repo_root, str(receipts["prospective_public"]["path"]))
    )
    authority = fixture.prototype._read_jsonl(
        _resolve(repo_root, str(receipts["prospective_private_authority"]["path"]))
    )
    releases = fixture.prototype._read_jsonl(
        _resolve(repo_root, str(receipts["prospective_releases"]["path"]))
    )
    views = fixture.prototype.StreamViews(
        public,
        authority,
        releases,
        deepcopy(dict(upstream["stream_manifest"]["prospective"])),
    )
    errors = fixture.stream_conformance_errors(views, "prospective")
    if errors:
        raise ValueError("sealed_stream_conformance:" + ",".join(errors))
    return views


def _segment(index: int) -> str:
    """Name the fixed warmup, initial, shifted, and recurrent intervals."""

    if index < WARMUP_COUNT:
        return "warmup"
    if index < 384:
        return "initial_future"
    if index < 768:
        return "shift"
    return "recurrence"


def _planned_candidates(
    views: fixture.prototype.StreamViews,
    stream_ids: Sequence[str],
) -> tuple[dict[tuple[str, int], JsonDict], list[JsonDict]]:
    """Freeze common candidate and case identities before running any arm."""

    releases = {str(row["event_id"]): row for row in views.releases}
    by_stream = {
        stream_id: [row for row in views.public if row["stream_id"] == stream_id]
        for stream_id in stream_ids
    }
    plans: dict[tuple[str, int], JsonDict] = {}
    rows: list[JsonDict] = []
    for stream_id in stream_ids:
        events = by_stream[stream_id]
        warmup = [
            fixture._released_case(event, releases[str(event["event_id"])])
            for event in events[:WARMUP_COUNT]
        ]
        warmup_masks = fixture._fit_masks(
            warmup, dict.fromkeys(fixture.FAMILIES, fixture.FULL_MASK)
        )
        for offset in range(MAX_OPPORTUNITIES):
            nomination_indices, nomination_index, admission_indices, decision_index = (
                fixture._opportunity_schedule(offset)
            )
            nomination_cases = [
                fixture._released_case(events[index], releases[str(events[index]["event_id"])])
                for index in nomination_indices
            ]
            admission_cases = [
                fixture._released_case(events[index], releases[str(events[index]["event_id"])])
                for index in admission_indices
            ]
            candidate = fixture._fit_masks(nomination_cases, warmup_masks)
            opportunity_index = offset + 1
            plan = {
                "stream_id": stream_id,
                "opportunity_index": opportunity_index,
                "nomination_index": nomination_index,
                "decision_index": decision_index,
                "nomination_cases": nomination_cases,
                "admission_cases": admission_cases,
                "candidate_masks": candidate,
                "candidate_state_hash": fixture.mask_hash(candidate),
                "nomination_case_ids_sha256": transactional.sha256_json(
                    [row["event_id"] for row in nomination_cases]
                ),
                "admission_case_ids_sha256": transactional.sha256_json(
                    [row["event_id"] for row in admission_cases]
                ),
            }
            plans[(stream_id, opportunity_index)] = plan
            rows.append(
                {
                    "stream_id": stream_id,
                    "opportunity_index": opportunity_index,
                    "candidate_state_hash": plan["candidate_state_hash"],
                    "nomination_case_ids_sha256": plan["nomination_case_ids_sha256"],
                    "admission_case_ids_sha256": plan["admission_case_ids_sha256"],
                    "nomination_label_count": FRESH_LABELS_PER_OPPORTUNITY,
                    "admission_label_count": FRESH_LABELS_PER_OPPORTUNITY,
                    "alpha": fixture.opportunity_alpha(opportunity_index),
                    "threshold": DEFAULT_THRESHOLD,
                    "frozen_before_arm_execution": True,
                }
            )
    return plans, rows


def run_common_candidate_diagnostic(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check that all arms receive the same frozen opportunity inputs."""

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stream_id"]), int(row["opportunity_index"]))].append(row)
    violations = []
    fields = (
        "candidate_state_hash",
        "nomination_case_ids_sha256",
        "admission_case_ids_sha256",
        "nomination_label_count",
        "admission_label_count",
        "alpha",
        "threshold",
    )
    for key, group in sorted(groups.items()):
        if any(
            len({json.dumps(row.get(field), sort_keys=True) for row in group}) != 1
            for field in fields
        ):
            violations.append(f"{key[0]}:opportunity-{key[1]}")
    return {
        "checked_opportunity_count": len(groups),
        "violation_count": len(violations),
        "violations": violations,
        "passed": not violations,
        "ran_before_benchmark": all(row.get("frozen_before_arm_execution") is True for row in rows),
    }


def _enrich_opportunities(
    rows: Sequence[Mapping[str, Any]],
    plans: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[JsonDict]:
    """Add fixed identities, counterfactual gain, and every charged action."""

    enriched = []
    for value in rows:
        row = deepcopy(dict(value))
        plan = plans[(str(row["stream_id"]), int(row["opportunity_index"]))]
        row["committed_candidate_state_hash"] = row["candidate_state_hash"]
        row["candidate_state_hash"] = plan["candidate_state_hash"]
        row["nomination_case_ids_sha256"] = plan["nomination_case_ids_sha256"]
        row["admission_case_ids_sha256"] = plan["admission_case_ids_sha256"]
        row["alpha"] = fixture.opportunity_alpha(int(row["opportunity_index"]))
        row["threshold"] = DEFAULT_THRESHOLD
        row["candidate_future_error"] = int(row["candidate_later_error"])
        row["incumbent_future_error"] = int(row["incumbent_later_error"])
        row["available_gain"] = row["incumbent_future_error"] - row["candidate_future_error"]
        row["harmful_admission"] = row["available_gain"] < 0 and row["decision"] == "accept"
        row["missed_beneficial_opportunity"] = (
            row["available_gain"] > 0 and row["decision"] != "accept"
        )
        row["zero_available_gain"] = row["available_gain"] == 0
        row["acquisition_label_cost"] = FRESH_LABELS_PER_OPPORTUNITY
        row["admission_label_cost"] = FRESH_LABELS_PER_OPPORTUNITY
        row["pending_queue_bytes"] = len(
            _canonical_bytes(
                {
                    "candidate": row["candidate_state_hash"],
                    "incumbent": row["incumbent_state_hash"],
                    "nomination": row["nomination_case_ids_sha256"],
                    "admission": row["admission_case_ids_sha256"],
                }
            )
        )
        row["update_decision_cost"] = 1
        row["rejection_cost"] = int(row["decision"] != "accept")
        row["commit_cost"] = int(row["decision"] == "accept")
        row["recovery_delay_events"] = int(row["decision_index"]) - int(row["nomination_index"])
        row["learner_read_counterfactual"] = False
        row["evaluator_only_fields"] = [
            "candidate_future_error",
            "incumbent_future_error",
            "available_gain",
            "true_opportunity_class",
        ]
        row["candidate_state_changed"] = row["candidate_state_hash"] != row["incumbent_state_hash"]
        row["admitted_state_change"] = (
            row["candidate_state_changed"] and row["decision"] == "accept"
        )
        enriched.append(row)
    return enriched


def _replay_event_rows(
    views: fixture.prototype.StreamViews,
    stream_id: str,
    opportunities: Sequence[Mapping[str, Any]],
    plans: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[JsonDict]:
    """Rebuild each prediction before applying the recorded admission action."""

    events = [row for row in views.public if row["stream_id"] == stream_id]
    authority = {str(row["event_id"]): row for row in views.authority}
    releases = {str(row["event_id"]): row for row in views.releases}
    released_at: dict[int, int] = defaultdict(int)
    for event in events:
        released_at[int(releases[str(event["event_id"])]["release_index"])] += 1
    warmup = [
        fixture._released_case(event, releases[str(event["event_id"])])
        for event in events[:WARMUP_COUNT]
    ]
    warmup_masks = fixture._fit_masks(warmup, dict.fromkeys(fixture.FAMILIES, fixture.FULL_MASK))
    states = {arm: deepcopy(warmup_masks) for arm in ARMS}
    cumulative_nomination: list[JsonDict] = []
    decisions = {(str(row["arm"]), int(row["decision_index"])): row for row in opportunities}
    plan_by_decision = {
        int(plan["decision_index"]): plan for key, plan in plans.items() if key[0] == stream_id
    }
    truth0 = authority[str(events[0]["event_id"])]
    rows: list[JsonDict] = []
    for index, event in enumerate(events):
        plan = plan_by_decision.get(index)
        for arm in ARMS:
            state = states[arm]
            prediction_start = time.perf_counter_ns()
            prediction = fixture.prototype.predict_masks(state, event)
            prediction_cost = time.perf_counter_ns() - prediction_start
            release_start = time.perf_counter_ns()
            release_count = released_at.get(index, 0)
            release_cost = time.perf_counter_ns() - release_start
            truth = authority[str(event["event_id"])]
            action = decisions.get((arm, index))
            rows.append(
                {
                    "unit_id": f"{stream_id}:{arm}:{index:04d}",
                    "stream_id": stream_id,
                    "seed": int(truth0["stream_seed"]),
                    "stratum": str(truth0["stratum"]),
                    "arm": arm,
                    "event_id": str(event["event_id"]),
                    "chronology_index": index,
                    "segment": _segment(index),
                    "prediction": prediction,
                    "prediction_state_hash": fixture.mask_hash(state),
                    "prediction_order": index * 3,
                    "label_release_index": int(releases[str(event["event_id"])]["release_index"]),
                    "label_release_order": int(releases[str(event["event_id"])]["release_index"])
                    * 3
                    + 1,
                    "prediction_frozen_before_release": True,
                    "released_label_count_after_prediction": release_count,
                    "admission_action_after_prediction": None
                    if action is None
                    else action["decision"],
                    "evaluator_exact_label": str(truth["exact_label"]),
                    "full_denominator_error": int(prediction != truth["exact_label"]),
                    "false_accept": int(
                        prediction == "accept" and truth["exact_label"] == "reject"
                    ),
                    "abstention": int(prediction == "abstain"),
                    "learner_read_private_authority": False,
                    "learner_read_current_label": False,
                    "evaluator_only_fields": [
                        "evaluator_exact_label",
                        "full_denominator_error",
                        "false_accept",
                    ],
                    "prediction_cost_ns": prediction_cost,
                    "release_processing_cost_ns": release_cost,
                    "censored": False,
                }
            )
        if plan is None:
            continue
        cumulative_nomination.extend(plan["nomination_cases"])
        candidate = deepcopy(dict(plan["candidate_masks"]))
        for arm in ARMS:
            decision = str(decisions[(arm, index)]["decision"])
            if arm == "full_reference":
                states[arm] = fixture._fit_masks(cumulative_nomination, states[arm])
            elif arm == "reset" or (arm in fixture.ADMISSION_ARMS and decision == "accept"):
                states[arm] = deepcopy(candidate)
    return rows


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read JSON objects while malformed evidence fails closed."""

    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("raw_rows_unavailable") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("raw_rows_unavailable")
    return rows


def prequential_row_errors(
    rows: Sequence[Mapping[str, Any]], stream_ids: Sequence[str]
) -> list[str]:
    """Check matrix completion, chronology, prediction seals, and authority isolation."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    expected = len(stream_ids) * len(ARMS) * EVENTS_PER_STREAM
    add(len(rows) != expected, "event_row_count")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("stream_id")), str(row.get("arm")))].append(row)
    expected_groups = {(stream_id, arm) for stream_id in stream_ids for arm in ARMS}
    add(set(groups) != expected_groups, "stream_arm_matrix")
    add(
        any(
            sorted(int(row.get("chronology_index", -1)) for row in group)
            != list(range(EVENTS_PER_STREAM))
            for group in groups.values()
        ),
        "chronology",
    )
    add(
        any(
            row.get("prediction_frozen_before_release") is not True
            or int(row.get("prediction_order", 1)) >= int(row.get("label_release_order", 0))
            or row.get("learner_read_private_authority") is not False
            or row.get("learner_read_current_label") is not False
            for row in rows
        ),
        "authority_or_release_order",
    )
    add(
        any(row.get("segment") != _segment(int(row.get("chronology_index", -1))) for row in rows),
        "segment",
    )
    add(any(row.get("censored") is not False for row in rows), "censoring")
    return errors


def opportunity_row_errors(
    rows: Sequence[Mapping[str, Any]], stream_ids: Sequence[str]
) -> list[str]:
    """Check every opportunity, counterfactual, cost, quota, and common identity."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    expected = len(stream_ids) * len(ARMS) * MAX_OPPORTUNITIES
    add(len(rows) != expected, "opportunity_row_count")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("stream_id")), str(row.get("arm")))].append(row)
    add(
        any(
            sorted(int(row.get("opportunity_index", 0)) for row in group)
            != list(range(1, MAX_OPPORTUNITIES + 1))
            for group in groups.values()
        ),
        "opportunity_sequence",
    )
    add(
        any(
            row.get("label_overlap_count") != 0
            or row.get("all_admission_labels_released") is not True
            or row.get("learner_read_counterfactual") is not False
            or row.get("future_label_used_for_nomination") is not False
            or row.get("private_regime_used") is not False
            for row in rows
        ),
        "label_or_authority_isolation",
    )
    add(
        any(
            int(row.get("acquisition_label_cost", 0)) != FRESH_LABELS_PER_OPPORTUNITY
            or int(row.get("admission_label_cost", 0)) != FRESH_LABELS_PER_OPPORTUNITY
            or int(row.get("pending_queue_bytes", 0)) <= 0
            or int(row.get("update_decision_cost", 0)) != 1
            or int(row.get("rejection_cost", 0)) + int(row.get("commit_cost", 0)) != 1
            or int(row.get("recovery_delay_events", 0)) <= 0
            or int(row.get("memory_bytes", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
            for row in rows
        ),
        "cost_or_capacity",
    )
    add(run_common_candidate_diagnostic(rows)["passed"] is not True, "common_candidate")
    return errors


def reduce_prequential_rows(
    event_rows: Sequence[Mapping[str, Any]], opportunity_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Reduce each stream-arm without dropping warmup, recurrence, or costs."""

    event_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    opportunity_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        event_groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    for row in opportunity_rows:
        opportunity_groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    reduced = []
    for key in sorted(event_groups, key=lambda item: (item[0], ARMS.index(item[1]))):
        events = event_groups[key]
        opportunities = opportunity_groups[key]
        future = [row for row in events if int(row["chronology_index"]) >= WARMUP_COUNT]
        recurrence = [row for row in events if row["segment"] == "recurrence"]
        warmup = [row for row in events if row["segment"] == "warmup"]
        reduced.append(
            {
                "unit_id": f"{key[0]}:{key[1]}",
                "stream_id": key[0],
                "seed": int(events[0]["seed"]),
                "stratum": str(events[0]["stratum"]),
                "arm": key[1],
                "metric": "prospective_full_denominator_error",
                "event_count": len(events),
                "future_event_count": len(future),
                "future_error": sum(int(row["full_denominator_error"]) for row in future),
                "future_error_rate": sum(int(row["full_denominator_error"]) for row in future)
                / len(future),
                "false_accept": sum(int(row["false_accept"]) for row in future),
                "false_accept_rate": sum(int(row["false_accept"]) for row in future) / len(future),
                "abstention": sum(int(row["abstention"]) for row in future),
                "warmup_event_count": len(warmup),
                "warmup_error_rate": sum(int(row["full_denominator_error"]) for row in warmup)
                / len(warmup),
                "recurrence_event_count": len(recurrence),
                "recurrence_error": sum(int(row["full_denominator_error"]) for row in recurrence),
                "recurrence_error_rate": sum(
                    int(row["full_denominator_error"]) for row in recurrence
                )
                / len(recurrence),
                "nomination_label_count": sum(
                    int(row["acquisition_label_cost"]) for row in opportunities
                ),
                "admission_label_count": sum(
                    int(row["admission_label_cost"]) for row in opportunities
                ),
                "total_paid_label_count": sum(
                    int(row["acquisition_label_cost"]) + int(row["admission_label_cost"])
                    for row in opportunities
                ),
                "opportunity_count": len(opportunities),
                "accepted_update_count": sum(row["decision"] == "accept" for row in opportunities),
                "rejected_update_count": sum(row["decision"] == "reject" for row in opportunities),
                "deferred_update_count": sum(row["decision"] == "defer" for row in opportunities),
                "harmful_update_accepted_count": sum(
                    bool(row["harmful_admission"]) for row in opportunities
                ),
                "useful_opportunity_not_admitted_count": sum(
                    bool(row["missed_beneficial_opportunity"]) for row in opportunities
                ),
                "zero_available_gain_count": sum(
                    bool(row["zero_available_gain"]) for row in opportunities
                ),
                "maximum_memory_bytes": max(int(row["memory_bytes"]) for row in opportunities),
                "maximum_pending_queue_bytes": max(
                    int(row["pending_queue_bytes"]) for row in opportunities
                ),
                "update_decision_cost": sum(
                    int(row["update_decision_cost"]) for row in opportunities
                ),
                "rejection_cost": sum(int(row["rejection_cost"]) for row in opportunities),
                "commit_cost": sum(int(row["commit_cost"]) for row in opportunities),
                "recovery_delay_events": sum(
                    int(row["recovery_delay_events"]) for row in opportunities
                ),
                "prediction_cost_ns": sum(int(row["prediction_cost_ns"]) for row in events),
                "release_processing_cost_ns": sum(
                    int(row["release_processing_cost_ns"]) for row in events
                ),
                "final_state_hash": str(events[-1]["prediction_state_hash"]),
                "censored": False,
            }
        )
    return reduced


def build_opportunity_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Use every round as denominator, including neutral zero-gain cases."""

    result = []
    arms = sorted({str(row["arm"]) for row in rows}, key=ARMS.index)
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        for opportunity_index in [*range(1, MAX_OPPORTUNITIES + 1), "overall"]:
            selected = (
                arm_rows
                if opportunity_index == "overall"
                else [row for row in arm_rows if int(row["opportunity_index"]) == opportunity_index]
            )
            denominator = len(selected)
            harmful = sum(bool(row["harmful_admission"]) for row in selected)
            missed = sum(bool(row["missed_beneficial_opportunity"]) for row in selected)
            zero = sum(bool(row["zero_available_gain"]) for row in selected)
            result.append(
                {
                    "arm": arm,
                    "opportunity_index": opportunity_index,
                    "denominator_unit": "stream_opportunity",
                    "opportunity_denominator": denominator,
                    "harmful_admission_count": harmful,
                    "harmful_admission_rate": harmful / denominator,
                    "missed_beneficial_opportunity_count": missed,
                    "missed_beneficial_opportunity_rate": missed / denominator,
                    "zero_available_gain_count": zero,
                    "zero_available_gain_rate": zero / denominator,
                }
            )
    return result


def run_learning_panel(
    views: fixture.prototype.StreamViews,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
    measurement_limit_s: float = MEASUREMENT_LIMIT_S,
) -> LearningPanel:
    """Run the fixed seven-arm panel and checkpoint every complete stream."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    plans, diagnostic_rows = _planned_candidates(views, selected)
    diagnostic = run_common_candidate_diagnostic(diagnostic_rows)
    if diagnostic["passed"] is not True or diagnostic["ran_before_benchmark"] is not True:
        raise ValueError("common_candidate_diagnostic_failed")
    all_events: list[JsonDict] = []
    all_opportunities: list[JsonDict] = []
    completed: list[str] = []
    censored: list[str] = []
    started = time.monotonic()
    last_heartbeat = started
    for offset, stream_id in enumerate(selected):
        if time.monotonic() - started > measurement_limit_s:
            censored.extend(selected[offset:])
            break
        shard = paths.stream_shards / f"{stream_id}.json"
        checkpoint = _load_object(shard)
        if checkpoint.get("schema") == SCHEMA and checkpoint.get("status") == "complete":
            events = checkpoint.get("event_rows", [])
            opportunities = checkpoint.get("opportunity_rows", [])
        else:
            panel = fixture.run_admission_panel(views, stream_ids=(stream_id,), progress=False)
            opportunities = _enrich_opportunities(panel.opportunity_rows, plans)
            events = _replay_event_rows(views, stream_id, opportunities, plans)
            checkpoint = {
                "schema": SCHEMA,
                "status": "complete",
                "stream_id": stream_id,
                "event_rows": events,
                "opportunity_rows": opportunities,
            }
            _atomic_write(shard, _canonical_bytes(checkpoint))
        if prequential_row_errors(events, (stream_id,)):
            raise ValueError(f"prequential_conformance:{stream_id}")
        if opportunity_row_errors(opportunities, (stream_id,)):
            raise ValueError(f"opportunity_conformance:{stream_id}")
        all_events.extend(events)
        all_opportunities.extend(opportunities)
        completed.append(stream_id)
        now = time.monotonic()
        if progress:
            print(
                f"phase 3 benchmark unit {offset + 1}/{len(selected)} completed_rows={len(all_events)} elapsed_s={now - started:.3f}",
                flush=True,
            )
        if now - last_heartbeat >= 60:
            print(
                f"phase 3 benchmark heartbeat completed_units={len(completed)} completed_rows={len(all_events)} elapsed_s={now - started:.3f}",
                flush=True,
            )
            last_heartbeat = now
        _atomic_write(
            paths.provisional,
            _canonical_bytes(
                {
                    "schema": SCHEMA,
                    "status": "in_progress",
                    "completed_stream_ids": completed,
                    "censored_stream_ids": censored,
                    "completed_event_arm_rows": len(all_events),
                    "elapsed_s": now - started,
                }
            ),
        )
    rows = reduce_prequential_rows(all_events, all_opportunities) if all_events else []
    _atomic_write(paths.prequential_rows, fixture.prototype.jsonl_bytes(all_events))
    _atomic_write(paths.opportunity_rows, fixture.prototype.jsonl_bytes(all_opportunities))
    _atomic_write(paths.diagnostic_rows, fixture.prototype.jsonl_bytes(diagnostic_rows))
    maximum_memory = max((int(row["memory_bytes"]) for row in all_opportunities), default=0)
    return LearningPanel(
        all_events,
        all_opportunities,
        rows,
        diagnostic,
        completed,
        censored,
        maximum_memory,
    )


def independent_reduce(
    prequential_path: Path, opportunity_path: Path
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Cold-reduce disk rows without trusting producer headline metrics."""

    events = _read_jsonl(prequential_path)
    opportunities = _read_jsonl(opportunity_path)
    stream_ids = tuple(sorted({str(row.get("stream_id")) for row in events}))
    event_errors = prequential_row_errors(events, stream_ids)
    opportunity_errors = opportunity_row_errors(opportunities, stream_ids)
    if event_errors or opportunity_errors:
        raise ValueError("raw_row_conformance:" + ",".join(event_errors + opportunity_errors))
    return reduce_prequential_rows(events, opportunities), build_opportunity_summary_rows(
        opportunities
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return the deterministic nearest-rank bootstrap percentile."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _bootstrap_interval(values: Sequence[float], draws: int, salt: str) -> JsonDict:
    """Resample whole paired streams with a frozen independent seed."""

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
    """Build stream-paired overall and recurrence-stratum intervals."""

    comparisons = []
    for comparison_id, metric, treatment, control in COMPARISON_SPECS:
        for stratum in (None, "separated_recurrence", "overlapping_recurrence"):
            selected = [row for row in rows if stratum is None or row["stratum"] == stratum]
            by_unit = {(str(row["stream_id"]), str(row["arm"])): row for row in selected}
            stream_ids = sorted({str(row["stream_id"]) for row in selected})
            if not stream_ids:
                continue
            differences = [
                float(by_unit[(stream_id, treatment)][metric])
                - float(by_unit[(stream_id, control)][metric])
                for stream_id in stream_ids
            ]
            interval = _bootstrap_interval(
                differences,
                draws,
                f"{comparison_id}:{stratum or 'overall'}",
            )
            comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "treatment_arm": treatment,
                    "control_arm": control,
                    "stratum": stratum or "overall",
                    "independent_unit": "stream",
                    "independent_unit_count": len(stream_ids),
                    "bootstrap_resamples": draws,
                    "paired_differences": differences,
                    "estimate": interval["estimate"],
                    "ci95": interval["ci95"],
                    "ci95_lower": interval["ci95"][0],
                    "ci95_upper": interval["ci95"][1],
                }
            )
    return comparisons


def causal_summary(
    event_rows: Sequence[Mapping[str, Any]], opportunity_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Count admitted state changes and later public prediction changes."""

    paired = [row for row in opportunity_rows if row["arm"] == "paired_gated"]
    accepted = [row for row in paired if row["decision"] == "accept"]
    changed = [row for row in accepted if row["admitted_state_change"]]
    boundaries: dict[str, int] = {}
    for row in changed:
        stream_id = str(row["stream_id"])
        boundaries[stream_id] = min(
            boundaries.get(stream_id, EVENTS_PER_STREAM), int(row["decision_index"])
        )
    by_key = {
        (str(row["stream_id"]), str(row["arm"]), int(row["chronology_index"])): row
        for row in event_rows
    }
    later_changes = 0
    for stream_id, boundary in boundaries.items():
        for index in range(boundary + 1, EVENTS_PER_STREAM):
            later_changes += int(
                by_key[(stream_id, "paired_gated", index)]["prediction"]
                != by_key[(stream_id, "frozen_warmup", index)]["prediction"]
            )
    stream_ids = tuple(sorted({str(row["stream_id"]) for row in event_rows}))
    chronology = len(prequential_row_errors(event_rows, stream_ids))
    resource = len(opportunity_row_errors(opportunity_rows, stream_ids))
    return {
        "paired_admission_count": len(accepted),
        "admitted_state_change_count": len(changed),
        "later_changed_prediction_count": later_changes,
        "chronology_violation_count": chronology,
        "quota_or_memory_violation_count": resource,
    }


def _overall_comparison(
    comparisons: Sequence[Mapping[str, Any]], comparison_id: str
) -> Mapping[str, Any]:
    """Return the overall row for one frozen comparison."""

    return next(
        row
        for row in comparisons
        if row["comparison_id"] == comparison_id and row["stratum"] == "overall"
    )


def score_acceptance_gates(
    comparisons: Sequence[Mapping[str, Any]], causal: Mapping[str, Any]
) -> dict[str, JsonDict]:
    """Score frozen learning gates without mixing them with run completion."""

    observed = {name: _overall_comparison(comparisons, name) for name in SCIENTIFIC_GATE_NAMES[:6]}
    definitions = (
        (
            "future_error_vs_reset",
            "ci95_upper<0",
            observed["future_error_vs_reset"]["ci95_upper"],
            float(observed["future_error_vs_reset"]["ci95_upper"]) < 0,
        ),
        (
            "future_error_vs_unconditional_recognition",
            "ci95_upper<0",
            observed["future_error_vs_unconditional_recognition"]["ci95_upper"],
            float(observed["future_error_vs_unconditional_recognition"]["ci95_upper"]) < 0,
        ),
        (
            "false_accept_vs_reset",
            "ci95_upper<=0",
            observed["false_accept_vs_reset"]["ci95_upper"],
            float(observed["false_accept_vs_reset"]["ci95_upper"]) <= 0,
        ),
        (
            "false_accept_vs_unconditional_recognition",
            "ci95_upper<=0",
            observed["false_accept_vs_unconditional_recognition"]["ci95_upper"],
            float(observed["false_accept_vs_unconditional_recognition"]["ci95_upper"]) <= 0,
        ),
        (
            "recurrence_degradation_vs_frozen_warmup",
            "estimate<=0.02",
            observed["recurrence_degradation_vs_frozen_warmup"]["estimate"],
            float(observed["recurrence_degradation_vs_frozen_warmup"]["estimate"]) <= 0.02,
        ),
        (
            "recurrence_error_vs_label_shuffled_admission",
            "estimate<0",
            observed["recurrence_error_vs_label_shuffled_admission"]["estimate"],
            float(observed["recurrence_error_vs_label_shuffled_admission"]["estimate"]) < 0,
        ),
        (
            "admitted_causal_change",
            "admissions>0 and state_changes>0 and later_prediction_changes>0",
            [
                causal["paired_admission_count"],
                causal["admitted_state_change_count"],
                causal["later_changed_prediction_count"],
            ],
            int(causal["paired_admission_count"]) > 0
            and int(causal["admitted_state_change_count"]) > 0
            and int(causal["later_changed_prediction_count"]) > 0,
        ),
        (
            "chronology_and_resource_safety",
            "chronology_violations==0 and quota_or_memory_violations==0",
            [causal["chronology_violation_count"], causal["quota_or_memory_violation_count"]],
            int(causal["chronology_violation_count"]) == 0
            and int(causal["quota_or_memory_violation_count"]) == 0,
        ),
    )
    return {
        name: {
            "expected": expected,
            "observed": value,
            "passed": passed,
            "pass": passed,
            "principle": "Keep this frozen scientific criterion separate from run completion.",
        }
        for name, expected, value, passed in definitions
    }


def classify_result(gates: Mapping[str, Mapping[str, Any]]) -> tuple[int, str, str]:
    """Return circular favorable evidence or an honest completed null."""

    value = int(all(gates[name]["passed"] is True for name in SCIENTIFIC_GATE_NAMES))
    if value:
        return (
            1,
            "circular_positive",
            "complete_circular_positive: independently admitted constraint state improved every frozen criterion under exact evaluator authority",
        )
    failed = [name for name in SCIENTIFIC_GATE_NAMES if gates[name]["passed"] is not True]
    return (
        0,
        "null",
        "complete_null: admission learning completed but frozen value gates failed: "
        + ",".join(failed),
    )


def _e2e_rows(panel: LearningPanel, paths: ExperimentPaths) -> list[JsonDict]:
    """Score the complete journal-to-future-error path from disk evidence."""

    reduced, opportunities = independent_reduce(paths.prequential_rows, paths.opportunity_rows)
    stream_count = len(panel.completed_stream_ids)
    return [
        {
            "stage": "complete_stream",
            "passed": len(panel.event_rows) == stream_count * len(ARMS) * EVENTS_PER_STREAM,
        },
        {
            "stage": "ordered_prediction_release_journal",
            "passed": prequential_row_errors(panel.event_rows, panel.completed_stream_ids) == [],
        },
        {
            "stage": "common_candidate_admission",
            "passed": panel.diagnostic["passed"] is True
            and any(row["decision"] == "accept" for row in panel.opportunity_rows),
        },
        {
            "stage": "future_error_rows",
            "passed": len(reduced) == stream_count * len(ARMS)
            and all(
                row["future_event_count"] == EVENTS_PER_STREAM - WARMUP_COUNT for row in reduced
            ),
        },
        {
            "stage": "opportunity_denominator",
            "passed": all(row["opportunity_denominator"] > 0 for row in opportunities),
        },
        {"stage": "cold_evaluator_recomputation", "passed": reduced == panel.rows},
    ]


def run_e2e_controls(root: Path) -> list[JsonDict]:
    """Run one complete sealed stream through journal, admission, and cold reduction."""

    paths = ExperimentPaths.under(root / "private_results")
    upstream = _load_object(REPO_ROOT / DEFAULT_UPSTREAM_ARTIFACT)
    views = load_sealed_views(REPO_ROOT, upstream)
    panel = run_learning_panel(views, paths, stream_ids=("prospective-01",), progress=False)
    rows = _e2e_rows(panel, paths)
    _atomic_write(root / "e2e_receipts.json", _canonical_bytes({"schema": SCHEMA, "rows": rows}))
    return rows


def _sample_budget(
    stream_ids: Sequence[str], completed: Sequence[str], censored: Sequence[str], draws: int
) -> JsonDict:
    """Declare fixed units, attempts, censoring, bootstrap, and stopping."""

    return {
        "fixed_global_stream_count": STREAM_COUNT,
        "planned_stream_count": len(stream_ids),
        "attempted_stream_count": len(completed) + len(censored),
        "completed_stream_count": len(completed),
        "censored_stream_count": len(censored),
        "censored_stream_ids": list(censored),
        "arms_per_stream": len(ARMS),
        "events_per_stream": EVENTS_PER_STREAM,
        "planned_event_arm_rows": len(stream_ids) * len(ARMS) * EVENTS_PER_STREAM,
        "completed_event_arm_rows": len(completed) * len(ARMS) * EVENTS_PER_STREAM,
        "planned_opportunity_rows": len(stream_ids) * len(ARMS) * MAX_OPPORTUNITIES,
        "completed_opportunity_rows": len(completed) * len(ARMS) * MAX_OPPORTUNITIES,
        "bootstrap_resamples": draws,
        "measurement_budget_s": MEASUREMENT_LIMIT_S,
        "stopping_rule": "all 24 sealed streams once within 900 seconds; retain censored identities",
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every precondition and the first exact external failure."""

    summary = gate_summary(checks)
    first = next((dict(row) for row in checks if row.get("passed") is not True), None)
    return {
        **summary,
        "checks": [dict(row) for row in checks],
        "first_failure": first,
        "failed_checks": [dict(row) for row in checks if row.get("passed") is not True],
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str | None],
    stream_ids: Sequence[str],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create required fields before blocked or measured classification."""

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
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": MODEL_INVOKED,
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
        "random_seed": {
            "experiment": RANDOM_SEED,
            "stream_seeds": list(fixture.STREAM_SEEDS[: len(stream_ids)]),
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, (), stream_ids, BOOTSTRAP_RESAMPLES),
        "acceptance_gate_results": {},
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "admission_run_complete_score": 0,
        "admission_value_score": 0,
        "continuous_self_learning_task": True,
        "prequential_rows_path": None,
        "opportunity_rows_path": None,
        "comparison_rows": [],
        "no_model_weight_mutation": True,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str | None],
    stream_ids: Sequence[str],
    *,
    started_at: str,
    duration_s: float,
) -> JsonDict:
    """Return row-free terminal evidence for an external prerequisite failure."""

    artifact = _base_artifact(
        checks,
        hashes,
        stream_ids,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=duration_s,
    )
    first = artifact["gate_check_summary"].get("first_failure") or {
        "upstream": "external",
        "field": "precondition",
    }
    artifact["honest_verdict"] = (
        f"blocked_{first.get('upstream', 'external')}_{first.get('field', 'precondition')}".replace(
            "/", "_"
        ).replace(" ", "_")
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable configuration, sources, raw evidence, gates, and receipts."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_spans_s",
    }
    return transactional.sha256_json(
        {key: value for key, value in artifact.items() if key not in excluded}
    )


def _receipt(path: Path, row_count: int | None = None) -> JsonDict:
    """Describe current exact sidecar bytes and optional row count."""

    result: JsonDict = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }
    if row_count is not None:
        result["row_count"] = row_count
    return result


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    bootstrap_draws: int = BOOTSTRAP_RESAMPLES,
    progress: bool = False,
) -> JsonDict:
    """Authenticate, diagnose, replay, cold-reduce, score, and seal evidence."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    monotonic_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    if progress:
        _progress(0, "start", "authenticate Exp7281, sealed manifests, and output ownership")
    phase_start = time.monotonic()
    checks, hashes, upstream = collect_preconditions(repo_root, paths)
    spans["phase_0_preconditions"] = time.monotonic() - phase_start
    _atomic_write(
        paths.provisional,
        _canonical_bytes({"schema": SCHEMA, "status": "in_progress", "phase": 0, "checks": checks}),
    )
    if gate_summary(checks)["passed"] is not True:
        if progress:
            _progress(0, "end", "external precondition failed; no measurement rows were written")
        return build_blocked_artifact(
            checks,
            hashes,
            selected,
            started_at=started_at,
            duration_s=time.monotonic() - monotonic_start,
        )
    if progress:
        _progress(0, "end", "all external checks passed")
        _progress(1, "start", "confirm zero current model work")
        print("phase 1 BEFORE model load: no model load scheduled", flush=True)
        print("phase 1 AFTER model load: attempted and completed loads remain zero", flush=True)
        print("phase 1 BEFORE generation: no generation scheduled", flush=True)
        print(
            "phase 1 AFTER generation: attempted and completed generations remain zero", flush=True
        )
        _progress(1, "end", "MODEL_SPECS is empty and every current counter is zero")

    phase_start = time.monotonic()
    if progress:
        _progress(2, "start", "BEFORE common-candidate diagnostic")
        print(
            "phase 2 BEFORE benchmark: freeze common candidates and opportunity inputs", flush=True
        )
    views = load_sealed_views(repo_root, upstream)
    plans, diagnostic_rows = _planned_candidates(views, selected)
    diagnostic = run_common_candidate_diagnostic(diagnostic_rows)
    if diagnostic["passed"] is not True:
        raise ValueError("common_candidate_diagnostic_failed")
    spans["phase_2_common_candidate"] = time.monotonic() - phase_start
    if progress:
        print("phase 2 AFTER benchmark: common-candidate diagnostic passed", flush=True)
        _progress(2, "end", f"opportunities={diagnostic['checked_opportunity_count']}")

    phase_start = time.monotonic()
    if progress:
        _progress(3, "start", "BEFORE seven-arm 24-stream CPU benchmark")
        print("phase 3 BEFORE benchmark: ordered prediction and admission replay", flush=True)
    panel = run_learning_panel(views, paths, stream_ids=selected, progress=progress)
    spans["phase_3_seven_arm_benchmark"] = time.monotonic() - phase_start
    if progress:
        print("phase 3 AFTER benchmark: bounded replay returned", flush=True)
        _progress(
            3,
            "end",
            f"completed_event_rows={len(panel.event_rows)} censored_streams={len(panel.censored_stream_ids)}",
        )

    phase_start = time.monotonic()
    if progress:
        _progress(4, "start", "BEFORE cold evaluator recomputation")
        print("phase 4 BEFORE benchmark: reduce exact raw disk rows", flush=True)
    reduced, opportunity_summaries = independent_reduce(
        paths.prequential_rows, paths.opportunity_rows
    )
    if reduced != panel.rows:
        raise ValueError("cold_evaluator_reducer_mismatch")
    comparisons = build_comparison_rows(reduced, draws=bootstrap_draws)
    causal = causal_summary(panel.event_rows, panel.opportunity_rows)
    science_gates = score_acceptance_gates(comparisons, causal)
    value_score, verdict_class, verdict = classify_result(science_gates)
    spans["phase_4_cold_reduction_and_bootstrap"] = time.monotonic() - phase_start
    if progress:
        print("phase 4 AFTER benchmark: cold reducer and paired bootstrap completed", flush=True)
        _progress(4, "end", f"comparisons={len(comparisons)} value_score={value_score}")

    phase_start = time.monotonic()
    if progress:
        _progress(5, "start", "run journal-to-opportunity E2E controls")
    e2e_rows = _e2e_rows(panel, paths)
    _atomic_write(paths.e2e_sidecar, _canonical_bytes({"schema": SCHEMA, "rows": e2e_rows}))
    prequential_errors = prequential_row_errors(panel.event_rows, selected)
    opportunity_errors = opportunity_row_errors(panel.opportunity_rows, selected)
    expected_event_rows = len(selected) * len(ARMS) * EVENTS_PER_STREAM
    expected_opportunity_rows = len(selected) * len(ARMS) * MAX_OPPORTUNITIES
    completion_gates = {
        "complete_stream_arm_matrix": {
            "expected": [expected_event_rows, expected_opportunity_rows, 0],
            "observed": [
                len(panel.event_rows),
                len(panel.opportunity_rows),
                len(panel.censored_stream_ids),
            ],
            "passed": len(panel.event_rows) == expected_event_rows
            and len(panel.opportunity_rows) == expected_opportunity_rows
            and not panel.censored_stream_ids,
            "principle": "Account for every planned event-arm and opportunity without deleting slow units.",
        },
        "common_candidate_diagnostic": {
            "expected": [len(selected) * MAX_OPPORTUNITIES, 0, True],
            "observed": [
                diagnostic["checked_opportunity_count"],
                diagnostic["violation_count"],
                diagnostic["ran_before_benchmark"],
            ],
            "passed": diagnostic["passed"] is True and diagnostic["ran_before_benchmark"] is True,
            "principle": "Freeze candidates and all admission inputs before outcome comparison.",
        },
        "prequential_and_opportunity_conformance": {
            "expected": [[], []],
            "observed": [prequential_errors, opportunity_errors],
            "passed": not prequential_errors and not opportunity_errors,
            "principle": "Protect chronology, authority separation, quotas, capacity, and complete costs.",
        },
        "cold_evaluator_recomputation": {
            "expected": transactional.sha256_json(panel.rows),
            "observed": transactional.sha256_json(reduced),
            "passed": reduced == panel.rows,
            "principle": "Recompute headline inputs from raw disk evidence before publication.",
        },
        "e2e_journal_to_future_error": {
            "expected": len(e2e_rows),
            "observed": sum(row["passed"] is True for row in e2e_rows),
            "passed": all(row["passed"] is True for row in e2e_rows),
            "principle": "Exercise the complete stream, journal, admission, future error, and denominator path.",
        },
    }
    for gate in (*completion_gates.values(), *science_gates.values()):
        gate["pass"] = gate["passed"]
    run_complete = int(all(gate["passed"] is True for gate in completion_gates.values()))
    if not run_complete:
        value_score = 0
        verdict_class = "null"
        verdict = "complete_null: admission measurement ended with accounted task-owned censoring"
    spans["phase_5_e2e_and_gates"] = time.monotonic() - phase_start
    if progress:
        _progress(
            5,
            "end",
            f"e2e_passed={all(row['passed'] for row in e2e_rows)} run_complete={run_complete}",
        )

    prequential_receipt = _receipt(paths.prequential_rows, len(panel.event_rows))
    opportunity_receipt = _receipt(paths.opportunity_rows, len(panel.opportunity_rows))
    diagnostic_receipt = _receipt(paths.diagnostic_rows, len(diagnostic_rows))
    e2e_receipt = _receipt(paths.e2e_sidecar)
    for receipt in (prequential_receipt, opportunity_receipt, diagnostic_receipt, e2e_receipt):
        hashes[str(receipt["path"])] = str(receipt["sha256"])
    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        hashes,
        selected,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=time.monotonic() - monotonic_start,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "rows": reduced,
            "sample_size_budget": _sample_budget(
                selected, panel.completed_stream_ids, panel.censored_stream_ids, bootstrap_draws
            ),
            "acceptance_gate_results": {**completion_gates, **science_gates},
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "admission_run_complete_score": run_complete,
            "admission_value_score": value_score,
            "prequential_rows_path": str(paths.prequential_rows),
            "opportunity_rows_path": str(paths.opportunity_rows),
            "comparison_rows": comparisons,
            "opportunity_summary_rows": opportunity_summaries,
            "common_candidate_diagnostic": diagnostic,
            "causal_summary": causal,
            "admission_contract": deepcopy(upstream["admission_contract"]),
            "upstream_artifact": {
                "path": str(_resolve(repo_root, DEFAULT_UPSTREAM_ARTIFACT)),
                "sha256": EXPECTED_UPSTREAM_SHA256,
                "admission_fixture_ready_score": upstream["admission_fixture_ready_score"],
            },
            "prequential_rows_receipt": prequential_receipt,
            "opportunity_rows_receipt": opportunity_receipt,
            "diagnostic_rows_receipt": diagnostic_receipt,
            "e2e_sidecar_receipt": e2e_receipt,
            "validation_receipts": [
                {
                    "command": f"independent_reduce {paths.prequential_rows} {paths.opportunity_rows}",
                    "exit_code": 0,
                    "classification": "passed",
                    "duration_s": spans["phase_4_cold_reduction_and_bootstrap"],
                    "log_sha256": transactional.sha256_json([reduced, opportunity_summaries]),
                },
                {
                    "command": "e2e_journal_to_future_error",
                    "exit_code": 0,
                    "classification": "passed",
                    "duration_s": spans["phase_5_e2e_and_gates"],
                    "log_sha256": transactional.sha256_json(e2e_rows),
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
    """Cold-check schema, rows, gates, costs, files, and terminal class."""

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
            artifact.get(key) != 0
            for key in (
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
        or artifact.get("no_model_weight_mutation") is not True,
        "learning_contract",
    )
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    receipts = artifact.get("validation_receipts", [])
    required_receipt = {"command", "exit_code", "classification", "duration_s", "log_sha256"}
    add(
        not isinstance(receipts, list)
        or any(
            set(row) != required_receipt
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
            or artifact.get("admission_run_complete_score") != 0
            or artifact.get("admission_value_score") != 0
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
    run_complete = artifact.get("admission_run_complete_score")
    value = artifact.get("admission_value_score")
    complete_contract = (
        run_complete in {0, 1}
        and value in {0, 1}
        and (value == 0 or run_complete == 1)
        and artifact.get("verdict_class") == ("circular_positive" if value == 1 else "null")
        and str(artifact.get("honest_verdict", "")).startswith("complete_")
    )
    add(not complete_contract or artifact.get("verdict_class") == "positive", "complete_contract")
    selected = tuple(
        expected_stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    rows = artifact.get("rows", [])
    expected_units = {(stream_id, arm) for stream_id in selected for arm in ARMS}
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units
        or any(
            row.get("censored") is not False
            or int(row.get("event_count", 0)) != EVENTS_PER_STREAM
            or int(row.get("future_event_count", 0)) != EVENTS_PER_STREAM - WARMUP_COUNT
            or int(row.get("recurrence_event_count", 0)) != 256
            or int(row.get("nomination_label_count", 0)) != NOMINATION_LABEL_BUDGET
            or int(row.get("admission_label_count", 0)) != ADMISSION_LABEL_BUDGET
            or int(row.get("maximum_memory_bytes", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
            for row in rows
        ),
        "rows",
    )
    comparisons = artifact.get("comparison_rows", [])
    add(
        not isinstance(comparisons, list)
        or {row.get("comparison_id") for row in comparisons} != {row[0] for row in COMPARISON_SPECS}
        or any(
            not row.get("paired_differences") or int(row.get("independent_unit_count", 0)) <= 0
            for row in comparisons
        ),
        "comparison_rows",
    )
    gates = artifact.get("acceptance_gate_results", {})
    add(
        not isinstance(gates, Mapping)
        or not gates
        or any(
            gate.get("passed") != gate.get("pass")
            or not {"expected", "observed", "passed", "pass", "principle"} <= set(gate)
            for gate in gates.values()
        ),
        "acceptance_gate_results",
    )
    science_pass = int(
        all(gates.get(name, {}).get("passed") is True for name in SCIENTIFIC_GATE_NAMES)
    )
    add(value != science_pass, "admission_value_score")
    causal = artifact.get("causal_summary", {})
    add(value == 1 and int(causal.get("paired_admission_count", 0)) == 0, "zero_admission_positive")
    if check_files:
        sidecars = [
            artifact.get("prequential_rows_receipt", {}),
            artifact.get("opportunity_rows_receipt", {}),
            artifact.get("diagnostic_rows_receipt", {}),
            artifact.get("e2e_sidecar_receipt", {}),
        ]
        add(
            any(
                _sha256_path(_resolve(repo_root, str(row.get("path", "")))) != row.get("sha256")
                for row in sidecars
            ),
            "sidecar_hashes",
        )
        try:
            reduced, opportunity_summaries = independent_reduce(
                _resolve(repo_root, str(artifact.get("prequential_rows_path", ""))),
                _resolve(repo_root, str(artifact.get("opportunity_rows_path", ""))),
            )
        except ValueError:
            reduced, opportunity_summaries = [], []
        add(
            reduced != rows or opportunity_summaries != artifact.get("opportunity_summary_rows"),
            "cold_reducer",
        )
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
    """Attach exact validation command evidence and refresh the checksum."""

    required = {"command", "exit_code", "classification", "duration_s", "log_sha256"}
    if any(
        set(receipt) != required
        or not isinstance(receipt["command"], str)
        or not isinstance(receipt["exit_code"], int)
        or not isinstance(receipt["classification"], str)
        or not isinstance(receipt["duration_s"], (int, float))
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt["log_sha256"])) is None
        for receipt in receipts
    ):
        raise ValueError("validation_receipt_schema")
    changed = deepcopy(dict(artifact))
    changed["validation_receipts"] = [dict(receipt) for receipt in receipts]
    changed["reproducibility_checksum"] = reproducibility_checksum(changed)
    return changed


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
) -> None:
    """Cold-validate and atomically publish one terminal artifact."""

    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=expected_stream_ids,
        check_files=artifact.get("status") == "complete",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_write(path, _canonical_bytes(dict(artifact)))


def _validation_commands(candidate: Path) -> list[list[str]]:
    """Return focused coverage, affected checks, E2E, and artifact validation."""

    python = str(REPO_ROOT / ".venv/bin/python")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    module = "python/carnot/experiment_7282_v640_admission_learning.py"
    test = str(TEST_PATH)
    wrapper = str(WRAPPER_PATH)
    return [
        [coverage, "erase"],
        [
            coverage,
            "run",
            f"--include={REPO_ROOT / module}",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-n",
            "0",
            "--basetemp=/tmp/carnot-exp7282-coverage",
            test,
            "-q",
        ],
        [
            coverage,
            "report",
            f"--include={REPO_ROOT / module}",
            "--show-missing",
            "--fail-under=100",
        ],
        [
            python,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "--no-cov",
            "-n",
            "0",
            "--basetemp=/tmp/carnot-exp7282-affected",
            "tests/python/test_experiment_7281_v640_admission_prototype.py::test_scenario_cl_7281_panel_preserves_quotas_controls_and_rows",
            "tests/python/test_experiment_7268_v639_recognition_learning.py::test_scenario_cl_7268_bootstrap_and_value_gates_are_fail_closed",
            "tests/python/test_experiment_7269_v639_recognition_audit.py::test_bounds_keep_overlap_recall_false_accepts_and_zero_effects",
            "-q",
        ],
        [ruff, "check", module, test, wrapper],
        [ruff, "format", "--check", module, test, wrapper],
        [mypy, module, wrapper],
        [python, "scripts/check_spec_coverage.py", test],
        [
            python,
            "-u",
            wrapper,
            "--date",
            RUN_DATE,
            "--e2e-worker",
            "--output-root",
            "/tmp/carnot-exp7282-e2e-validation",
        ],
        [
            python,
            "-u",
            wrapper,
            "--date",
            RUN_DATE,
            "--validate",
            "--artifact-path",
            str(candidate),
        ],
        [python, "scripts/adversarial_verify.py", "--json", str(candidate)],
        [python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date and private validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--e2e-worker", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--stream-id", action="append")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, validate, and publish only terminal admission evidence."""

    print("phase 0 immediate: Exp7282 admission learning started", flush=True)
    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run_date_must_be_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    if args.e2e_worker:
        _progress(1, "start", "BEFORE private complete-stream E2E benchmark")
        rows = run_e2e_controls(paths.provisional.parent / "worker_e2e")
        _progress(1, "end", f"AFTER private E2E stages={len(rows)}")
        return 0
    if args.validate:
        if args.artifact_path is None:
            raise SystemExit("artifact_path_required")
        _progress(1, "start", "BEFORE cold artifact validation")
        artifact = _load_object(args.artifact_path)
        expected = tuple(args.stream_id) if args.stream_id else None
        errors = validate_artifact(artifact, expected_stream_ids=expected, check_files=True)
        if errors:
            raise SystemExit("artifact_validation_failed:" + ",".join(errors))
        _progress(1, "end", "AFTER cold artifact validation passed")
        return 0
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact)
        _progress(6, "end", f"wrote blocked terminal artifact {paths.artifact}")
        return 0
    _progress(6, "start", "write measured candidate under raw evidence")
    _atomic_write(paths.terminal_candidate, _canonical_bytes(artifact))
    _progress(6, "end", f"candidate={paths.terminal_candidate}")
    _progress(7, "start", "BEFORE focused tests, static checks, E2E, and artifact checks")
    receipts = list(artifact["validation_receipts"])
    for command in _validation_commands(paths.terminal_candidate):
        receipts.append(fixture._command_receipt(command))
    artifact = attach_validation_receipts(artifact, receipts)
    _atomic_write(
        paths.provisional,
        _canonical_bytes(
            {"schema": SCHEMA, "status": "in_progress", "phase": 7, "validation_receipts": receipts}
        ),
    )
    failed = [row for row in receipts if row["exit_code"] != 0]
    if failed:
        raise RuntimeError(
            "focused_validation_failed:" + ",".join(str(row["command"]) for row in failed)
        )
    _progress(7, "end", "AFTER all focused validations passed")
    _progress(8, "start", "BEFORE final cold validation and atomic terminal write")
    write_artifact(paths.artifact, artifact)
    _progress(8, "end", f"AFTER atomic terminal write {paths.artifact}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
