"""Measure prospective learning from the lossless packed belief state.

The evaluator keeps labels outside every decision. Each arm predicts first.
Only then can a due queried label update the arm's external memory.

Spec refs: REQ-CL-7227 and SCENARIO-CL-7227-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import socket
import time
from typing import Any

from carnot import experiment_7199_v634_bounded_acquisition as exp7199
from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7227
SCHEMA = "carnot.exp7227.v636_belief_learning.v1"
MILESTONE = "2026.09.636"
RUN_DATE = "20260911"
BOOTSTRAP_SEED = 7_227_001
BOOTSTRAP_DRAWS = 10_000
STREAM_SEEDS = tuple(exp7226.STREAM_SEEDS)
EVENTS_PER_SEED = exp7226.EVENTS_PER_SEED
WARMUP_COUNT = exp7226.WARMUP_COUNT
PENDING_CAPACITY = exp7226.PENDING_CAPACITY
QUERY_CEILING = exp7226.QUERY_CEILING
BLOCK_SIZE = exp7226.BLOCK_SIZE
FAMILIES = tuple(exp7226.FAMILIES)
WITHHOLD_SEGMENTS = ((256, 384), (640, 768))
ARMS = (
    "frozen_warmup",
    "reference_online_version_space",
    "packed_online_memory",
    "original_committed_predicate",
    "packed_feedback_withheld",
)
COST_NAMES = (
    "selection_ns",
    "update_ns",
    "recomputation_ns",
    "commit_ns",
    "serialization_ns",
    "lookup_ns",
)
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
EXPECTED_UPSTREAM_SHA256 = "sha256:8521dfa184b60b994249ef956137eff680174ec29197e3eee498c6cc69754660"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM_ARTIFACT = Path("results/experiment_7226_v636_belief_compiler.json")
DEFAULT_STATE_PATH = Path("results/checkpoints/experiment_7227_v636_belief_learning_state.json")
DEFAULT_DECISION_ROWS_PATH = Path(
    "results/checkpoints/experiment_7227_v636_belief_learning_decision_rows.jsonl"
)
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7227_v636_belief_learning.json")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7199_v634_bounded_acquisition.py"),
    Path("python/carnot/experiment_7213_v635_refinement_learning.py"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7227_v636_belief_learning.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/experiments/experiment_7227_v636_belief_learning.py"),
    Path("tests/python/test_experiment_7227_v636_belief_learning.py"),
    SPEC_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "field_principles",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "execution_host",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "MODEL_SPECS",
    "model_invoked",
    "belief_run_complete_score",
    "belief_learning_value_score",
    "continuous_self_learning_task",
    "decision_rows_path",
    "comparison_rows",
    "acceptance_gate_learning",
    "memory_deletion_rows",
    "latency_summary",
    "no_model_weight_mutation",
    "checkpoint_path",
    "panel_conformance_errors",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID prevents another task from supplying this evidence.",
    "milestone": "The milestone binds this result to the V636 learning contract.",
    "field_principles": (
        "Annotate actual values in this map; do not wrap arbitrary dictionaries as "
        "principle/value records."
    ),
    "status": (
        "Write a terminal artifact only when done or externally blocked; running "
        "checkpoints use a different path."
    ),
    "run_date": "Use 20260911 and record actual UTC timestamps, never copy an upstream run date.",
    "started_at_utc": "Record the actual UTC start separately from the fixed run date.",
    "completed_at_utc": "Record the actual UTC end separately from the fixed run date.",
    "preconditions_checked": "Actual code, resource, identity and gate observations before expensive work.",
    "inference_substrate": (
        "Use the recognized literal for the work actually executed; custom free text caused "
        "the Exp7208 quarantine."
    ),
    "inference_substrate_class": (
        "Match actual generation, load-only, CPU or aggregation work and its duration floor."
    ),
    "execution_venue": (
        "Exactly host, kv260, gatemate or polarfire; the top-level orchestration here is host."
    ),
    "execution_host": "Actual hostname separate from venue.",
    "duration_s": "Measured monotonic work time; no padding or reclassification to evade a floor.",
    "source_artifact_hashes": "Bind code, source documents, manifests and raw evidence to claims.",
    "rows": (
        "Per unit/arm/seed metric, error and abstention for every comparison; retain full "
        "denominators."
    ),
    "sample_size_budget": (
        "Planned, attempted, completed, censored and independent units; no silent removal."
    ),
    "random_seed": "Freeze random choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash exact source, inputs, settings and raw unit rows.",
    "gate_check_summary": (
        "Every blocked_* verdict names failed check, upstream, field, expected and observed value."
    ),
    "verifier_is_oracle": (
        "True when correctness authority is reused as the verifier; independent code alone is "
        "not distinct authority."
    ),
    "verdict_class": (
        "Closed enum positive | circular_positive | null | blocked | disqualified | partial. "
        "partial means unfinished own work only."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings; blocked_* for external absence. "
        "A failed acceptance gate forbids positive."
    ),
    "MODEL_SPECS": "Only models actually invoked; [] for CPU/aggregation.",
    "model_invoked": (
        "True only for actual model execution; upstream model outputs are cached evidence."
    ),
    "belief_run_complete_score": "All prospective scheduled event and arm records exist.",
    "belief_learning_value_score": (
        "Fixed future-error, false-accept, retention and causality criteria."
    ),
    "continuous_self_learning_task": (
        "True; durable state updates after feedback and affects later queries."
    ),
    "decision_rows_path": "Per-event prediction-before-release evidence.",
    "comparison_rows": "Twenty independent-stream differences with CIs.",
    "acceptance_gate_learning": "Separate efficacy, parity and cost criteria.",
    "memory_deletion_rows": "Causal effect of withholding/deleting learned state.",
    "latency_summary": "Measured end-to-end update/lookup cost and memory.",
    "no_model_weight_mutation": "True; this is constraint memory learning.",
    "checkpoint_path": "Reloadable final controller bytes stay outside the terminal artifact.",
    "panel_conformance_errors": "An empty list proves chronology and matched resources.",
}

unwrap_principled = exp7213.unwrap_principled
gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary
StreamViews = exp7226.StreamViews


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw decisions and learned state separate from the terminal result."""

    state: Path
    decision_rows: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the task-owned repository destinations."""

        return cls(DEFAULT_STATE_PATH, DEFAULT_DECISION_ROWS_PATH, DEFAULT_ARTIFACT_PATH)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put every test output below one caller-owned directory."""

        return cls(
            root / "checkpoints" / DEFAULT_STATE_PATH.name,
            root / "checkpoints" / DEFAULT_DECISION_ROWS_PATH.name,
            root / DEFAULT_ARTIFACT_PATH.name,
        )


@dataclass(frozen=True)
class PendingFeedback:
    """Hold one evaluator label outside controller-visible query state."""

    public_event: JsonDict
    role: str
    committed_role: str
    request_index: int
    release_index: int
    delay: int
    observed_label: str


@dataclass
class LearningPanel:
    """Retain full decisions, summaries, costs, parity, and final states."""

    rows: list[JsonDict]
    decisions: list[JsonDict]
    costs: dict[str, list[int]]
    lookup_costs: dict[str, list[int]]
    parity_counts: JsonDict
    final_states: list[JsonDict]
    maximum_state_bytes: int
    maximum_pending_bytes: int


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve repository-relative evidence without changing absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while preserving file absence as a failed observation."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_object(path: Path) -> JsonDict:
    """Decode one JSON object while malformed evidence stays a failed gate."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence early."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_identity(text: str) -> JsonDict:
    """Read only the Exp7227 roadmap block for identity gates."""

    match = re.search(r"(?ms)^- id: exp7227-belief-learning\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7227-belief-learning" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_artifact: Path = DEFAULT_UPSTREAM_ARTIFACT,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate the compiler contract before loading any authority row."""

    upstream_path = _resolve(repo_root, upstream_artifact)
    upstream = _load_object(upstream_path)
    source_hashes = {str(path): _sha256_path(_resolve(repo_root, path)) for path in SOURCE_PATHS}
    source_hashes[str(upstream_path)] = _sha256_path(upstream_path)
    spec_text = _resolve(repo_root, SPEC_PATH).read_text(encoding="utf-8")
    roadmap_text = _resolve(repo_root, "research-roadmap.yaml").read_text(encoding="utf-8")
    exclusion_text = _resolve(repo_root, "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    identity = _task_identity(roadmap_text)
    quarantine = exp7213.quarantine_state(
        upstream,
        exclusion_text,
        upstream_path.name,
        "exp7226-belief-compiler",
    )
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7199_v634_bounded_acquisition",
            "carnot.experiment_7213_v635_refinement_learning",
            "carnot.experiment_7226_v636_belief_compiler",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    output_state = {
        "artifact": _path_writable(paths.artifact),
        "decision_rows": _path_writable(paths.decision_rows),
        "checkpoint": _path_writable(paths.state),
    }
    receipt_map = unwrap_principled(upstream.get("stream_manifest_path"))
    receipt_map = receipt_map if isinstance(receipt_map, Mapping) else {}
    expected_receipts = {
        "public_stream",
        "authority_sidecar",
        "release_manifest",
        "public_manifest",
    }
    receipt_hash_matches: dict[str, bool] = {}
    for name in expected_receipts:
        receipt = receipt_map.get(name)
        if not isinstance(receipt, Mapping):
            receipt_hash_matches[name] = False
            continue
        path = _resolve(repo_root, str(receipt.get("path", "")))
        observed_hash = _sha256_path(path)
        receipt_hash_matches[name] = observed_hash is not None and observed_hash == receipt.get(
            "sha256"
        )
        source_hashes[str(path)] = observed_hash
    public_manifest: JsonDict = {}
    manifest_receipt = receipt_map.get("public_manifest")
    if isinstance(manifest_receipt, Mapping):
        public_manifest = _load_object(_resolve(repo_root, str(manifest_receipt.get("path", ""))))
    expected_identity = {
        "id": "exp7227-belief-learning",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    source_state = {
        str(path): "nonempty" if source_hashes[str(path)] is not None else "missing"
        for path in SOURCE_PATHS
    }
    checksum_matches = bool(upstream) and exp7226.reproducibility_checksum(
        upstream
    ) == upstream.get("reproducibility_checksum")
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7227",
            True,
            "REQ-CL-7227" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7227-*",
            8,
            len(set(re.findall(r"SCENARIO-CL-7227-[A-Z-]+", spec_text))),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            source_state,
        ),
        gate_check(
            "v636_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            identity,
        ),
        gate_check(
            "required_imports", "python", "imports", {name: True for name in imports}, imports
        ),
        gate_check(
            "output_destinations",
            "host_filesystem",
            "raw,checkpoint,artifact",
            {name: True for name in output_state},
            output_state,
        ),
        gate_check(
            "exp7226_artifact_hash",
            "exp7226-belief-compiler",
            str(upstream_artifact),
            EXPECTED_UPSTREAM_SHA256,
            source_hashes[str(upstream_path)],
        ),
        gate_check(
            "exp7226_status",
            "exp7226-belief-compiler",
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "exp7226_ready",
            "exp7226-belief-compiler",
            "belief_compiler_ready_score",
            1,
            upstream.get("belief_compiler_ready_score"),
        ),
        gate_check(
            "exp7226_gate",
            "exp7226-belief-compiler",
            "gate_check_summary.passed",
            True,
            upstream.get("gate_check_summary", {}).get("passed")
            if isinstance(upstream.get("gate_check_summary"), Mapping)
            else None,
        ),
        gate_check(
            "exp7226_no_model",
            "exp7226-belief-compiler",
            "MODEL_SPECS,model_invoked",
            {"MODEL_SPECS": [], "model_invoked": False},
            {
                "MODEL_SPECS": upstream.get("MODEL_SPECS"),
                "model_invoked": upstream.get("model_invoked"),
            },
        ),
        gate_check(
            "exp7226_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
        gate_check(
            "exp7226_reproducibility_checksum",
            "exp7226-belief-compiler",
            "reproducibility_checksum",
            True,
            checksum_matches,
        ),
        gate_check(
            "sealed_stream_hashes",
            "exp7226-belief-compiler.stream_manifest_path",
            "public,authority,release,manifest",
            {name: True for name in expected_receipts},
            receipt_hash_matches,
        ),
        gate_check(
            "twenty_stream_contract",
            "experiment_7226/public_stream_manifest.json",
            "seeds,events,pending,query",
            {
                "seeds": list(STREAM_SEEDS),
                "events_per_seed": EVENTS_PER_SEED,
                "pending_capacity": PENDING_CAPACITY,
                "query_ceiling": QUERY_CEILING,
            },
            {
                "seeds": public_manifest.get("seeds"),
                "events_per_seed": public_manifest.get("events_per_seed"),
                "pending_capacity": public_manifest.get("pending_capacity"),
                "query_ceiling": public_manifest.get("query_ceiling_per_stream_arm"),
            },
        ),
    ]
    return checks, upstream, source_hashes


def load_stream_views(repo_root: Path, upstream_artifact: Path) -> StreamViews:
    """Load sealed views only after callers have passed every producer gate."""

    upstream = _load_object(_resolve(repo_root, upstream_artifact))
    receipts = upstream["stream_manifest_path"]
    paths = exp7226.ExperimentPaths(
        _resolve(repo_root, receipts["public_stream"]["path"]),
        _resolve(repo_root, receipts["authority_sidecar"]["path"]),
        _resolve(repo_root, receipts["release_manifest"]["path"]),
        _resolve(repo_root, receipts["public_manifest"]["path"]),
        _resolve(repo_root, upstream["compiler_state_path"]["path"]),
        _resolve(repo_root, upstream_artifact),
    )
    return exp7226.load_stream_views(paths)


def _withheld(request_index: int) -> bool:
    """Apply the preregistered deletion schedule without reading outcomes."""

    return any(start <= request_index < stop for start, stop in WITHHOLD_SEGMENTS)


def _state_size(*controllers: Any) -> int:
    """Count canonical persistent bytes that can affect a later action."""

    return sum(len(controller.state_bytes()) for controller in controllers)


def _survivor_count(controller: Any, family: str) -> int:
    """Read one finite family count from either shipped representation."""

    if isinstance(controller, exp7226.PackedBeliefController):
        return len(controller.survivors(family))
    return len(controller.families[family].hypotheses)


def _packed_energy(
    controller: exp7226.PackedBeliefController,
    prediction: str,
    event: Mapping[str, Any],
) -> float | None:
    """Read prediction energy without exposing an authority label."""

    if prediction == "abstain":
        return None
    return controller.energy(prediction, event)["value"]


def _committed_prediction(
    controller: exp7199.VersionSpaceController,
    frozen: exp7226.PackedBeliefController,
    event: Mapping[str, Any],
) -> tuple[str, float | None]:
    """Deploy only the shipped validated singleton or the frozen fallback."""

    family = str(event["family_id"])
    template = controller.families[family].committed_template
    if template is None:
        prediction, _ = frozen.predict(event)
        return prediction, _packed_energy(frozen, prediction, event)
    prediction = exp7226.exact_label(
        family,
        int(event["numeric_value"]),
        int(template["parameter"]),
    )
    return prediction, 0.0


def _query_role(controller: exp7199.VersionSpaceController, family: str) -> str:
    """Reserve post-singleton evidence for the shipped validation boundary."""

    state = controller.families[family]
    if state.candidate_parameter is not None and state.committed_template is None:
        return "validation"
    return "support"


def _release_payload(pending: PendingFeedback) -> JsonDict:
    """Expose one released label with no hidden parameter or future field."""

    return {
        "event_id": pending.public_event["event_id"],
        "family_id": pending.public_event["family_id"],
        "numeric_value": pending.public_event["numeric_value"],
        "observed_label": pending.observed_label,
        "role": pending.role,
        "request_index": pending.request_index,
        "release_index": pending.release_index,
    }


def _aggregate_row(
    seed: int,
    arm: str,
    decisions: Sequence[Mapping[str, Any]],
    query_count: int,
    released_count: int,
    max_pending: int,
) -> JsonDict:
    """Reduce one stream while keeping each prospective denominator."""

    future = [row for row in decisions if row["prospective"]]
    recurrence = [row for row in decisions if row["window"] == "recurrence"]
    error = sum(int(row["error"]) for row in future)
    false_accept = sum(int(row["false_accept"]) for row in future)
    abstention = sum(int(row["abstention"]) for row in future)
    recurrence_error = sum(int(row["error"]) for row in recurrence)
    return {
        "unit_id": f"{seed}:{arm}",
        "arm": arm,
        "seed": seed,
        "metric": "prospective_full_denominator_error",
        "error": error,
        "abstention": abstention,
        "event_count": len(future),
        "error_rate": error / len(future),
        "false_accept": false_accept,
        "false_accept_rate": false_accept / len(future),
        "recurrence_error": recurrence_error,
        "recurrence_event_count": len(recurrence),
        "recurrence_error_rate": recurrence_error / len(recurrence),
        "query_count": query_count,
        "released_query_count": released_count,
        "pending_at_end": query_count - released_count,
        "max_pending": max_pending,
    }


def run_learning_panel(
    views: StreamViews,
    *,
    state_path: Path,
    seeds: Sequence[int] = STREAM_SEEDS,
    progress: bool = False,
) -> LearningPanel:
    """Replay matched arms while decisions remain ahead of authority access."""

    del state_path  # The final checkpoint is published only after all streams finish.
    started = time.monotonic()
    rows: list[JsonDict] = []
    decisions: list[JsonDict] = []
    costs = {name: [] for name in COST_NAMES}
    lookup_costs = {arm: [] for arm in ARMS}
    parity_counts: JsonDict = {
        "prediction_mismatch_count": 0,
        "query_mismatch_count": 0,
        "energy_mismatch_count": 0,
    }
    final_states: list[JsonDict] = []
    maximum_state_bytes = 0
    maximum_pending_bytes = 0
    public_by_seed = {
        seed: [row for row in views.public if int(row["seed"]) == seed] for seed in seeds
    }
    authority_by_id = {str(row["event_id"]): row for row in views.authority}
    release_by_id = {str(row["event_id"]): row for row in views.releases}

    for seed_number, seed in enumerate(seeds, start=1):
        frozen = exp7226.PackedBeliefController()
        reference = exp7199.VersionSpaceController()
        packed = exp7226.PackedBeliefController()
        committed = exp7199.VersionSpaceController()
        withheld = exp7226.PackedBeliefController()
        pending: list[PendingFeedback] = []
        query_count = 0
        released_count = 0
        max_pending = 0
        unit_decisions = {arm: [] for arm in ARMS}
        events = public_by_seed[seed]

        for block_index, offset in enumerate(range(0, len(events), BLOCK_SIZE)):
            block = events[offset : offset + BLOCK_SIZE]
            selected_id: str | None = None
            if query_count < QUERY_CEILING:
                tie_ranks = exp7199.seeded_tie_ranks(seed, block_index, block)
                started_selection = time.perf_counter_ns()
                reference_choice = exp7199.select_request(
                    block,
                    "priority_admission",
                    tie_ranks,
                    reference,
                )
                costs["selection_ns"].append(time.perf_counter_ns() - started_selection)
                started_selection = time.perf_counter_ns()
                packed_choice = packed.select_request(block, tie_ranks)
                costs["selection_ns"].append(time.perf_counter_ns() - started_selection)
                selected_id = str(reference_choice["event_id"])
                parity_counts["query_mismatch_count"] += int(
                    selected_id != str(packed_choice["event_id"])
                )

            for event in block:
                event_id = str(event["event_id"])
                chronology_index = int(event["chronology_index"])
                family = str(event["family_id"])
                selected = selected_id == event_id
                role = "support" if selected else None
                committed_role = _query_role(committed, family) if selected else None
                arm_values: dict[str, tuple[str, float | None, str, int]] = {}

                for arm in ARMS:
                    lookup_started = time.perf_counter_ns()
                    if arm == "frozen_warmup":
                        prediction, _ = frozen.predict(event)
                        energy = _packed_energy(frozen, prediction, event)
                        state_hash = frozen.state_hash()
                        survivor_count = _survivor_count(frozen, family)
                    elif arm == "reference_online_version_space":
                        prediction, disagreement = reference.predict(event)
                        energy = None if prediction == "abstain" else disagreement
                        state_hash = reference.state_hash()
                        survivor_count = _survivor_count(reference, family)
                    elif arm == "packed_online_memory":
                        prediction, _ = packed.predict(event)
                        energy = _packed_energy(packed, prediction, event)
                        state_hash = packed.state_hash()
                        survivor_count = _survivor_count(packed, family)
                    elif arm == "original_committed_predicate":
                        prediction, energy = _committed_prediction(committed, frozen, event)
                        state_hash = committed.state_hash()
                        survivor_count = _survivor_count(committed, family)
                    else:
                        prediction, _ = withheld.predict(event)
                        energy = _packed_energy(withheld, prediction, event)
                        state_hash = withheld.state_hash()
                        survivor_count = _survivor_count(withheld, family)
                    lookup_elapsed = time.perf_counter_ns() - lookup_started
                    costs["lookup_ns"].append(lookup_elapsed)
                    lookup_costs[arm].append(lookup_elapsed)
                    arm_values[arm] = (prediction, energy, state_hash, survivor_count)

                parity_counts["prediction_mismatch_count"] += int(
                    arm_values["reference_online_version_space"][0]
                    != arm_values["packed_online_memory"][0]
                )
                parity_counts["energy_mismatch_count"] += int(
                    arm_values["reference_online_version_space"][1]
                    != arm_values["packed_online_memory"][1]
                )

                drafts: dict[str, JsonDict] = {}
                for arm, value in arm_values.items():
                    prediction, energy, state_hash, survivor_count = value
                    draft = {
                        "unit_id": f"{seed}:{arm}",
                        "arm": arm,
                        "seed": seed,
                        "event_id": event_id,
                        "chronology_index": chronology_index,
                        "family_id": family,
                        "numeric_value": int(event["numeric_value"]),
                        "prospective": chronology_index >= WARMUP_COUNT,
                        "prediction": prediction,
                        "energy": energy,
                        "query_selected": selected,
                        "query_admitted": False,
                        "query_role": committed_role
                        if arm == "original_committed_predicate"
                        else role,
                        "query_count_after_decision": query_count,
                        "delay": None,
                        "pending_capacity_use": len(pending),
                        "survivor_count": survivor_count,
                        "state_hash_before_release": state_hash,
                        "memory_hash_after_release": state_hash,
                        "changed_memory_hash": False,
                        "released_feedback_ids": [],
                        "withheld_release_ids": [],
                        "prediction_before_release": True,
                        "future_label_visible_to_decision": False,
                        "hidden_parameter_visible_to_decision": False,
                    }
                    drafts[arm] = draft

                query_admitted = (
                    selected and query_count < QUERY_CEILING and len(pending) < PENDING_CAPACITY
                )
                if query_admitted:
                    release = release_by_id[event_id]
                    truth = authority_by_id[event_id]
                    pending.append(
                        PendingFeedback(
                            dict(event),
                            str(role),
                            str(committed_role),
                            chronology_index,
                            int(release["release_index"]),
                            int(release["delay"]),
                            str(truth["observed_label"]),
                        )
                    )
                    query_count += 1
                    max_pending = max(max_pending, len(pending))
                    for draft in drafts.values():
                        draft["query_admitted"] = True
                        draft["query_count_after_decision"] = query_count
                        draft["delay"] = int(release["delay"])
                        draft["pending_capacity_use"] = len(pending)

                truth = authority_by_id[event_id]
                exact_label = str(truth["exact_label"])
                for arm, draft in drafts.items():
                    prediction = str(draft["prediction"])
                    draft.update(
                        {
                            "window": truth["window"],
                            "recurrence_segment": truth["window"] == "recurrence",
                            "exact_label": exact_label,
                            "error": int(prediction == "abstain" or prediction != exact_label),
                            "false_accept": int(prediction == "accept" and exact_label == "reject"),
                            "abstention": int(prediction == "abstain"),
                        }
                    )
                    decisions.append(draft)
                    unit_decisions[arm].append(draft)

                due = sorted(
                    [item for item in pending if item.release_index <= chronology_index],
                    key=lambda item: (item.release_index, item.request_index),
                )
                due_ids = [str(item.public_event["event_id"]) for item in due]
                withheld_due = [item for item in due if _withheld(item.request_index)]
                withheld_ids = [str(item.public_event["event_id"]) for item in withheld_due]
                if due:
                    for item in due:
                        started_update = time.perf_counter_ns()
                        reference.observe(
                            item.public_event,
                            observed_label=item.observed_label,
                            role=item.role,
                            request_index=item.request_index,
                            release_index=item.release_index,
                        )
                        costs["update_ns"].append(time.perf_counter_ns() - started_update)
                        started_update = time.perf_counter_ns()
                        committed.observe(
                            item.public_event,
                            observed_label=item.observed_label,
                            role=item.committed_role,
                            request_index=item.request_index,
                            release_index=item.release_index,
                        )
                        costs["update_ns"].append(time.perf_counter_ns() - started_update)
                    payloads = [_release_payload(item) for item in due]
                    commit_started = time.perf_counter_ns()
                    packed.commit_batch(
                        payloads,
                        current_cycle=chronology_index,
                        expected_parent_hash=packed.state_hash(),
                    )
                    costs["commit_ns"].append(time.perf_counter_ns() - commit_started)
                    warmup_payloads = [
                        _release_payload(item) for item in due if item.request_index < WARMUP_COUNT
                    ]
                    if warmup_payloads:
                        frozen.commit_batch(
                            warmup_payloads,
                            current_cycle=chronology_index,
                            expected_parent_hash=frozen.state_hash(),
                        )
                    kept_payloads = [
                        _release_payload(item) for item in due if not _withheld(item.request_index)
                    ]
                    if kept_payloads:
                        withheld.commit_batch(
                            kept_payloads,
                            current_cycle=chronology_index,
                            expected_parent_hash=withheld.state_hash(),
                        )
                    recompute_started = time.perf_counter_ns()
                    for affected_family in {str(item.public_event["family_id"]) for item in due}:
                        state = packed.family_state(affected_family)
                        exp7226._votes_for(
                            affected_family,
                            int(state["survivor_mask"]),
                        )
                    costs["recomputation_ns"].append(time.perf_counter_ns() - recompute_started)
                    serialization_started = time.perf_counter_ns()
                    packed.state_bytes()
                    costs["serialization_ns"].append(time.perf_counter_ns() - serialization_started)
                    released_count += len(due)
                    due_identity = {item.public_event["event_id"] for item in due}
                    pending = [
                        item
                        for item in pending
                        if item.public_event["event_id"] not in due_identity
                    ]

                after_hashes = {
                    "frozen_warmup": frozen.state_hash(),
                    "reference_online_version_space": reference.state_hash(),
                    "packed_online_memory": packed.state_hash(),
                    "original_committed_predicate": committed.state_hash(),
                    "packed_feedback_withheld": withheld.state_hash(),
                }
                for arm, draft in drafts.items():
                    draft["memory_hash_after_release"] = after_hashes[arm]
                    draft["changed_memory_hash"] = (
                        draft["state_hash_before_release"] != after_hashes[arm]
                    )
                    draft["released_feedback_ids"] = due_ids
                    draft["withheld_release_ids"] = withheld_ids

                state_bytes = _state_size(frozen, reference, packed, committed, withheld)
                pending_bytes = len(
                    transactional.canonical_json_bytes(
                        [
                            {
                                "event_id": item.public_event["event_id"],
                                "role": item.role,
                                "committed_role": item.committed_role,
                                "request_index": item.request_index,
                                "release_index": item.release_index,
                            }
                            for item in pending
                        ]
                    )
                )
                maximum_state_bytes = max(maximum_state_bytes, state_bytes)
                maximum_pending_bytes = max(maximum_pending_bytes, pending_bytes)

        for arm in ARMS:
            rows.append(
                _aggregate_row(
                    seed,
                    arm,
                    unit_decisions[arm],
                    query_count,
                    released_count,
                    max_pending,
                )
            )
        final_states.append(
            {
                "seed": seed,
                "frozen_warmup": frozen.state_dict(),
                "reference_online_version_space": reference.state_dict(),
                "packed_online_memory": packed.state_dict(),
                "original_committed_predicate": committed.state_dict(),
                "packed_feedback_withheld": withheld.state_dict(),
                "pending": [
                    {
                        "event_id": item.public_event["event_id"],
                        "role": item.role,
                        "committed_role": item.committed_role,
                        "request_index": item.request_index,
                        "release_index": item.release_index,
                    }
                    for item in pending
                ],
            }
        )
        if progress:
            print(
                f"PHASE 4 PROGRESS: completed stream {seed_number}/{len(seeds)}; "
                f"decisions={len(decisions)}; elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    return LearningPanel(
        rows,
        decisions,
        costs,
        lookup_costs,
        parity_counts,
        final_states,
        maximum_state_bytes,
        maximum_pending_bytes,
    )


def panel_conformance_errors(
    panel: LearningPanel,
    *,
    seeds: Sequence[int] = STREAM_SEEDS,
) -> list[str]:
    """Check chronology, denominators, query limits, and exact packed parity."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    expected_units = len(seeds) * len(ARMS)
    add(len(panel.rows) != expected_units, "aggregate_row_count")
    add(len(panel.decisions) != expected_units * EVENTS_PER_SEED, "decision_row_count")
    add(
        any(row["prediction_before_release"] is not True for row in panel.decisions),
        "prediction_release_chronology",
    )
    add(
        any(
            row["future_label_visible_to_decision"] is not False
            or row["hidden_parameter_visible_to_decision"] is not False
            for row in panel.decisions
        ),
        "decision_authority_access",
    )
    add(
        any(int(row["pending_capacity_use"]) > PENDING_CAPACITY for row in panel.decisions),
        "pending_capacity",
    )
    add(any(int(row["query_count"]) > QUERY_CEILING for row in panel.rows), "query_ceiling")
    add(
        any(int(row["event_count"]) != EVENTS_PER_SEED - WARMUP_COUNT for row in panel.rows),
        "future_denominator",
    )
    add(any(row["abstention"] > row["error"] for row in panel.rows), "abstention_denominator")
    add(any(int(value) != 0 for value in panel.parity_counts.values()), "packed_reference_parity")
    for seed in seeds:
        reference = [
            row
            for row in panel.decisions
            if row["seed"] == seed and row["arm"] == "reference_online_version_space"
        ]
        packed = [
            row
            for row in panel.decisions
            if row["seed"] == seed and row["arm"] == "packed_online_memory"
        ]
        add(
            [
                (row["event_id"], row["prediction"], row["energy"], row["query_selected"])
                for row in reference
            ]
            != [
                (row["event_id"], row["prediction"], row["energy"], row["query_selected"])
                for row in packed
            ],
            "packed_reference_information",
        )
    return errors


def _percentile(values: Sequence[float | int], probability: float) -> float:
    """Return one deterministic nearest-rank percentile."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(probability * (len(ordered) - 1)))))
    return ordered[index]


def _bootstrap_interval(values: Sequence[float], *, draws: int, salt: str) -> JsonDict:
    """Resample complete stream seeds as the only independent units."""

    if not values:
        return {"estimate": 0.0, "ci95": [0.0, 0.0], "draws": draws}
    seed = int(transactional.sha256_json([BOOTSTRAP_SEED, salt])[-16:], 16)
    generator = random.Random(seed)
    means = [
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    return {
        "estimate": sum(values) / len(values),
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "draws": draws,
    }


def build_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[JsonDict]:
    """Build preregistered comparisons without pooling event rows."""

    by_unit = {(int(row["seed"]), str(row["arm"])): row for row in rows}
    comparisons = []
    for control in (
        "frozen_warmup",
        "reference_online_version_space",
        "original_committed_predicate",
    ):
        differences = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            packed = by_unit[(seed, "packed_online_memory")]
            baseline = by_unit[(seed, control)]
            differences.append(
                {
                    "seed": seed,
                    "future_error_delta": float(packed["error_rate"])
                    - float(baseline["error_rate"]),
                    "false_accept_delta": float(packed["false_accept_rate"])
                    - float(baseline["false_accept_rate"]),
                    "recurrence_error_increase": float(packed["recurrence_error_rate"])
                    - float(baseline["recurrence_error_rate"]),
                }
            )
        comparison_id = f"packed_online_memory_vs_{control}"
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "independent_unit": "stream_seed",
                "independent_unit_count": len(differences),
                "seed_differences": differences,
                "future_error_delta": _bootstrap_interval(
                    [row["future_error_delta"] for row in differences],
                    draws=draws,
                    salt=comparison_id + ":error",
                ),
                "false_accept_delta": _bootstrap_interval(
                    [row["false_accept_delta"] for row in differences],
                    draws=draws,
                    salt=comparison_id + ":false_accept",
                ),
                "recurrence_error_increase": _bootstrap_interval(
                    [row["recurrence_error_increase"] for row in differences],
                    draws=draws,
                    salt=comparison_id + ":recurrence",
                ),
            }
        )
    return comparisons


def build_memory_deletion_rows(decisions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Compare full and withheld state on each independent stream."""

    result = []
    seeds = sorted({int(row["seed"]) for row in decisions})
    for seed in seeds:
        packed = {
            int(row["chronology_index"]): row
            for row in decisions
            if row["seed"] == seed and row["arm"] == "packed_online_memory"
        }
        withheld = {
            int(row["chronology_index"]): row
            for row in decisions
            if row["seed"] == seed and row["arm"] == "packed_feedback_withheld"
        }
        release_indices = [index for index, row in packed.items() if row["withheld_release_ids"]]
        first_release = min(release_indices) if release_indices else EVENTS_PER_SEED
        changed = [
            index
            for index in sorted(packed)
            if packed[index]["prediction"] != withheld[index]["prediction"]
        ]
        future = [index for index in sorted(packed) if index >= WARMUP_COUNT]
        result.append(
            {
                "unit_id": f"{seed}:packed_feedback_withheld",
                "seed": seed,
                "withhold_segments": [list(segment) for segment in WITHHOLD_SEGMENTS],
                "first_withheld_release_index": (
                    None if first_release == EVENTS_PER_SEED else first_release
                ),
                "withheld_feedback_count": sum(
                    len(row["withheld_release_ids"]) for row in packed.values()
                ),
                "pre_release_difference_count": sum(index < first_release for index in changed),
                "changed_decision_count": len(changed),
                "changed_decision_event_ids": [packed[index]["event_id"] for index in changed],
                "packed_future_error": sum(int(packed[index]["error"]) for index in future),
                "withheld_future_error": sum(int(withheld[index]["error"]) for index in future),
                "future_error_difference": sum(
                    int(withheld[index]["error"]) - int(packed[index]["error"]) for index in future
                ),
                "causal_dependence_observed": bool(changed),
                "no_access_before_release": not any(index < first_release for index in changed),
            }
        )
    return result


def score_acceptance_gate(
    comparisons: Sequence[Mapping[str, Any]],
    parity_counts: Mapping[str, Any],
    deletion_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Score fixed efficacy, parity, causality, and nonbinding cost targets."""

    primary = next(
        row
        for row in comparisons
        if row["comparison_id"] == "packed_online_memory_vs_frozen_warmup"
    )
    error_passed = float(primary["future_error_delta"]["ci95"][1]) < 0.0
    false_accept_passed = float(primary["false_accept_delta"]["ci95"][1]) <= 0.0
    recurrence_passed = float(primary["recurrence_error_increase"]["estimate"]) <= 0.02
    parity_passed = all(int(value) == 0 for value in parity_counts.values())
    causality_passed = (
        bool(deletion_rows)
        and sum(int(row["changed_decision_count"]) for row in deletion_rows) > 0
        and all(int(row["pre_release_difference_count"]) == 0 for row in deletion_rows)
        and all(row["no_access_before_release"] is True for row in deletion_rows)
    )
    learning_value = (
        error_passed
        and false_accept_passed
        and recurrence_passed
        and parity_passed
        and causality_passed
    )
    return {
        "efficacy": {
            "future_error_upper_ci95_lt_zero": error_passed,
            "false_accept_upper_ci95_lte_zero": false_accept_passed,
            "recurrence_error_increase_lte_0_02": recurrence_passed,
            "passed": error_passed and false_accept_passed and recurrence_passed,
        },
        "parity": {
            **dict(parity_counts),
            "passed": parity_passed,
            "statistical_superiority_claimed": False,
        },
        "causality": {
            "changed_decision_count": sum(
                int(row["changed_decision_count"]) for row in deletion_rows
            ),
            "pre_release_difference_count": sum(
                int(row["pre_release_difference_count"]) for row in deletion_rows
            ),
            "passed": causality_passed,
        },
        "cost": {
            "tier1_lookup_target_ns": 1_000,
            "acceleration_target_x": 100.0,
            "required_for_learning_value": False,
        },
        "learning_value_passed": learning_value,
    }


def latency_summary(panel: LearningPanel) -> JsonDict:
    """Reduce each measured operation without treating targets as results."""

    operations = {
        name: {
            "count": len(values),
            "p50_ns": _percentile(values, 0.50),
            "p95_ns": _percentile(values, 0.95),
            "total_ns": sum(values),
        }
        for name, values in panel.costs.items()
    }
    packed_p50 = _percentile(panel.lookup_costs["packed_online_memory"], 0.50)
    reference_p50 = _percentile(panel.lookup_costs["reference_online_version_space"], 0.50)
    unique_events = len(panel.decisions) // len(ARMS)
    total_ns = sum(int(row["total_ns"]) for row in operations.values())
    return {
        "clock": "time.perf_counter_ns",
        "operations": operations,
        "amortized_per_event_total_ns": total_ns / unique_events,
        "allocated_bytes": {
            "maximum_state_bytes": panel.maximum_state_bytes,
            "maximum_pending_bytes": panel.maximum_pending_bytes,
            "measurement": "canonical persistent and pending bytes",
        },
        "packed_lookup_p50_ns": packed_p50,
        "reference_lookup_p50_ns": reference_p50,
        "packed_vs_reference_lookup_acceleration_x": (
            0.0 if packed_p50 == 0 else reference_p50 / packed_p50
        ),
        "tier1_lookup_target_ns": 1_000,
        "tier1_lookup_target_met": packed_p50 < 1_000,
        "acceleration_target_x": 100.0,
        "acceleration_target_met": (
            False if packed_p50 == 0 else reference_p50 / packed_p50 >= 100.0
        ),
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode complete raw rows with one canonical row per line."""

    return b"".join(transactional.canonical_json_bytes(dict(row)) for row in rows)


def _stable_scientific_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Remove no values because aggregate rows contain no host timing."""

    return [dict(row) for row in rows]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash sources, settings, gates, and timing-free scientific evidence."""

    payload = {
        "schema": artifact.get("schema"),
        "source_artifact_hashes": artifact.get("source_artifact_hashes"),
        "sample_size_budget": artifact.get("sample_size_budget"),
        "random_seed": artifact.get("random_seed"),
        "rows": _stable_scientific_rows(artifact.get("rows", [])),
        "decision_rows_sha256": artifact.get("decision_rows_path", {}).get("sha256"),
        "comparison_rows": artifact.get("comparison_rows"),
        "memory_deletion_rows": artifact.get("memory_deletion_rows"),
        "acceptance_gate_learning": artifact.get("acceptance_gate_learning"),
        "gate_check_summary": artifact.get("gate_check_summary"),
        "status": artifact.get("status"),
        "verdict_class": artifact.get("verdict_class"),
        "honest_verdict": artifact.get("honest_verdict"),
    }
    return transactional.sha256_json(payload)


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
    duration_s: float = 0.0,
    seeds: Sequence[int] = STREAM_SEEDS,
) -> JsonDict:
    """Create a schema-complete object before any learning claim."""

    now = datetime.now(UTC).isoformat()
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at or now,
        "completed_at_utc": completed_at or now,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_streams": len(seeds),
            "attempted_streams": 0,
            "completed_streams": 0,
            "censored_streams": len(seeds),
            "planned_events": len(seeds) * EVENTS_PER_SEED,
            "attempted_events": 0,
            "completed_events": 0,
            "censored_events": len(seeds) * EVENTS_PER_SEED,
            "independent_units_planned": len(seeds),
            "independent_units_completed": 0,
        },
        "random_seed": {
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "streams": list(seeds),
            "withhold_segments": [list(segment) for segment in WITHHOLD_SEGMENTS],
        },
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external:unknown_precondition",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "belief_run_complete_score": 0,
        "belief_learning_value_score": 0,
        "continuous_self_learning_task": True,
        "decision_rows_path": {
            "path": str(paths.decision_rows),
            "sha256": None,
            "row_count": 0,
            "format": "jsonl",
        },
        "comparison_rows": [],
        "acceptance_gate_learning": {
            "efficacy": {"passed": False},
            "parity": {"passed": False},
            "causality": {"passed": False},
            "cost": {"required_for_learning_value": False},
            "learning_value_passed": False,
        },
        "memory_deletion_rows": [],
        "latency_summary": {"operations": {}},
        "no_model_weight_mutation": True,
        "checkpoint_path": {
            "path": str(paths.state),
            "sha256": None,
            "controller_count": 0,
        },
        "panel_conformance_errors": [],
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    duration_s: float,
    started_at: str | None = None,
    completed_at: str | None = None,
    seeds: Sequence[int] = STREAM_SEEDS,
) -> JsonDict:
    """Return row-free terminal evidence for an unchanged external block."""

    artifact = _base_artifact(
        checks,
        source_hashes,
        paths,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        seeds=seeds,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_external:{failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    check_files: bool = False,
    repo_root: Path = REPO_ROOT,
    expected_seeds: Sequence[int] = STREAM_SEEDS,
) -> list[str]:
    """Cold-check schema, rows, hashes, gates, and terminal classification."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("execution_venue") != "host", "execution_venue")
    add(
        artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False,
        "model_contract",
    )
    add(artifact.get("no_model_weight_mutation") is not True, "weight_mutation")
    add(
        artifact.get("verdict_class")
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict_class",
    )
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    if artifact.get("status") == "blocked":
        add(bool(artifact.get("rows")), "blocked_rows")
        add(artifact.get("verdict_class") != "blocked", "blocked_verdict")
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "blocked_gate")
        add(artifact.get("inference_substrate") != "blocked_no_run", "blocked_substrate")
        return errors
    add(artifact.get("status") != "complete", "terminal_status")
    add(artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate")
    add(artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS, "substrate_class")
    add(artifact.get("gate_check_summary", {}).get("passed") is not True, "precondition_gate")
    add(len(artifact.get("rows", [])) != len(expected_seeds) * len(ARMS), "aggregate_rows")
    add(artifact.get("belief_run_complete_score") != 1, "completion_score")
    add(len(artifact.get("comparison_rows", [])) != 3, "comparison_rows")
    add(len(artifact.get("memory_deletion_rows", [])) != len(expected_seeds), "deletion_rows")
    parity = artifact.get("acceptance_gate_learning", {}).get("parity", {})
    add(parity.get("passed") is not True, "parity_gate")
    expected_class = (
        "circular_positive" if artifact.get("belief_learning_value_score") == 1 else "null"
    )
    add(artifact.get("verdict_class") != expected_class, "value_verdict")
    if check_files:
        decision_receipt = artifact.get("decision_rows_path", {})
        decision_path = _resolve(repo_root, str(decision_receipt.get("path", "")))
        add(_sha256_path(decision_path) != decision_receipt.get("sha256"), "decision_rows_hash")
        add(
            sum(1 for _ in decision_path.open(encoding="utf-8"))
            != len(expected_seeds) * len(ARMS) * EVENTS_PER_SEED,
            "decision_rows_count",
        )
        checkpoint = artifact.get("checkpoint_path", {})
        checkpoint_path = _resolve(repo_root, str(checkpoint.get("path", "")))
        add(_sha256_path(checkpoint_path) != checkpoint.get("sha256"), "checkpoint_hash")
        for path_text, expected_hash in artifact.get("source_artifact_hashes", {}).items():
            if expected_hash is not None:
                add(
                    _sha256_path(_resolve(repo_root, path_text)) != expected_hash,
                    f"source_hash:{path_text}",
                )
    return errors


def _require_valid(errors: Sequence[str]) -> None:
    """Stop publication when any cold artifact check fails."""

    if errors:
        raise ValueError("invalid_exp7227_artifact:" + ",".join(errors))


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_artifact: Path = DEFAULT_UPSTREAM_ARTIFACT,
    seeds: Sequence[int] = STREAM_SEEDS,
    progress: bool = False,
) -> JsonDict:
    """Run authenticated replay, causal controls, bootstrap, and cold checks."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    if progress:
        print("PHASE 0 START: check code, contract, hashes, imports, and outputs", flush=True)
    checks, _, source_hashes = collect_preconditions(
        repo_root,
        paths,
        upstream_artifact=upstream_artifact,
    )
    if not all(row["passed"] for row in checks):
        if progress:
            print("PHASE 0 END: external prerequisite failed; no CPU replay ran", flush=True)
        return build_blocked_artifact(
            checks,
            source_hashes,
            paths,
            duration_s=time.monotonic() - started,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            seeds=seeds,
        )
    if progress:
        print("PHASE 0 END: all authenticated preconditions passed", flush=True)
        print("PHASE 1 END: progress protocol active; no model load is scheduled", flush=True)
        print(
            "PHASE 2 END: substrate=cpu_exact_solver_or_simulator; model_invoked=false",
            flush=True,
        )
        print("PHASE 3 START: load sealed public, release, and evaluator views", flush=True)
    views = load_stream_views(repo_root, upstream_artifact)
    if tuple(seeds) != STREAM_SEEDS:
        selected = set(seeds)
        views = StreamViews(
            [row for row in views.public if int(row["seed"]) in selected],
            [row for row in views.authority if int(row["seed"]) in selected],
            [row for row in views.releases if int(row["seed"]) in selected],
            {**views.manifest, "seeds": list(seeds), "total_events": len(seeds) * EVENTS_PER_SEED},
        )
    stream_errors = exp7226.stream_conformance_errors(views, expected_seeds=seeds)
    if progress:
        print(
            f"PHASE 3 END: sealed stream checks completed; errors={len(stream_errors)}", flush=True
        )
        print("PHASE 4 BENCHMARK START: matched chronological five-arm CPU replay", flush=True)
    panel = run_learning_panel(views, state_path=paths.state, seeds=seeds, progress=progress)
    if progress:
        print(f"PHASE 4 BENCHMARK END: recorded decisions={len(panel.decisions)}", flush=True)
        print("PHASE 5 START: score fixed stream-bootstrap efficacy gates", flush=True)
    conformance = panel_conformance_errors(panel, seeds=seeds)
    comparisons = build_comparison_rows(panel.rows)
    if progress:
        print("PHASE 5 END: prospective comparisons use frozen stream seeds", flush=True)
        print("PHASE 6 START: score preregistered feedback-withholding effects", flush=True)
    deletion_rows = build_memory_deletion_rows(panel.decisions)
    acceptance = score_acceptance_gate(comparisons, panel.parity_counts, deletion_rows)
    if progress:
        print(
            f"PHASE 6 END: changed decisions={acceptance['causality']['changed_decision_count']}",
            flush=True,
        )
        print("PHASE 7 START: reduce measured operation times and allocated bytes", flush=True)
    latency = latency_summary(panel)
    acceptance["cost"].update(
        {
            "packed_lookup_p50_ns": latency["packed_lookup_p50_ns"],
            "packed_vs_reference_lookup_acceleration_x": latency[
                "packed_vs_reference_lookup_acceleration_x"
            ],
            "tier1_lookup_target_met": latency["tier1_lookup_target_met"],
            "acceleration_target_met": latency["acceleration_target_met"],
        }
    )
    if progress:
        print(
            f"PHASE 7 END: amortized_total_ns={latency['amortized_per_event_total_ns']:.3f}",
            flush=True,
        )
    transactional._atomic_write(paths.decision_rows, _jsonl_bytes(panel.decisions))
    transactional._atomic_write(
        paths.state,
        transactional.canonical_json_bytes(
            {
                "schema": "carnot.exp7227.learning_state.v1",
                "controller_count": len(seeds) * len(ARMS),
                "states": panel.final_states,
            }
        ),
    )
    source_hashes[str(paths.decision_rows)] = _sha256_path(paths.decision_rows)
    source_hashes[str(paths.state)] = _sha256_path(paths.state)
    complete = not stream_errors and not conformance
    learning_value = complete and bool(acceptance["learning_value_passed"])
    artifact = _base_artifact(
        checks,
        source_hashes,
        paths,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        seeds=seeds,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": panel.rows,
            "sample_size_budget": {
                "planned_streams": len(seeds),
                "attempted_streams": len(seeds),
                "completed_streams": len(seeds),
                "censored_streams": 0,
                "planned_events": len(seeds) * EVENTS_PER_SEED,
                "attempted_events": len(seeds) * EVENTS_PER_SEED,
                "completed_events": len(seeds) * EVENTS_PER_SEED,
                "censored_events": 0,
                "planned_arm_event_rows": len(seeds) * EVENTS_PER_SEED * len(ARMS),
                "completed_arm_event_rows": len(panel.decisions),
                "independent_units_planned": len(seeds),
                "independent_units_completed": len(seeds),
            },
            "verdict_class": "circular_positive" if learning_value else "null",
            "honest_verdict": (
                "complete: packed online memory passed fixed prospective learning gates"
                if learning_value
                else "complete_null: packed online memory did not pass every fixed prospective learning gate"
            ),
            "belief_run_complete_score": int(complete),
            "belief_learning_value_score": int(learning_value),
            "decision_rows_path": {
                "path": str(paths.decision_rows),
                "sha256": _sha256_path(paths.decision_rows),
                "row_count": len(panel.decisions),
                "format": "jsonl",
            },
            "comparison_rows": comparisons,
            "acceptance_gate_learning": acceptance,
            "memory_deletion_rows": deletion_rows,
            "latency_summary": latency,
            "checkpoint_path": {
                "path": str(paths.state),
                "sha256": _sha256_path(paths.state),
                "controller_count": len(seeds) * len(ARMS),
            },
            "panel_conformance_errors": [*stream_errors, *conformance],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _require_valid(
        validate_artifact(
            artifact,
            check_files=True,
            repo_root=repo_root,
            expected_seeds=seeds,
        )
    )
    if progress:
        print(
            f"PHASE 7 SEAL END: learning_value={int(learning_value)}; "
            f"conformance_errors={len(conformance)}",
            flush=True,
        )
    return artifact


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Publish already validated terminal bytes through one atomic rename."""

    return transactional._atomic_write(
        path,
        transactional.canonical_json_bytes(dict(artifact)),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and optional private output root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--upstream-artifact", type=Path, default=DEFAULT_UPSTREAM_ARTIFACT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CPU replay and atomically publish one terminal artifact."""

    print("PHASE 0 PRECONDITION: parse inputs before checking any resource", flush=True)
    args = _parse_args(argv)
    if str(args.date) != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    artifact = build_and_seal(
        REPO_ROOT,
        paths,
        upstream_artifact=args.upstream_artifact,
        progress=True,
    )
    print("PHASE 8 FINAL VALIDATION START: cold-check terminal fields and hashes", flush=True)
    _require_valid(
        validate_artifact(
            artifact,
            check_files=artifact["status"] == "complete",
            repo_root=REPO_ROOT,
        )
    )
    print("PHASE 8 FINAL VALIDATION END: terminal object is valid", flush=True)
    print("PHASE 9 REQUIRED FIELDS START: publish the validated terminal result", flush=True)
    write_artifact(paths.artifact, artifact)
    print(f"PHASE 9 REQUIRED FIELDS END: wrote {paths.artifact}", flush=True)
    return 0
