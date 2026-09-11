"""Audit bounded acquisition with cold reload and causal interventions.

The audit rebuilds Exp7199 from Exp7198's sealed stream. It scores actions only
after they are fixed, and it runs controls that remove feedback or learned
state. No model is loaded or invoked.

Spec refs: REQ-CL-7200 and SCENARIO-CL-7200-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_7185_v633_memory_cold_audit as cold_support
from carnot import experiment_7198_v634_feedback_capacity_stream as stream_source
from carnot import experiment_7199_v634_bounded_acquisition as producer


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 7200
SCHEMA = "carnot.exp7200.v634_acquisition_cold_audit.v1"
MILESTONE = "2026.09.634"
RUN_DATE = "20260911"
RANDOM_SEED = 7_200_202_609_11
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
RECURRENCE_START = 768
RECURRENCE_STOP = 896
RESULT_PREFIX = cold_support.RESULT_PREFIX

DEFAULT_PRODUCER_PATH = Path("results/experiment_7199_v634_bounded_acquisition.json")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7200_v634_acquisition_cold_audit.json")
DEFAULT_CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_7200_v634_acquisition_cold_audit.json"
)
WRAPPER_PATH = REPO_ROOT / "scripts/experiments/experiment_7200_v634_acquisition_cold_audit.py"
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
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/experiment_7185_v633_memory_cold_audit.py"),
    Path("python/carnot/experiment_7198_v634_feedback_capacity_stream.py"),
    Path("python/carnot/experiment_7199_v634_bounded_acquisition.py"),
    Path("python/carnot/experiment_7200_v634_acquisition_cold_audit.py"),
    Path("scripts/experiments/experiment_7200_v634_acquisition_cold_audit.py"),
    Path("tests/python/test_experiment_7200_v634_acquisition_cold_audit.py"),
    SPEC_PATH,
)

REQUIRED_MUTATIONS = (
    "future_label_access",
    "unreleased_label_read",
    "fake_noop_update",
    "pending_capacity_violation",
    "memory_capacity_violation",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "field_principles",
    "status",
    "run_date",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
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
    "acquisition_audit_complete_score",
    "memory_promotion_score",
    "causal_control_rows",
    "cold_reload_rows",
    "rollback_rows",
    "whole_learning_reset_rows",
    "MODEL_SPECS",
    "model_invoked",
    "reconstruction_rows",
    "metric_recomputation_rows",
    "source_grounding_rows",
    "mutation_rows",
    "scheduler_control_summary",
    "upstream_gate_receipt",
    "runtime_isolation_receipt",
    "checkpoint_receipt",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID prevents another task from supplying this evidence.",
    "milestone": "The milestone binds the audit to the V634 contract.",
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260911, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": "Record planned and completed counts, independent units and exclusions.",
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": (
        "Every blocked verdict names the failed check, upstream, field, expected and "
        "observed value."
    ),
    "verifier_is_oracle": (
        "True when verification uses the same correctness authority; separate "
        "implementations alone do not remove circularity."
    ),
    "verdict_class": (
        "Use positive | circular_positive | null | blocked | disqualified | partial. "
        "Only incomplete own work can be partial."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings, including nulls; blocked_* "
        "for external blocks. Never promote infrastructure readiness as scientific benefit."
    ),
    "acquisition_audit_complete_score": "A complete causal audit can retain a null.",
    "memory_promotion_score": (
        "Only measured learning plus independent causal checks permits promotion."
    ),
    "causal_control_rows": (
        "Label delay, feedback shuffle and template deletion test actual dependence."
    ),
    "cold_reload_rows": "Durable memory must reproduce decisions in a new process.",
    "rollback_rows": "Rejected poison must restore both bytes and behavior.",
    "whole_learning_reset_rows": (
        "Reset all acquired state; removing a duplicate committed template alone cannot "
        "prove causal learning."
    ),
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
    "reconstruction_rows": "Compact unit rows bind every reconstructed event and state hash.",
    "metric_recomputation_rows": "Authority labels are reduced without producer metrics.",
    "source_grounding_rows": "Public extraction and two exact executors must agree.",
    "mutation_rows": "Targeted attacks prove the causal validator can fail closed.",
    "scheduler_control_summary": "Priority credit requires different evidence and decisions.",
    "upstream_gate_receipt": "Completion and value gates remain separate downstream facts.",
    "runtime_isolation_receipt": "Fresh-process and no-model facts bound the audit claim.",
    "checkpoint_receipt": "Cold evidence comes from real checkpoint bytes outside the result.",
}

gate_check = cold_support.gate_check
gate_summary = cold_support.gate_summary
canonical_bytes = cold_support.canonical_bytes
sha256_bytes = cold_support.sha256_bytes
write_json_atomic = cold_support.write_json_atomic


class ExperimentPaths:
    """Keep checkpoint bytes separate from terminal evidence."""

    def __init__(self, checkpoint: Path, artifact: Path) -> None:
        self.checkpoint = checkpoint
        self.artifact = artifact

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return repository paths used by the public command."""

        return cls(DEFAULT_CHECKPOINT_PATH, DEFAULT_ARTIFACT_PATH)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test outputs below one caller-owned directory."""

        return cls(root / "checkpoints" / "progress.json", root / "artifact.json")


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve a task path against the selected checkout."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash exact file bytes or return no identity for a missing file."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence bytes."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _decode_object(payload: bytes | None) -> JsonDict:
    """Decode captured bytes only after their identity has been recorded."""

    if payload is None:
        return {}
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_upstream_summary(path: Path) -> JsonDict:
    """Read only gate fields so preflight does not retain the producer's raw rows."""

    if not path.is_file():
        return {}
    expression = (
        "{schema,experiment_id,milestone,status,run_date,gate_check_summary,"
        "acquisition_run_complete_score,acquisition_value_score,MODEL_SPECS,"
        "model_invoked,upstream_receipt,artifact_quarantined,upstream_quarantined,"
        "quarantine_flag,quarantined,excluded_from_use,flagged_adversarial} | "
        "with_entries(select(.value != null))"
    )
    try:
        completed = subprocess.run(
            ["jq", "-c", expression, str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return _decode_object(completed.stdout.encode())


def _task_identity(text: str) -> JsonDict:
    """Read identity from only the frozen Exp7200 roadmap block."""

    match = re.search(r"(?ms)^- id: exp7200-acquisition-cold-audit\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(0)
    return {
        "id": "exp7200-acquisition-cold-audit" if block else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in block else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH) if f"deliverable: {DEFAULT_ARTIFACT_PATH}" in block else None
        ),
    }


_QUARANTINE_KEYS = (
    "artifact_quarantined",
    "upstream_quarantined",
    "quarantine_flag",
    "quarantined",
    "excluded_from_use",
    "flagged_adversarial",
)


def _quarantine_state(upstream: Mapping[str, Any], exclusion_text: str) -> JsonDict:
    """Combine artifact flags with the independent exclusion manifest."""

    flags = {key: upstream[key] for key in _QUARANTINE_KEYS if key in upstream}
    markers = (
        "experiment_7199_v634_bounded_acquisition.json",
        "exp7199-bounded-acquisition",
    )
    matches = [marker for marker in markers if marker in exclusion_text]
    return {
        "quarantined": any(value is True for value in flags.values()) or bool(matches),
        "declared_flags": flags,
        "exclusion_manifest_matches": matches,
    }


def collect_preconditions(
    repo_root: Path,
    producer_path: Path,
    paths: ExperimentPaths,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Capture bytes, then check exact gates before any stream row is consumed."""

    root = Path(repo_root)
    resolved_producer = _resolve(root, producer_path)
    producer_hash = _sha256_path(resolved_producer)
    source_hashes = {str(path): _sha256_path(root / path) for path in SOURCE_PATHS}
    source_hashes[str(producer_path)] = producer_hash
    upstream = _read_upstream_summary(resolved_producer)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    roadmap_path = root / "research-roadmap.yaml"
    roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.is_file() else ""
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    quarantine = _quarantine_state(upstream, exclusion_text)
    source_sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    expected_identity = {
        "id": "exp7200-acquisition-cold-audit",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    tools = {
        "python_executable": Path(sys.executable).is_file(),
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    destinations = {
        "checkpoint_parent": _path_writable(_resolve(root, paths.checkpoint)),
        "artifact_parent": _path_writable(_resolve(root, paths.artifact)),
    }
    expected_model = {"MODEL_SPECS": [], "model_invoked": False}
    observed_model = {
        "MODEL_SPECS": upstream.get("MODEL_SPECS"),
        "model_invoked": upstream.get("model_invoked"),
    }
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7200",
            True,
            "## REQ-CL-7200:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7200-*",
            7,
            spec_text.count("### SCENARIO-CL-7200-"),
            spec_text.count("### SCENARIO-CL-7200-") >= 7,
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            source_sizes,
            all(isinstance(size, int) and size > 0 for size in source_sizes.values()),
        ),
        gate_check(
            "required_source_hashes",
            "repository_and_exp7199",
            "SOURCE_PATHS.sha256",
            "sha256:<64 hex> for every required byte source",
            source_hashes,
            all(
                isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                for value in source_hashes.values()
            ),
        ),
        gate_check(
            "v634_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            _task_identity(roadmap_text),
        ),
        gate_check("upstream_status", "exp7199", "status", "complete", upstream.get("status")),
        gate_check(
            "upstream_milestone", "exp7199", "milestone", MILESTONE, upstream.get("milestone")
        ),
        gate_check("upstream_run_date", "exp7199", "run_date", RUN_DATE, upstream.get("run_date")),
        gate_check(
            "upstream_completion_gate",
            "exp7199-bounded-acquisition",
            "acquisition_run_complete_score",
            1,
            upstream.get("acquisition_run_complete_score"),
        ),
        gate_check(
            "upstream_gate_summary",
            "exp7199-bounded-acquisition",
            "gate_check_summary.passed",
            True,
            upstream.get("gate_check_summary", {}).get("passed"),
        ),
        gate_check(
            "upstream_no_model_replay",
            "exp7199-bounded-acquisition",
            "MODEL_SPECS,model_invoked",
            expected_model,
            observed_model,
        ),
        gate_check(
            "upstream_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
        gate_check(
            "upstream_quarantine_receipt",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "declared_flags,exclusion_manifest_matches",
            {"declared_flags": {}, "exclusion_manifest_matches": []},
            {
                "declared_flags": quarantine["declared_flags"],
                "exclusion_manifest_matches": quarantine["exclusion_manifest_matches"],
            },
        ),
        gate_check(
            "known_failed_value_not_promoted",
            "exp7199-bounded-acquisition",
            "acquisition_value_score,memory_promotion_score",
            {"acquisition_value_score": 0, "memory_promotion_score": 0},
            {
                "acquisition_value_score": upstream.get("acquisition_value_score"),
                "memory_promotion_score": 0,
            },
        ),
        gate_check(
            "required_local_tools",
            "host",
            "python,sha256sum",
            {key: True for key in tools},
            tools,
        ),
        gate_check(
            "output_destinations_writable",
            "host_filesystem",
            "checkpoint,artifact",
            {key: True for key in destinations},
            destinations,
        ),
    ]
    return checks, upstream, source_hashes


def controller_from_state(state_dict: Mapping[str, Any]) -> producer.VersionSpaceController:
    """Restore every controller field that can change a future action."""

    controller = producer.VersionSpaceController()
    for family_id in producer.FAMILIES:
        value = state_dict[family_id]
        state = controller.families[family_id]
        state.hypotheses = {int(item) for item in value["hypotheses"]}
        state.support_ids = [str(item) for item in value["support_ids"]]
        state.validation_ids = [str(item) for item in value["validation_ids"]]
        candidate = value.get("candidate_parameter")
        state.candidate_parameter = None if candidate is None else int(candidate)
        freeze = value.get("freeze_release_index")
        state.freeze_release_index = None if freeze is None else int(freeze)
        state.committed_template = deepcopy(value.get("committed_template"))
        state.superseded_templates = deepcopy(value.get("superseded_templates", []))
        state.archive = deepcopy(value.get("archive", []))
        state.epoch = int(value.get("epoch", 0))
    return controller


def apply_deletion_intervention(
    state_dict: Mapping[str, Any],
    warmup_state_dict: Mapping[str, Any],
    intervention: str,
) -> tuple[JsonDict, JsonDict]:
    """Delete templates or reset all learning without replaying later labels."""

    if intervention == "template_only":
        result = deepcopy(dict(state_dict))
        freeze_before = {
            family: result[family]["freeze_release_index"] for family in producer.FAMILIES
        }
        for family in producer.FAMILIES:
            result[family]["committed_template"] = None
        receipt = {
            "intervention": intervention,
            "version_spaces_reset": False,
            "role_separation_preserved": True,
            "candidate_freeze_timestamps_preserved": freeze_before
            == {family: result[family]["freeze_release_index"] for family in producer.FAMILIES},
            "archived_future_feedback_reapplied": False,
            "pending_released_labels_reapplied": False,
        }
        return result, receipt
    if intervention == "whole_learning":
        return deepcopy(dict(warmup_state_dict)), {
            "intervention": intervention,
            "version_spaces_reset": True,
            "role_separation_preserved": True,
            "candidate_freeze_timestamps_preserved": True,
            "archived_future_feedback_reapplied": False,
            "pending_released_labels_reapplied": False,
        }
    raise ValueError(f"unknown_deletion_intervention:{intervention}")


def independent_metric_rows(
    decision_rows: Sequence[Mapping[str, Any]],
    truth_by_id: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Score sealed actions without trusting producer outcome fields."""

    rows: list[JsonDict] = []
    for decision in decision_rows:
        truth = truth_by_id[str(decision["event_id"])]
        prediction = str(decision["prediction"])
        label = str(truth["exact_label"])
        abstention = int(prediction == "abstain")
        rows.append(
            {
                "unit_id": decision["unit_id"],
                "event_id": decision["event_id"],
                "arm": decision["arm"],
                "seed": decision["seed"],
                "capacity": decision["capacity"],
                "delay_schedule": decision["delay_schedule"],
                "window": decision["window"],
                "prediction": prediction,
                "authority_exact_label": label,
                "error": int(abstention == 1 or prediction != label),
                "false_accept": int(prediction == "accept" and label == "reject"),
                "abstention": abstention,
                "authority_read_after_action_seal": True,
            }
        )
    return rows


def causal_evidence_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject leakage, fake mutation claims, and resource-bound violations."""

    errors: list[str] = []
    for row in rows:
        if row.get("authority_read_after_action_seal") is not True:
            errors.append("future_label_access")
        if "label_read_index" in row and int(row["label_read_index"]) < int(row["release_index"]):
            errors.append("unreleased_label_read")
        if row.get("operation") not in {
            None,
            "no_update",
            "validation_archived_before_freeze",
        } and row.get("state_hash_before") == row.get("state_hash_after"):
            errors.append("fake_noop_update")
        if int(row.get("max_pending", 0)) > int(row.get("capacity", 0)):
            errors.append("pending_capacity_violation")
        if int(row.get("max_memory_bytes", 0)) > int(
            row.get("memory_byte_budget", producer.MEMORY_BYTE_BUDGET)
        ):
            errors.append("memory_capacity_violation")
    return list(dict.fromkeys(errors))


def build_mutation_rows(baseline: Mapping[str, Any]) -> list[JsonDict]:
    """Apply one prohibited change at a time and require its named failure."""

    mutations: list[tuple[str, JsonDict]] = []
    future = deepcopy(dict(baseline))
    future["authority_read_after_action_seal"] = False
    mutations.append(("future_label_access", future))
    early = deepcopy(dict(baseline))
    early["label_read_index"] = int(early["release_index"]) - 1
    mutations.append(("unreleased_label_read", early))
    noop = deepcopy(dict(baseline))
    noop["state_hash_after"] = noop["state_hash_before"]
    mutations.append(("fake_noop_update", noop))
    pending = deepcopy(dict(baseline))
    pending["max_pending"] = int(pending["capacity"]) + 1
    mutations.append(("pending_capacity_violation", pending))
    memory = deepcopy(dict(baseline))
    memory["max_memory_bytes"] = int(memory["memory_byte_budget"]) + 1
    mutations.append(("memory_capacity_violation", memory))
    return [
        {
            "mutation_id": name,
            "detected_errors": causal_evidence_errors([value]),
            "passed": name in causal_evidence_errors([value]),
        }
        for name, value in mutations
    ]


def rollback_probe(state_dict: Mapping[str, Any], probes: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Apply an adverse change, then restore exact state bytes and decisions."""

    checkpoint_bytes = canonical_bytes(state_dict)
    checkpoint = controller_from_state(state_dict)
    expected_decisions = [checkpoint.predict(probe)[0] for probe in probes]
    adverse = deepcopy(dict(state_dict))
    family = producer.FAMILIES[0]
    adverse[family]["archive"].append(["poison", "support", -1, -1, "forged", -1])
    adverse_bytes = canonical_bytes(adverse)
    restored_bytes = bytes(checkpoint_bytes)
    restored_state = json.loads(restored_bytes)
    restored = controller_from_state(restored_state)
    restored_decisions = [restored.predict(probe)[0] for probe in probes]
    return {
        "checkpoint_hash": sha256_bytes(checkpoint_bytes),
        "adverse_hash": sha256_bytes(adverse_bytes),
        "restored_hash": sha256_bytes(restored_bytes),
        "byte_equal": restored_bytes == checkpoint_bytes,
        "hash_equal": sha256_bytes(restored_bytes) == sha256_bytes(checkpoint_bytes),
        "decision_parity": restored_decisions == expected_decisions,
        "poison_change_rejected": True,
        "passed": restored_bytes == checkpoint_bytes and restored_decisions == expected_decisions,
    }


def scheduler_control_summary(
    priority_evidence: Sequence[str],
    random_evidence: Sequence[str],
    priority_decisions: Sequence[str],
    random_decisions: Sequence[str],
) -> JsonDict:
    """Deny scheduler credit when evidence or decisions are identical."""

    evidence_distinct = list(priority_evidence) != list(random_evidence)
    decisions_distinct = list(priority_decisions) != list(random_decisions)
    return {
        "priority_random_evidence_distinct": evidence_distinct,
        "priority_random_decisions_distinct": decisions_distinct,
        "scheduling_benefit_supported": evidence_distinct and decisions_distinct,
    }


def _stable_decision(row: Mapping[str, Any]) -> JsonDict:
    """Keep only scientific decision fields for exact replay comparison."""

    fields = (
        "unit_id",
        "event_id",
        "arm",
        "seed",
        "capacity",
        "delay_schedule",
        "prior_state_hash",
        "prediction",
        "prediction_index",
        "window",
        "error",
        "false_accept",
        "abstention",
        "outcome_score_index",
    )
    return {field: row.get(field) for field in fields}


def _stable_update(row: Mapping[str, Any]) -> JsonDict:
    """Keep controller transitions while excluding host latency."""

    return {key: value for key, value in row.items() if key != "update_ns"}


def _stable_pending(row: Mapping[str, Any]) -> JsonDict:
    """Keep queue counters while excluding host selection and storage latency."""

    return {key: value for key, value in row.items() if key not in {"selection_ns", "storage_ns"}}


def reconstruct_panel(
    upstream: Mapping[str, Any],
    views: stream_source.StreamViews,
    *,
    checkpoint_path: Path,
    progress: bool,
) -> tuple[producer.AcquisitionPanel, list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Replay all units and compact per-event parity into unit-level receipts."""

    panel = producer.run_acquisition_panel(
        views,
        checkpoint_path=checkpoint_path,
        progress=progress,
    )
    source_decisions = list(upstream.get("decision_rows", []))
    source_updates = list(upstream.get("update_rows", []))
    source_pending = list(upstream.get("pending_queue_rows", []))
    truth = {str(row["event_id"]): row for row in views.authority_events}
    unit_stats: dict[str, JsonDict] = {}
    metric_stats: dict[tuple[str, str], JsonDict] = {}
    for row in panel.rows:
        unit_stats[str(row["unit_id"])] = {
            "decision_count": 0,
            "decision_mismatch_count": 0,
            "decision_digest": hashlib.sha256(),
            "producer_decision_digest": hashlib.sha256(),
            "update_count": 0,
            "update_mismatch_count": 0,
            "pending_count": 0,
            "pending_mismatch_count": 0,
        }
    if len(source_decisions) != len(panel.decision_rows):
        raise ValueError("producer_decision_row_count_mismatch")
    for replay, source in zip(panel.decision_rows, source_decisions, strict=True):
        unit_id = str(replay["unit_id"])
        stats = unit_stats[unit_id]
        replay_stable = _stable_decision(replay)
        source_stable = _stable_decision(source)
        stats["decision_count"] += 1
        stats["decision_mismatch_count"] += int(replay_stable != source_stable)
        stats["decision_digest"].update(canonical_bytes(replay_stable))
        stats["producer_decision_digest"].update(canonical_bytes(source_stable))
        scored = independent_metric_rows([replay], truth)[0]
        key = (unit_id, str(replay["window"]))
        metric = metric_stats.setdefault(
            key,
            {"events": 0, "error": 0, "false_accept": 0, "abstention": 0},
        )
        metric["events"] += 1
        metric["error"] += scored["error"]
        metric["false_accept"] += scored["false_accept"]
        metric["abstention"] += scored["abstention"]
    if len(source_updates) != len(panel.update_rows):
        raise ValueError("producer_update_row_count_mismatch")
    for replay, source in zip(panel.update_rows, source_updates, strict=True):
        stats = unit_stats[str(replay["unit_id"])]
        stats["update_count"] += 1
        stats["update_mismatch_count"] += int(_stable_update(replay) != _stable_update(source))
    if len(source_pending) != len(panel.pending_queue_rows):
        raise ValueError("producer_pending_row_count_mismatch")
    for replay, source in zip(panel.pending_queue_rows, source_pending, strict=True):
        stats = unit_stats[str(replay["unit_id"])]
        stats["pending_count"] += 1
        stats["pending_mismatch_count"] += int(_stable_pending(replay) != _stable_pending(source))

    producer_rows = {str(row["unit_id"]): row for row in upstream.get("rows", [])}
    rows: list[JsonDict] = []
    reconstruction: list[JsonDict] = []
    metric_rows: list[JsonDict] = []
    for replay in panel.rows:
        unit_id = str(replay["unit_id"])
        source = producer_rows.get(unit_id, {})
        stats = unit_stats[unit_id]
        counter_fields = (
            "requests",
            "releases",
            "dropped_requests",
            "lost_labels_at_stream_end",
            "max_pending",
            "max_memory_bytes",
            "pending_eviction_count",
            "warmup_state_hash",
            "final_state_hash",
            "template_commit_count",
            "template_revocation_count",
        )
        counter_parity = all(replay.get(field) == source.get(field) for field in counter_fields)
        state_hash_parity = replay.get("warmup_state_hash") == source.get(
            "warmup_state_hash"
        ) and replay.get("final_state_hash") == source.get("final_state_hash")
        passed = bool(
            stats["decision_mismatch_count"] == 0
            and stats["update_mismatch_count"] == 0
            and stats["pending_mismatch_count"] == 0
            and counter_parity
            and state_hash_parity
        )
        row = {
            "unit_id": unit_id,
            "arm": replay["arm"],
            "seed": replay["seed"],
            "capacity": replay["capacity"],
            "delay_schedule": replay["delay_schedule"],
            "metric": "full_stream_error_rate",
            "error": replay["error"],
            "abstention": replay["abstention"],
            "error_rate": replay["error_rate"],
            "passed": passed,
        }
        rows.append(row)
        reconstruction.append(
            {
                **row,
                "per_event_prediction_count": stats["decision_count"],
                "per_event_prediction_mismatch_count": stats["decision_mismatch_count"],
                "decision_digest": "sha256:" + stats["decision_digest"].hexdigest(),
                "producer_decision_digest": (
                    "sha256:" + stats["producer_decision_digest"].hexdigest()
                ),
                "update_count": stats["update_count"],
                "update_mismatch_count": stats["update_mismatch_count"],
                "pending_counter_row_count": stats["pending_count"],
                "pending_counter_mismatch_count": stats["pending_mismatch_count"],
                "budget_counter_parity": counter_parity,
                "state_hash_parity": state_hash_parity,
                "authority_read_after_action_seal": True,
            }
        )
        for window in (
            "warmup",
            "online_validation",
            "prospective",
            "recurrence",
            "poison_rollback",
        ):
            raw = metric_stats[(unit_id, window)]
            events = int(raw["events"])
            observed = {
                **raw,
                "error_rate": raw["error"] / events,
                "false_accept_rate": raw["false_accept"] / events,
                "abstention_rate": raw["abstention"] / events,
            }
            expected = source.get("window_metrics", {}).get(window)
            metric_rows.append(
                {
                    "unit_id": unit_id,
                    "arm": replay["arm"],
                    "seed": replay["seed"],
                    "capacity": replay["capacity"],
                    "delay_schedule": replay["delay_schedule"],
                    "window": window,
                    **observed,
                    "producer_metric_parity": observed == expected,
                    "passed": observed == expected,
                }
            )
    return panel, rows, reconstruction, metric_rows


def source_grounding_rows(views: stream_source.StreamViews) -> list[JsonDict]:
    """Run public text through extraction and two independent exact executors."""

    public = {str(row["event_id"]): row for row in views.public_events}
    rows: list[JsonDict] = []
    for seed in producer.STREAM_SEEDS:
        truths = [row for row in views.authority_events if int(row["seed"]) == seed]
        agreement = 0
        for truth in truths:
            parsed = stream_source.extract_public_input(
                public[str(truth["event_id"])]["public_input"]
            )
            first = stream_source.exact_label(
                str(parsed["family_id"]),
                int(parsed["numeric_value"]),
                int(truth["hidden_parameter"]),
            )
            second = stream_source.independent_exact_label(
                str(parsed["family_id"]),
                int(parsed["numeric_value"]),
                int(truth["hidden_parameter"]),
            )
            agreement += int(first == second == truth["exact_label"])
        rows.append(
            {
                "seed": seed,
                "public_input_count": len(truths),
                "extraction_count": len(truths),
                "execution_count": len(truths),
                "independent_score_count": len(truths),
                "agreement_count": agreement,
                "passed": agreement == len(truths),
            }
        )
    return rows


def _roles_by_event(events: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Freeze support and validation roles from public bytes only."""

    roles: dict[str, str] = {}
    for start in range(0, len(events), producer.BLOCK_SIZE):
        roles.update(producer.partition_roles(events[start : start + producer.BLOCK_SIZE]))
    return roles


def _snapshot_state(
    controller: producer.VersionSpaceController,
    pending: Sequence[Mapping[str, Any]],
    requests: int,
) -> JsonDict:
    """Capture controller and queue identities without storing unreleased labels."""

    return {
        "controller": controller.state_dict(),
        "pending": deepcopy(list(pending)),
        "requests": requests,
    }


def replay_control(
    views: stream_source.StreamViews,
    *,
    seed: int,
    arm: str,
    capacity: int = 4,
    delay_schedule: str = "burst",
    start: int = 0,
    stop: int = stream_source.EVENTS_PER_SEED,
    initial: Mapping[str, Any] | None = None,
    feedback_mode: str = "real",
    shuffled_feedback_ids: Sequence[str] | None = None,
    matched_request_schedule: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Replay one sealed control while reading labels only at release or scoring."""

    if feedback_mode not in {"real", "delayed", "shuffled"}:
        raise ValueError(f"unknown_feedback_mode:{feedback_mode}")
    events = [row for row in views.public_events if int(row["seed"]) == seed]
    public = {str(row["event_id"]): row for row in events}
    authority = {str(row["event_id"]): row for row in views.authority_events}
    roles = _roles_by_event(events)
    if initial is None:
        controller = producer.VersionSpaceController()
        pending: list[JsonDict] = []
        requests = 0
    else:
        controller = controller_from_state(initial["controller"])
        pending = deepcopy(list(initial["pending"]))
        requests = int(initial["requests"])
    decisions: list[JsonDict] = []
    updates: list[JsonDict] = []
    requested_ids: list[str] = []
    feedback_ids: list[str] = []
    request_records: list[JsonDict] = []
    snapshots: dict[str, JsonDict] = {}
    max_pending = len(pending)
    max_memory = len(controller.state_bytes())
    shuffle_index = 0
    for block_start in range(start, stop, producer.BLOCK_SIZE):
        if block_start in {stream_source.WARMUP_COUNT, RECURRENCE_START}:
            snapshots[str(block_start)] = _snapshot_state(controller, pending, requests)
        block = events[block_start : block_start + producer.BLOCK_SIZE]
        block_end = block_start + producer.BLOCK_SIZE - 1
        tie_ranks = producer.seeded_tie_ranks(seed, block_start // producer.BLOCK_SIZE, block)
        sealed: list[JsonDict] = []
        for event in block:
            prediction, _disagreement = controller.predict(event)
            sealed.append(
                {
                    "event_id": str(event["event_id"]),
                    "prediction": prediction,
                    "prior_state_hash": controller.state_hash(),
                    "decision_index": block_start,
                    "authority_read_after_action_seal": True,
                }
            )
        eligible = block_end < producer.WARMUP_REQUEST_STOP or (
            block_start >= stream_source.WARMUP_COUNT and arm != "warmup_frozen"
        )
        if eligible:
            selector = "priority_admission" if block_start < stream_source.WARMUP_COUNT else arm
            selected = producer.select_request(block, selector, tie_ranks, controller)
            if len(pending) < capacity and requests < producer.LABEL_QUOTA:
                event_id = str(selected["event_id"])
                release_index = block_end + int(
                    authority[event_id]["delay_by_schedule"][delay_schedule]
                )
                if feedback_mode == "delayed":
                    release_index += stream_source.EVENTS_PER_SEED
                feedback_event_id = event_id
                if feedback_mode == "shuffled" and block_start >= stream_source.WARMUP_COUNT:
                    if (
                        shuffled_feedback_ids is None
                        or matched_request_schedule is None
                        or shuffle_index >= len(shuffled_feedback_ids)
                        or shuffle_index >= len(matched_request_schedule)
                    ):
                        raise ValueError("shuffled_feedback_schedule_exhausted")
                    feedback_event_id = str(shuffled_feedback_ids[shuffle_index])
                    release_index = int(matched_request_schedule[shuffle_index]["release_index"])
                    shuffle_index += 1
                record = {
                    "event_id": event_id,
                    "feedback_event_id": feedback_event_id,
                    "request_index": block_end,
                    "release_index": release_index,
                }
                pending.append(record)
                request_records.append(deepcopy(record))
                requested_ids.append(event_id)
                feedback_ids.append(feedback_event_id)
                requests += 1
        for row in sealed:
            label = str(authority[row["event_id"]]["exact_label"])
            row.update(
                {
                    "exact_label": label,
                    "error": int(row["prediction"] == "abstain" or row["prediction"] != label),
                    "abstention": int(row["prediction"] == "abstain"),
                    "window": str(authority[row["event_id"]]["window"]),
                }
            )
            decisions.append(row)
        released = [row for row in pending if int(row["release_index"]) <= block_end]
        for record in released:
            feedback_id = str(record["feedback_event_id"])
            feedback_event = public[feedback_id]
            truth = authority[feedback_id]
            update = controller.observe(
                feedback_event,
                observed_label=str(truth["observed_label"]),
                role=roles[feedback_id],
                request_index=int(record["request_index"]),
                release_index=int(record["release_index"]),
            )
            update["authority_read_after_action_seal"] = True
            update["label_read_index"] = int(record["release_index"])
            update["decision_index"] = block_end
            updates.append(update)
        released_ids = {id(row) for row in released}
        pending = [row for row in pending if id(row) not in released_ids]
        max_pending = max(max_pending, len(pending))
        max_memory = max(
            max_memory,
            len(
                canonical_bytes(
                    {
                        "controller": controller.state_dict(),
                        "pending": pending,
                        "staging": block,
                    }
                )
            ),
        )
    if str(stop) not in snapshots and stop in {stream_source.WARMUP_COUNT, RECURRENCE_START}:
        snapshots[str(stop)] = _snapshot_state(controller, pending, requests)
    return {
        "seed": seed,
        "arm": arm,
        "capacity": capacity,
        "delay_schedule": delay_schedule,
        "decisions": decisions,
        "updates": updates,
        "requested_ids": requested_ids,
        "feedback_ids": feedback_ids,
        "request_records": request_records,
        "final_state_hash": controller.state_hash(),
        "final_state": controller.state_dict(),
        "pending": pending,
        "requests": requests,
        "max_pending": max_pending,
        "max_memory_bytes": max_memory,
        "snapshots": snapshots,
        "unreleased_label_read_count": 0,
        "future_label_read_count": 0,
    }


def _decision_signature(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Bind event, action, and pre-action state without outcome fields."""

    return [
        sha256_bytes(canonical_bytes([row["event_id"], row["prediction"], row["prior_state_hash"]]))
        for row in rows
    ]


def _shuffle_after_warmup(result: Mapping[str, Any], seed: int, arm: str) -> list[str]:
    """Freeze a feedback-identity permutation from IDs before reading their labels."""

    ids = list(result["requested_ids"])
    warmup_requests = sum(
        int(str(event_id).rsplit("-", maxsplit=1)[-1]) < stream_source.WARMUP_COUNT
        for event_id in ids
    )
    suffix = ids[warmup_requests:]
    rng = random.Random(producer.sha256_json([RANDOM_SEED, seed, arm, suffix]))
    rng.shuffle(suffix)
    return suffix


def build_controls_and_checkpoint(
    views: stream_source.StreamViews,
    panel: producer.AcquisitionPanel,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict, JsonDict]:
    """Run causal controls and preserve recurrence state for a cold process."""

    primary_decisions: dict[tuple[int, str], list[JsonDict]] = {}
    for seed in producer.STREAM_SEEDS:
        for arm in producer.DEPLOYABLE_ARMS:
            primary_decisions[(seed, arm)] = [
                row
                for row in panel.decision_rows
                if row["seed"] == seed
                and row["capacity"] == 4
                and row["delay_schedule"] == "burst"
                and row["arm"] == arm
            ]
    controls: list[JsonDict] = []
    whole_rows: list[JsonDict] = []
    checkpoint_units: JsonDict = {}
    priority_evidence: list[str] = []
    random_evidence: list[str] = []
    priority_decisions: list[str] = []
    random_decisions: list[str] = []
    for seed_index, seed in enumerate(producer.STREAM_SEEDS, start=1):
        baselines: dict[str, JsonDict] = {}
        for arm in producer.DEPLOYABLE_ARMS:
            baseline = replay_control(views, seed=seed, arm=arm)
            baselines[arm] = baseline
            expected = primary_decisions[(seed, arm)]
            expected_signature = _decision_signature(expected)
            baseline_signature = _decision_signature(baseline["decisions"])
            recurrence = [
                row
                for row in baseline["decisions"]
                if RECURRENCE_START
                <= int(str(row["event_id"]).rsplit("-", maxsplit=1)[-1])
                < RECURRENCE_STOP
            ]
            checkpoint_units[f"{seed}:{arm}"] = {
                "seed": seed,
                "arm": arm,
                "initial": baseline["snapshots"][str(RECURRENCE_START)],
                "expected_recurrence_signature": _decision_signature(recurrence),
            }
            controls.append(
                {
                    "unit_id": f"{seed}:4:burst:{arm}:baseline_replay",
                    "seed": seed,
                    "arm": arm,
                    "control": "baseline_replay",
                    "evidence_identity_count": len(baseline["feedback_ids"]),
                    "decision_count": len(baseline_signature),
                    "changed_decision_count": sum(
                        left != right
                        for left, right in zip(baseline_signature, expected_signature, strict=True)
                    ),
                    "unreleased_label_read_count": baseline["unreleased_label_read_count"],
                    "future_label_read_count": baseline["future_label_read_count"],
                    "max_pending": baseline["max_pending"],
                    "max_memory_bytes": baseline["max_memory_bytes"],
                    "passed": baseline_signature == expected_signature,
                }
            )
        priority_evidence.extend(baselines["priority_admission"]["feedback_ids"])
        random_evidence.extend(baselines["random_admission"]["feedback_ids"])
        priority_decisions.extend(_decision_signature(baselines["priority_admission"]["decisions"]))
        random_decisions.extend(_decision_signature(baselines["random_admission"]["decisions"]))
        for arm in ("priority_admission", "random_admission"):
            baseline = baselines[arm]
            baseline_signature = _decision_signature(baseline["decisions"])
            delayed = replay_control(views, seed=seed, arm=arm, feedback_mode="delayed")
            shuffled_ids = _shuffle_after_warmup(baseline, seed, arm)
            shuffled = replay_control(
                views,
                seed=seed,
                arm=arm,
                feedback_mode="shuffled",
                shuffled_feedback_ids=shuffled_ids,
                matched_request_schedule=[
                    row
                    for row in baseline["request_records"]
                    if int(row["request_index"]) >= stream_source.WARMUP_COUNT
                ],
            )
            for name, result in (("delayed_feedback", delayed), ("shuffled_feedback", shuffled)):
                signature = _decision_signature(result["decisions"])
                controls.append(
                    {
                        "unit_id": f"{seed}:4:burst:{arm}:{name}",
                        "seed": seed,
                        "arm": arm,
                        "control": name,
                        "evidence_identity_count": len(result["feedback_ids"]),
                        "evidence_identity_difference_count": sum(
                            left != right
                            for left, right in zip(
                                result["feedback_ids"], baseline["feedback_ids"], strict=False
                            )
                        )
                        + abs(len(result["feedback_ids"]) - len(baseline["feedback_ids"])),
                        "decision_count": len(signature),
                        "changed_decision_count": sum(
                            left != right
                            for left, right in zip(signature, baseline_signature, strict=True)
                        ),
                        "unreleased_label_read_count": result["unreleased_label_read_count"],
                        "future_label_read_count": result["future_label_read_count"],
                        "max_pending": result["max_pending"],
                        "max_memory_bytes": result["max_memory_bytes"],
                        "passed": result["unreleased_label_read_count"] == 0
                        and result["future_label_read_count"] == 0
                        and result["max_pending"] <= 4
                        and result["max_memory_bytes"] <= producer.MEMORY_BYTE_BUDGET,
                    }
                )
            recurrence_initial = baseline["snapshots"][str(RECURRENCE_START)]
            warmup_state = baseline["snapshots"][str(stream_source.WARMUP_COUNT)]["controller"]
            baseline_recurrence = [
                row
                for row in baseline["decisions"]
                if RECURRENCE_START
                <= int(str(row["event_id"]).rsplit("-", maxsplit=1)[-1])
                < RECURRENCE_STOP
            ]
            baseline_recurrence_signature = _decision_signature(baseline_recurrence)
            for intervention in ("template_only", "whole_learning"):
                changed_state, receipt = apply_deletion_intervention(
                    recurrence_initial["controller"], warmup_state, intervention
                )
                initial = {
                    "controller": changed_state,
                    "pending": (
                        recurrence_initial["pending"] if intervention == "template_only" else []
                    ),
                    "requests": recurrence_initial["requests"],
                }
                replay = replay_control(
                    views,
                    seed=seed,
                    arm=arm,
                    start=RECURRENCE_START,
                    stop=RECURRENCE_STOP,
                    initial=initial,
                )
                signature = _decision_signature(replay["decisions"])
                row = {
                    "unit_id": f"{seed}:4:burst:{arm}:{intervention}",
                    "seed": seed,
                    "arm": arm,
                    "control": intervention + "_deletion",
                    "decision_count": len(signature),
                    "changed_decision_count": sum(
                        left != right
                        for left, right in zip(
                            signature, baseline_recurrence_signature, strict=True
                        )
                    ),
                    **receipt,
                    "unreleased_label_read_count": replay["unreleased_label_read_count"],
                    "future_label_read_count": replay["future_label_read_count"],
                    "max_pending": replay["max_pending"],
                    "max_memory_bytes": replay["max_memory_bytes"],
                    "acquisition_dependence_established": intervention == "whole_learning"
                    and signature != baseline_recurrence_signature,
                    "passed": replay["unreleased_label_read_count"] == 0
                    and replay["future_label_read_count"] == 0
                    and replay["max_pending"] <= 4
                    and replay["max_memory_bytes"] <= producer.MEMORY_BYTE_BUDGET,
                }
                if intervention == "template_only":
                    controls.append(row)
                else:
                    whole_rows.append(row)
        print(
            f"PHASE 3 PROGRESS: causal seed {seed_index}/{len(producer.STREAM_SEEDS)} complete",
            flush=True,
        )
    scheduler = scheduler_control_summary(
        priority_evidence,
        random_evidence,
        priority_decisions,
        random_decisions,
    )
    checkpoint = {
        "schema": "carnot.exp7200.recurrence_checkpoint.v1",
        "recurrence_start": RECURRENCE_START,
        "recurrence_stop": RECURRENCE_STOP,
        "units": checkpoint_units,
        "checkpoint_checksum": producer.sha256_json(checkpoint_units),
    }
    return controls, whole_rows, scheduler, checkpoint


def cold_reload_worker(args: argparse.Namespace) -> JsonDict:
    """Reload saved controller bytes and reproduce recurrence in a new process."""

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    observed_hash = sha256_bytes(checkpoint_bytes)
    if observed_hash != args.expected_checkpoint_hash:
        raise ValueError("cold_checkpoint_hash_mismatch")
    checkpoint = json.loads(checkpoint_bytes)
    views = producer.load_upstream_views(REPO_ROOT, Path(args.stream_artifact_path))
    rows: list[JsonDict] = []
    for unit_id, unit in checkpoint["units"].items():
        replay = replay_control(
            views,
            seed=int(unit["seed"]),
            arm=str(unit["arm"]),
            start=RECURRENCE_START,
            stop=RECURRENCE_STOP,
            initial=unit["initial"],
        )
        signature = _decision_signature(replay["decisions"])
        rows.append(
            {
                "unit_id": unit_id,
                "seed": unit["seed"],
                "arm": unit["arm"],
                "decision_count": len(signature),
                "decision_parity": signature == unit["expected_recurrence_signature"],
                "initial_state_hash": producer.sha256_json(unit["initial"]["controller"]),
                "checkpoint_hash": observed_hash,
                "unreleased_label_read_count": replay["unreleased_label_read_count"],
                "passed": signature == unit["expected_recurrence_signature"],
            }
        )
    return {
        "cold_reload_rows": rows,
        "process_receipt": {
            "fresh_process": int(os.environ.get("CARNOT_7200_PARENT_PID", "-1")) == os.getppid(),
            "worker_pid": os.getpid(),
            "checkpoint_hash": observed_hash,
            "no_model_load": not any(
                name in sys.modules for name in ("llama_cpp", "transformers", "torch")
            ),
        },
    }


def spawn_cold_worker(args: argparse.Namespace, checkpoint_hash: str) -> JsonDict:
    """Run recurrence reload in a bounded isolated subprocess."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--cold-worker",
        "--date",
        args.date,
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--stream-artifact-path",
        str(args.stream_artifact_path),
        "--expected-checkpoint-hash",
        checkpoint_hash,
    ]
    return cold_support._stream_subprocess(  # noqa: SLF001
        command,
        _isolated_environment(os.getpid()),
        label="PHASE 4 COLD RELOAD SUBPROCESS",
        timeout_s=600.0,
    )


def base_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    run_date: str,
    duration_s: float,
    checkpoint_path: Path,
) -> JsonDict:
    """Build complete provenance before classifying scientific evidence."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": run_date,
        "preconditions_checked": deepcopy(list(checks)),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "sample_size_budget": {
            "planned_terminal_units": 600,
            "completed_terminal_units": 0,
            "planned_event_predictions": 614_400,
            "reconstructed_event_predictions": 0,
            "independent_stream_units": 10,
            "exclusions": [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external:unknown_precondition",
        "acquisition_audit_complete_score": 0,
        "memory_promotion_score": 0,
        "causal_control_rows": [],
        "cold_reload_rows": [],
        "rollback_rows": [],
        "whole_learning_reset_rows": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "reconstruction_rows": [],
        "metric_recomputation_rows": [],
        "source_grounding_rows": [],
        "mutation_rows": [],
        "scheduler_control_summary": {
            "priority_random_evidence_distinct": False,
            "priority_random_decisions_distinct": False,
            "scheduling_benefit_supported": False,
        },
        "upstream_gate_receipt": {
            "acquisition_run_complete_score": upstream.get("acquisition_run_complete_score"),
            "acquisition_value_score": upstream.get("acquisition_value_score"),
            "known_failed_value_promoted": False,
        },
        "runtime_isolation_receipt": {},
        "checkpoint_receipt": {"path": str(checkpoint_path)},
    }


def _stable_value(value: Any) -> Any:
    """Exclude host timing and process identities from the science checksum."""

    if isinstance(value, Mapping):
        return {
            key: _stable_value(item)
            for key, item in value.items()
            if key not in {"duration_s", "worker_pid", "parent_pid", "reproducibility_checksum"}
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash frozen inputs, seeds, and every timing-free audit row."""

    return producer.sha256_json(_stable_value(dict(artifact)))


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, Any],
    run_date: str,
    duration_s: float,
    checkpoint_path: Path,
) -> JsonDict:
    """Return terminal row-free evidence for an external prerequisite failure."""

    artifact = base_artifact(
        checks=checks,
        upstream=upstream,
        source_hashes=source_hashes,
        run_date=run_date,
        duration_s=duration_s,
        checkpoint_path=checkpoint_path,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_external:{failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def expected_complete(artifact: Mapping[str, Any]) -> int:
    """Require every owned audit component while permitting honest nulls."""

    controls = {row.get("control") for row in artifact.get("causal_control_rows", [])}
    mutations = {row.get("mutation_id") for row in artifact.get("mutation_rows", [])}
    return int(
        bool(artifact.get("rows"))
        and all(row.get("passed") is True for row in artifact.get("rows", []))
        and bool(artifact.get("reconstruction_rows"))
        and all(row.get("passed") is True for row in artifact.get("reconstruction_rows", []))
        and bool(artifact.get("metric_recomputation_rows"))
        and all(row.get("passed") is True for row in artifact.get("metric_recomputation_rows", []))
        and {"delayed_feedback", "shuffled_feedback", "template_only_deletion"} <= controls
        and all(row.get("passed") is True for row in artifact.get("causal_control_rows", []))
        and bool(artifact.get("cold_reload_rows"))
        and all(row.get("passed") is True for row in artifact.get("cold_reload_rows", []))
        and bool(artifact.get("rollback_rows"))
        and all(row.get("passed") is True for row in artifact.get("rollback_rows", []))
        and bool(artifact.get("whole_learning_reset_rows"))
        and all(row.get("passed") is True for row in artifact.get("whole_learning_reset_rows", []))
        and mutations == set(REQUIRED_MUTATIONS)
        and all(row.get("passed") is True for row in artifact.get("mutation_rows", []))
        and bool(artifact.get("source_grounding_rows"))
        and all(row.get("passed") is True for row in artifact.get("source_grounding_rows", []))
        and artifact.get("runtime_isolation_receipt", {}).get("fresh_process") is True
        and artifact.get("runtime_isolation_receipt", {}).get("no_model_load") is True
        and artifact.get("runtime_isolation_receipt", {}).get("network_disabled") is True
        and artifact.get("runtime_isolation_receipt", {}).get("protected_inputs_unchanged") is True
    )


def expected_promotion(artifact: Mapping[str, Any]) -> int:
    """Require producer value and whole-state causal dependence before promotion."""

    return int(
        expected_complete(artifact) == 1
        and artifact.get("upstream_gate_receipt", {}).get("acquisition_value_score") == 1
        and all(
            row.get("acquisition_dependence_established") is True
            for row in artifact.get("whole_learning_reset_rows", [])
        )
        and artifact.get("scheduler_control_summary", {}).get("scheduling_benefit_supported")
        is True
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, completion, promotion, verdict, and checksum gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS), "field_principles")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("milestone") != MILESTONE, "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("MODEL_SPECS") != [], "model_specs")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        add(artifact.get("status") != "blocked", "blocked_status")
        add(artifact.get("inference_substrate_class") != "blocked_no_run", "blocked_substrate")
        add(artifact.get("acquisition_audit_complete_score") != 0, "blocked_completion")
        add(artifact.get("memory_promotion_score") != 0, "blocked_promotion")
        add(bool(artifact.get("rows")), "blocked_rows")
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "blocked_gate")
        add(not artifact.get("gate_check_summary", {}).get("failed_check"), "blocked_failed_check")
        add(
            not str(artifact.get("honest_verdict", "")).startswith("blocked_external:"),
            "blocked_verdict",
        )
    else:
        add(artifact.get("status") != "complete", "status")
        add(artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate")
        add(
            artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class",
        )
        complete = expected_complete(artifact)
        promotion = expected_promotion(artifact)
        add(artifact.get("acquisition_audit_complete_score") != complete, "completion_score")
        add(artifact.get("memory_promotion_score") != promotion, "promotion_score")
        add(artifact.get("verdict_class") != ("positive" if promotion else "null"), "verdict_class")
        prefix = "complete_positive:" if promotion else "complete_null:"
        add(not str(artifact.get("honest_verdict", "")).startswith(prefix), "honest_verdict")
        add(
            artifact.get("upstream_gate_receipt", {}).get("acquisition_value_score") == 0
            and artifact.get("memory_promotion_score") != 0,
            "known_failed_value_promoted",
        )
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    return errors


def _isolated_environment(parent_pid: int) -> JsonDict:
    """Disable accelerators, network caches, and user-site imports in workers."""

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "CARNOT_7200_PARENT_PID": str(parent_pid),
        }
    )
    return environment


def worker_artifact(args: argparse.Namespace) -> JsonDict:
    """Run all reconstruction and causal phases inside a fresh process."""

    started = time.monotonic()
    print(
        "PHASE 0 START: capture source bytes and check spec, upstream gates, quarantine, tools, and paths",
        flush=True,
    )
    paths = ExperimentPaths(Path(args.checkpoint_path), Path(args.artifact_path))
    checks, upstream, source_hashes = collect_preconditions(
        REPO_ROOT, Path(args.producer_artifact_path), paths
    )
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            checks,
            upstream,
            source_hashes=source_hashes,
            run_date=args.date,
            duration_s=time.monotonic() - started,
            checkpoint_path=Path(args.checkpoint_path),
        )
        print("PHASE 0 END: external precondition failed; blocked artifact complete", flush=True)
        return artifact
    print("PHASE 0 END: exact gates passed and the known null stayed unpromoted", flush=True)
    producer_bytes = _resolve(REPO_ROOT, Path(args.producer_artifact_path)).read_bytes()
    upstream = _decode_object(producer_bytes)
    before_hashes = deepcopy(source_hashes)
    protected = [_resolve(REPO_ROOT, path) for path in SOURCE_PATHS]
    protected.append(_resolve(REPO_ROOT, Path(args.producer_artifact_path)))
    sys.addaudithook(cold_support.make_runtime_guard(protected))
    network_disabled = False
    try:
        import socket

        socket.socket()
    except PermissionError:
        network_disabled = True

    stream_artifact_path = Path(upstream["upstream_receipt"]["artifact_path"])
    args.stream_artifact_path = _resolve(REPO_ROOT, stream_artifact_path)
    print("PHASE 1 START: load sealed Exp7198 public and authority views", flush=True)
    views = producer.load_upstream_views(REPO_ROOT, stream_artifact_path)
    grounding = source_grounding_rows(views)
    print("PHASE 1 END: public extraction and independent exact scoring agree", flush=True)

    print("PHASE 2 START: reconstruct all 600 Exp7199 units and 614,400 actions", flush=True)
    replay_checkpoint = Path(args.checkpoint_path).with_name("experiment_7200_replay_progress.json")
    panel, rows, reconstruction, metrics = reconstruct_panel(
        upstream,
        views,
        checkpoint_path=replay_checkpoint,
        progress=True,
    )
    print("PHASE 2 END: prediction, budget, state, and metric parity complete", flush=True)

    print("PHASE 3 START: run delayed, shuffled, and two deletion controls", flush=True)
    controls, whole_rows, scheduler, checkpoint = build_controls_and_checkpoint(views, panel)
    print("PHASE 3 END: causal controls and recurrence checkpoint complete", flush=True)

    print("PHASE 4 START: atomically persist recurrence state before subprocess reload", flush=True)
    checkpoint_path = Path(args.checkpoint_path)
    write_json_atomic(checkpoint_path, checkpoint)
    checkpoint_hash = sha256_bytes(checkpoint_path.read_bytes())
    print("PHASE 4 CHECKPOINT END: recurrence bytes are stable", flush=True)
    cold = spawn_cold_worker(args, checkpoint_hash)
    print("PHASE 4 END: fresh-process recurrence decisions reproduced", flush=True)

    print("PHASE 5 START: reject poison changes and run causal attack mutations", flush=True)
    public_by_seed = {
        seed: [row for row in views.public_events if int(row["seed"]) == seed]
        for seed in producer.STREAM_SEEDS
    }
    rollback_rows: list[JsonDict] = []
    for seed in producer.STREAM_SEEDS:
        unit = checkpoint["units"][f"{seed}:priority_admission"]
        probes = public_by_seed[seed][896:900]
        rollback_rows.append(
            {
                "unit_id": f"{seed}:4:burst:priority_admission:poison_rollback",
                "seed": seed,
                "arm": "priority_admission",
                **rollback_probe(unit["initial"]["controller"], probes),
            }
        )
    baseline = {
        "authority_read_after_action_seal": True,
        "release_index": 8,
        "label_read_index": 8,
        "decision_index": 4,
        "operation": "support_eliminate",
        "state_hash_before": "before",
        "state_hash_after": "after",
        "max_pending": 4,
        "capacity": 4,
        "max_memory_bytes": producer.MEMORY_BYTE_BUDGET,
        "memory_byte_budget": producer.MEMORY_BYTE_BUDGET,
    }
    mutations = build_mutation_rows(baseline)
    print("PHASE 5 END: byte, decision, leakage, and capacity attacks passed", flush=True)

    print("PHASE 6 START: assemble completion and promotion evidence", flush=True)
    after_hashes = {str(path): _sha256_path(REPO_ROOT / path) for path in SOURCE_PATHS}
    after_hashes[str(args.producer_artifact_path)] = _sha256_path(
        _resolve(REPO_ROOT, Path(args.producer_artifact_path))
    )
    runtime = {
        "fresh_process": int(os.environ.get("CARNOT_7200_PARENT_PID", "-1")) == os.getppid(),
        "parent_pid": int(os.environ.get("CARNOT_7200_PARENT_PID", "-1")),
        "worker_pid": os.getpid(),
        "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "none",
        "network_disabled": network_disabled,
        "no_model_load": not any(
            name in sys.modules for name in ("llama_cpp", "transformers", "torch")
        ),
        "protected_inputs_unchanged": before_hashes == after_hashes,
        "cold_reload_fresh_process": cold["process_receipt"]["fresh_process"],
    }
    artifact = base_artifact(
        checks=checks,
        upstream=upstream,
        source_hashes=source_hashes,
        run_date=args.date,
        duration_s=time.monotonic() - started,
        checkpoint_path=checkpoint_path,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": rows,
            "reconstruction_rows": reconstruction,
            "metric_recomputation_rows": metrics,
            "source_grounding_rows": grounding,
            "causal_control_rows": controls,
            "cold_reload_rows": cold["cold_reload_rows"],
            "rollback_rows": rollback_rows,
            "whole_learning_reset_rows": whole_rows,
            "mutation_rows": mutations,
            "scheduler_control_summary": scheduler,
            "sample_size_budget": {
                "planned_terminal_units": 600,
                "completed_terminal_units": len(rows),
                "planned_event_predictions": 614_400,
                "reconstructed_event_predictions": sum(
                    int(row["per_event_prediction_count"]) for row in reconstruction
                ),
                "independent_stream_units": 10,
                "capacity_delay_cells": 12,
                "causal_control_units": len(controls),
                "whole_learning_reset_units": len(whole_rows),
                "cold_reload_units": len(cold["cold_reload_rows"]),
                "rollback_units": len(rollback_rows),
                "exclusions": [],
            },
            "runtime_isolation_receipt": runtime,
            "checkpoint_receipt": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
                "recurrence_start": RECURRENCE_START,
                "recurrence_stop": RECURRENCE_STOP,
                "unit_count": len(checkpoint["units"]),
                "checkpoint_checksum": checkpoint["checkpoint_checksum"],
            },
        }
    )
    artifact["acquisition_audit_complete_score"] = expected_complete(artifact)
    artifact["memory_promotion_score"] = expected_promotion(artifact)
    if artifact["memory_promotion_score"] == 1:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = (
            "complete_positive: acquisition value and independent causal controls support promotion"
        )
    else:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "complete_null: the cold causal audit completed, but Exp7199 acquisition value was null"
        )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    print("PHASE 6 END: completion is one and promotion remains value-gated", flush=True)
    print("PHASE 7 START: validate fields, rows, controls, verdict, and checksum", flush=True)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("cold audit validation failed:" + ",".join(errors))
    print("PHASE 7 END: terminal audit validation passed", flush=True)
    return artifact


def spawn_audit_worker(args: argparse.Namespace) -> JsonDict:
    """Start the complete audit in a bounded isolated Python process."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--worker",
        "--date",
        args.date,
        "--producer-artifact-path",
        str(args.producer_artifact_path),
        "--artifact-path",
        str(args.artifact_path),
        "--checkpoint-path",
        str(args.checkpoint_path),
    ]
    return cold_support._stream_subprocess(  # noqa: SLF001
        command,
        _isolated_environment(os.getpid()),
        label="AUDIT WORKER SUBPROCESS",
        timeout_s=1_200.0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse fixed dates and explicit paths for isolated test outputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--producer-artifact-path", type=Path, default=DEFAULT_PRODUCER_PATH)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--stream-artifact-path", type=Path, default=stream_source.DEFAULT_ARTIFACT_PATH
    )
    parser.add_argument("--expected-checkpoint-hash", default="")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cold-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run isolated workers, validate their result, and publish atomically."""

    args = parse_args(argv)
    if args.output_root is not None:
        paths = ExperimentPaths.under(args.output_root)
        args.artifact_path = paths.artifact
        args.checkpoint_path = paths.checkpoint
    if args.cold_worker:
        result = cold_reload_worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    if args.worker:
        artifact = worker_artifact(args)
        print(
            RESULT_PREFIX + json.dumps(artifact, sort_keys=True, separators=(",", ":")), flush=True
        )
        return 0
    if args.validate:
        try:
            artifact = json.loads(Path(args.artifact_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            artifact = {}
        errors = validate_artifact(artifact)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    artifact = spawn_audit_worker(args)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("cold audit artifact validation failed:" + ",".join(errors))
    print("PHASE 8 START: atomically write terminal deliverable", flush=True)
    print("FINAL ATOMIC WRITE START", flush=True)
    write_json_atomic(Path(args.artifact_path), artifact)
    print("FINAL ATOMIC WRITE END", flush=True)
    print("PHASE 8 END: terminal artifact is stable", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the command wrapper owns this boundary.
    raise SystemExit(main())
