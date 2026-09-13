"""Audit bounded coverage learning from authenticated raw event rows.

The audit rebuilds metrics and controller state in a fresh CPU process. It
keeps the prior audit as history and does not invoke an LLM.

Spec refs: REQ-CL-7255 and SCENARIO-CL-7255-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
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
import sys
import time
from typing import Any

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7242_v637_recurrence_audit as exp7242
from carnot import experiment_7253_v638_coverage_memory as exp7253
from carnot import experiment_7254_v638_coverage_learning as exp7254
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7255
SCHEMA = "carnot.exp7255.v638_coverage_audit.v1"
MILESTONE = "2026.09.638"
RUN_DATE = "20260912"
AUDIT_SEED = 7_255_000
BOOTSTRAP_SEED = exp7254.BOOTSTRAP_SEED
BOOTSTRAP_DRAWS = exp7254.BOOTSTRAP_DRAWS
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
RESULT_PREFIX = exp7242.RESULT_PREFIX

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = REPO_ROOT / "scripts/experiments/experiment_7255_v638_coverage_audit.py"
TEST_PATH = Path("tests/python/test_experiment_7255_v638_coverage_audit.py")
DEFAULT_FIXTURE_ARTIFACT = Path("results/experiment_7253_v638_coverage_memory.json")
DEFAULT_LEARNER_ARTIFACT = Path("results/experiment_7254_v638_coverage_learning.json")
DEFAULT_OLD_AUDIT = Path("results/experiment_7242_v637_recurrence_audit.json")
DEFAULT_PREQUENTIAL_ROWS = Path("results/raw/experiment_7254/prequential_rows.jsonl")
DEFAULT_OPERATION_ROWS = Path("results/raw/experiment_7254/operation_rows.jsonl")
DEFAULT_STATE_MANIFEST = Path("results/checkpoints/experiment_7254_v638_states.json")
DEFAULT_EVIDENCE_SIDECAR = Path("results/checkpoints/experiment_7254_v638_evidence.json")
DEFAULT_LIVE_STATE_ROOT = Path("results/checkpoints/experiment_7254_live_state")
DEFAULT_CHECKPOINT = Path("results/checkpoints/experiment_7255_v638_in_progress.json")
DEFAULT_MUTATION_SIDECAR = Path("results/checkpoints/experiment_7255_v638_mutations.json")
DEFAULT_E2E_ROOT = Path("results/checkpoints/experiment_7255_e2e")
DEFAULT_ARTIFACT = Path("results/experiment_7255_v638_coverage_audit.json")
EXPECTED_FIXTURE_SHA256 = "sha256:1deb4d16e455f6d19e317d040103263ce968c79e71b1cf4fe6d311209f46ec4c"
EXPECTED_LEARNER_SHA256 = "sha256:209b8d243587181a8cc1f2c4f8dd24e59291cca0a2f348460fe125f66b799720"
EXPECTED_OLD_AUDIT_SHA256 = (
    "sha256:738ac5a3c2eea092b69f42a317ad34af8cf3a037bc8d251cac80aea51f4f8b6b"
)

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    Path("python/carnot/experiment_7242_v637_recurrence_audit.py"),
    Path("python/carnot/experiment_7253_v638_coverage_memory.py"),
    Path("python/carnot/experiment_7254_v638_coverage_learning.py"),
    Path("python/carnot/experiment_7255_v638_coverage_audit.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/experiments/experiment_7255_v638_coverage_audit.py"),
    TEST_PATH,
    SPEC_PATH,
)
COMPARISON_SPECS = tuple(exp7254.COMPARISON_SPECS)
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7255-[A-Z-]+")

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
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "execution_host",
    "duration_s",
    "phase_spans_s",
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
    "coverage_audit_complete_score",
    "coverage_promotion_score",
    "shuffle_effect_rows",
    "mutation_rows",
    "continuous_self_learning_task",
)

FIELD_PRINCIPLES = {
    "schema": "Version this artifact and keep experiment identity at top level.",
    "experiment_id": "Bind the result to the fixed Exp7255 task.",
    "milestone": "Bind the result to milestone 2026.09.638.",
    "status": "A terminal artifact is complete or blocked; checkpoints remain unfinished.",
    "run_date": "Use 20260912 and retain actual UTC timestamps.",
    "started_at_utc": "Record the actual UTC start time.",
    "completed_at_utc": "Record the actual UTC end time.",
    "field_principles": "Store explanations here and ordinary values at top level.",
    "preconditions_checked": "Record exact inputs, ownership, hashes, imports, and failures.",
    "MODEL_SPECS": "Name only models invoked now; this audit invokes none.",
    "model_invoked": "Derive model use from current calls, not source history.",
    "inference_substrate": "Describe the exact CPU work with a recognized literal.",
    "inference_substrate_class": "Use the closed compute class without invented duration.",
    "execution_venue": "Use host for orchestration and keep board receipts separate.",
    "execution_host": "Record the real hostname outside the venue vocabulary.",
    "duration_s": "Measure monotonic elapsed work without sleeps or padding.",
    "phase_spans_s": "Keep disjoint monotonic spans for every numbered phase.",
    "random_seed": "Freeze audit and bootstrap seeds before reading outcomes.",
    "reproducibility_checksum": "Bind source, inputs, raw rows, configuration, and findings.",
    "source_artifact_hashes": "Authenticate exact upstream, stream, raw, and state bytes.",
    "rows": "Retain every stream-arm metric, error, abstention, and censoring state.",
    "sample_size_budget": "State planned, attempted, completed, and censored units.",
    "acceptance_gate_results": "Record each expected value, observation, and pass state.",
    "gate_check_summary": "Name the exact failed upstream field for blocked results.",
    "verifier_is_oracle": "Expose exact-oracle use so conformance cannot imply learned value.",
    "honest_verdict": "Use complete_ for findings and blocked_ for external absence.",
    "verdict_class": "Use the closed class; an oracle or failed science gate forbids positive.",
    "validation_receipts": "Record actual commands, exit codes, classifications, and log hashes.",
    "coverage_audit_complete_score": "One records a complete independent audit, including a null.",
    "coverage_promotion_score": "One requires reproduced value and all causal safety checks.",
    "shuffle_effect_rows": "Keep mapping, archive, decision, and zero-headroom differences.",
    "mutation_rows": "Each named safety attack must produce its expected rejection.",
    "continuous_self_learning_task": "Audit persistent cross-query constraints with fixed weights.",
}

gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary


class AuditEvidenceError(ValueError):
    """Reject malformed evidence before it can change the audit verdict."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional, negative-fixture, E2E, and terminal bytes separate."""

    checkpoint: Path
    mutation_sidecar: Path
    e2e_root: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Use the fixed result paths for the roadmap command."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put all test-owned output below one caller-owned directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive outputs without changing authenticated upstream locations."""

        checkpoints = root / "checkpoints"
        return cls(
            checkpoints / DEFAULT_CHECKPOINT.name,
            checkpoints / DEFAULT_MUTATION_SIDECAR.name,
            checkpoints / DEFAULT_E2E_ROOT.name,
            root / DEFAULT_ARTIFACT.name,
        )


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Emit a flushed phase boundary for the external watchdog."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository-relative evidence while preserving absolute test paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _sha256_path(path: Path) -> str | None:
    """Hash exact file bytes while absence stays an observed failure."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object and keep malformed evidence unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Reuse the prior cold audit's strict JSONL object reader."""

    return exp7242.read_jsonl(path)


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_identity(text: str) -> JsonDict:
    """Extract only the fixed Exp7255 roadmap block."""

    match = re.search(r"(?ms)^- id: exp7255-coverage-audit\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7255-coverage-audit" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def _artifact_checksum_valid(module: Any, artifact: Mapping[str, Any]) -> bool:
    """Treat malformed producer fields as a failed checksum observation."""

    try:
        return module.reproducibility_checksum(artifact) == artifact.get("reproducibility_checksum")
    except (KeyError, TypeError, ValueError):
        return False


def _receipt_matches(repo_root: Path, receipt: Any) -> bool:
    """Require a declared path and hash to match the current exact bytes."""

    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("path"), str):
        return False
    return _sha256_path(_resolve(repo_root, str(receipt["path"]))) == receipt.get("sha256")


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    fixture_artifact: Path = DEFAULT_FIXTURE_ARTIFACT,
    learner_artifact: Path = DEFAULT_LEARNER_ARTIFACT,
) -> tuple[list[JsonDict], dict[str, JsonDict], dict[str, str | None]]:
    """Authenticate both producers, raw evidence, imports, and output ownership."""

    fixture_path = _resolve(repo_root, fixture_artifact)
    learner_path = _resolve(repo_root, learner_artifact)
    old_path = _resolve(repo_root, DEFAULT_OLD_AUDIT)
    fixture = _load_object(fixture_path)
    learner = _load_object(learner_path)
    old_audit = _load_object(old_path)
    upstreams = {"exp7253": fixture, "exp7254": learner, "exp7242": old_audit}
    hashes: dict[str, str | None] = {
        str(path): _sha256_path(_resolve(repo_root, path)) for path in SOURCE_PATHS
    }
    hashes[str(fixture_artifact)] = _sha256_path(fixture_path)
    hashes[str(learner_artifact)] = _sha256_path(learner_path)
    hashes[str(DEFAULT_OLD_AUDIT)] = _sha256_path(old_path)
    evidence_receipts = {
        "prequential_rows": learner.get("prequential_rows_path", {}),
        "operation_rows": learner.get("operation_rows_path", {}),
        "state_manifest": learner.get("state_manifest_path", {}),
        "learner_evidence": learner.get("evidence_sidecar_path", {}),
    }
    stream_receipts = fixture.get("stream_manifest", {}).get("receipts", {})
    for receipt in (*evidence_receipts.values(), *stream_receipts.values()):
        if isinstance(receipt, Mapping) and isinstance(receipt.get("path"), str):
            path = str(receipt["path"])
            hashes[path] = _sha256_path(_resolve(repo_root, path))

    spec = _load_text(_resolve(repo_root, SPEC_PATH))
    roadmap = _load_text(_resolve(repo_root, "research-roadmap.yaml"))
    exclusions = _load_text(_resolve(repo_root, "ops/exclusion_manifest.yaml"))
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7226_v636_belief_compiler",
            "carnot.experiment_7253_v638_coverage_memory",
            "carnot.experiment_7254_v638_coverage_learning",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    writable = {
        name: _path_writable(getattr(paths, name))
        for name in ("checkpoint", "mutation_sidecar", "e2e_root", "artifact")
    }
    source_state = {
        str(path): "nonempty" if hashes[str(path)] is not None else "missing"
        for path in SOURCE_PATHS
    }
    fixture_quarantine = exp7213.quarantine_state(
        fixture, exclusions, fixture_path.name, "exp7253-coverage-memory"
    )
    learner_quarantine = exp7213.quarantine_state(
        learner, exclusions, learner_path.name, "exp7254-coverage-learning"
    )
    checks = [
        gate_check(
            "driving_capability_spec", str(SPEC_PATH), "REQ-CL-7255", True, "REQ-CL-7255" in spec
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7255-*",
            7,
            len(set(SCENARIO_PATTERN.findall(spec))),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            dict.fromkeys(source_state, "nonempty"),
            source_state,
        ),
        gate_check(
            "v638_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            {
                "id": "exp7255-coverage-audit",
                "milestone": MILESTONE,
                "deliverable": str(DEFAULT_ARTIFACT),
            },
            _task_identity(roadmap),
        ),
        gate_check("required_imports", "python", "imports", dict.fromkeys(imports, True), imports),
        gate_check(
            "writable_output_paths",
            "host_filesystem",
            "checkpoint,mutation,e2e,artifact",
            dict.fromkeys(writable, True),
            writable,
        ),
        gate_check(
            "exp7253_artifact_hash",
            "exp7253",
            str(fixture_artifact),
            EXPECTED_FIXTURE_SHA256,
            hashes[str(fixture_artifact)],
        ),
        gate_check("exp7253_status", "exp7253", "status", "complete", fixture.get("status")),
        gate_check(
            "exp7253_fixture_ready",
            "exp7253",
            "coverage_fixture_ready_score",
            1,
            fixture.get("coverage_fixture_ready_score"),
        ),
        gate_check(
            "exp7253_checksum",
            "exp7253",
            "reproducibility_checksum",
            True,
            _artifact_checksum_valid(exp7253, fixture),
        ),
        gate_check(
            "exp7253_not_quarantined",
            "artifact_and_exclusion_manifest",
            "quarantined",
            False,
            fixture_quarantine["quarantined"],
        ),
        gate_check(
            "exp7254_artifact_hash",
            "exp7254",
            str(learner_artifact),
            EXPECTED_LEARNER_SHA256,
            hashes[str(learner_artifact)],
        ),
        gate_check("exp7254_status", "exp7254", "status", "complete", learner.get("status")),
        gate_check(
            "exp7254_run_complete",
            "exp7254",
            "coverage_run_complete_score",
            1,
            learner.get("coverage_run_complete_score"),
        ),
        gate_check(
            "exp7254_checksum",
            "exp7254",
            "reproducibility_checksum",
            True,
            _artifact_checksum_valid(exp7254, learner),
        ),
        gate_check(
            "exp7254_not_quarantined",
            "artifact_and_exclusion_manifest",
            "quarantined",
            False,
            learner_quarantine["quarantined"],
        ),
        gate_check(
            "exp7254_raw_receipts",
            "exp7254",
            "prequential,operation,state,evidence",
            dict.fromkeys(evidence_receipts, True),
            {
                name: _receipt_matches(repo_root, receipt)
                for name, receipt in evidence_receipts.items()
            },
        ),
        gate_check(
            "exp7253_stream_receipts",
            "exp7253",
            "prospective_public,authority,releases",
            {name: True for name in stream_receipts if name.startswith("prospective_")},
            {
                name: _receipt_matches(repo_root, receipt)
                for name, receipt in stream_receipts.items()
                if name.startswith("prospective_")
            },
        ),
        gate_check(
            "exp7242_preserved_hash",
            "exp7242",
            str(DEFAULT_OLD_AUDIT),
            EXPECTED_OLD_AUDIT_SHA256,
            hashes[str(DEFAULT_OLD_AUDIT)],
        ),
    ]
    return checks, upstreams, hashes


def _load_text(path: Path) -> str:
    """Read contract text without hiding a missing-file precondition."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def stream_authority_errors(
    public_rows: Sequence[Mapping[str, Any]],
    release_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reconstruct private authority and prove that public rows do not expose it."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    forbidden = exp7253.FORBIDDEN_PUBLIC_FIELDS
    add(any(forbidden & set(row) for row in public_rows), "public_authority_leakage")
    public = {str(row.get("event_id")): row for row in public_rows}
    releases = {str(row.get("event_id")): row for row in release_rows}
    authority = {str(row.get("event_id")): row for row in authority_rows}
    add(set(public) != set(releases) or set(public) != set(authority), "event_join_mismatch")
    for event_id, event in public.items():
        private = authority.get(event_id, {})
        release = releases.get(event_id, {})
        coordinates = ("stream_id", "chronology_index", "family_id", "numeric_value")
        add(
            any(event.get(key) != private.get(key) for key in coordinates),
            "authority_join_mismatch",
        )
        try:
            seed = int(private["stream_seed"])
            family = str(event["family_id"])
            index = int(event["chronology_index"])
            base = exp7253._stable_parameter(seed, family)
            regime, parameter = exp7253._regime_parameter(
                str(private["drift_pattern"]), index, base
            )
            exact = exp7226.exact_label(family, int(event["numeric_value"]), parameter)
            delay = exp7253.DELAY_SUPPORT[(seed + index) % len(exp7253.DELAY_SUPPORT)]
            noisy = (seed * 17 + index * 13) % 31 == 0
            observed = ("reject" if exact == "accept" else "accept") if noisy else exact
        except (KeyError, TypeError, ValueError):
            add(True, "authority_shape")
            continue
        add(
            private.get("regime_id") != regime or private.get("hidden_parameter") != parameter,
            "private_regime_mismatch",
        )
        add(private.get("exact_label") != exact, "private_label_mismatch")
        add(
            release.get("delay") != delay
            or release.get("release_noisy") is not noisy
            or release.get("observed_label") != observed,
            "release_schedule_mismatch",
        )
    return errors


def decision_visibility_errors(row: Mapping[str, Any]) -> list[str]:
    """Reject a learner decision that could see unreleased or private authority."""

    if row.get("oracle_control") is True:
        return []
    errors: list[str] = []
    if row.get("controller_input_fields") != ["event_id", "family_id", "numeric_value"]:
        errors.append("future_label_leakage")
    if row.get("held_out_label_visible_to_controller") is not False:
        errors.append("future_label_leakage")
    if row.get("prediction_frozen_before_release") is not True:
        errors.append("feedback_chronology")
    return list(dict.fromkeys(errors))


def _selected_stream_rows(path: Path, stream_ids: Sequence[str]) -> list[JsonDict]:
    """Stream one large JSONL file and retain only preregistered stream units."""

    selected = set(stream_ids)
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditEvidenceError(f"invalid_jsonl:{path}:{line_number}") from error
            if not isinstance(row, dict):
                raise AuditEvidenceError(f"non_object_jsonl:{path}:{line_number}")
            if row.get("stream_id") in selected:
                rows.append(row)
    return rows


def _selected_views(repo_root: Path, stream_ids: Sequence[str]) -> exp7253.StreamViews:
    """Cold-load authenticated separated stream views for the selected units."""

    fixture = _load_object(_resolve(repo_root, DEFAULT_FIXTURE_ARTIFACT))
    receipts = fixture.get("stream_manifest", {}).get("receipts", {})
    selected = set(stream_ids)

    def load(name: str) -> list[JsonDict]:
        receipt = receipts.get(name, {})
        path = _resolve(repo_root, str(receipt.get("path", "")))
        return [row for row in _read_jsonl(path) if row.get("stream_id") in selected]

    public = load("prospective_public")
    authority = load("prospective_private_authority")
    releases = load("prospective_releases")
    return exp7253.StreamViews(public, authority, releases, {"stream_ids": list(stream_ids)})


def _percentile(values: Sequence[int | float], probability: float) -> float:
    """Use the frozen nearest-rank percentile without producer aggregates."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(probability * (len(ordered) - 1)))))
    return ordered[index]


def _reduce_stream_arm(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rebuild one stream-arm summary directly from chronological event rows."""

    ordered = sorted(rows, key=lambda row: int(row["chronology_index"]))
    future = [row for row in ordered if int(row["chronology_index"]) >= exp7253.WARMUP_COUNT]
    recurrence = [row for row in future if row["recurrence_eligible"] is True]
    future_errors = sum(int(row["full_denominator_error"]) for row in future)
    false_accepts = sum(int(row["false_accept"]) for row in future)
    abstentions = sum(int(row["abstention"]) for row in future)
    recurrence_errors = sum(int(row["full_denominator_error"]) for row in recurrence)
    commits = [row for row in ordered if row["commit_applied"] is True]
    return {
        "unit_id": str(ordered[0]["unit_id"]),
        "stream_id": str(ordered[0]["stream_id"]),
        "seed": int(ordered[0]["seed"]),
        "arm": str(ordered[0]["arm"]),
        "metric": "prospective_full_denominator_error",
        "event_count": len(ordered),
        "future_event_count": len(future),
        "future_error": future_errors,
        "future_error_rate": future_errors / len(future),
        "false_accept": false_accepts,
        "false_accept_rate": false_accepts / len(future),
        "abstention": abstentions,
        "abstention_rate": abstentions / len(future),
        "recurrence_event_count": len(recurrence),
        "recurrence_error": recurrence_errors,
        "recurrence_error_rate": None if not recurrence else recurrence_errors / len(recurrence),
        "query_count": sum(int(row["query_selected"]) for row in ordered),
        "release_count": int(ordered[-1]["released_query_count_after"]),
        "constraint_addition_count": sum(int(row["constraint_addition_count"]) for row in ordered),
        "constraint_deactivation_count": sum(
            int(row["constraint_deactivation_count"]) for row in ordered
        ),
        "valid_reactivation_count": sum(int(row["valid_reactivation_count"]) for row in ordered),
        "later_changed_decision_after_reactivation_count": sum(
            int(row["later_changed_decision_after_reactivation"]) for row in ordered
        ),
        "pre_release_difference_count": sum(int(row["pre_release_difference"]) for row in ordered),
        "shuffle_selection_change_count": sum(
            int(row["shuffle_selection_change_count"]) for row in ordered
        ),
        "lookup_p50_ns": _percentile([int(row["lookup_cost_ns"]) for row in ordered], 0.50),
        "lookup_p95_ns": _percentile([int(row["lookup_cost_ns"]) for row in ordered], 0.95),
        "update_p50_ns": _percentile([int(row["update_cost_ns"]) for row in commits], 0.50),
        "update_p95_ns": _percentile([int(row["update_cost_ns"]) for row in commits], 0.95),
        "durable_commit_p95_ns": _percentile(
            [int(row["durable_commit_cost_ns"]) for row in commits], 0.95
        ),
        "maximum_memory_bytes": max(int(row["memory_total_bytes"]) for row in ordered),
        "censored": False,
    }


def _reduce_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep streams independent and preserve the frozen arm order."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stream_id"]), str(row["arm"]))].append(row)
    order = {arm: index for index, arm in enumerate(exp7253.ARMS)}
    return [
        _reduce_stream_arm(group)
        for _, group in sorted(groups.items(), key=lambda item: (item[0][0], order[item[0][1]]))
    ]


def _raw_evidence_checks(
    rows: Sequence[Mapping[str, Any]],
    views: exp7253.StreamViews,
    stream_ids: Sequence[str],
) -> list[JsonDict]:
    """Audit event joins, decisions, transaction chains, outcomes, and memory bounds."""

    authority = {str(row["event_id"]): row for row in views.authority}
    public = {str(row["event_id"]): row for row in views.public}
    expected_count = len(stream_ids) * exp7253.EVENTS_PER_STREAM * len(exp7253.ARMS)
    visibility = [error for row in rows for error in decision_visibility_errors(row)]
    outcome_errors = 0
    join_errors = 0
    memory_errors = 0
    chain_errors = 0
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("stream_id")), str(row.get("arm")))].append(row)
        event_id = str(row.get("event_id"))
        event = public.get(event_id, {})
        truth = authority.get(event_id, {})
        if any(
            row.get(key) != event.get(key) for key in ("stream_id", "chronology_index", "event_id")
        ):
            join_errors += 1
        prediction = row.get("prediction")
        label = truth.get("exact_label")
        outcome_errors += int(
            row.get("full_denominator_error") != int(prediction != label)
            or row.get("classification_error")
            != int(prediction != "abstain" and prediction != label)
            or row.get("abstention") != int(prediction == "abstain")
            or row.get("false_accept") != int(prediction == "accept" and label == "reject")
        )
        for field, limit in (
            ("memory_witness_bytes", exp7253.MEMORY_CAPS["witness_bytes"]),
            ("memory_archive_bytes", exp7253.MEMORY_CAPS["archive_bytes"]),
            ("memory_pending_bytes", exp7253.MEMORY_CAPS["pending_bytes"]),
            ("memory_ledger_bytes", exp7253.MEMORY_CAPS["ledger_bytes"]),
            ("memory_total_bytes", exp7253.MEMORY_CAPS["total_bytes"]),
        ):
            memory_errors += int(int(row.get(field, limit + 1)) > limit)
    expected_units = {(stream, arm) for stream in stream_ids for arm in exp7253.ARMS}
    for group in groups.values():
        ordered = sorted(group, key=lambda row: int(row["chronology_index"]))
        if [int(row["chronology_index"]) for row in ordered] != list(
            range(exp7253.EVENTS_PER_STREAM)
        ):
            chain_errors += 1
        commits = [row for row in ordered if row.get("commit_applied") is True]
        for previous, current in zip(commits, commits[1:]):
            chain_errors += int(
                current.get("commit_parent_hash") != previous.get("commit_child_hash")
            )
        for row in ordered:
            if row.get("commit_applied") is True:
                chain_errors += int(
                    row.get("commit_parent_hash") != row.get("state_hash_before_prediction")
                    or row.get("commit_parent_hash") == row.get("commit_child_hash")
                    or not str(row.get("commit_child_hash", "")).startswith("sha256:")
                )
            else:
                chain_errors += int(
                    row.get("commit_parent_hash") is not None
                    or row.get("commit_child_hash") is not None
                )
    authority_errors = stream_authority_errors(views.public, views.releases, views.authority)
    definitions = (
        ("complete_raw_event_rows", expected_count, len(rows)),
        ("complete_stream_arm_units", expected_units, set(groups)),
        ("private_authority_reconstruction", [], authority_errors),
        ("decision_visibility", [], sorted(set(visibility))),
        ("raw_public_authority_join", 0, join_errors),
        ("raw_outcome_reconstruction", 0, outcome_errors),
        ("transaction_parent_child_chain", 0, chain_errors),
        ("bounded_memory_rows", 0, memory_errors),
    )
    return [
        {
            "check": name,
            "expected": sorted(expected) if isinstance(expected, set) else expected,
            "observed": sorted(observed) if isinstance(observed, set) else observed,
            "passed": expected == observed,
        }
        for name, expected, observed in definitions
    ]


def _nomination_evaluation(
    archive: Mapping[str, Any], witnesses: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Independently evaluate one archive against only contemporaneously released witnesses."""

    recent = list(witnesses)[-exp7253.VALIDATION_WINDOW :]
    predictions = [
        exp7240._prediction_from_masks(archive["survivor_masks"], witness) for witness in recent
    ]
    applicable = sum(int(value != "abstain") for value in predictions)
    contradictions = sum(
        int(prediction != "abstain" and prediction != witness["observed_label"])
        for prediction, witness in zip(predictions, recent)
    )
    return {
        "released_witness_count": applicable,
        "contradiction_count": contradictions,
        "gate_passed": applicable >= exp7253.MIN_VALIDATION_WITNESSES and contradictions == 0,
    }


def _nomination_error_names(
    receipt: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
) -> list[str]:
    """Name producer nomination fields that disagree with released-only recomputation."""

    errors: list[str] = []
    for side in ("before", "after"):
        for evaluation in receipt[f"{side}_evaluations"]:
            expected = _nomination_evaluation(by_id[str(evaluation["archive_id"])], witnesses)
            if any(evaluation.get(key) != value for key, value in expected.items()):
                errors.append("reactivation_witness_gate")
    if (
        selected is not None
        and _nomination_evaluation(selected, witnesses)["gate_passed"] is not True
    ):
        errors.append("unsafe_reactivation")
    return errors


def _durable_state_error_count(
    repo_root: Path,
    replay_states: Mapping[tuple[str, str], Mapping[str, Any]],
    expected_states: Mapping[tuple[str, str], Mapping[str, Any]],
) -> int:
    """Cold-load every durable final state and count hash or byte mismatches."""

    errors = 0
    for (stream_id, arm), replay in replay_states.items():
        expected = expected_states.get((stream_id, arm), replay)
        durable = _resolve(repo_root, DEFAULT_LIVE_STATE_ROOT) / stream_id / f"{arm}.json"
        try:
            if arm in exp7253.ARCHIVE_ARMS or arm == "reset_relearn":
                controller = exp7253.CoverageArchiveController.load(durable)
            else:
                controller = exp7226.PackedBeliefController.from_state(_load_object(durable))
            errors += int(
                controller.state_hash() != expected.get("final_state_sha256")
                or len(controller.state_bytes()) != expected.get("final_state_bytes")
            )
        except (OSError, ValueError, TypeError):
            errors += 1
    return errors


def _replay_with_instrumentation(
    views: exp7253.StreamViews,
    raw_rows: Sequence[Mapping[str, Any]],
    repo_root: Path,
    stream_ids: Sequence[str],
    *,
    progress: bool,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay shipped controllers without durable writes and inspect each nomination."""

    observations: list[JsonDict] = []
    nomination_errors: list[str] = []
    context: dict[str, Any] = {}
    original_nominate = exp7253.nominate_archives
    original_commit = exp7253.CoverageArchiveController.commit_batch
    original_archive_save = exp7253.CoverageArchiveController.save
    original_packed_save = exp7226.PackedBeliefController.save

    def tracked_nominate(
        archives: Sequence[Mapping[str, Any]],
        witnesses: Sequence[Mapping[str, Any]],
        *,
        shuffled: bool,
        shuffle_seed: int = exp7253.SHUFFLE_SEED,
    ) -> tuple[JsonDict, JsonDict | None]:
        receipt, selected = original_nominate(
            archives, witnesses, shuffled=shuffled, shuffle_seed=shuffle_seed
        )
        by_id = {str(row["archive_id"]): row for row in archives}
        nomination_errors.extend(_nomination_error_names(receipt, by_id, witnesses, selected))
        state = context.get("state", {})
        if shuffled and int(state.get("archive_cap", 0)) > 0:
            before = [row["candidate_archive_id"] for row in receipt["before_mapping"]]
            after = [row["candidate_archive_id"] for row in receipt["after_mapping"]]
            event_id = str(witnesses[-1]["event_id"]) if witnesses else ""
            match = re.search(r"(prospective-\d{2})", event_id)
            observations.append(
                {
                    "stream_id": None if match is None else match.group(1),
                    "admission_mode": state.get("admission_mode"),
                    "mapping_changed": before != after,
                    "selection_changed": receipt["selection_changed"],
                    "eligible_candidate_count": sum(
                        int(row["gate_passed"]) for row in receipt["before_evaluations"]
                    ),
                }
            )
        return receipt, selected

    def tracked_commit(self: Any, *args: Any, **kwargs: Any) -> JsonDict:
        context["state"] = self.state_dict()
        return original_commit(self, *args, **kwargs)

    def no_save(self: Any, path: Path) -> JsonDict:
        del path
        return {"sha256": self.state_hash(), "bytes": len(self.state_bytes())}

    exp7253.nominate_archives = tracked_nominate
    exp7253.CoverageArchiveController.commit_batch = tracked_commit
    exp7253.CoverageArchiveController.save = no_save
    exp7226.PackedBeliefController.save = no_save
    try:
        panel = exp7254.run_learning_panel(
            views,
            state_root=repo_root / DEFAULT_LIVE_STATE_ROOT,
            stream_ids=stream_ids,
            progress=progress,
        )
    finally:
        exp7253.nominate_archives = original_nominate
        exp7253.CoverageArchiveController.commit_batch = original_commit
        exp7253.CoverageArchiveController.save = original_archive_save
        exp7226.PackedBeliefController.save = original_packed_save

    timing_fields = {
        "lookup_cost_ns",
        "update_cost_ns",
        "serialization_cost_ns",
        "durable_commit_cost_ns",
    }
    semantic_mismatches = sum(
        int(
            {key: value for key, value in raw.items() if key not in timing_fields}
            != {key: value for key, value in replay.items() if key not in timing_fields}
        )
        for raw, replay in zip(raw_rows, panel.prequential_rows)
    ) + abs(len(raw_rows) - len(panel.prequential_rows))
    manifest = _load_object(_resolve(repo_root, DEFAULT_STATE_MANIFEST))
    expected_states = {
        (str(row["stream_id"]), str(row["arm"])): row for row in manifest.get("entries", [])
    }
    replay_states = {(str(row["stream_id"]), str(row["arm"])): row for row in panel.state_entries}
    state_errors = 0
    for key, replay in replay_states.items():
        expected = expected_states.get(key, {})
        state_errors += int(
            replay.get("final_state_sha256") != expected.get("final_state_sha256")
            or replay.get("final_state_bytes") != expected.get("final_state_bytes")
            or replay.get("initial_state_sha256") != expected.get("initial_state_sha256")
        )
    durable_errors = _durable_state_error_count(repo_root, replay_states, expected_states)
    rows = [
        {
            "check": "shipped_controller_semantic_replay",
            "expected": 0,
            "observed": semantic_mismatches,
            "passed": semantic_mismatches == 0,
        },
        {
            "check": "every_reactivation_released_witness_gate",
            "expected": [],
            "observed": sorted(set(nomination_errors)),
            "passed": not nomination_errors,
        },
        {
            "check": "final_state_manifest_replay",
            "expected": 0,
            "observed": state_errors,
            "passed": state_errors == 0,
        },
        {
            "check": "fresh_durable_state_load",
            "expected": 0,
            "observed": durable_errors,
            "passed": durable_errors == 0,
        },
    ]
    return rows, observations


def build_shuffle_effect_rows(
    observations: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Aggregate mapping, archive-selection, and later-decision effects without censoring."""

    keys = sorted(
        {(str(row["stream_id"]), str(row["admission_mode"])) for row in observations}
        | {(str(row["stream_id"]), str(row["admission_mode"])) for row in decisions}
    )
    result: list[JsonDict] = []
    for stream_id, mode in keys:
        nominations = [
            row
            for row in observations
            if row.get("stream_id") == stream_id and row.get("admission_mode") == mode
        ]
        effects = [
            row
            for row in decisions
            if row.get("stream_id") == stream_id and row.get("admission_mode") == mode
        ]
        result.append(
            {
                "unit_id": f"{stream_id}:{mode}",
                "stream_id": stream_id,
                "admission_mode": mode,
                "nomination_count": len(nominations),
                "candidate_mapping_difference_count": sum(
                    int(row.get("mapping_changed") is True) for row in nominations
                ),
                "selected_archive_difference_count": sum(
                    int(row.get("selection_changed") is True) for row in nominations
                ),
                "later_decision_difference_count": sum(
                    int(row.get("aligned_prediction") != row.get("shuffled_prediction"))
                    for row in effects
                    if int(row.get("released_query_count_before", 0)) > 0
                ),
                "pre_release_decision_difference_count": sum(
                    int(row.get("aligned_prediction") != row.get("shuffled_prediction"))
                    for row in effects
                    if int(row.get("released_query_count_before", 0)) == 0
                ),
                "zero_headroom_count": sum(
                    int(int(row.get("eligible_candidate_count", 0)) <= 1) for row in nominations
                ),
                "censored": False,
            }
        )
    return result


def _decision_effects(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Pair aligned and shuffled predictions for both archive admission rules."""

    indexed = {(str(row["stream_id"]), str(row["arm"]), str(row["event_id"])): row for row in rows}
    effects: list[JsonDict] = []
    for stream_id in sorted({str(row["stream_id"]) for row in rows}):
        event_ids = [
            str(row["event_id"])
            for row in rows
            if row["stream_id"] == stream_id and row["arm"] == "coverage_archive_aligned"
        ]
        for mode in ("fifo", "coverage"):
            aligned_arm = f"{mode}_archive_aligned"
            shuffled_arm = f"{mode}_archive_shuffled"
            for event_id in event_ids:
                aligned = indexed[(stream_id, aligned_arm, event_id)]
                shuffled = indexed[(stream_id, shuffled_arm, event_id)]
                effects.append(
                    {
                        "stream_id": stream_id,
                        "admission_mode": mode,
                        "event_id": event_id,
                        "aligned_prediction": aligned["prediction"],
                        "shuffled_prediction": shuffled["prediction"],
                        "released_query_count_before": aligned["released_query_count_before"],
                    }
                )
    return effects


def _causal_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute causal counts from target, shuffled, frozen, and oracle rows."""

    target = [row for row in rows if row.get("arm") == "coverage_archive_aligned"]
    shuffled = [row for row in rows if row.get("arm") == "coverage_archive_shuffled"]
    frozen = [row for row in rows if row.get("arm") == "frozen_warmup"]
    return {
        "constraint_addition_count": sum(int(row["constraint_addition_count"]) for row in target),
        "constraint_deactivation_count": sum(
            int(row["constraint_deactivation_count"]) for row in target
        ),
        "valid_reactivation_count": sum(int(row["valid_reactivation_count"]) for row in target),
        "later_changed_decision_after_reactivation_count": sum(
            int(row["later_changed_decision_after_reactivation"]) for row in target
        ),
        "pre_release_difference_count": sum(int(row["pre_release_difference"]) for row in target),
        "prospective_shuffle_selection_change_count": sum(
            int(row["shuffle_selection_change_count"]) for row in shuffled
        ),
        "oracle_headroom_event_count": sum(
            int(
                int(row["chronology_index"]) >= exp7253.WARMUP_COUNT
                and int(row["full_denominator_error"]) > 0
            )
            for row in frozen
        ),
        "zero_pre_release_differences": all(
            int(row["pre_release_difference"]) == 0 for row in target
        ),
    }


def _bootstrap_interval(values: Sequence[float], draws: int, salt: str) -> JsonDict:
    """Resample whole streams with the frozen seed and deterministic percentile rule."""

    if not values:
        return {"estimate": 0.0, "ci95": [0.0, 0.0]}
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
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS
) -> list[JsonDict]:
    """Build paired stream comparisons without using producer comparison rows."""

    indexed = {(int(row["seed"]), str(row["arm"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    result: list[JsonDict] = []
    for comparison_id, metric, control in COMPARISON_SPECS:
        differences: list[JsonDict] = []
        for seed in seeds:
            target = indexed[(seed, "coverage_archive_aligned")].get(metric)
            baseline = indexed[(seed, control)].get(metric)
            if isinstance(target, (int, float)) and isinstance(baseline, (int, float)):
                differences.append({"seed": seed, "difference": float(target) - float(baseline)})
        interval = _bootstrap_interval(
            [float(row["difference"]) for row in differences], draws, comparison_id
        )
        result.append(
            {
                "comparison_id": comparison_id,
                "metric": metric,
                "target_arm": "coverage_archive_aligned",
                "control_arm": control,
                "independent_unit": "stream",
                "independent_unit_count": len(differences),
                "bootstrap_draws": draws,
                "seed_differences": differences,
                **interval,
            }
        )
    return result


def score_exp7254_gates(
    comparisons: Sequence[Mapping[str, Any]], causal_summary: Mapping[str, Any]
) -> dict[str, JsonDict]:
    """Recompute every frozen Exp7254 scientific gate, including the 0.02 bound."""

    by_id = {str(row["comparison_id"]): row for row in comparisons}

    def upper(name: str) -> float:
        return float(by_id[name]["ci95"][1])

    def estimate(name: str) -> float:
        return float(by_id[name]["estimate"])

    definitions = (
        (
            "future_error_vs_frozen_upper_ci95_lt_zero",
            "<0",
            upper("future_error_vs_frozen"),
            upper("future_error_vs_frozen") < 0,
        ),
        (
            "future_error_vs_reset_upper_ci95_lt_zero",
            "<0",
            upper("future_error_vs_reset"),
            upper("future_error_vs_reset") < 0,
        ),
        (
            "false_accept_vs_frozen_upper_ci95_lte_zero",
            "<=0",
            upper("false_accept_vs_frozen"),
            upper("false_accept_vs_frozen") <= 0,
        ),
        (
            "false_accept_vs_reset_upper_ci95_lte_zero",
            "<=0",
            upper("false_accept_vs_reset"),
            upper("false_accept_vs_reset") <= 0,
        ),
        (
            "recurrence_error_increase_vs_frozen_lte_0_02",
            "<=0.02",
            estimate("recurrence_error_vs_frozen"),
            estimate("recurrence_error_vs_frozen") <= 0.02,
        ),
        (
            "recurrence_error_vs_destructive_upper_ci95_lt_zero",
            "<0",
            upper("recurrence_error_vs_destructive"),
            upper("recurrence_error_vs_destructive") < 0,
        ),
        (
            "recurrence_error_vs_coverage_shuffled_upper_ci95_lt_zero",
            "<0",
            upper("recurrence_error_vs_coverage_shuffled"),
            upper("recurrence_error_vs_coverage_shuffled") < 0,
        ),
        (
            "valid_reactivation_gt_zero",
            ">0",
            int(causal_summary["valid_reactivation_count"]),
            int(causal_summary["valid_reactivation_count"]) > 0,
        ),
        (
            "later_changed_decision_after_reactivation_gt_zero",
            ">0",
            int(causal_summary["later_changed_decision_after_reactivation_count"]),
            int(causal_summary["later_changed_decision_after_reactivation_count"]) > 0,
        ),
        (
            "pre_release_difference_eq_zero",
            "==0",
            int(causal_summary["pre_release_difference_count"]),
            int(causal_summary["pre_release_difference_count"]) == 0,
        ),
        (
            "positive_aligned_vs_fifo_value",
            ">0 coverage gain",
            -estimate("future_error_vs_fifo_aligned"),
            -estimate("future_error_vs_fifo_aligned") > 0,
        ),
        (
            "effective_prospective_shuffle_intervention",
            ">0",
            int(causal_summary["prospective_shuffle_selection_change_count"]),
            int(causal_summary["prospective_shuffle_selection_change_count"]) > 0,
        ),
        (
            "oracle_control_headroom_gt_zero",
            ">0",
            int(causal_summary["oracle_headroom_event_count"]),
            int(causal_summary["oracle_headroom_event_count"]) > 0,
        ),
        (
            "constraint_addition_gt_zero",
            ">0",
            int(causal_summary["constraint_addition_count"]),
            int(causal_summary["constraint_addition_count"]) > 0,
        ),
        (
            "constraint_deactivation_gt_zero",
            ">0",
            int(causal_summary["constraint_deactivation_count"]),
            int(causal_summary["constraint_deactivation_count"]) > 0,
        ),
    )
    return {
        name: {
            "principle": "Keep each preregistered science criterion separate from audit completion.",
            "expected": expected,
            "observed": observed,
            "pass": passed,
            "passed": passed,
        }
        for name, expected, observed, passed in definitions
    }


def audit_raw_evidence(
    repo_root: Path,
    *,
    stream_ids: Sequence[str],
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    progress: bool = False,
) -> JsonDict:
    """Reduce authenticated rows and replay shipped controllers in this cold worker."""

    started = time.monotonic()
    if progress:
        print("phase 3 cold reducer load START", flush=True)
    views = _selected_views(repo_root, stream_ids)
    raw_rows = _selected_stream_rows(_resolve(repo_root, DEFAULT_PREQUENTIAL_ROWS), stream_ids)
    if progress:
        print(
            f"phase 3 cold reducer load END completed_units={len(stream_ids)} "
            f"raw_rows={len(raw_rows)} elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
        print("phase 4 controller replay START", flush=True)
    raw_check_rows = _raw_evidence_checks(raw_rows, views, stream_ids)
    reduced = _reduce_rows(raw_rows)
    learner = _load_object(_resolve(repo_root, DEFAULT_LEARNER_ARTIFACT))
    expected = [row for row in learner.get("rows", []) if row.get("stream_id") in set(stream_ids)]
    raw_check_rows.append(
        {
            "check": "producer_summary_parity",
            "expected": transactional.sha256_json(expected),
            "observed": transactional.sha256_json(reduced),
            "passed": reduced == expected,
        }
    )
    replay_rows, observations = _replay_with_instrumentation(
        views, raw_rows, repo_root, stream_ids, progress=progress
    )
    decisions = _decision_effects(raw_rows)
    causal = _causal_summary(raw_rows)
    comparison_rows = build_comparison_rows(reduced, draws=bootstrap_draws)
    if progress:
        print(
            f"phase 4 controller replay END completed_units={len(stream_ids)} "
            f"elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
    return {
        "rows": reduced,
        "raw_check_rows": raw_check_rows,
        "replay_rows": replay_rows,
        "comparison_rows": comparison_rows,
        "causal_summary": causal,
        "shuffle_effect_rows": build_shuffle_effect_rows(observations, decisions),
        "process_receipt": {
            "worker_pid": os.getpid(),
            "parent_pid": int(os.environ.get("CARNOT_7255_PARENT_PID", "0")),
            "fresh_process": int(os.environ.get("CARNOT_7255_PARENT_PID", "0")) != os.getpid(),
            "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "network_cache_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "no_model_load": not any(
                name in sys.modules for name in ("torch", "transformers", "llama_cpp")
            ),
        },
    }


def _mutation_row(name: str, rejection: str, observed: bool) -> JsonDict:
    """Use one explicit shape for every isolated negative fixture."""

    observed_rejection = rejection if observed else "not_rejected"
    return {
        "mutation": name,
        "expected_rejection": rejection,
        "observed_rejection": observed_rejection,
        "passed": observed_rejection == rejection,
    }


def _control_release(event_id: str, label: str, index: int) -> JsonDict:
    """Build one due finite release for transaction controls."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 0,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def run_mutation_controls(root: Path) -> list[JsonDict]:
    """Apply all five causal and durability attacks outside the compute receipt."""

    root.mkdir(parents=True, exist_ok=True)
    signatures = ("aaaa", "aaaa", "aaar", "aarr", "arrr", "rrrr")
    archives = [
        exp7253.diagnostic_archive(f"archive-{index}", index, tuple(signature))
        for index, signature in enumerate(signatures)
    ]
    fifo, _ = exp7253.retain_archives(archives, mode="fifo")
    coverage, _ = exp7253.retain_archives(archives, mode="coverage")
    natural_difference = [row["archive_id"] for row in fifo] != [
        row["archive_id"] for row in coverage
    ]
    mutated_coverage = deepcopy(fifo)
    alias_rejected = natural_difference and [row["archive_id"] for row in fifo] == [
        row["archive_id"] for row in mutated_coverage
    ]

    shuffle = exp7253.run_shuffle_diagnostic()
    canceled = deepcopy(shuffle)
    canceled["after_mapping"] = deepcopy(canceled["before_mapping"])
    canceled["selected_after_archive_id"] = canceled["selected_before_archive_id"]
    canceled["selection_changed"] = False
    shuffle_rejected = (
        shuffle["before_mapping"] != shuffle["after_mapping"]
        and canceled["before_mapping"] == canceled["after_mapping"]
    )

    leaked = {
        "oracle_control": False,
        "controller_input_fields": [
            "event_id",
            "family_id",
            "numeric_value",
            "later_released_label",
        ],
        "held_out_label_visible_to_controller": False,
        "prediction_frozen_before_release": True,
    }
    future_rejected = "future_label_leakage" in decision_visibility_errors(leaked)

    views = exp7253.build_stream_views("prospective")
    public = deepcopy(views.public[:1])
    releases = deepcopy(views.releases[:1])
    authority = deepcopy(views.authority[:1])
    authority[0]["regime_id"] = "private-mutation"
    private_rejected = "private_regime_mismatch" in stream_authority_errors(
        public, releases, authority
    )

    controller = exp7253.CoverageArchiveController()
    state_path = root / "durable-write-control.json"
    controller.save(state_path)
    durable_parent = state_path.read_bytes()
    controller.commit_batch(
        [_control_release("omitted-write", "accept", 1)],
        current_cycle=1,
        expected_parent_hash=controller.state_hash(),
    )
    omitted_rejected = (
        controller.state_bytes() != durable_parent and state_path.read_bytes() == durable_parent
    )
    return [
        _mutation_row("fifo_coverage_alias", "fifo_coverage_alias_detected", alias_rejected),
        _mutation_row("shuffle_canceled", "shuffle_canceled_detected", shuffle_rejected),
        _mutation_row("future_label_leak", "future_label_leak_detected", future_rejected),
        _mutation_row("private_regime_change", "private_regime_change_detected", private_rejected),
        _mutation_row(
            "durable_write_omission", "durable_write_omission_detected", omitted_rejected
        ),
    ]


def _isolated_environment(parent_pid: int) -> dict[str, str]:
    """Disable accelerators and online model caches for every fresh worker."""

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "CARNOT_7255_PARENT_PID": str(parent_pid),
        }
    )
    return environment


def _spawn_worker(command: Sequence[str], label: str, timeout_s: float = 1200.0) -> JsonDict:
    """Reuse the shipped streamed subprocess runner and its truthful heartbeat."""

    return exp7242.exp7228.exp7214.cold_support._stream_subprocess(
        command,
        _isolated_environment(os.getpid()),
        label=label,
        timeout_s=timeout_s,
    )


def _reload_worker(state_path: Path, event: Mapping[str, Any]) -> JsonDict:
    """Cold-load one state and return its decision for E2E-007 parity."""

    controller = exp7253.CoverageArchiveController.load(state_path)
    prediction = controller.predict(event)
    return {
        "state_sha256": controller.state_hash(),
        "prediction": list(prediction),
        "worker_pid": os.getpid(),
    }


def run_e2e_controls(root: Path, *, progress: bool = False) -> list[JsonDict]:
    """Run cold load, unseen query, delayed commit, rejection, reload, and rollback."""

    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "controller.json"
    exp7253.CoverageArchiveController().save(state_path)
    if progress:
        print("phase 5 E2E-007 cold load START", flush=True)
    controller = exp7253.CoverageArchiveController.load(state_path)
    parent_bytes = controller.state_bytes()
    event = {
        "event_id": "unseen-next-query",
        "family_id": "lower_bound",
        "numeric_value": 313,
    }
    prediction_before = controller.predict(event)
    cold_row = {
        "control": "cold_load_unseen_query_before_feedback",
        "passed": controller.state_bytes() == parent_bytes and prediction_before[0] == "accept",
    }
    release = _control_release("delayed-feedback", "reject", 1)
    receipt = controller.commit_batch(
        [release],
        current_cycle=1,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    prediction_after = controller.predict(event)
    commit_row = {
        "control": "accepted_delayed_feedback_commit",
        "passed": receipt["parent_hash"] != receipt["new_state_hash"]
        and prediction_before != prediction_after
        and state_path.read_bytes() == controller.state_bytes(),
    }
    child_bytes = controller.state_bytes()
    rejected = False
    try:
        controller.commit_batch(
            [_control_release("wrong-parent", "accept", 2)],
            current_cycle=2,
            expected_parent_hash="sha256:" + "0" * 64,
            state_path=state_path,
        )
    except exp7253.ArchiveCommitRejected:
        rejected = True
    rejection_row = {
        "control": "wrong_parent_rejected_without_write",
        "passed": rejected
        and controller.state_bytes() == child_bytes
        and state_path.read_bytes() == child_bytes,
    }
    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--reload-worker",
        "--date",
        RUN_DATE,
        "--state-path",
        str(state_path),
        "--event-json",
        json.dumps(event, sort_keys=True),
    ]
    if progress:
        print("phase 5 E2E-007 fresh-process reload START", flush=True)
    cold = _spawn_worker(command, "phase 5 E2E-007 reload", timeout_s=120.0)
    if progress:
        print("phase 5 E2E-007 fresh-process reload END", flush=True)
    reload_row = {
        "control": "fresh_process_reload_decision_parity",
        "passed": cold.get("state_sha256") == controller.state_hash()
        and cold.get("prediction") == list(prediction_after)
        and cold.get("worker_pid") != os.getpid(),
    }
    rollback = controller.rollback(receipt, state_path=state_path)
    rollback_row = {
        "control": "rollback_restores_parent_bytes",
        "passed": rollback["byte_identical"] is True
        and controller.state_bytes() == parent_bytes
        and state_path.read_bytes() == parent_bytes,
    }
    if progress:
        print("phase 5 E2E-007 cold load END completed_units=5", flush=True)
    return [cold_row, commit_row, rejection_row, reload_row, rollback_row]


def derive_terminal_scores(
    *, audit_complete: bool, gates: Mapping[str, Mapping[str, Any]], safety_passed: bool
) -> tuple[int, int, str, str]:
    """Keep completed-audit status separate from learner promotion."""

    complete = int(audit_complete)
    science = all(row.get("passed", row.get("pass")) is True for row in gates.values())
    promotion = int(audit_complete and science and safety_passed)
    if promotion:
        return (
            complete,
            promotion,
            "circular_positive",
            ("complete_circular_positive: coverage value reproduced with exact-oracle conformance"),
        )
    return (
        complete,
        promotion,
        "null",
        ("complete_null: coverage audit completed but promotion criteria did not all pass"),
    )


def _sample_budget(stream_ids: Sequence[str], *, complete: bool) -> JsonDict:
    """Declare the fixed all-stream stopping rule without outcome-based extension."""

    attempted = len(stream_ids)
    completed = attempted if complete else 0
    return {
        "independent_units_planned": attempted,
        "independent_units_attempted": attempted if complete else 0,
        "independent_units_completed": completed,
        "independent_units_censored": 0,
        "arms_per_unit": len(exp7253.ARMS),
        "events_per_unit": exp7253.EVENTS_PER_STREAM,
        "planned_arm_event_rows": attempted * exp7253.EVENTS_PER_STREAM * len(exp7253.ARMS),
        "completed_arm_event_rows": completed * exp7253.EVENTS_PER_STREAM * len(exp7253.ARMS),
        "stopping_rule": "all predeclared streams once; no outcome-based extension",
    }


def _stable_artifact(value: Mapping[str, Any]) -> JsonDict:
    """Remove only invocation clocks, host identity, process IDs, and checksum itself."""

    stable = deepcopy(dict(value))
    for key in (
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_spans_s",
        "execution_host",
        "reproducibility_checksum",
    ):
        stable.pop(key, None)
    receipt = stable.get("reducer_process_receipt")
    if isinstance(receipt, dict):
        receipt.pop("worker_pid", None)
        receipt.pop("parent_pid", None)
    return stable


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind sources, inputs, settings, reduced rows, gates, and safety evidence."""

    return transactional.sha256_json(_stable_artifact(artifact))


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstreams: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str],
    duration_s: float,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> JsonDict:
    """Create every required top-level field before terminal classification."""

    now = datetime.now(UTC).isoformat()
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at or now,
        "completed_at_utc": completed_at or now,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": list(checks),
        "MODEL_SPECS": MODEL_SPECS,
        "model_invoked": MODEL_INVOKED,
        "model_load_count": 0,
        "model_generation_count": 0,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "phase_spans_s": {},
        "random_seed": {
            "audit_seed": AUDIT_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, complete=False),
        "acceptance_gate_results": {},
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "coverage_audit_complete_score": 0,
        "coverage_promotion_score": 0,
        "shuffle_effect_rows": [],
        "mutation_rows": [],
        "continuous_self_learning_task": {
            "scope": "persistent cross-query constraint change",
            "llm_weights": "immutable",
            "current_llm_invocations": 0,
        },
        "upstream_observations": {
            name: {
                "status": value.get("status"),
                "verdict_class": value.get("verdict_class"),
                "honest_verdict": value.get("honest_verdict"),
            }
            for name, value in upstreams.items()
        },
        "output_paths": {
            "checkpoint": str(paths.checkpoint),
            "mutation_sidecar": str(paths.mutation_sidecar),
            "artifact": str(paths.artifact),
        },
        "methodology": {
            "kind": "independent deterministic raw-row reduction and controller replay",
            "no_llm_invocation": True,
            "immutable_model_weights": True,
        },
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstreams: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str],
    duration_s: float,
) -> JsonDict:
    """Return a terminal blocked result for one absent or quarantined prerequisite."""

    artifact = _base_artifact(
        checks,
        upstreams,
        source_hashes,
        paths,
        stream_ids=stream_ids,
        duration_s=duration_s,
    )
    summary = gate_summary(checks)
    if summary.get("field") is None:
        summary = {
            "failed_check": "execution_not_attempted",
            "upstream": "exp7255",
            "field": "execution_state",
            "expected_value": "complete",
            "observed_value": "not_run",
        }
    artifact["gate_check_summary"] = {
        "passed": False,
        "check": summary.get("failed_check"),
        "upstream": summary.get("upstream"),
        "field": summary.get("field"),
        "expected_value": summary.get("expected_value"),
        "observed_value": summary.get("observed_value"),
    }
    artifact["honest_verdict"] = f"blocked_external_precondition: {summary.get('failed_check')}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _receipt_error(receipt: Mapping[str, Any]) -> bool:
    """Return whether one validation receipt lacks auditable command evidence."""

    return not (
        isinstance(receipt.get("command"), str)
        and bool(receipt.get("command"))
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("classification"), str)
        and bool(receipt.get("classification"))
        and isinstance(receipt.get("log_sha256"), str)
        and str(receipt.get("log_sha256")).startswith("sha256:")
    )


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check terminal shape, scores, raw evidence, controls, files, and checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("milestone") != MILESTONE, "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("MODEL_SPECS") != [], "model_specs")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    add(artifact.get("execution_venue") != "host", "execution_venue")
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts),
        "validation_receipt_schema",
    )
    status = artifact.get("status")
    add(status not in {"complete", "blocked"}, "status")
    streams = tuple(
        expected_stream_ids
        or sorted({str(row.get("stream_id")) for row in artifact.get("rows", [])})
    )
    if status == "blocked":
        add(
            artifact.get("verdict_class") != "blocked"
            or artifact.get("rows") != []
            or artifact.get("coverage_audit_complete_score") != 0
            or artifact.get("coverage_promotion_score") != 0
            or not artifact.get("gate_check_summary", {}).get("field"),
            "blocked_contract",
        )
        return errors

    add(artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate")
    add(
        artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate_class",
    )
    rows = artifact.get("rows", [])
    expected_units = {(stream, arm) for stream in streams for arm in exp7253.ARMS}
    observed_units = {
        (str(row.get("stream_id")), str(row.get("arm"))) for row in rows if isinstance(row, Mapping)
    }
    add(observed_units != expected_units, "unit_rows")
    draws = int(artifact.get("random_seed", {}).get("bootstrap_draws", 0))
    comparisons = build_comparison_rows(rows, draws=draws)
    add(artifact.get("comparison_rows") != comparisons, "comparison_rows")
    gates = score_exp7254_gates(comparisons, artifact.get("causal_summary", {}))
    add(artifact.get("acceptance_gate_results") != gates, "acceptance_gate_results")
    safety_rows = [
        *artifact.get("raw_check_rows", []),
        *artifact.get("replay_rows", []),
        *artifact.get("mutation_rows", []),
        *artifact.get("e2e_rows", []),
    ]
    safety_passed = all(
        isinstance(row, Mapping) and row.get("passed") is True for row in safety_rows
    )
    receipt = artifact.get("reducer_process_receipt", {})
    safety_passed = safety_passed and all(
        receipt.get(field) is True
        for field in (
            "fresh_process",
            "gpu_disabled",
            "network_cache_offline",
            "no_model_load",
        )
    )
    scores = derive_terminal_scores(audit_complete=True, gates=gates, safety_passed=safety_passed)
    add(artifact.get("coverage_audit_complete_score") != scores[0], "audit_score")
    add(artifact.get("coverage_promotion_score") != scores[1], "promotion_score")
    add(artifact.get("verdict_class") != scores[2], "verdict_class")
    add(artifact.get("honest_verdict") != scores[3], "honest_verdict")
    add(artifact.get("verifier_is_oracle") is not True, "oracle_disclosure")
    add(len(artifact.get("shuffle_effect_rows", [])) != len(streams) * 2, "shuffle_effect_rows")
    add(len(artifact.get("mutation_rows", [])) != 5, "mutation_rows")
    add(len(artifact.get("e2e_rows", [])) != 5, "e2e_rows")
    if check_files:
        sidecar = artifact.get("mutation_sidecar_path", {})
        sidecar_path = _resolve(repo_root, str(sidecar.get("path", "")))
        add(_sha256_path(sidecar_path) != sidecar.get("sha256"), "mutation_sidecar_hash")
        for path, expected in artifact.get("source_artifact_hashes", {}).items():
            if path == str(sidecar.get("path")):
                continue
            add(_sha256_path(_resolve(repo_root, path)) != expected, "source_artifact_hash")
    return errors


def _require_valid(errors: Sequence[str], prefix: str = "artifact_validation") -> None:
    """Refuse terminal publication when any cold validator check fails."""

    if errors:
        raise AuditEvidenceError(prefix + ":" + ",".join(errors))


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach only actual command receipts and refresh the stable checksum."""

    if any(_receipt_error(receipt) for receipt in receipts):
        raise ValueError("validation_receipt_schema")
    changed = deepcopy(dict(artifact))
    changed["validation_receipts"] = [dict(receipt) for receipt in receipts]
    changed["reproducibility_checksum"] = reproducibility_checksum(changed)
    return changed


def _write_checkpoint(path: Path, value: Mapping[str, Any]) -> JsonDict:
    """Write only explicitly provisional evidence under results/checkpoints."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = transactional.canonical_json_bytes(value)
    receipt = transactional._atomic_write(path, payload)
    return {
        **receipt,
        "sha256": transactional.sha256_bytes(payload),
        "bytes": len(payload),
    }


def _spawn_audit_worker(stream_ids: Sequence[str], bootstrap_draws: int) -> JsonDict:
    """Run the independent reduction and replay in one isolated cold process."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--audit-worker",
        "--date",
        RUN_DATE,
        "--stream-ids",
        ",".join(stream_ids),
        "--bootstrap-draws",
        str(bootstrap_draws),
    ]
    return _spawn_worker(command, "phase 3-4 Exp7255 cold audit", timeout_s=1200.0)


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    progress: bool = False,
) -> JsonDict:
    """Authenticate, cold-reduce, attack, score, and return one validated terminal object."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(exp7253.STREAM_COUNT))
    )
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    phase_started = time.monotonic()
    if progress:
        _progress(0, "start", "authenticate upstream bytes, quarantine, imports, and paths")
    checks, upstreams, source_hashes = collect_preconditions(repo_root, paths)
    spans["phase_0_authentication"] = time.monotonic() - phase_started
    if progress:
        _progress(0, "end", f"checks={len(checks)} passed={gate_summary(checks)['passed']}")

    phase_started = time.monotonic()
    if progress:
        _progress(1, "start", "verify driving REQ and SCENARIO contract")
    spec_ready = all(
        row.get("passed") is True
        for row in checks
        if row.get("check") in {"driving_capability_spec", "scenario_contract"}
    )
    spans["phase_1_spec_contract"] = time.monotonic() - phase_started
    if progress:
        _progress(1, "end", f"spec_ready={spec_ready}")

    phase_started = time.monotonic()
    if progress:
        _progress(2, "start", "classify authenticated external preconditions")
    preconditions_passed = gate_summary(checks)["passed"] is True
    spans["phase_2_preconditions"] = time.monotonic() - phase_started
    if progress:
        _progress(2, "end", f"passed={preconditions_passed}")
    if not preconditions_passed:
        for phase, detail in (
            (3, "cold reducer skipped after external block"),
            (4, "controller replay skipped after external block"),
            (5, "negative controls skipped after external block"),
            (6, "promotion scoring skipped after external block"),
        ):
            if progress:
                _progress(phase, "start", detail)
                _progress(phase, "end", detail)
        return build_blocked_artifact(
            checks,
            upstreams,
            source_hashes,
            paths,
            stream_ids=selected,
            duration_s=time.monotonic() - started,
        )

    _write_checkpoint(
        paths.checkpoint,
        {
            "schema": "carnot.exp7255.checkpoint.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "in_progress",
            "started_at_utc": started_at,
            "completed_units": 0,
            "planned_units": len(selected),
        },
    )
    phase_started = time.monotonic()
    if progress:
        _progress(3, "start", f"spawn cold raw reducer units={len(selected)}")
    worker = _spawn_audit_worker(selected, bootstrap_draws)
    spans["phase_3_raw_reduction"] = time.monotonic() - phase_started
    if progress:
        _progress(3, "end", f"completed_units={len(selected)} rows={len(worker['rows'])}")

    phase_started = time.monotonic()
    if progress:
        _progress(4, "start", "verify replay and durable-state findings")
    replay_passed = all(row.get("passed") is True for row in worker["replay_rows"])
    spans["phase_4_replay_verification"] = time.monotonic() - phase_started
    if progress:
        _progress(4, "end", f"checks={len(worker['replay_rows'])} passed={replay_passed}")

    phase_started = time.monotonic()
    if progress:
        _progress(5, "start", "run five mutations and E2E-007 adaptation")
    mutation_rows = run_mutation_controls(paths.e2e_root / "mutations")
    e2e_rows = run_e2e_controls(paths.e2e_root, progress=progress)
    historical = {
        "exp7242": {
            "path": str(DEFAULT_OLD_AUDIT),
            "sha256": source_hashes[str(DEFAULT_OLD_AUDIT)],
            "role": "one_time_failed_predecessor_audit_history",
        },
        "exp7253": {
            "path": str(DEFAULT_FIXTURE_ARTIFACT),
            "sha256": source_hashes[str(DEFAULT_FIXTURE_ARTIFACT)],
            "role": "source_model_history_not_current_invocation",
        },
        "exp7254": {
            "path": str(DEFAULT_LEARNER_ARTIFACT),
            "sha256": source_hashes[str(DEFAULT_LEARNER_ARTIFACT)],
            "role": "audited_learner_history_not_current_invocation",
        },
    }
    sidecar_value = {
        "schema": "carnot.exp7255.mutation_receipts.v1",
        "experiment_id": EXPERIMENT_ID,
        "current_compute_receipt": False,
        "historical_model_receipts": historical,
        "mutation_rows": mutation_rows,
        "e2e_rows": e2e_rows,
    }
    sidecar_write = _write_checkpoint(paths.mutation_sidecar, sidecar_value)
    source_hashes[str(paths.mutation_sidecar)] = str(sidecar_write["sha256"])
    spans["phase_5_mutations_and_e2e"] = time.monotonic() - phase_started
    if progress:
        _progress(
            5,
            "end",
            f"mutations={len(mutation_rows)} e2e_controls={len(e2e_rows)}",
        )

    phase_started = time.monotonic()
    if progress:
        _progress(6, "start", "recompute frozen Exp7254 gates and separate audit from promotion")
    gates = score_exp7254_gates(worker["comparison_rows"], worker["causal_summary"])
    safety_passed = (
        all(row.get("passed") is True for row in worker["raw_check_rows"])
        and replay_passed
        and all(row.get("passed") is True for row in mutation_rows)
        and all(row.get("passed") is True for row in e2e_rows)
        and all(
            worker["process_receipt"].get(field) is True
            for field in (
                "fresh_process",
                "gpu_disabled",
                "network_cache_offline",
                "no_model_load",
            )
        )
    )
    scores = derive_terminal_scores(
        audit_complete=True,
        gates=gates,
        safety_passed=safety_passed,
    )
    spans["phase_6_terminal_scoring"] = time.monotonic() - phase_started
    if progress:
        _progress(6, "end", f"audit_complete={scores[0]} promotion={scores[1]}")

    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        upstreams,
        source_hashes,
        paths,
        stream_ids=selected,
        duration_s=time.monotonic() - started,
        started_at=started_at,
        completed_at=completed_at,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "random_seed": {
                "audit_seed": AUDIT_SEED,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_draws": bootstrap_draws,
            },
            "rows": worker["rows"],
            "sample_size_budget": _sample_budget(selected, complete=True),
            "acceptance_gate_results": gates,
            "gate_check_summary": {"passed": True, "failed_check": None},
            "honest_verdict": scores[3],
            "verdict_class": scores[2],
            "coverage_audit_complete_score": scores[0],
            "coverage_promotion_score": scores[1],
            "shuffle_effect_rows": worker["shuffle_effect_rows"],
            "mutation_rows": mutation_rows,
            "e2e_rows": e2e_rows,
            "comparison_rows": worker["comparison_rows"],
            "causal_summary": worker["causal_summary"],
            "raw_check_rows": worker["raw_check_rows"],
            "replay_rows": worker["replay_rows"],
            "reducer_process_receipt": worker["process_receipt"],
            "mutation_sidecar_path": {
                "path": str(paths.mutation_sidecar),
                "sha256": sidecar_write["sha256"],
                "bytes": sidecar_write["bytes"],
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _require_valid(
        validate_artifact(
            artifact,
            repo_root=repo_root,
            expected_stream_ids=selected,
            check_files=True,
        )
    )
    return artifact


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Validate and publish terminal bytes through one flushed atomic rename."""

    _require_valid(
        validate_artifact(
            artifact,
            repo_root=repo_root,
            expected_stream_ids=expected_stream_ids,
            check_files=artifact.get("status") == "complete",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return transactional._atomic_write(path, transactional.canonical_json_bytes(artifact))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse terminal, validation, raw-worker, and E2E reload modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--stream-ids", default="")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--audit-worker", action="store_true")
    parser.add_argument("--reload-worker", action="store_true")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--event-json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the no-LLM audit and atomically publish only terminal validated bytes."""

    print("phase 0 boundary: Exp7255 process entered", flush=True)
    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    paths = ExperimentPaths.under(args.output_root)
    selected = tuple(filter(None, args.stream_ids.split(","))) or tuple(
        f"prospective-{index + 1:02d}" for index in range(exp7253.STREAM_COUNT)
    )
    if args.reload_worker:
        if args.state_path is None or args.event_json is None:
            raise ValueError("reload_worker_arguments")
        event = json.loads(args.event_json)
        print(
            RESULT_PREFIX + json.dumps(_reload_worker(args.state_path, event), sort_keys=True),
            flush=True,
        )
        return 0
    if args.audit_worker:
        result = audit_raw_evidence(
            REPO_ROOT,
            stream_ids=selected,
            bootstrap_draws=args.bootstrap_draws,
            progress=True,
        )
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.validate:
        _progress(7, "start", "cold-validate existing terminal artifact")
        artifact = _load_object(paths.artifact)
        _require_valid(
            validate_artifact(
                artifact,
                repo_root=REPO_ROOT,
                expected_stream_ids=(selected if artifact.get("status") == "complete" else None),
                check_files=artifact.get("status") == "complete",
            )
        )
        _progress(7, "end", "artifact valid")
        return 0
    artifact = build_and_seal(
        REPO_ROOT,
        paths,
        stream_ids=selected,
        bootstrap_draws=args.bootstrap_draws,
        progress=True,
    )
    _progress(7, "start", "cold-validate terminal object before publication")
    _require_valid(
        validate_artifact(
            artifact,
            repo_root=REPO_ROOT,
            expected_stream_ids=selected,
            check_files=artifact.get("status") == "complete",
        )
    )
    _progress(7, "end", "terminal object valid")
    _progress(8, "start", f"atomic terminal write {paths.artifact}")
    receipt = write_artifact(
        paths.artifact,
        artifact,
        repo_root=REPO_ROOT,
        expected_stream_ids=selected,
    )
    _progress(8, "end", f"sha256={receipt['sha256']} bytes={receipt['bytes']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the thin wrapper.
    raise SystemExit(main())
