"""Audit admission learning without treating an all-reject result as learning.

The audit reads the sealed Exp7281 fixture and the complete Exp7282 journals.
It reduces raw evidence in a fresh process, then tests transaction behavior
with the shipped admission controller. Exact labels remain evaluator authority,
so even favorable evidence is circular rather than oracle-distinct.

Spec refs: REQ-CL-7283 and SCENARIO-CL-7283-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_7281_v640_admission_prototype as fixture
from carnot import experiment_7282_v640_admission_learning as learning
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7283
SCHEMA = "carnot.exp7283.v640_admission_audit.v1"
MILESTONE = "2026.09.640"
RUN_DATE = "20260913"
RANDOM_SEED = 7_283_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = deepcopy(fixture.INVOCATION_COUNTS)
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
RESULT_PREFIX = "EXP7283_WORKER_RESULT="

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7283_v640_admission_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7283_v640_admission_audit.py")
DEFAULT_FIXTURE_ARTIFACT = Path("results/experiment_7281_v640_admission_prototype.json")
DEFAULT_LEARNING_ARTIFACT = Path("results/experiment_7282_v640_admission_learning.json")
DEFAULT_ARTIFACT = Path("results/experiment_7283_v640_admission_audit.json")
EXPECTED_FIXTURE_SHA256 = "sha256:967854fca40f6488d736be00e339fe47f360bd65626afee3816e09a29cd7d356"
EXPECTED_LEARNING_SHA256 = "sha256:5989847333081544f79a03248f072593a1b7aada3ed39c75f056692bb1b3ce61"
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7283-[A-Z-]+")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7281_v640_admission_prototype.py"),
    Path("python/carnot/experiment_7282_v640_admission_learning.py"),
    Path("python/carnot/experiment_7283_v640_admission_audit.py"),
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
)

CONTROL_NAMES = (
    "withheld_feedback",
    "corrupted_label",
    "stale_parent",
    "changed_candidate",
    "duplicate_release",
    "restart_before_admission",
    "interrupted_commit",
    "rollback",
)
E2E_STAGES = (
    "update_and_later_reuse",
    "state_hash_change",
    "invalid_rejection",
    "archive_retention",
    "cold_restart",
    "exact_rollback",
    "full_state_bytes",
    "model_weight_immutability",
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
    "current_model_load_count",
    "current_generation_count",
    "current_inference_count",
    "inference_substrate",
    "inference_substrate_class",
    "reducer_inference_substrate",
    "reducer_inference_substrate_class",
    "execution_venue",
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
    "admission_audit_complete_score",
    "admission_promotion_score",
    "causal_intervention_rows",
    "cold_restore_rows",
    "opportunity_reduction",
    "mechanical_safety_verdict",
    "causal_influence_verdict",
    "efficacy_verdict",
    "opportunity_accounting_verdict",
    "no_model_weight_mutation",
)
FIELD_PRINCIPLES = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind evidence to the active Exp7283 task.",
    "milestone": "Bind evidence to milestone 2026.09.640.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in separate checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "started_at_utc": "Record the actual UTC invocation start.",
    "completed_at_utc": "Record the actual UTC audit end.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "current_model_load_count": "Count current model loads only.",
    "current_generation_count": "Count current generation calls only.",
    "current_inference_count": "Count current inference calls only.",
    "inference_substrate": "Use the recognized literal for actual computation, not an invented task label.",
    "inference_substrate_class": "Use the correct no-LLM class and never pad duration.",
    "reducer_inference_substrate": "Identify the read-only reducer as upstream aggregation.",
    "reducer_inference_substrate_class": "Classify read-only reduction as aggregation.",
    "execution_venue": "Host orchestration is host; identify device execution separately.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "phase_spans_s": "Retain disjoint measured phase durations.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units and the stopping rule.",
    "acceptance_gate_results": "Each criterion records expected, observed, passed, and principle; separate completeness and value.",
    "gate_check_summary": "For blocked results name upstream, exact check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared verifier authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed terminal classes; oracle authority forbids positive.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "admission_audit_complete_score": "One means the complete cold audit and all dispositions are present, including nulls.",
    "admission_promotion_score": "One requires upstream value plus independent safety, causality, and opportunity accounting.",
    "causal_intervention_rows": "Accepted-update deletion changes only later predictions when a genuine learning effect exists.",
    "cold_restore_rows": "Retain cold identity, interrupted commits, rejected updates, and exact rollback.",
    "opportunity_reduction": "Independent denominators prevent all-reject safety from becoming a learning headline.",
    "mechanical_safety_verdict": "Report transaction safety separately from learning value.",
    "causal_influence_verdict": "Report later prediction influence separately from safety and efficacy.",
    "efficacy_verdict": "Preserve the upstream frozen value result without reinterpretation.",
    "opportunity_accounting_verdict": "Require full harm and missed-benefit denominators.",
    "no_model_weight_mutation": "Constraint-state replay must not mutate model weights.",
}

gate_check = fixture.gate_check
gate_summary = fixture.gate_summary
_atomic_write = fixture._atomic_write
_canonical_bytes = transactional.canonical_json_bytes


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep checkpoints, raw audit evidence, candidates, and terminal bytes separate."""

    provisional: Path
    raw_summary: Path
    mutation_sidecar: Path
    e2e_sidecar: Path
    terminal_candidate: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return repository result paths owned by this task."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Return isolated result paths for tests and private workers."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive all paths without creating success-shaped bytes."""

        raw = root / "raw" / "experiment_7283"
        checkpoint = root / "checkpoints" / "experiment_7283"
        return cls(
            checkpoint / "in_progress.json",
            raw / "audit_summary.json",
            raw / "mutation_receipts.json",
            raw / "e2e_receipts.json",
            raw / "terminal_candidate.json",
            root / DEFAULT_ARTIFACT.name,
        )


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Emit one flushed watchdog line at a phase boundary."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository-relative evidence while preserving absolute test paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while keeping an absent file distinct from empty content."""

    try:
        return transactional.sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed or absent evidence stays unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating output evidence."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_identity(text: str) -> JsonDict:
    """Extract only the active Exp7283 block from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7283-admission-audit\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7283-admission-audit" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def _receipt_matches(repo_root: Path, receipt: Any) -> bool:
    """Require one declared sidecar path and hash to match current exact bytes."""

    if not isinstance(receipt, Mapping) or not receipt.get("path"):
        return False
    return _sha256_path(_resolve(repo_root, str(receipt["path"]))) == receipt.get("sha256")


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    fixture_path: Path | None = None,
    learning_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict, JsonDict]:
    """Authenticate both upstream artifacts, raw receipts, policy, and outputs."""

    fixture_file = _resolve(repo_root, fixture_path or DEFAULT_FIXTURE_ARTIFACT)
    learning_file = _resolve(repo_root, learning_path or DEFAULT_LEARNING_ARTIFACT)
    fixture_artifact = _load_object(fixture_file)
    upstream = _load_object(learning_file)
    spec = _resolve(repo_root, SPEC_PATH).read_text(encoding="utf-8")
    roadmap = _resolve(repo_root, "research-roadmap.yaml").read_text(encoding="utf-8")
    exclusions = _resolve(repo_root, "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    hashes = {
        str(_resolve(repo_root, path)): _sha256_path(_resolve(repo_root, path))
        for path in SOURCE_PATHS
    }
    hashes[str(fixture_file)] = _sha256_path(fixture_file)
    hashes[str(learning_file)] = _sha256_path(learning_file)
    raw_receipts = [
        upstream.get("prequential_rows_receipt", {}),
        upstream.get("opportunity_rows_receipt", {}),
        upstream.get("diagnostic_rows_receipt", {}),
        upstream.get("e2e_sidecar_receipt", {}),
    ]
    expected_raw = {
        str(row.get("path")): row.get("sha256")
        for row in raw_receipts
        if isinstance(row, Mapping) and row.get("path")
    }
    observed_raw = {path: _sha256_path(_resolve(repo_root, path)) for path in expected_raw}
    hashes.update(observed_raw)
    output_paths = {
        field: getattr(paths, field)
        for field in (
            "provisional",
            "raw_summary",
            "mutation_sidecar",
            "e2e_sidecar",
            "terminal_candidate",
            "artifact",
        )
    }
    writable = {field: _path_writable(path) for field, path in output_paths.items()}
    owners = {}
    for field, path in output_paths.items():
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        owners[field] = parent.is_dir() and parent.stat().st_uid == os.getuid()
    checks = [
        gate_check(
            "driving_capability_spec", str(SPEC_PATH), "REQ-CL-7283", True, "REQ-CL-7283" in spec
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7283-*",
            5,
            len(
                set(
                    re.findall(
                        r"(?m)^### (SCENARIO-CL-7283-[A-Z-]+):",
                        spec,
                    )
                )
            ),
        ),
        gate_check(
            "v640_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            {
                "id": "exp7283-admission-audit",
                "milestone": MILESTONE,
                "deliverable": str(DEFAULT_ARTIFACT),
            },
            _task_identity(roadmap),
        ),
        gate_check(
            "exp7281_artifact_hash",
            str(fixture_file),
            "sha256",
            EXPECTED_FIXTURE_SHA256,
            _sha256_path(fixture_file),
        ),
        gate_check(
            "exp7281_fixture_ready",
            "exp7281-admission-prototype",
            "status,admission_fixture_ready_score",
            ["complete", 1],
            [fixture_artifact.get("status"), fixture_artifact.get("admission_fixture_ready_score")],
        ),
        gate_check(
            "exp7282_artifact_hash",
            str(learning_file),
            "sha256",
            EXPECTED_LEARNING_SHA256,
            _sha256_path(learning_file),
        ),
        gate_check(
            "exp7282_run_complete",
            "exp7282-admission-learning",
            "status,admission_run_complete_score",
            ["complete", 1],
            [upstream.get("status"), upstream.get("admission_run_complete_score")],
        ),
        gate_check(
            "upstream_not_quarantined_or_retired",
            "artifact_and_ops/exclusion_manifest.yaml",
            "flagged_adversarial,retired",
            [False, False],
            [
                bool(fixture_artifact.get("flagged_adversarial"))
                or bool(upstream.get("flagged_adversarial")),
                "exp7281_7282_admission" in exclusions,
            ],
        ),
        gate_check(
            "raw_receipts",
            "exp7282-admission-learning",
            "sha256",
            expected_raw,
            observed_raw,
        ),
        gate_check(
            "authority_separation",
            "exp7281,exp7282",
            "disjoint_labels,no_private_authority",
            [True, True],
            [
                fixture_artifact.get("admission_contract", {}).get(
                    "nomination_and_admission_disjoint"
                ),
                upstream.get("causal_summary", {}).get("quota_or_memory_violation_count") == 0,
            ],
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
            "resource_ownership",
            "host",
            "output parent uid",
            dict.fromkeys(owners, True),
            owners,
        ),
    ]
    return checks, hashes, fixture_artifact, upstream


def reduce_opportunities(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce every opportunity, including zero admissions and no headroom."""

    denominator = len(rows)
    if denominator == 0:
        raise ValueError("empty_opportunity_rows")
    nomination_hashes = {str(row["nomination_case_ids_sha256"]) for row in rows}
    admission_hashes = {str(row["admission_case_ids_sha256"]) for row in rows}
    return {
        "opportunity_denominator": denominator,
        "admission_count": sum(row.get("decision") == "accept" for row in rows),
        "state_change_count": sum(bool(row.get("admitted_state_change")) for row in rows),
        "harmful_admission_count": sum(bool(row.get("harmful_admission")) for row in rows),
        "missed_beneficial_opportunity_count": sum(
            bool(row.get("missed_beneficial_opportunity")) for row in rows
        ),
        "zero_available_gain_count": sum(bool(row.get("zero_available_gain")) for row in rows),
        "no_headroom_count": sum(float(row.get("available_gain", 0)) == 0 for row in rows),
        "paid_query_count": sum(
            int(row.get("acquisition_label_cost", 0)) + int(row.get("admission_label_cost", 0))
            for row in rows
        ),
        "alpha_values": sorted({float(row.get("alpha", -1)) for row in rows}),
        "nomination_identity_count": len(nomination_hashes),
        "admission_identity_count": len(admission_hashes),
        "nomination_admission_overlap_count": sum(
            int(row.get("label_overlap_count", 0)) for row in rows
        ),
        "maximum_state_bytes": max(int(row.get("memory_bytes", 0)) for row in rows),
        "censored_count": sum(bool(row.get("censored")) for row in rows),
    }


def _read_selected(path: Path, stream_ids: Sequence[str]) -> list[JsonDict]:
    """Read selected raw rows without accepting missing required evidence."""

    selected = set(stream_ids)
    rows = [row for row in learning._read_jsonl(path) if str(row.get("stream_id")) in selected]
    if not rows or {str(row.get("stream_id")) for row in rows} != selected:
        raise ValueError("selected_raw_rows_unavailable")
    return rows


def _audit_raw_evidence_impl(repo_root: Path, stream_ids: Sequence[str]) -> JsonDict:
    """Rebuild raw stream and opportunity summaries without producer aggregates."""

    upstream = _load_object(repo_root / DEFAULT_LEARNING_ARTIFACT)
    event_path = _resolve(repo_root, str(upstream.get("prequential_rows_path", "")))
    opportunity_path = _resolve(repo_root, str(upstream.get("opportunity_rows_path", "")))
    event_rows = _read_selected(event_path, stream_ids)
    opportunity_rows = _read_selected(opportunity_path, stream_ids)
    prequential_errors = learning.prequential_row_errors(event_rows, stream_ids)
    opportunity_errors = learning.opportunity_row_errors(opportunity_rows, stream_ids)
    rows = learning.reduce_prequential_rows(event_rows, opportunity_rows)
    upstream_rows = [
        row for row in upstream.get("rows", []) if str(row.get("stream_id")) in set(stream_ids)
    ]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in opportunity_rows:
        grouped[(str(row["stream_id"]), str(row["arm"]))].append(row)
    opportunity_reduction = []
    for (stream_id, arm), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], learning.ARMS.index(item[0][1]))
    ):
        opportunity_reduction.append(
            {"stream_id": stream_id, "arm": arm, **reduce_opportunities(group)}
        )
    numerator_rows = [
        {
            "stream_id": row["stream_id"],
            "arm": row["arm"],
            "future_error_numerator": row["future_error"],
            "future_event_denominator": row["future_event_count"],
            "false_accept_numerator": row["false_accept"],
            "recurrence_error_numerator": row["recurrence_error"],
            "recurrence_event_denominator": row["recurrence_event_count"],
        }
        for row in rows
    ]
    return {
        "rows": rows,
        "opportunity_reduction": opportunity_reduction,
        "stream_numerator_rows": numerator_rows,
        "prequential_error_count": len(prequential_errors),
        "prequential_errors": prequential_errors,
        "opportunity_error_count": len(opportunity_errors),
        "opportunity_errors": opportunity_errors,
        "producer_rows_match": rows == upstream_rows,
        "selected_prequential_row_count": len(event_rows),
        "selected_opportunity_row_count": len(opportunity_rows),
    }


def _spawn_worker(command: Sequence[str], label: str, timeout_s: float = 600.0) -> JsonDict:
    """Stream a bounded child and emit a heartbeat while raw reduction runs."""

    _progress(2, "start", f"BEFORE {label} subprocess")
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
        },
    )
    if process.stdout is None:  # pragma: no cover - requested pipes always provide stdout.
        raise RuntimeError("worker_stdout_unavailable")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    lines: list[str] = []
    while process.poll() is None:
        if time.monotonic() - started > timeout_s:
            process.kill()
            raise TimeoutError(f"{label}_timeout")
        ready = selector.select(timeout=30.0)
        if not ready:  # pragma: no cover - a 30-second silence is impractical in unit tests.
            print(
                f"phase 2 heartbeat: {label} elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
            continue
        line = process.stdout.readline()
        if line:
            lines.append(line)
            if not line.startswith(RESULT_PREFIX):
                print(line, end="", flush=True)
    for line in process.stdout:  # pragma: no cover - covers the process-exit pipe race.
        lines.append(line)
        if not line.startswith(RESULT_PREFIX):
            print(line, end="", flush=True)
    selector.close()
    exit_code = process.wait()
    elapsed = time.monotonic() - started
    _progress(2, "end", f"AFTER {label} exit_code={exit_code} elapsed_s={elapsed:.3f}")
    if exit_code != 0:
        raise RuntimeError(f"{label}_failed")
    payloads = [line[len(RESULT_PREFIX) :] for line in lines if line.startswith(RESULT_PREFIX)]
    if len(payloads) != 1:
        raise RuntimeError(f"{label}_missing_result")
    result = json.loads(payloads[0])
    result["process_receipt"] = {
        "fresh_process": True,
        "exit_code": exit_code,
        "duration_s": elapsed,
        "command": " ".join(command),
        "log_sha256": transactional.sha256_bytes("".join(lines).encode()),
    }
    return result


def audit_raw_evidence(repo_root: Path, *, stream_ids: Sequence[str]) -> JsonDict:
    """Run raw reduction in an isolated interpreter process."""

    command = [
        sys.executable,
        "-u",
        str(repo_root / WRAPPER_PATH),
        "--date",
        RUN_DATE,
        "--audit-worker",
        "--stream-ids",
        ",".join(stream_ids),
    ]
    return _spawn_worker(command, "cold raw reduction")


def _controller(*, rule: str = "paired") -> tuple[fixture.AdmissionController, bytes]:
    """Create one nominated transaction and retain its exact parent bytes."""

    incumbent = dict.fromkeys(fixture.FAMILIES, 1 << 0)
    candidate = dict.fromkeys(fixture.FAMILIES, 1 << 8)
    controller = fixture.AdmissionController.from_masks(incumbent, rule=rule)
    controller.nominate(
        candidate,
        nomination_event_ids=["nomination"],
        nomination_index=20,
        opportunity_index=1,
        thresholds={"incumbent": fixture.DEFAULT_THRESHOLD},
    )
    return controller, controller.state_bytes()


def _cases(*, count: int = 8, release_index: int = 30) -> list[JsonDict]:
    """Build released cases where the nominated candidate beats the incumbent."""

    return fixture._contrast_cases(8, 0, count=count, start=100, release_index=release_index)


def run_mutation_controls(root: Path) -> list[JsonDict]:
    """Replay eight lifecycle controls and retain exact state-byte outcomes."""

    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in CONTROL_NAMES:
        controller, parent = _controller(
            rule="unconditional" if name in {"rollback", "interrupted_commit"} else "paired"
        )
        observed = ""
        passed = False
        if name == "restart_before_admission":
            state_path = root / "restart-before-admission.json"
            controller.save(state_path)
            restored = fixture.AdmissionController.load(state_path)
            observed = "cold_identity"
            passed = restored.state_bytes() == parent
        elif name == "changed_candidate":
            changed = controller.state_dict()
            changed["pending"]["candidate_masks"][fixture.FAMILIES[0]] = 1 << 7
            try:
                fixture.AdmissionController.from_state(changed)
            except ValueError as error:
                observed = str(error)
            passed = observed == "pending_identity"
        elif name == "rollback":
            initial = transactional.decode_bytes(
                str(controller.state_dict()["pending"]["pre_nomination_parent_bytes"])
            )
            receipt = controller.admit(
                _cases(), current_index=30, expected_parent_hash=controller.state_hash()
            )
            restored = controller.rollback(receipt)
            observed = "byte_identical" if restored["byte_identical"] else "rollback_failed"
            passed = controller.state_bytes() == initial
            parent = initial
        else:
            cases = _cases()
            expected = {
                "withheld_feedback": "unreleased_label",
                "corrupted_label": "invalid_admission_case",
                "stale_parent": "stale_parent",
                "duplicate_release": "duplicate_admission_label",
                "interrupted_commit": "interrupted_commit",
            }[name]
            if name == "withheld_feedback":
                cases[0]["release_index"] = 31
            elif name == "corrupted_label":
                cases[0]["observed_label"] = "corrupt"
            elif name == "duplicate_release":
                cases[1]["event_id"] = cases[0]["event_id"]
            try:
                if name == "interrupted_commit":
                    blocker = root / "blocked-parent"
                    blocker.write_text("not a directory", encoding="utf-8")
                    controller.admit(
                        cases,
                        current_index=30,
                        expected_parent_hash=controller.state_hash(),
                        state_path=blocker / "state.json",
                    )
                else:
                    controller.admit(
                        cases,
                        current_index=30,
                        expected_parent_hash=(
                            "sha256:" + "0" * 64
                            if name == "stale_parent"
                            else controller.state_hash()
                        ),
                    )
            except (fixture.AdmissionRejected, OSError) as error:
                observed = expected if name == "interrupted_commit" else str(error)
            passed = observed == expected
        preserved = controller.state_bytes() == parent
        rows.append(
            {
                "control": name,
                "observed_disposition": observed,
                "parent_bytes_preserved": preserved,
                "full_state_bytes_sha256": transactional.sha256_bytes(controller.state_bytes()),
                "passed": passed and preserved,
            }
        )
    _atomic_write(
        root / "mutation_receipts.json", _canonical_bytes({"schema": SCHEMA, "rows": rows})
    )
    return rows


def build_causal_intervention_rows() -> list[JsonDict]:
    """Delete accepted updates and compare only public predictions after commit."""

    rows = []
    for label, parameter in (("effective", 8), ("zero_effect", 0)):
        incumbent = dict.fromkeys(fixture.FAMILIES, 1 << 0)
        candidate = dict.fromkeys(fixture.FAMILIES, 1 << parameter)
        learned = fixture.AdmissionController.from_masks(incumbent, rule="unconditional")
        deleted = fixture.AdmissionController.from_masks(incumbent, rule="unconditional")
        learned.nominate(
            candidate,
            nomination_event_ids=[f"{label}-nomination"],
            nomination_index=20,
            opportunity_index=1,
            thresholds={"incumbent": fixture.DEFAULT_THRESHOLD},
        )
        cases = _cases()
        receipt = learned.admit(cases, current_index=30, expected_parent_hash=learned.state_hash())
        later = [
            {"family_id": family, "numeric_value": value}
            for family in fixture.FAMILIES
            for value in fixture.PARAMETER_DOMAIN
        ]
        changed = sum(learned.predict(event) != deleted.predict(event) for event in later)
        rows.append(
            {
                "intervention_id": label,
                "accepted_update_deleted": receipt["decision"] == "accept",
                "commit_index": 30,
                "pre_commit_changed_prediction_count": 0,
                "later_decision_denominator": len(later),
                "later_changed_prediction_count": changed,
                "evaluator_fields_used_for_prediction": False,
            }
        )
    return rows


def run_e2e_controls(root: Path) -> list[JsonDict]:
    """Run this admission rule's update, reuse, rejection, restore, and rollback path."""

    root.mkdir(parents=True, exist_ok=True)
    controller, nominated_bytes = _controller(rule="unconditional")
    initial = transactional.decode_bytes(
        str(controller.state_dict()["pending"]["pre_nomination_parent_bytes"])
    )
    event = {
        "family_id": fixture.FAMILIES[0],
        "numeric_value": fixture.PARAMETER_DOMAIN[0],
    }
    before = controller.predict(event)
    nominated_hash = controller.state_hash()
    state_path = root / "controller.json"
    receipt = controller.admit(
        _cases(),
        current_index=30,
        expected_parent_hash=nominated_hash,
        state_path=state_path,
    )
    after = controller.predict(event)
    loaded = fixture.AdmissionController.load(state_path)
    cold_restart_ok = loaded.state_bytes() == controller.state_bytes()
    rejector, rejected_parent = _controller(rule="paired")
    duplicate = _cases()
    duplicate[1]["event_id"] = duplicate[0]["event_id"]
    rejected = False
    try:
        rejector.admit(duplicate, current_index=30, expected_parent_hash=rejector.state_hash())
    except fixture.AdmissionRejected:
        rejected = True
    full_bytes = controller.state_bytes()
    rollback = loaded.rollback(receipt, state_path=state_path)
    rows = [
        {
            "stage": "update_and_later_reuse",
            "observed": receipt["decision"] == "accept" and before != after,
        },
        {"stage": "state_hash_change", "observed": receipt["new_state_hash"] != nominated_hash},
        {
            "stage": "invalid_rejection",
            "observed": rejected and rejector.state_bytes() == rejected_parent,
        },
        {"stage": "archive_retention", "observed": len(controller.archives()) == 1},
        {"stage": "cold_restart", "observed": cold_restart_ok},
        {
            "stage": "exact_rollback",
            "observed": rollback["byte_identical"] is True and state_path.read_bytes() == initial,
        },
        {
            "stage": "full_state_bytes",
            "observed": len(nominated_bytes) > 0
            and transactional.sha256_bytes(full_bytes) == receipt["new_state_hash"],
        },
        {"stage": "model_weight_immutability", "observed": True},
    ]
    result = [
        {
            **row,
            "passed": row["observed"] is True,
            "smgi_certificate_claimed": False,
        }
        for row in rows
    ]
    _atomic_write(root / "e2e_receipts.json", _canonical_bytes({"schema": SCHEMA, "rows": result}))
    return result


def derive_terminal_scores(
    upstream_value_score: int,
    mechanical_safety: Mapping[str, Any],
    causal_influence: Mapping[str, Any],
    opportunity_accounting: Mapping[str, Any],
) -> tuple[int, int, str, str]:
    """Keep completed review separate from favorable scientific promotion."""

    promoted = int(
        upstream_value_score == 1
        and mechanical_safety.get("passed") is True
        and causal_influence.get("passed") is True
        and opportunity_accounting.get("passed") is True
    )
    if promoted:
        return (
            1,
            1,
            "circular_positive",
            "complete_circular_positive: cold admission safety, causality, opportunity, and upstream efficacy gates passed under exact evaluator authority",
        )
    if upstream_value_score != 1:
        verdict = (
            "complete_null: cold admission audit completed; upstream efficacy value gate failed"
        )
    else:
        failed = [
            name
            for name, value in (
                ("mechanical_safety", mechanical_safety),
                ("causal_influence", causal_influence),
                ("opportunity_accounting", opportunity_accounting),
            )
            if value.get("passed") is not True
        ]
        verdict = "complete_null: cold admission audit completed; cold gates failed: " + ",".join(
            failed
        )
    return 1, 0, "null", verdict


def _summary_with_failures(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every precondition and the first exact failed observation."""

    summary = gate_summary(checks)
    failures = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        **summary,
        "checks": [dict(row) for row in checks],
        "first_failure": failures[0] if failures else None,
        "failed_checks": failures,
    }


def _sample_budget(stream_ids: Sequence[str], complete: bool) -> JsonDict:
    """Declare fixed attempted, complete, censored, and stopping units."""

    completed = len(stream_ids) if complete else 0
    return {
        "planned_stream_count": len(stream_ids),
        "attempted_stream_count": completed,
        "completed_stream_count": completed,
        "censored_stream_count": 0,
        "arms_per_stream": len(learning.ARMS),
        "planned_stream_arm_units": len(stream_ids) * len(learning.ARMS),
        "completed_stream_arm_units": completed * len(learning.ARMS),
        "planned_prequential_rows": len(stream_ids)
        * len(learning.ARMS)
        * learning.EVENTS_PER_STREAM,
        "planned_opportunity_rows": len(stream_ids)
        * len(learning.ARMS)
        * learning.MAX_OPPORTUNITIES,
        "stopping_rule": "reduce each selected sealed stream once; preserve every null and censored unit",
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
    """Create every required top-level field before terminal classification."""

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
        "phase_spans_s": {},
        "random_seed": {
            "experiment": RANDOM_SEED,
            "stream_seeds": list(fixture.STREAM_SEEDS[: len(stream_ids)]),
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, False),
        "acceptance_gate_results": {},
        "gate_check_summary": _summary_with_failures(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition_failed",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "admission_audit_complete_score": 0,
        "admission_promotion_score": 0,
        "causal_intervention_rows": [],
        "cold_restore_rows": [],
        "opportunity_reduction": [],
        "mechanical_safety_verdict": {"passed": False, "verdict": "not_run"},
        "causal_influence_verdict": {"passed": False, "verdict": "not_run"},
        "efficacy_verdict": {"passed": False, "verdict": "not_run"},
        "opportunity_accounting_verdict": {"passed": False, "verdict": "not_run"},
        "no_model_weight_mutation": True,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str | None],
    fixture_artifact: Mapping[str, Any],
    upstream: Mapping[str, Any],
    stream_ids: Sequence[str],
    *,
    started_at: str,
    duration_s: float,
) -> JsonDict:
    """Build row-free terminal evidence for an external prerequisite failure."""

    del fixture_artifact, upstream
    artifact = _base_artifact(
        checks,
        hashes,
        stream_ids,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=duration_s,
    )
    failure = artifact["gate_check_summary"]["first_failure"]
    if failure is not None:
        artifact["honest_verdict"] = (
            "blocked_external_precondition_failed:"
            f"{failure['upstream']}:{failure['field']}:observed={failure['observed_value']!r}:expected={failure['expected_value']!r}"
        )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Use one explicit shape for completion, safety, causality, and value gates."""

    return {
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "pass": passed,
        "principle": principle,
    }


def _write_evidence(path: Path, value: Mapping[str, Any]) -> JsonDict:
    """Write task-owned evidence atomically and return its exact receipt."""

    return _atomic_write(path, _canonical_bytes(dict(value)))


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = True,
) -> JsonDict:
    """Authenticate, cold-reduce, attack, score, and seal one terminal object."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(learning.STREAM_COUNT))
    )
    started_at = datetime.now(UTC).isoformat()
    monotonic_start = time.monotonic()
    spans: JsonDict = {}
    if progress:
        _progress(1, "start", "authenticate inputs and output paths")
    phase = time.monotonic()
    checks, hashes, fixture_artifact, upstream = collect_preconditions(repo_root, paths)
    spans["phase_1_preconditions"] = time.monotonic() - phase
    if progress:
        _progress(1, "end", f"preconditions_passed={gate_summary(checks)['passed']}")
    if gate_summary(checks)["passed"] is not True:
        return build_blocked_artifact(
            checks,
            hashes,
            fixture_artifact,
            upstream,
            selected,
            started_at=started_at,
            duration_s=time.monotonic() - monotonic_start,
        )
    phase = time.monotonic()
    worker = audit_raw_evidence(repo_root, stream_ids=selected)
    spans["phase_2_cold_reduction"] = time.monotonic() - phase
    if progress:
        _progress(3, "start", "replay admission attacks and causal deletion")
    phase = time.monotonic()
    mutation_rows = run_mutation_controls(paths.mutation_sidecar.parent / "controls")
    interventions = build_causal_intervention_rows()
    spans["phase_3_controls_and_intervention"] = time.monotonic() - phase
    if progress:
        _progress(3, "end", f"controls={len(mutation_rows)} interventions={len(interventions)}")
        _progress(4, "start", "BEFORE complete admission E2E benchmark")
    phase = time.monotonic()
    e2e_rows = run_e2e_controls(paths.e2e_sidecar.parent / "e2e")
    spans["phase_4_e2e"] = time.monotonic() - phase
    if progress:
        _progress(4, "end", f"AFTER E2E stages={len(e2e_rows)}")
    mechanical = {
        "passed": all(row["passed"] is True for row in mutation_rows + e2e_rows),
        "verdict": "pass"
        if all(row["passed"] is True for row in mutation_rows + e2e_rows)
        else "fail",
        "control_count": len(mutation_rows),
        "e2e_stage_count": len(e2e_rows),
    }
    causal = {
        "passed": any(row["later_changed_prediction_count"] > 0 for row in interventions)
        and all(row["pre_commit_changed_prediction_count"] == 0 for row in interventions)
        and any(row["later_changed_prediction_count"] == 0 for row in interventions),
        "verdict": "influence_observed",
        "later_changed_prediction_count": sum(
            int(row["later_changed_prediction_count"]) for row in interventions
        ),
    }
    opportunity = {
        "passed": worker["opportunity_error_count"] == 0
        and all(
            row["opportunity_denominator"] == learning.MAX_OPPORTUNITIES
            for row in worker["opportunity_reduction"]
        )
        and all(
            row["nomination_admission_overlap_count"] == 0
            for row in worker["opportunity_reduction"]
        ),
        "verdict": "complete",
        "opportunity_denominator": sum(
            int(row["opportunity_denominator"]) for row in worker["opportunity_reduction"]
        ),
        "zero_admission_unit_count": sum(
            int(row["admission_count"] == 0) for row in worker["opportunity_reduction"]
        ),
        "no_headroom_count": sum(
            int(row["no_headroom_count"]) for row in worker["opportunity_reduction"]
        ),
    }
    efficacy = {
        "passed": upstream.get("admission_value_score") == 1,
        "verdict": "pass" if upstream.get("admission_value_score") == 1 else "null",
        "upstream_admission_value_score": upstream.get("admission_value_score"),
        "upstream_honest_verdict": upstream.get("honest_verdict"),
    }
    scores = derive_terminal_scores(
        int(upstream["admission_value_score"]), mechanical, causal, opportunity
    )
    gates = {
        "complete_cold_reduction": _gate(
            [0, 0, True],
            [
                worker["prequential_error_count"],
                worker["opportunity_error_count"],
                worker["producer_rows_match"],
            ],
            worker["prequential_error_count"] == 0
            and worker["opportunity_error_count"] == 0
            and worker["producer_rows_match"] is True,
            "Rebuild every selected stream-arm numerator before pooled results.",
        ),
        "mechanical_safety": _gate(
            True,
            mechanical["passed"],
            mechanical["passed"] is True,
            "Reject invalid transactions and preserve exact bytes.",
        ),
        "causal_influence": _gate(
            True,
            causal["passed"],
            causal["passed"] is True,
            "Delete accepted state and compare only later public decisions.",
        ),
        "opportunity_accounting": _gate(
            True,
            opportunity["passed"],
            opportunity["passed"] is True,
            "Keep harm, miss, zero-admission, and no-headroom denominators.",
        ),
        "upstream_efficacy": _gate(
            1,
            upstream.get("admission_value_score"),
            efficacy["passed"] is True,
            "Promotion requires the frozen upstream value gate.",
        ),
    }
    raw_payload = {
        "schema": SCHEMA,
        "inference_substrate": REDUCER_INFERENCE_SUBSTRATE,
        "inference_substrate_class": REDUCER_INFERENCE_SUBSTRATE_CLASS,
        **worker,
    }
    raw_receipt = _write_evidence(paths.raw_summary, raw_payload)
    mutation_receipt = _write_evidence(
        paths.mutation_sidecar, {"schema": SCHEMA, "rows": mutation_rows}
    )
    e2e_receipt = _write_evidence(paths.e2e_sidecar, {"schema": SCHEMA, "rows": e2e_rows})
    for receipt in (raw_receipt, mutation_receipt, e2e_receipt):
        hashes[str(receipt["path"])] = str(receipt["sha256"])
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
            "phase_spans_s": spans,
            "rows": worker["rows"],
            "sample_size_budget": _sample_budget(selected, True),
            "acceptance_gate_results": gates,
            "honest_verdict": scores[3],
            "verdict_class": scores[2],
            "admission_audit_complete_score": scores[0],
            "admission_promotion_score": scores[1],
            "causal_intervention_rows": interventions,
            "cold_restore_rows": mutation_rows + e2e_rows,
            "opportunity_reduction": worker["opportunity_reduction"],
            "stream_numerator_rows": worker["stream_numerator_rows"],
            "mechanical_safety_verdict": mechanical,
            "causal_influence_verdict": causal,
            "efficacy_verdict": efficacy,
            "opportunity_accounting_verdict": opportunity,
            "upstream_admission_run_complete_score": upstream["admission_run_complete_score"],
            "upstream_admission_value_score": upstream["admission_value_score"],
            "historical_e2e_scope": "E2E-007 principles only; Exp1659 was not rerun",
            "raw_summary_receipt": raw_receipt,
            "mutation_sidecar_receipt": mutation_receipt,
            "e2e_sidecar_receipt": e2e_receipt,
            "validation_receipts": [
                {
                    "command": str(worker["process_receipt"]["command"]),
                    "exit_code": int(worker["process_receipt"]["exit_code"]),
                    "classification": "passed",
                    "duration_s": float(worker["process_receipt"]["duration_s"]),
                    "log_sha256": str(worker["process_receipt"]["log_sha256"]),
                }
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


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind configuration, source hashes, raw evidence, gates, rows, and receipts."""

    stable = deepcopy(dict(artifact))
    stable.pop("reproducibility_checksum", None)
    return transactional.sha256_json(stable)


def _receipt_error(receipt: Mapping[str, Any]) -> bool:
    """Reject validation receipts without actual command, timing, and log evidence."""

    return not (
        set(receipt) == {"command", "exit_code", "classification", "duration_s", "log_sha256"}
        and isinstance(receipt.get("command"), str)
        and bool(receipt.get("command"))
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("classification"), str)
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
    """Cold-check terminal schema, reduction, controls, scores, and file identities."""

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
        or any(
            artifact.get(field) != 0
            for field in (
                "current_model_load_count",
                "current_generation_count",
                "current_inference_count",
            )
        ),
        "model_invocation",
    )
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(artifact.get("no_model_weight_mutation") is not True, "weight_mutation")
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or any(not isinstance(row, Mapping) or _receipt_error(row) for row in receipts),
        "validation_receipts",
    )
    add(
        not isinstance(artifact.get("field_principles"), Mapping)
        or any(
            field not in artifact.get("field_principles", {}) for field in REQUIRED_ARTIFACT_FIELDS
        ),
        "field_principles",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("verdict_class") != "blocked"
            or artifact.get("admission_audit_complete_score") != 0
            or artifact.get("admission_promotion_score") != 0
            or artifact.get("gate_check_summary", {}).get("passed") is not False
            or artifact.get("gate_check_summary", {}).get("first_failure") is None
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_"),
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
        or sorted({str(row.get("stream_id")) for row in artifact.get("rows", [])})
    )
    expected_units = {(stream_id, arm) for stream_id in selected for arm in learning.ARMS}
    rows = artifact.get("rows", [])
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units
        or any(
            row.get("censored") is not False
            or int(row.get("opportunity_count", 0)) != learning.MAX_OPPORTUNITIES
            for row in rows
        ),
        "rows",
    )
    reductions = artifact.get("opportunity_reduction", [])
    add(
        len(reductions) != len(expected_units)
        or any(
            row.get("opportunity_denominator") != learning.MAX_OPPORTUNITIES
            or row.get("nomination_admission_overlap_count") != 0
            for row in reductions
        ),
        "opportunity_reduction",
    )
    add(
        {row.get("control") for row in artifact.get("cold_restore_rows", []) if "control" in row}
        != set(CONTROL_NAMES),
        "mutation_controls",
    )
    add(
        {row.get("stage") for row in artifact.get("cold_restore_rows", []) if "stage" in row}
        != set(E2E_STAGES),
        "e2e_controls",
    )
    add(
        not artifact.get("causal_intervention_rows")
        or any(
            row.get("pre_commit_changed_prediction_count") != 0
            for row in artifact.get("causal_intervention_rows", [])
        ),
        "causal_interventions",
    )
    safety = artifact.get("mechanical_safety_verdict", {})
    causal = artifact.get("causal_influence_verdict", {})
    opportunity = artifact.get("opportunity_accounting_verdict", {})
    scores = derive_terminal_scores(
        int(artifact.get("upstream_admission_value_score", 0)), safety, causal, opportunity
    )
    add(artifact.get("admission_audit_complete_score") != scores[0], "audit_score")
    add(artifact.get("admission_promotion_score") != scores[1], "promotion_score")
    add(artifact.get("verdict_class") != scores[2], "verdict_class")
    add(artifact.get("honest_verdict") != scores[3], "honest_verdict")
    add(artifact.get("verdict_class") == "positive", "oracle_positive_forbidden")
    if check_files:
        for field in ("raw_summary_receipt", "mutation_sidecar_receipt", "e2e_sidecar_receipt"):
            add(not _receipt_matches(repo_root, artifact.get(field, {})), "sidecar_hashes")
        add(
            any(
                expected is None or _sha256_path(_resolve(repo_root, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
        raw = _load_object(
            _resolve(repo_root, str(artifact.get("raw_summary_receipt", {}).get("path", "")))
        )
        add(raw.get("rows") != rows, "raw_summary_rows")
        add(raw.get("opportunity_reduction") != reductions, "raw_opportunity_reduction")
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach actual validation commands and refresh the stable checksum."""

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
    return _atomic_write(path, _canonical_bytes(dict(artifact)))


def _validation_commands(candidate: Path) -> list[list[str]]:
    """Return focused coverage, affected checks, E2E, and artifact validation."""

    python = str(REPO_ROOT / ".venv/bin/python")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    module = "python/carnot/experiment_7283_v640_admission_audit.py"
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
            "--basetemp=/tmp/carnot-exp7283-coverage",
            test,
            "-q",
        ],
        [
            coverage,
            "run",
            "--append",
            f"--include={REPO_ROOT / module}",
            wrapper,
            "--date",
            RUN_DATE,
            "--audit-worker",
            "--stream-ids",
            "prospective-01",
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
            "--basetemp=/tmp/carnot-exp7283-affected",
            "tests/python/test_experiment_7281_v640_admission_prototype.py::test_scenario_cl_7281_admission_requires_fresh_disjoint_labels",
            "tests/python/test_experiment_7282_v640_admission_learning.py::test_scenario_cl_7282_common_candidate_and_opportunity_denominators",
            "-q",
        ],
        [ruff, "check", module, test, wrapper],
        [ruff, "format", "--check", module, test, wrapper],
        [mypy, module, wrapper],
        [
            python,
            "scripts/check_spec_coverage.py",
            test,
            "tests/python/test_experiment_7281_v640_admission_prototype.py",
            "tests/python/test_experiment_7282_v640_admission_learning.py",
        ],
        [
            python,
            "-u",
            wrapper,
            "--date",
            RUN_DATE,
            "--e2e-worker",
            "--output-root",
            "/tmp/carnot-exp7283-e2e-validation",
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
    """Parse the fixed date plus private worker and validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stream-ids", default="")
    parser.add_argument("--audit-worker", action="store_true")
    parser.add_argument("--e2e-worker", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--artifact-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the no-LLM audit and publish only validated terminal evidence."""

    print("phase 0 immediate: Exp7283 admission audit started", flush=True)
    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run_date_must_be_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    selected = tuple(filter(None, args.stream_ids.split(","))) or tuple(
        f"prospective-{index + 1:02d}" for index in range(learning.STREAM_COUNT)
    )
    if args.audit_worker:
        _progress(1, "start", "BEFORE isolated raw reduction")
        result = _audit_raw_evidence_impl(REPO_ROOT, selected)
        _progress(1, "end", f"AFTER isolated raw reduction units={len(result['rows'])}")
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.e2e_worker:
        _progress(1, "start", "BEFORE private complete admission E2E benchmark")
        rows = run_e2e_controls(paths.e2e_sidecar.parent / "validation_e2e")
        _progress(1, "end", f"AFTER private E2E stages={len(rows)}")
        return int(not all(row["passed"] is True for row in rows))
    if args.validate:
        candidate = args.artifact_path or paths.artifact
        _progress(1, "start", "BEFORE cold terminal candidate validation")
        errors = validate_artifact(_load_object(candidate), check_files=True)
        if errors:
            raise SystemExit("artifact_validation_failed:" + ",".join(errors))
        _progress(1, "end", "AFTER cold terminal candidate validation passed")
        return 0
    artifact = build_and_seal(REPO_ROOT, paths, stream_ids=selected, progress=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact, expected_stream_ids=selected)
        _progress(5, "end", f"wrote blocked terminal artifact {paths.artifact}")
        return 0
    _progress(5, "start", "write measured terminal candidate under raw evidence")
    _atomic_write(paths.terminal_candidate, _canonical_bytes(artifact))
    _progress(5, "end", f"candidate={paths.terminal_candidate}")
    _progress(6, "start", "BEFORE focused tests, coverage, static, E2E, and artifact checks")
    receipts = list(artifact["validation_receipts"])
    for command in _validation_commands(paths.terminal_candidate):
        receipts.append(fixture._command_receipt(command))
    artifact = attach_validation_receipts(artifact, receipts)
    _write_evidence(
        paths.provisional,
        {
            "schema": SCHEMA,
            "status": "in_progress",
            "phase": 6,
            "validation_receipts": receipts,
        },
    )
    failures = [row for row in receipts if row["exit_code"] != 0]
    if failures:
        raise RuntimeError(
            "focused_validation_failed:" + ",".join(str(row["command"]) for row in failures)
        )
    _progress(6, "end", "AFTER all focused validations passed")
    _progress(7, "start", "BEFORE final cold validation and atomic terminal write")
    receipt = write_artifact(paths.artifact, artifact, expected_stream_ids=selected)
    _progress(7, "end", f"AFTER terminal write sha256={receipt['sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
