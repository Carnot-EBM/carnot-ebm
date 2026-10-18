"""Measure host delayed-acceptance cost while retaining board dispositions.

This experiment runs no board command and invokes no language model. It uses
the fixed graph generator from Exp7187 and the corrected transition law from
Exp7188. Device and transfer timings in the break-even table are hypotheses,
not measurements.

Spec refs: REQ-ISING-7203 and SCENARIO-ISING-7203-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot import experiment_7188_v633_quantized_transition_audit as correction


JsonDict = dict[str, Any]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from conductor_gates import _is_quarantined as repository_is_quarantined  # noqa: E402


RUN_DATE = "20260911"
TASK_ID = "exp7203-hardware-correction"
MILESTONE = "2026.09.634"
RESULT_PATH = Path("results/experiment_7203_v634_hardware_correction.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7203_v634_hardware_correction.json")
SPEC_PATH = Path("openspec/capabilities/ising-backend/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
UPSTREAM_CORRECTION_PATH = Path("results/experiment_7188_v633_quantized_transition_audit.json")
UPSTREAM_BOARD_PATH = Path("results/experiment_7190_v633_board_placement_receipt.json")

SIZES = (16, 32, 64)
SEEDS = tuple(range(7_203_001, 7_203_011))
BITS = (4, 8, 16)
K = 2
BETA = 1.0
EQUAL_WORK = "equal_work"
EQUAL_WALL = "equal_wall"
BUDGET_KINDS = (EQUAL_WORK, EQUAL_WALL)
FULL_ARM = correction.FULL_ARM
CORRECTED_ARM = correction.CORRECTED_ARM
ARMS = (FULL_ARM, CORRECTED_ARM)
EQUAL_WORK_PROPOSALS = 1_024
EQUAL_WALL_SECONDS = 0.100
HOST_MEASUREMENT_CAP_SECONDS = 900.0
LAW_TOLERANCE = 1.0e-10
DEVICE_LATENCY_NS = (10, 100, 1_000, 10_000)
TRANSFER_LATENCY_NS = (100, 1_000, 10_000, 100_000)
HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "Board continuity and quantized-correction cost envelope",
    "track": "hardware",
    "requires_gpu": False,
    "per_unit_rows": True,
    "milestone": MILESTONE,
    "deliverable": str(RESULT_PATH),
    "gated_on": [],
    "prior_failures": [
        {
            "experiment_id": "exp7146-gatemate-changed-state-continuity",
            "verdict": "blocked_no_new_operator_physical_state_receipt_after_exp6559",
            "addressed_by": (
                "Keep a read-only inherited board disposition and perform a new CPU "
                "correction-cost envelope; no repeated physical probe."
            ),
            "retire_if_same_verdict": True,
        }
    ],
    "operator_override": (
        "2026-05-29 operator directive (standing): active hardware continuity; scope "
        "overlap with exp7146 is a read-only disposition. New work measures corrected "
        "host costs without retrying unchanged JTAG."
    ),
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/known-issues.md"),
    Path("research-hardware-wishlist.md"),
    ROADMAP_PATH,
    SPEC_PATH,
    UPSTREAM_CORRECTION_PATH,
    UPSTREAM_BOARD_PATH,
    Path("results/experiment_6559_gatemate_changed_state_continuity.json"),
    Path("results/experiment_7146_v627_gatemate_changed_state.json"),
    Path("results/experiment_5861_attached_board_state_receipts.json"),
    Path("results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"),
    Path("results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"),
    Path("results/experiment_3867_polarfire_soc_smoke_v4.json"),
    Path("python/carnot/analysis/pbit_sampler_portability.py"),
    Path("python/carnot/experiment_7187_v633_slice_sampler.py"),
    Path("python/carnot/experiment_7188_v633_quantized_transition_audit.py"),
    Path("python/carnot/experiment_7203_v634_hardware_correction.py"),
    Path("scripts/conductor_gates.py"),
    Path("scripts/experiments/experiment_7203_v634_hardware_correction.py"),
    Path("tests/python/test_experiment_7203_v634_hardware_correction.py"),
)

FIELD_PRINCIPLES = {
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
    "sample_size_budget": (
        "Record planned and completed counts, independent units and exclusions."
    ),
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
    "hardware_envelope_complete_score": (
        "Complete visibility and host measurements do not imply board readiness."
    ),
    "board_rows": "Each board retains its own evidence and next prerequisite.",
    "operator_state_receipt": (
        "A changed physical state must be backed by operator-authored evidence."
    ),
    "correction_cost_rows": ("Quantization comparisons retain full-target correction and cost."),
    "break_even_rows": "Separate measured host costs from hypothetical device inputs.",
    "hardware_execution_claimed": (
        "False: this task measures CPU correction and reads board receipts."
    ),
    "topology_fit": "Unknown unless an explicit fixed-graph mapping exists.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
}

REQUIRED_ARTIFACT_FIELDS = set(FIELD_PRINCIPLES) | {
    "task_id",
    "milestone",
    "spec_refs",
    "host_identity",
    "exact_law_rows",
    "device_timing_available",
    "hardware_command_count",
    "hardware_operations_issued",
    "power_savings_claimed",
    "tsu_execution_claimed",
    "fpga_execution_claimed",
    "soft_spin_execution_claimed",
    "equal_effective_sample_throughput_established",
    "mixing_speed_established",
}


def progress(phase: int, boundary: str, operation: str) -> None:
    """Flush each boundary so the experiment cannot look stalled."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Encode stable finite JSON for hashes and compact progress receipts."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as exc:
        raise ValueError("nonfinite value in canonical JSON") from exc


def sha256_json(value: Any) -> str:
    """Return one tagged digest for a stable JSON value."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact source bytes instead of accepting a path as provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind the artifact while excluding only the field that stores its digest."""

    material = dict(payload)
    material["reproducibility_checksum"] = ""
    return sha256_json(material)


def _read_json(path: Path) -> JsonDict:
    """Read one required JSON object and reject any other top-level shape."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _task_contract(root: Path) -> JsonDict | None:
    """Read only the exact V634 fields that define this independent task."""

    path = root / ROADMAP_PATH
    if not path.is_file():
        return None
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        return None
    task = next(
        (
            item
            for item in document["tasks"]
            if isinstance(item, Mapping) and item.get("id") == TASK_ID
        ),
        None,
    )
    if task is None:
        return None
    keys = tuple(EXPECTED_TASK_CONTRACT)
    return {key: task.get(key, [] if key == "gated_on" else None) for key in keys}


def upstream_gate(payload: Mapping[str, Any], upstream: str, field: str, expected: Any) -> JsonDict:
    """Apply repository quarantine authority before trusting a readiness field."""

    quarantined = bool(repository_is_quarantined(dict(payload)))
    actual = payload.get(field)
    return {
        "check": f"{Path(upstream).stem}_not_quarantined",
        "upstream": upstream,
        "field": field,
        "expected_value": {"value": expected, "quarantined": False},
        "observed_value": {"value": actual, "quarantined": quarantined},
        "passed": not quarantined and actual == expected,
    }


def _check(
    name: str, upstream: str, field: str, expected: Any, observed: Any, passed: bool
) -> JsonDict:
    """Create one complete gate row for terminal failure diagnostics."""

    return {
        "check": name,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(passed),
    }


def collect_preconditions(
    root: Path, *, output_path: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], dict[str, str]]:
    """Check bytes, contracts, tools, storage, readiness, and quarantine first."""

    checks: list[JsonDict] = []
    progress(0, "check", "required source byte counts")
    sizes = {
        path.as_posix(): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    checks.append(
        _check(
            "required_source_bytes",
            "repository",
            "byte_count",
            "every required source is nonempty",
            sizes,
            all(size is not None and size > 0 for size in sizes.values()),
        )
    )

    progress(0, "check", "driving capability requirement")
    spec_path = root / SPEC_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    spec_observed = {"exists": spec_path.is_file(), "req_present": "REQ-ISING-7203" in spec_text}
    checks.append(
        _check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ISING-7203",
            {"exists": True, "req_present": True},
            spec_observed,
            all(spec_observed.values()),
        )
    )

    progress(0, "check", "exact V634 roadmap task")
    task = _task_contract(root)
    checks.append(
        _check(
            "same_milestone_gate_fields",
            ROADMAP_PATH.as_posix(),
            "task_contract",
            EXPECTED_TASK_CONTRACT,
            task,
            task == EXPECTED_TASK_CONTRACT,
        )
    )

    progress(0, "check", "Python NumPy and YAML tools")
    tool_observed = {
        "python": bool(sys.executable),
        "python_version": platform.python_version(),
        "numpy": bool(np.__version__),
        "numpy_version": np.__version__,
        "pyyaml": bool(yaml.__version__),
        "pyyaml_version": yaml.__version__,
    }
    checks.append(
        _check(
            "required_tools",
            "host_python_environment",
            "python_numpy_pyyaml",
            {"python": True, "numpy": True, "pyyaml": True},
            tool_observed,
            bool(sys.executable and np.__version__ and yaml.__version__),
        )
    )

    progress(0, "check", "result and checkpoint output directories")
    output_observed = {
        "result_parent": output_path.parent.is_dir() and os.access(output_path.parent, os.W_OK),
        "checkpoint_parent": checkpoint_path.parent.is_dir()
        and os.access(checkpoint_path.parent, os.W_OK),
        "checkpoint_under_results_checkpoints": (
            checkpoint_path.parent.resolve() == (root / "results" / "checkpoints").resolve()
            or root.resolve() not in checkpoint_path.resolve().parents
        ),
    }
    checks.append(
        _check(
            "output_directories",
            "filesystem",
            "result_and_checkpoint_parent",
            {
                "result_parent": True,
                "checkpoint_parent": True,
                "checkpoint_under_results_checkpoints": True,
            },
            output_observed,
            all(output_observed.values()),
        )
    )

    progress(0, "check", "source hashes")
    hashes = {
        path.as_posix(): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if sizes[path.as_posix()] is not None and sizes[path.as_posix()] > 0
    }
    checks.append(
        _check(
            "source_artifact_hashes",
            "required_source_bytes",
            "sha256",
            len(REQUIRED_SOURCE_PATHS),
            len(hashes),
            len(hashes) == len(REQUIRED_SOURCE_PATHS)
            and all(HASH_PATTERN.fullmatch(value) for value in hashes.values()),
        )
    )

    progress(0, "check", "Exp7188 quarantine before readiness")
    try:
        correction_artifact = _read_json(root / UPSTREAM_CORRECTION_PATH)
    except (OSError, ValueError, json.JSONDecodeError):
        correction_artifact = {}
    correction_quarantine = upstream_gate(
        correction_artifact,
        UPSTREAM_CORRECTION_PATH.as_posix(),
        "corrected_kernel_ready_score",
        1,
    )
    correction_quarantine["check"] = "exp7188_not_quarantined"
    checks.append(correction_quarantine)

    progress(0, "check", "Exp7188 exact readiness fields")
    correction_fields = {
        "status": correction_artifact.get("status"),
        "quantized_audit_complete_score": correction_artifact.get("quantized_audit_complete_score"),
        "corrected_kernel_ready_score": correction_artifact.get("corrected_kernel_ready_score"),
    }
    expected_correction_fields = {
        "status": "complete",
        "quantized_audit_complete_score": 1,
        "corrected_kernel_ready_score": 1,
    }
    checks.append(
        _check(
            "exp7188_exact_gate_fields",
            UPSTREAM_CORRECTION_PATH.as_posix(),
            "status_and_ready_scores",
            expected_correction_fields,
            correction_fields if correction_quarantine["passed"] else "not_consumed_quarantined",
            correction_quarantine["passed"] and correction_fields == expected_correction_fields,
        )
    )

    progress(0, "check", "Exp7190 quarantine before readiness")
    try:
        board_artifact = _read_json(root / UPSTREAM_BOARD_PATH)
    except (OSError, ValueError, json.JSONDecodeError):
        board_artifact = {}
    board_quarantine = upstream_gate(
        board_artifact,
        UPSTREAM_BOARD_PATH.as_posix(),
        "board_placement_receipt_complete_score",
        1,
    )
    board_quarantine["check"] = "exp7190_not_quarantined"
    checks.append(board_quarantine)

    progress(0, "check", "Exp7190 exact readiness fields")
    board_fields = {
        "status": board_artifact.get("status"),
        "board_placement_receipt_complete_score": board_artifact.get(
            "board_placement_receipt_complete_score"
        ),
        "board_names": sorted(
            row.get("board")
            for row in board_artifact.get("board_rows", [])
            if isinstance(row, Mapping) and isinstance(row.get("board"), str)
        ),
    }
    expected_board_fields = {
        "status": "complete",
        "board_placement_receipt_complete_score": 1,
        "board_names": ["GateMate", "KV260", "PolarFire"],
    }
    checks.append(
        _check(
            "exp7190_exact_gate_fields",
            UPSTREAM_BOARD_PATH.as_posix(),
            "status_score_and_board_names",
            expected_board_fields,
            board_fields if board_quarantine["passed"] else "not_consumed_quarantined",
            board_quarantine["passed"] and board_fields == expected_board_fields,
        )
    )

    progress(0, "check", "known failed board values remain unpromoted")
    board_by_name = {
        row.get("board"): row
        for row in board_artifact.get("board_rows", [])
        if board_quarantine["passed"] and isinstance(row, Mapping)
    }
    failed_observed = {
        "GateMate": board_by_name.get("GateMate", {}).get("disposition"),
        "PolarFire": board_by_name.get("PolarFire", {}).get("disposition"),
    }
    failed_expected = {
        "GateMate": "blocked_inherited_no_new_physical_state",
        "PolarFire": "blocked_missing_raw_dispatch_transcript",
    }
    checks.append(
        _check(
            "known_failed_upstream_values_not_promoted",
            UPSTREAM_BOARD_PATH.as_posix(),
            "board_dispositions",
            failed_expected,
            failed_observed,
            failed_observed == failed_expected,
        )
    )
    return checks, hashes


def _first_failed(checks: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Return one normalized failure for a terminal blocked artifact."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return None
    return {
        "passed": False,
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "field": failed.get("field"),
        "expected_value": failed.get("expected_value"),
        "observed_value": failed.get("observed_value"),
    }


def _common_row(
    *, unit_id: str, arm: str, seed: int | None, metric: Any, error: Any, abstention: bool
) -> JsonDict:
    """Keep the minimum comparison identity beside every evidence row."""

    return {
        "unit_id": unit_id,
        "arm": arm,
        "seed": seed,
        "metric": metric,
        "error": error,
        "abstention": abstention,
    }


def load_board_evidence(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Copy the latest board rows while re-hashing every cited local receipt."""

    artifact = _read_json(root / UPSTREAM_BOARD_PATH)
    source_rows = artifact.get("board_rows", [])
    if not isinstance(source_rows, list):
        raise ValueError("Exp7190 board_rows must be a list")
    rows: list[JsonDict] = []
    for source in source_rows:
        if not isinstance(source, Mapping) or not isinstance(source.get("board"), str):
            raise ValueError("Exp7190 board row is malformed")
        row = deepcopy(dict(source))
        evidence_path = root / str(row.get("evidence_path"))
        if not evidence_path.is_file():
            raise ValueError(f"board evidence is missing: {evidence_path}")
        row["source_hash"] = sha256_file(evidence_path)
        row["source_artifact_hash"] = sha256_file(root / UPSTREAM_BOARD_PATH)
        row["receipt_date"] = row.get("recorded_date")
        row["evidence_type"] = (
            "programmable_logic_terminal_transcript"
            if row["board"] == "KV260"
            else "operator_physical_state_receipt_audit"
            if row["board"] == "GateMate"
            else "ssh_and_board_cpu_dispatch_receipt"
        )
        row["unresolved_prerequisite"] = row.get("exact_next_prerequisite")
        row.update(
            _common_row(
                unit_id=f"board:{row['board']}",
                arm="aggregation_from_upstream_artifacts",
                seed=None,
                metric=row.get("terminal_criterion_met"),
                error=(
                    None
                    if row.get("terminal_criterion_met") is True
                    else row.get("exact_next_prerequisite")
                ),
                abstention=row.get("terminal_criterion_met") is not True,
            )
        )
        rows.append(row)

    by_board = {row["board"]: row for row in rows}
    if set(by_board) != {"KV260", "GateMate", "PolarFire"}:
        raise ValueError("Exp7190 must supply exactly the three attached boards")
    kv260 = by_board["KV260"]
    transcript_candidates = [
        root / path
        for path in kv260.get("supporting_evidence_paths", [])
        if "transcript" in str(path)
    ]
    transcript_hash = (
        sha256_file(transcript_candidates[0])
        if transcript_candidates and transcript_candidates[0].is_file()
        else None
    )
    if not (
        kv260.get("terminal_criterion_met") is True
        and transcript_hash == kv260.get("raw_transcript_hash")
    ):
        kv260["disposition"] = "blocked_invalid_kv260_graduation_receipt"
        kv260["programmable_logic_sampling_observed"] = False
        kv260["metric"] = False
        kv260["error"] = "KV260 transcript hash does not support graduation"
        kv260["abstention"] = True

    prior_operator = artifact.get("operator_state_receipt", {})
    if not isinstance(prior_operator, Mapping):
        prior_operator = {}
    newer = prior_operator.get("newer_than_exp6559") is True
    operator = {
        **dict(prior_operator),
        "newer_than_exp6559": newer,
        "compared_to_experiment": "Exp6559",
        "cutoff_source_hash": sha256_file(
            root / "results/experiment_6559_gatemate_changed_state_continuity.json"
        ),
        "latest_search_source": "results/experiment_7146_v627_gatemate_changed_state.json",
        "latest_search_source_hash": sha256_file(
            root / "results/experiment_7146_v627_gatemate_changed_state.json"
        ),
        "authorized_action_for_later_task": (
            "one bounded GateMate detect in a separate hardware task" if newer else None
        ),
        "hardware_command_count": 0,
        "hardware_operations_issued": [],
    }
    return rows, operator


def graph_metrics(instance: slices.SliceInstance) -> JsonDict:
    """Count the complete graph and fields without changing its placement shape."""

    validation = slices.validate_instance(instance)
    degree = [0] * instance.n
    for left, right, _coupling in instance.edges:
        degree[left] += 1
        degree[right] += 1
    return {
        "edge_count": len(instance.edges),
        "maximum_degree": max(degree, default=0),
        "edges_dropped": 0,
        "nonzero_field_count": sum(field != 0.0 for field in instance.fields),
        "frustrated_triangle": validation["frustrated_triangle"],
        "degree_limit": 16,
        "degree_limit_necessary_passed": max(degree, default=0) <= 16,
        "explicit_parent_graph_mapping_present": False,
        "topology_fit": "topology_unknown",
    }


def changed_coefficient_count(
    instance: slices.SliceInstance, quantized: correction.QuantizedInstance
) -> int:
    """Count decoded coefficients that differ from their full-precision inputs."""

    originals = tuple(edge[2] for edge in instance.edges) + tuple(instance.fields)
    decoded = tuple(code * quantized.scale for code in quantized.edge_codes) + tuple(
        code * quantized.scale for code in quantized.field_codes
    )
    return sum(
        original != approximate for original, approximate in zip(originals, decoded, strict=True)
    )


def exact_grid_control(n: int) -> slices.SliceInstance:
    """Create a full graph on the quantizer grid as a no-distortion control."""

    base = slices.make_frustrated_instance(n, SEEDS[0])
    edges = tuple(
        (left, right, 1.0 if coupling > 0.0 else -1.0) for left, right, coupling in base.edges
    )
    fields = tuple(1.0 if index % 2 == 0 else -1.0 for index in range(n))
    control = slices.SliceInstance(n=n, seed=7_203_000, edges=edges, fields=fields)
    slices.validate_instance(control)
    return control


def _small_law_row(
    instance: slices.SliceInstance, bits: int, *, exact_grid_negative_control: bool
) -> JsonDict:
    """Recompute one finite target and the corrected pair-swap transition law."""

    states = slices.enumerate_slice(instance.n, K)
    quantized = correction.quantize_instance(instance, bits)
    full_energies = tuple(
        correction.full_precision_authority_energy(instance, state) for state in states
    )
    approximate_energies = tuple(correction.quantized_energy(quantized, state) for state in states)
    full_target = correction.distribution_from_energies(full_energies, BETA)
    approximate_target = correction.distribution_from_energies(approximate_energies, BETA)
    full_matrix = correction.build_transition_matrix(
        states, BETA, full_energies, approximate_energies, FULL_ARM
    )
    corrected_matrix = correction.build_transition_matrix(
        states, BETA, full_energies, approximate_energies, CORRECTED_ARM
    )
    full_diagnostics = correction.transition_diagnostics(full_matrix, full_target)
    corrected_diagnostics = correction.transition_diagnostics(corrected_matrix, full_target)
    changed = changed_coefficient_count(instance, quantized)
    distortion = correction.total_variation(full_target, approximate_target)
    change_contract_passed = changed == 0 if exact_grid_negative_control else changed > 0
    passed = bool(
        change_contract_passed
        and full_diagnostics["stationary_residual_max"] <= LAW_TOLERANCE
        and corrected_diagnostics["stationary_residual_max"] <= LAW_TOLERANCE
        and corrected_diagnostics["detailed_balance_error_max"] <= LAW_TOLERANCE
        and (not exact_grid_negative_control or distortion <= LAW_TOLERANCE)
    )
    row = {
        "row_type": "exact_law",
        "n": instance.n,
        "k": K,
        "beta": BETA,
        "graph_seed": instance.seed,
        "precision_bits": bits,
        "instance_hash": instance.instance_hash,
        "state_count": len(states),
        "exact_grid_negative_control": exact_grid_negative_control,
        "changed_coefficient_count": changed,
        "quantized_target_tv_from_full": distortion,
        "full_stationary_residual_max": full_diagnostics["stationary_residual_max"],
        "full_detailed_balance_error_max": full_diagnostics["detailed_balance_error_max"],
        "corrected_stationary_residual_max": corrected_diagnostics["stationary_residual_max"],
        "corrected_detailed_balance_error_max": corrected_diagnostics["detailed_balance_error_max"],
        "passed": passed,
    }
    row.update(
        _common_row(
            unit_id=(
                f"n8:exact-grid:bits{bits}"
                if exact_grid_negative_control
                else f"n8:seed{instance.seed}:bits{bits}"
            ),
            arm=CORRECTED_ARM,
            seed=instance.seed,
            metric=corrected_diagnostics["stationary_residual_max"],
            error=None if passed else "small-law correction contract failed",
            abstention=False,
        )
    )
    return row


def run_small_law_checks(
    *, seeds: Sequence[int] = SEEDS, bits_values: Sequence[int] = BITS
) -> list[JsonDict]:
    """Run the n=8 correction authority and one exact-grid control per precision."""

    rows = [
        _small_law_row(
            slices.make_frustrated_instance(8, seed),
            bits,
            exact_grid_negative_control=False,
        )
        for seed in seeds
        for bits in bits_values
    ]
    control = exact_grid_control(8)
    rows.extend(
        _small_law_row(control, bits, exact_grid_negative_control=True) for bits in bits_values
    )
    return rows


def _derived_seed(*parts: object) -> int:
    """Domain-separate fixed random streams without reading benchmark outcomes."""

    material = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def _accept_uniform(log_threshold: float, uniform: float) -> bool:
    """Compare one frozen uniform variate in log space."""

    return math.log(max(uniform, sys.float_info.min)) < log_threshold


def _trace_update(digest: Any, state: Sequence[int]) -> None:
    """Hash every retained state without keeping a large trace in memory."""

    digest.update(bytes(1 if spin == 1 else 0 for spin in state))


def run_chain(
    instance: slices.SliceInstance,
    *,
    bits: int,
    arm: str,
    budget_kind: str,
    seed: int,
    proposal_limit: int = EQUAL_WORK_PROPOSALS,
    wall_budget_s: float = EQUAL_WALL_SECONDS,
) -> JsonDict:
    """Measure one full or corrected chain and retain every rejected state."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if budget_kind not in BUDGET_KINDS:
        raise ValueError(f"unknown budget: {budget_kind}")
    if proposal_limit <= 0:
        raise ValueError("proposal_limit must be positive")
    if not math.isfinite(wall_budget_s) or wall_budget_s <= 0.0:
        raise ValueError("wall_budget_s must be positive and finite")
    slices.validate_instance(instance)
    quantized = correction.quantize_instance(instance, bits)
    state_rng = random.Random(_derived_seed(seed, instance.n, bits, budget_kind, "state"))
    proposal_rng = random.Random(_derived_seed(seed, instance.n, bits, budget_kind, "proposal"))
    acceptance_rng = random.Random(_derived_seed(seed, instance.n, bits, budget_kind, "acceptance"))
    positive = set(state_rng.sample(range(instance.n), K))
    current = tuple(1 if index in positive else -1 for index in range(instance.n))

    full_energy_elapsed_ns = 0
    cheap_energy_elapsed_ns = 0
    energy_started = time.perf_counter_ns()
    current_full = correction.full_precision_authority_energy(instance, current)
    full_energy_elapsed_ns += time.perf_counter_ns() - energy_started
    full_energy_calls = 1
    current_approximate = 0.0
    cheap_energy_calls = 0
    if arm == CORRECTED_ARM:
        energy_started = time.perf_counter_ns()
        current_approximate = correction.quantized_energy(quantized, current)
        cheap_energy_elapsed_ns += time.perf_counter_ns() - energy_started
        cheap_energy_calls = 1

    trace = hashlib.sha256()
    _trace_update(trace, current)
    accepted = 0
    stage_one_accepted = 0
    cheap_rejected = 0
    retained_rejections = 0
    proposals = 0
    started = time.monotonic()
    while True:
        candidate, _forward, _reverse = slices.propose_pair_swap(current, proposal_rng)
        stage_one_uniform = acceptance_rng.random()
        stage_two_uniform = acceptance_rng.random()
        move = False
        if arm == FULL_ARM:
            energy_started = time.perf_counter_ns()
            candidate_full = correction.full_precision_authority_energy(instance, candidate)
            full_energy_elapsed_ns += time.perf_counter_ns() - energy_started
            full_energy_calls += 1
            move = _accept_uniform(
                min(0.0, -BETA * (candidate_full - current_full)), stage_one_uniform
            )
            if move:
                current_full = candidate_full
        else:
            energy_started = time.perf_counter_ns()
            candidate_approximate = correction.quantized_energy(quantized, candidate)
            cheap_energy_elapsed_ns += time.perf_counter_ns() - energy_started
            cheap_energy_calls += 1
            delta_approximate = candidate_approximate - current_approximate
            stage_one = min(0.0, -BETA * delta_approximate)
            if _accept_uniform(stage_one, stage_one_uniform):
                stage_one_accepted += 1
                energy_started = time.perf_counter_ns()
                candidate_full = correction.full_precision_authority_energy(instance, candidate)
                full_energy_elapsed_ns += time.perf_counter_ns() - energy_started
                full_energy_calls += 1
                stage_two = correction.delayed_acceptance_log_terms(
                    BETA, candidate_full - current_full, delta_approximate
                )[1]
                move = _accept_uniform(stage_two, stage_two_uniform)
                if move:
                    current_full = candidate_full
                    current_approximate = candidate_approximate
            else:
                cheap_rejected += 1
        if move:
            current = candidate
            accepted += 1
        else:
            retained_rejections += 1
        proposals += 1
        _trace_update(trace, current)
        elapsed = time.monotonic() - started
        if budget_kind == EQUAL_WORK and proposals >= proposal_limit:
            break
        if budget_kind == EQUAL_WALL and elapsed >= wall_budget_s:
            break

    elapsed = time.monotonic() - started
    cheap_s = cheap_energy_elapsed_ns / 1_000_000_000.0
    full_s = full_energy_elapsed_ns / 1_000_000_000.0
    return {
        "proposal_count": proposals,
        "accepted_move_count": accepted,
        "acceptance_rate": accepted / proposals,
        "cheap_stage_reject_count": cheap_rejected,
        "stage_one_accept_count": stage_one_accepted,
        "stage_one_acceptance_rate": (
            stage_one_accepted / proposals if arm == CORRECTED_ARM else None
        ),
        "full_energy_calls": full_energy_calls,
        "cheap_energy_calls": cheap_energy_calls,
        "full_energy_elapsed_s": full_s,
        "cheap_energy_elapsed_s": cheap_s,
        "elapsed_wall_time_s": elapsed,
        "latency_s_per_proposal": elapsed / proposals,
        "host_correction_latency_s_per_proposal": (
            max(0.0, elapsed - cheap_s) / proposals if arm == CORRECTED_ARM else elapsed / proposals
        ),
        "chain_state_count": proposals + 1,
        "rejected_state_retention_count": retained_rejections,
        "trace_sha256": "sha256:" + trace.hexdigest(),
        "proposal_limit": proposal_limit if budget_kind == EQUAL_WORK else None,
        "wall_budget_s": wall_budget_s if budget_kind == EQUAL_WALL else None,
        "matched_random_stream_contract": (
            "arms share state, proposal, and two-uniform stream seeds for each size, "
            "precision, budget, and seed"
        ),
    }


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist real progress through same-directory replacement."""

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_cost_panel(checkpoint_path: Path) -> list[JsonDict]:
    """Run all fixed cost cells with bounded elapsed work and saved progress."""

    rows: list[JsonDict] = []
    started = time.monotonic()
    expected = len(SIZES) * len(SEEDS) * len(BITS) * len(BUDGET_KINDS) * len(ARMS)
    for n in SIZES:
        for seed in SEEDS:
            instance = slices.make_frustrated_instance(n, seed)
            metrics = graph_metrics(instance)
            for bits in BITS:
                quantized = correction.quantize_instance(instance, bits)
                changed = changed_coefficient_count(instance, quantized)
                if changed <= 0:
                    raise ValueError(
                        f"non-control condition changed no coefficient: n={n} seed={seed} bits={bits}"
                    )
                for budget_kind in BUDGET_KINDS:
                    for arm in ARMS:
                        elapsed_total = time.monotonic() - started
                        if elapsed_total >= HOST_MEASUREMENT_CAP_SECONDS:
                            raise TimeoutError("host measurement exceeded 900 seconds")
                        unit_id = f"n{n}:seed{seed}:bits{bits}:{budget_kind}:{arm}"
                        progress(4, "benchmark start", unit_id)
                        measurement = run_chain(
                            instance,
                            bits=bits,
                            arm=arm,
                            budget_kind=budget_kind,
                            seed=seed,
                        )
                        progress(
                            4,
                            "benchmark end",
                            f"{unit_id} proposals={measurement['proposal_count']}",
                        )
                        row = {
                            "row_type": "correction_cost",
                            "n": n,
                            "k": K,
                            "beta": BETA,
                            "graph_seed": seed,
                            "precision_bits": bits,
                            "budget_kind": budget_kind,
                            "instance_hash": instance.instance_hash,
                            "changed_coefficient_count": changed,
                            "quantization_distortion_interpretable": True,
                            **metrics,
                            **measurement,
                        }
                        row.update(
                            _common_row(
                                unit_id=unit_id,
                                arm=arm,
                                seed=seed,
                                metric=measurement["latency_s_per_proposal"],
                                error=None,
                                abstention=False,
                            )
                        )
                        rows.append(row)
                        if len(rows) % 30 == 0 or len(rows) == expected:
                            _write_checkpoint(
                                checkpoint_path,
                                {
                                    "status": "running"
                                    if len(rows) < expected
                                    else "measurement_complete",
                                    "task_id": TASK_ID,
                                    "completed_cost_rows": len(rows),
                                    "planned_cost_rows": expected,
                                    "elapsed_s": time.monotonic() - started,
                                    "last_unit_id": unit_id,
                                },
                            )
    return rows


def build_break_even_rows(cost_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Combine measured host cost with explicit future-device hypotheses."""

    rows: list[JsonDict] = []
    for n in SIZES:
        for bits in BITS:
            exact = [
                row
                for row in cost_rows
                if row.get("n") == n
                and row.get("precision_bits") == bits
                and row.get("budget_kind") == EQUAL_WORK
                and row.get("arm") == FULL_ARM
            ]
            corrected = [
                row
                for row in cost_rows
                if row.get("n") == n
                and row.get("precision_bits") == bits
                and row.get("budget_kind") == EQUAL_WORK
                and row.get("arm") == CORRECTED_ARM
            ]
            if len(exact) != len(SEEDS) or len(corrected) != len(SEEDS):
                raise ValueError("break-even aggregation requires ten matched seed rows")
            exact_mean = float(np.mean([row["latency_s_per_proposal"] for row in exact]))
            correction_host_mean = float(
                np.mean([row["host_correction_latency_s_per_proposal"] for row in corrected])
            )
            stage_one_mean = float(np.mean([row["stage_one_acceptance_rate"] for row in corrected]))
            for device_ns in DEVICE_LATENCY_NS:
                for transfer_ns in TRANSFER_LATENCY_NS:
                    predicted = correction_host_mean + (device_ns + transfer_ns) / 1_000_000_000.0
                    margin = exact_mean - predicted
                    unit_id = f"n{n}:bits{bits}:device{device_ns}ns:transfer{transfer_ns}ns"
                    row = {
                        "row_type": "break_even",
                        "n": n,
                        "precision_bits": bits,
                        "source_budget_kind": EQUAL_WORK,
                        "source_seed_count": len(SEEDS),
                        "measured_exact_host_latency_s_per_proposal_mean": exact_mean,
                        "measured_correction_host_latency_s_per_proposal_mean": correction_host_mean,
                        "measured_stage_one_acceptance_rate_mean": stage_one_mean,
                        "hypothetical_device_compute_latency_ns": device_ns,
                        "hypothetical_transfer_latency_ns": transfer_ns,
                        "predicted_corrected_latency_s_per_proposal": predicted,
                        "break_even_margin_s_per_proposal": margin,
                        "hypothesis_breaks_even": margin > 0.0,
                        "device_timing_source": "explicit_hypothesis",
                        "measured_device_timing": "unknown",
                        "claim_unit": "per_proposed_transition",
                        "equal_effective_sample_throughput_established": False,
                        "mixing_speed_established": False,
                    }
                    row.update(
                        _common_row(
                            unit_id=unit_id,
                            arm="future_device_delayed_acceptance_hypothesis",
                            seed=None,
                            metric=margin,
                            error=None,
                            abstention=False,
                        )
                    )
                    rows.append(row)
    return rows


def _sample_budget(
    exact_law_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
    break_even_rows: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Keep planned and completed unit counts separate from interpretation."""

    equal_work = [row for row in cost_rows if row.get("budget_kind") == EQUAL_WORK]
    equal_wall = [row for row in cost_rows if row.get("budget_kind") == EQUAL_WALL]
    return {
        "planned": {
            "board_rows": 3,
            "exact_law_rows": 33,
            "cost_rows": 360,
            "break_even_rows": 144,
            "equal_work_proposals": 184_320,
            "equal_wall_seconds": 18.0,
        },
        "completed": {
            "board_rows": len(board_rows),
            "exact_law_rows": len(exact_law_rows),
            "cost_rows": len(cost_rows),
            "break_even_rows": len(break_even_rows),
            "equal_work_proposals": sum(int(row["proposal_count"]) for row in equal_work),
            "equal_wall_proposals": sum(int(row["proposal_count"]) for row in equal_wall),
            "equal_wall_elapsed_s": sum(float(row["elapsed_wall_time_s"]) for row in equal_wall),
        },
        "independent_units": {
            "graph_seed_count": len(SEEDS),
            "graph_seeds": list(SEEDS),
            "size_count": len(SIZES),
            "precision_count": len(BITS),
        },
        "exclusions": [],
        "claim_limit": (
            "The proposal counts measure transition cost. They do not establish independent "
            "sample count, effective-sample throughput, or mixing speed."
        ),
    }


def _base_artifact(
    *, root: Path, run_date: str, checks: list[JsonDict], hashes: dict[str, str], started: float
) -> JsonDict:
    """Create the full schema before choosing a blocked or measured terminal state."""

    return {
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "spec_refs": [
            "REQ-ISING-7203",
            "SCENARIO-ISING-7203-PREFLIGHT",
            "SCENARIO-ISING-7203-BOARDS",
            "SCENARIO-ISING-7203-SMALL-LAW",
            "SCENARIO-ISING-7203-COST",
            "SCENARIO-ISING-7203-BREAK-EVEN",
            "SCENARIO-ISING-7203-ARTIFACT",
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": checks,
        "inference_substrate": "blocked_before_cpu_measurement",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "host_identity": platform.node() or "unknown-host",
        "duration_s": time.monotonic() - started,
        "source_artifact_hashes": hashes,
        "rows": [],
        "sample_size_budget": _sample_budget([], [], [], []),
        "random_seed": {
            "graph_and_chain_seeds": list(SEEDS),
            "beta": BETA,
            "k": K,
            "stream_derivation": "sha256(seed,n,bits,budget,stream-role)",
        },
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "hardware_envelope_complete_score": 0,
        "board_rows": [],
        "operator_state_receipt": None,
        "exact_law_rows": [],
        "correction_cost_rows": [],
        "break_even_rows": [],
        "hardware_execution_claimed": False,
        "hardware_command_count": 0,
        "hardware_operations_issued": [],
        "topology_fit": "topology_unknown",
        "device_timing_available": False,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "power_savings_claimed": False,
        "tsu_execution_claimed": False,
        "fpga_execution_claimed": False,
        "soft_spin_execution_claimed": False,
        "equal_effective_sample_throughput_established": False,
        "mixing_speed_established": False,
        "root": str(root.resolve()),
    }


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    checkpoint_path: Path,
) -> JsonDict:
    """Build complete CPU evidence or stop at one diagnosed external block."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    started = time.monotonic()
    progress(0, "start", "preconditions")
    progress(0, "before", "source, gate, quarantine, tool, and output checks")
    checks, hashes = collect_preconditions(
        root, output_path=output_path, checkpoint_path=checkpoint_path
    )
    progress(0, "after", "source, gate, quarantine, tool, and output checks")
    progress(0, "end", "preconditions")
    artifact = _base_artifact(
        root=root, run_date=run_date, checks=checks, hashes=hashes, started=started
    )
    failed = _first_failed(checks)
    if failed is not None:
        for phase, operation in (
            (1, "LLM contract"),
            (2, "three-board receipt evidence"),
            (3, "GateMate operator-state comparison"),
            (4, "CPU exact-law and correction-cost benchmarks"),
            (5, "hypothetical per-transition break-even envelope"),
            (6, "hardware and topology claim limits"),
        ):
            progress(phase, "start", f"{operation}: blocked at phase 0")
            progress(phase, "end", f"{operation}: blocked at phase 0")
        artifact.update(
            {
                "status": "blocked_external_precondition",
                "duration_s": time.monotonic() - started,
                "gate_check_summary": failed,
                "verdict_class": "blocked",
                "honest_verdict": f"blocked_{failed['failed_check']}",
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    progress(1, "start", "LLM contract")
    progress(1, "end", "MODEL_SPECS empty and model_invoked false")
    progress(2, "start", "three-board receipt evidence")
    board_rows, operator_receipt = load_board_evidence(root)
    progress(2, "end", f"three-board receipt evidence rows={len(board_rows)}")
    progress(3, "start", "GateMate operator-state comparison")
    progress(3, "end", "Exp6559 boundary retained; hardware command count zero")
    progress(4, "start", "n=8 exact-law authority")
    exact_law_rows = run_small_law_checks()
    progress(4, "end", f"n=8 exact-law authority rows={len(exact_law_rows)}")
    if not all(row["passed"] for row in exact_law_rows):
        raise ValueError("small-law correction authority failed")
    progress(4, "before", "bounded CPU correction-cost benchmark panel")
    cost_rows = run_cost_panel(checkpoint_path)
    progress(4, "after", f"bounded CPU correction-cost rows={len(cost_rows)}")
    progress(5, "start", "hypothetical per-transition break-even envelope")
    break_even_rows = build_break_even_rows(cost_rows)
    progress(5, "end", f"hypothetical break-even rows={len(break_even_rows)}")
    progress(6, "start", "hardware and topology claim limits")
    rows = board_rows + exact_law_rows + cost_rows + break_even_rows
    complete = int(
        len(board_rows) == 3
        and len(exact_law_rows) == 33
        and len(cost_rows) == 360
        and len(break_even_rows) == 144
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": "cpu_exact_solver_or_simulator",
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "duration_s": time.monotonic() - started,
            "rows": rows,
            "sample_size_budget": _sample_budget(
                exact_law_rows, cost_rows, break_even_rows, board_rows
            ),
            "gate_check_summary": {
                "passed": True,
                "failed_check": None,
                "upstream": "repository_preflight",
                "field": "all_required_preconditions",
                "expected_value": "all pass",
                "observed_value": "all pass",
            },
            "verifier_is_oracle": True,
            "verdict_class": "circular_positive",
            "honest_verdict": (
                "complete: the full-target correction law passed the n=8 authority check "
                "and the bounded CPU cost envelope completed; device timing, topology fit, "
                "mixing speed, effective-sample throughput, and hardware power remain unknown"
            ),
            "hardware_envelope_complete_score": complete,
            "board_rows": board_rows,
            "operator_state_receipt": operator_receipt,
            "exact_law_rows": exact_law_rows,
            "correction_cost_rows": cost_rows,
            "break_even_rows": break_even_rows,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(6, "end", "hardware claims false and topology unknown")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute completion, row census, claim boundaries, and checksum."""

    errors: list[str] = []
    if not REQUIRED_ARTIFACT_FIELDS.issubset(artifact):
        errors.append("missing_required_fields")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if artifact.get("hardware_execution_claimed") is not False:
        errors.append("hardware_claim_invalid")
    if (
        artifact.get("hardware_command_count") != 0
        or artifact.get("hardware_operations_issued") != []
    ):
        errors.append("hardware_command_invalid")
    if artifact.get("topology_fit") != "topology_unknown":
        errors.append("topology_fit_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_contract_invalid")
    if any(
        artifact.get(field) is not False
        for field in (
            "device_timing_available",
            "power_savings_claimed",
            "tsu_execution_claimed",
            "fpga_execution_claimed",
            "soft_spin_execution_claimed",
            "equal_effective_sample_throughput_established",
            "mixing_speed_established",
        )
    ):
        errors.append("unsupported_claim_invalid")

    blocked = artifact.get("verdict_class") == "blocked"
    board_rows = artifact.get("board_rows", [])
    exact_law_rows = artifact.get("exact_law_rows", [])
    cost_rows = artifact.get("correction_cost_rows", [])
    break_even_rows = artifact.get("break_even_rows", [])
    rows = artifact.get("rows", [])
    if blocked:
        summary = artifact.get("gate_check_summary", {})
        if not (
            artifact.get("status") == "blocked_external_precondition"
            and artifact.get("inference_substrate_class") == "blocked_no_run"
            and artifact.get("hardware_envelope_complete_score") == 0
            and rows == []
            and board_rows == []
            and exact_law_rows == []
            and cost_rows == []
            and break_even_rows == []
            and isinstance(summary, Mapping)
            and summary.get("passed") is False
            and bool(summary.get("failed_check"))
        ):
            errors.append("blocked_state_invalid")
    else:
        if not (
            artifact.get("status") == "complete"
            and artifact.get("verdict_class") == "circular_positive"
            and artifact.get("inference_substrate_class") == "cpu_exact_solver_or_simulator"
            and artifact.get("gate_check_summary", {}).get("passed") is True
            and artifact.get("verifier_is_oracle") is True
            and str(artifact.get("honest_verdict", "")).startswith("complete:")
        ):
            errors.append("terminal_state_invalid")
        if not (
            isinstance(board_rows, list)
            and isinstance(exact_law_rows, list)
            and isinstance(cost_rows, list)
            and isinstance(break_even_rows, list)
            and len(board_rows) == 3
            and len(exact_law_rows) == 33
            and len(cost_rows) == 360
            and len(break_even_rows) == 144
        ):
            errors.append("row_census_invalid")
        if rows != board_rows + exact_law_rows + cost_rows + break_even_rows:
            errors.append("rows_invalid")
        if isinstance(rows, list) and any(
            not isinstance(row, Mapping)
            or not {"unit_id", "arm", "seed", "metric", "error", "abstention"}.issubset(row)
            for row in rows
        ):
            errors.append("row_identity_invalid")
        board_by_name = {row.get("board"): row for row in board_rows if isinstance(row, Mapping)}
        if not (
            set(board_by_name) == {"KV260", "GateMate", "PolarFire"}
            and board_by_name["KV260"].get("disposition") == "graduated_preserved"
            and board_by_name["GateMate"].get("disposition")
            == "blocked_inherited_no_new_physical_state"
            and board_by_name["PolarFire"].get("disposition")
            == "blocked_missing_raw_dispatch_transcript"
            and all(row.get("hardware_command_count") == 0 for row in board_by_name.values())
        ):
            errors.append("board_rows_invalid")
        if isinstance(exact_law_rows, list) and any(
            row.get("passed") is not True
            or (
                row.get("exact_grid_negative_control") is True
                and row.get("changed_coefficient_count") != 0
            )
            or (
                row.get("exact_grid_negative_control") is False
                and not (
                    isinstance(row.get("changed_coefficient_count"), int)
                    and row["changed_coefficient_count"] > 0
                )
            )
            for row in exact_law_rows
            if isinstance(row, Mapping)
        ):
            errors.append("exact_law_invalid")
        expected_cells = {
            (n, seed, bits, budget, arm)
            for n in SIZES
            for seed in SEEDS
            for bits in BITS
            for budget in BUDGET_KINDS
            for arm in ARMS
        }
        observed_cells = {
            (
                row.get("n"),
                row.get("seed"),
                row.get("precision_bits"),
                row.get("budget_kind"),
                row.get("arm"),
            )
            for row in cost_rows
            if isinstance(row, Mapping)
        }
        if observed_cells != expected_cells:
            errors.append("cost_cells_invalid")
        if isinstance(cost_rows, list) and any(
            row.get("edges_dropped") != 0
            or row.get("topology_fit") != "topology_unknown"
            or not (
                isinstance(row.get("changed_coefficient_count"), int)
                and row["changed_coefficient_count"] > 0
            )
            or row.get("chain_state_count") != row.get("proposal_count", 0) + 1
            or row.get("rejected_state_retention_count")
            != row.get("proposal_count", 0) - row.get("accepted_move_count", 0)
            or (
                row.get("budget_kind") == EQUAL_WORK
                and row.get("proposal_count") != EQUAL_WORK_PROPOSALS
            )
            or (
                row.get("budget_kind") == EQUAL_WALL
                and row.get("elapsed_wall_time_s", 0.0) < EQUAL_WALL_SECONDS
            )
            for row in cost_rows
            if isinstance(row, Mapping)
        ):
            errors.append("cost_row_invalid")
        if isinstance(break_even_rows, list) and any(
            row.get("device_timing_source") != "explicit_hypothesis"
            or row.get("measured_device_timing") != "unknown"
            or row.get("claim_unit") != "per_proposed_transition"
            or row.get("equal_effective_sample_throughput_established") is not False
            or row.get("mixing_speed_established") is not False
            for row in break_even_rows
            if isinstance(row, Mapping)
        ):
            errors.append("break_even_invalid")
        expected_complete = int(
            len(board_rows) == 3
            and len(exact_law_rows) == 33
            and len(cost_rows) == 360
            and len(break_even_rows) == 144
        )
        if artifact.get("hardware_envelope_complete_score") != expected_complete:
            errors.append("completion_score_invalid")

    checksum = artifact.get("reproducibility_checksum")
    if (
        not isinstance(checksum, str)
        or not HASH_PATTERN.fullmatch(checksum)
        or checksum != artifact_checksum(artifact)
    ):
        errors.append("checksum_invalid")
    return list(dict.fromkeys(errors))


def atomic_write(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate, flush, and replace so the terminal path is never partial."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp7203 artifact: {errors}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            dict(artifact), handle, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256_file(path), "atomic_replace": True}


def run_experiment(
    root: Path, run_date: str, *, output_path: Path, checkpoint_path: Path
) -> JsonDict:
    """Run visible bounded phases and publish one terminal artifact."""

    if run_date != RUN_DATE:
        progress(0, "start", "preconditions")
        raise ValueError(f"run date must be {RUN_DATE}")
    artifact = build_artifact(
        root, run_date, output_path=output_path, checkpoint_path=checkpoint_path
    )

    progress(7, "start", "artifact validation")
    errors = validate_artifact(artifact)
    progress(7, "end", f"artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7203 artifact: {errors}")
    progress(8, "start", "final atomic write")
    receipt = atomic_write(output_path, artifact)
    progress(8, "end", f"final atomic write sha256={receipt['sha256']}")
    return artifact


def _parser() -> argparse.ArgumentParser:
    """Define run and read-only validation modes on one entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or validate one selected artifact without rewriting it."""

    args = _parser().parse_args(argv)
    try:
        if args.validate is not None:
            artifact = _read_json(args.validate)
            errors = validate_artifact(artifact)
            if errors:
                print(f"validation_failed errors={errors}", flush=True)
                return 2
            print("validation_passed", flush=True)
            return 0
        root = args.root.resolve()
        output = args.output or root / RESULT_PATH
        checkpoint = args.checkpoint or root / CHECKPOINT_PATH
        if not output.is_absolute():
            output = root / output
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        artifact = run_experiment(root, args.date, output_path=output, checkpoint_path=checkpoint)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    print(
        canonical_json(
            {
                "hardware_envelope_complete_score": artifact["hardware_envelope_complete_score"],
                "output": str(output),
                "verdict_class": artifact["verdict_class"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository entrypoint calls main.
    raise SystemExit(main())
