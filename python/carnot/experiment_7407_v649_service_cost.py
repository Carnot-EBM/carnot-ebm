"""Measure the complete V649 decision service on the host CPU.

The measurement starts with fixed source text and ends after a serialized
typed-policy return. It compares only scalar and vectorized NumPy scoring. It
does not train, load an LLM, contact a board, or predict device performance.

Spec refs: REQ-REPORT-7407 and SCENARIO-REPORT-7407-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7367_v646_board_disposition as board_history
from carnot import experiment_7393_v648_hardware_placement as placement
from carnot.experiment_7382_v648_decision_protocol import typed_decision
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.verify.pcib_probe import PCIBProbe


JsonDict = dict[str, Any]

RUN_DATE = "20260919"
MILESTONE = "2026.09.649"
PHASE = 4
EXPERIMENT_ID = "exp7407-service-cost"
SCHEMA = "carnot.exp7407.v649.service_cost.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7407_v649_service_cost.json")
RAW_DIR = Path("results/raw/experiment_7407_v649_service_cost")
MODULE_PATH = Path("python/carnot/experiment_7407_v649_service_cost.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7407_v649_service_cost.py")
TEST_PATH = Path("tests/python/test_experiment_7407_v649_service_cost.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXP7385_PATH = Path("results/experiment_7385_v648_decision_training.json")
EXP7393_PATH = Path("results/experiment_7393_v648_hardware_placement.json")
EXP7399_PATH = Path("results/experiment_7399_v649_online_trial.json")
EXP7314_PATH = Path("results/experiment_7314_v642_board_continuity.json")
EXP7358_PATH = Path("results/experiment_7358_v646_validation_contract.json")
CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_7385_v648_decision_training/"
    "natural_prevalence_bernoulli_gibbs_7382001.json"
)
CORPUS_PATH = Path("data/fover_corpus_v4.json")
KV260_TRANSCRIPT_PATH = Path(
    "results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"
)
POLARFIRE_TRANSCRIPT_PATH = Path("results/raw/experiment_7231/polarfire_dispatch.json")

EXPECTED_EXP7385_SHA256 = "sha256:05cb5c7fb56fa5afa9ec8b450315ea00534229ec477890a84b412355f8ede87b"
EXPECTED_CHECKPOINT_SHA256 = (
    "sha256:a93d7eb86200729a6c1db97475d98c1ef0265d94566d10b25bc8e7abb9139df8"
)
EXPECTED_CORPUS_SHA256 = "sha256:c5710308eb72575591165ad1df672086e3d91ae3270c174c409e8c1ef48725e2"
EXPECTED_EXP7314_SHA256 = "sha256:c83cc85d16c082898992b3d34197d5d6bb07dfae5cd5bb9beca10bbbc20d51e6"
EXPECTED_KV260_TRANSCRIPT_SHA256 = (
    "sha256:b813db8619cf29c3350fe9e806f68e20f0c8a5c381b6c11cb8dc096f524f6e5e"
)
EXPECTED_POLARFIRE_TRANSCRIPT_SHA256 = (
    "sha256:d1c438da812a1c45d63d5d777050c5169dab6aac33bfee8b44f4f4152dd6b801"
)

RANDOM_SEED = 7_407_307
FLOAT64_TOLERANCE = 1e-10
BATCH_SIZES = (1, 32, 128)
PAIRED_BLOCKS = 30
BOOTSTRAP_DRAWS = 10_000
STAGE_NAMES = (
    "raw_text_to_pcib_features",
    "numeric_energy_probability",
    "typed_policy",
    "journal_state_serialization",
    "complete_return_materialization",
)
ARMS = ("scalar_numpy", "vectorized_numpy")
ZERO_INVOCATION_COUNTS = deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)
MISSING_RECEIPT = board_history.MISSING_RECEIPT

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("ops/hardware-bringup-prep.md"),
    Path("research-hardware-wishlist.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    EXP7385_PATH,
    EXP7393_PATH,
    EXP7314_PATH,
    EXP7358_PATH,
    CHECKPOINT_PATH,
    CORPUS_PATH,
    KV260_TRANSCRIPT_PATH,
    POLARFIRE_TRANSCRIPT_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7385_v648_decision_training.py"),
    Path("python/carnot/experiment_7393_v648_hardware_placement.py"),
    MODULE_PATH,
    TEST_PATH,
    WRAPPER_PATH,
)

VALIDATION_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_FIELDS = frozenset(
    {
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
        "inference_substrate_details",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "phase_spans",
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
        "flagged_adversarial",
        "validation_receipts",
        "repository_health",
        "field_principles",
        "promotion_score",
        "service_cost_capture_complete_score",
        "service_cost_rows",
        "board_rows",
        "hardware_ready_score",
        "hardware_value_score",
    }
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw measurements, sidecars, validation logs, and output separate."""

    artifact: Path
    raw_evidence: Path
    changed_state_search: Path
    historical_receipts: Path
    terminal_candidate: Path
    validation_dir: Path

    @classmethod
    def defaults(cls, root: Path = REPO_ROOT) -> ExperimentPaths:  # pragma: no cover
        """Resolve production paths below the selected repository."""

        return cls.under(root)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Resolve task-owned paths below a repository or test directory."""

        raw = root / RAW_DIR
        return cls(
            artifact=root / RESULT_PATH,
            raw_evidence=raw / "measured_service_rows.json",
            changed_state_search=raw / "gatemate_changed_state_search.json",
            historical_receipts=raw / "historical_model_receipts.json",
            terminal_candidate=raw / "measured_terminal_candidate.json",
            validation_dir=raw / "validation",
        )


def utc_now() -> str:  # pragma: no cover - real wall-clock boundary.
    """Return one actual UTC boundary for the live experiment receipt."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every live phase and slow-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7407] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so parsed equality cannot hide a source replacement."""

    return current_work_receipt.sha256_file(path)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for rows, fixtures, and derived evidence."""

    return current_work_receipt.canonical_hash(value)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete JSON object with an atomic local replacement."""

    current_work_receipt.atomic_json(path, value)


def load_json(path: Path) -> Any:
    """Load one JSON value and fail at the exact input path."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable_json:{path}") from error


def gate_row(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
    passed: bool | None = None,
) -> JsonDict:
    """Record one explicit comparison without hiding its category."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def _referenced_hash(board: Mapping[str, Any], path: Path) -> str | None:
    """Return the hash assigned to one original board transcript."""

    references = board.get("referenced_evidence")
    if not isinstance(references, list):
        return None
    match = [row for row in references if isinstance(row, Mapping) and row.get("path") == str(path)]
    return str(match[0].get("sha256")) if len(match) == 1 else None


def _board_by_name(artifact: Mapping[str, Any], board: str) -> JsonDict:
    """Select exactly one named board row from an authenticated artifact."""

    matches = [
        dict(row)
        for row in artifact.get("board_rows") or []
        if isinstance(row, Mapping) and row.get("board") == board
    ]
    if len(matches) != 1:
        raise ValueError(f"board_row_not_unique:{board}")
    return matches[0]


def _source_gate(path: Path, expected_hash: str | None = None) -> JsonDict:
    """Authenticate one mandatory source by presence and optional pinned hash."""

    present = path.is_file() and path.stat().st_size > 0
    observed = sha256_file(path) if present else None
    passed = present and (expected_hash is None or observed == expected_hash)
    return gate_row(
        f"source_bytes:{path.as_posix()}",
        "mandatory_precondition",
        expected_hash or "readable_nonempty_bytes",
        observed if expected_hash else ("readable_nonempty_bytes" if present else None),
        passed=passed,
    )


def collect_preconditions(
    root: Path,
    paths: ExperimentPaths,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate the fixed cost fixture and historical board boundaries."""

    root = root.resolve()
    for destination in paths.__dict__.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    pinned = {
        EXP7385_PATH: EXPECTED_EXP7385_SHA256,
        CHECKPOINT_PATH: EXPECTED_CHECKPOINT_SHA256,
        CORPUS_PATH: EXPECTED_CORPUS_SHA256,
        EXP7314_PATH: EXPECTED_EXP7314_SHA256,
        KV260_TRANSCRIPT_PATH: EXPECTED_KV260_TRANSCRIPT_SHA256,
        POLARFIRE_TRANSCRIPT_PATH: EXPECTED_POLARFIRE_TRANSCRIPT_SHA256,
    }
    checks: list[JsonDict] = []
    hashes: dict[str, str | None] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        row = _source_gate(root / relative, pinned.get(relative))
        checks.append(row)
        hashes[relative.as_posix()] = (
            sha256_file(root / relative) if (root / relative).is_file() else None
        )

    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        gate_row(
            "driving_requirement",
            "mandatory_precondition",
            "REQ-REPORT-7407",
            "REQ-REPORT-7407" if "REQ-REPORT-7407" in spec_text else None,
        )
    )

    training = load_json(root / EXP7385_PATH)
    mandatory_training = (
        isinstance(training, Mapping)
        and training.get("experiment_id") == "exp7385-decision-training"
        and training.get("status") == "complete_decision_training_null"
        and training.get("verdict_class") == "null"
        and training.get("flagged_adversarial") is False
        and training.get("decision_capture_complete_score") == 1
        and all(
            row.get("passed") is True
            for row in training.get("acceptance_gate_results") or []
            if isinstance(row, Mapping)
            and row.get("category") in {"completion", "required_validation", "safety"}
        )
    )
    checks.append(
        gate_row(
            "exp7385_fixed_service_eligibility",
            "mandatory_precondition",
            ["complete_decision_training_null", "null", False, 1, True],
            [
                training.get("status"),
                training.get("verdict_class"),
                training.get("flagged_adversarial"),
                training.get("decision_capture_complete_score"),
                mandatory_training,
            ],
            passed=mandatory_training,
        )
    )

    checkpoint = load_json(root / CHECKPOINT_PATH)
    checkpoint_ok = (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("schema") == "carnot.exp7385.numeric_checkpoint.v1"
        and checkpoint.get("arm") == "natural_prevalence_bernoulli_gibbs"
        and checkpoint.get("seed") == 7382001
        and checkpoint.get("architecture") == {"input_dim": 2, "hidden_dims": [4], "output_dim": 1}
    )
    checks.append(
        gate_row(
            "exp7385_numeric_checkpoint_identity",
            "mandatory_precondition",
            ["carnot.exp7385.numeric_checkpoint.v1", "natural_prevalence_bernoulli_gibbs", 7382001],
            [checkpoint.get("schema"), checkpoint.get("arm"), checkpoint.get("seed")],
            passed=checkpoint_ok,
        )
    )

    corpus = load_json(root / CORPUS_PATH)
    corpus_ok = (
        isinstance(corpus, list)
        and len(corpus) >= 128
        and all(
            isinstance(row, Mapping) and isinstance(row.get("step_text"), str)
            for row in corpus[:128]
        )
    )
    checks.append(
        gate_row(
            "fixed_source_texts",
            "mandatory_precondition",
            {"minimum_rows": 128, "text_field": "step_text"},
            {
                "minimum_rows": len(corpus) if isinstance(corpus, list) else 0,
                "text_field": "step_text" if corpus_ok else None,
            },
            passed=corpus_ok,
        )
    )

    hardware = load_json(root / EXP7393_PATH)
    history = load_json(root / EXP7314_PATH)
    history_receipts_ok = placement._authenticate_hardware_receipts(root, hardware) == {
        "KV260": True,
        "GateMate": True,
        "PolarFire": True,
    }
    kv260 = _board_by_name(history, "KV260")
    polarfire = _board_by_name(history, "PolarFire")
    original_hashes = {
        KV260_TRANSCRIPT_PATH.as_posix(): _referenced_hash(kv260, KV260_TRANSCRIPT_PATH),
        POLARFIRE_TRANSCRIPT_PATH.as_posix(): _referenced_hash(
            polarfire, POLARFIRE_TRANSCRIPT_PATH
        ),
    }
    transcripts_ok = original_hashes == {
        KV260_TRANSCRIPT_PATH.as_posix(): EXPECTED_KV260_TRANSCRIPT_SHA256,
        POLARFIRE_TRANSCRIPT_PATH.as_posix(): EXPECTED_POLARFIRE_TRANSCRIPT_SHA256,
    }
    claim_scope_ok = (
        kv260.get("processor_class") == "fpga_fabric"
        and kv260.get("fabric_execution_completed") is True
        and polarfire.get("processor_class") == "cpu"
        and polarfire.get("programmable_logic_sampling_observed") is False
        and polarfire.get("observed_state", {}).get("dispatch_completed") is True
    )
    hardware_ok = (
        isinstance(hardware, Mapping)
        and hardware.get("schema") == placement.SCHEMA
        and hardware.get("board_disposition_complete_score") == 1
        and hardware.get("flagged_adversarial") is False
        and history_receipts_ok
        and transcripts_ok
        and claim_scope_ok
    )
    checks.append(
        gate_row(
            "original_board_transcripts_and_claim_scopes",
            "mandatory_precondition",
            {"receipts": True, "transcripts": True, "claim_scopes": True},
            {
                "receipts": history_receipts_ok,
                "transcripts": transcripts_ok,
                "claim_scopes": claim_scope_ok,
            },
            passed=hardware_ok,
        )
    )

    boundary = load_json(root / EXP7358_PATH)
    boundary_ok = (
        isinstance(boundary, Mapping)
        and boundary.get("experiment_id") == "exp7358-validation-contract"
        and boundary.get("validation_contract_ready_score") == 1
        and boundary.get("flagged_adversarial") is False
    )
    checks.append(
        gate_row(
            "exp7358_affected_command_boundary",
            "mandatory_precondition",
            1,
            boundary.get("validation_contract_ready_score"),
            passed=boundary_ok,
        )
    )

    physical = board_history.search_changed_state_receipt(
        root,
        paths.changed_state_search,
        candidate_paths=candidate_paths,
    )
    hashes[str(paths.changed_state_search)] = sha256_file(paths.changed_state_search)
    changed = physical.get("exists") is True
    checks.append(
        gate_row(
            "gatemate_changed_physical_state_receipt",
            "external_prerequisite",
            deepcopy(board_history.PHYSICAL_RECEIPT_CONTRACT),
            physical
            if changed
            else {
                "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                "selected_source_path": physical.get("selected_source_path"),
                "absence": MISSING_RECEIPT,
            },
            passed=changed,
        )
    )

    optional_available = (root / EXP7399_PATH).is_file()
    optional: JsonDict = {
        "source_experiment": "Exp7399",
        "available": optional_available,
        "eligible": False,
        "disposition": "optional_absent_does_not_block_fixed_service",
    }
    if optional_available:
        hashes[EXP7399_PATH.as_posix()] = sha256_file(root / EXP7399_PATH)
        trial = load_json(root / EXP7399_PATH)
        optional["eligible"] = bool(
            isinstance(trial, Mapping)
            and trial.get("verdict_class") in {"positive", "circular_positive", "null"}
            and trial.get("flagged_adversarial") is False
            and trial.get("required_checks_passed") is True
        )
        optional["disposition"] = (
            "eligible_report_separately"
            if optional["eligible"]
            else "optional_present_ineligible_report_separately"
        )
    else:
        hashes[EXP7399_PATH.as_posix()] = None
    checks.append(
        gate_row(
            "optional_exp7399_state",
            "optional_context",
            "eligible_or_absent",
            optional["disposition"],
            passed=not optional_available or optional["eligible"],
        )
    )

    texts = [str(row["step_text"]) for row in corpus[:128]] if corpus_ok else []
    context: JsonDict = {
        "fixed_service": {
            "eligible": mandatory_training and checkpoint_ok and corpus_ok,
            "source_experiment": "Exp7385",
            "source_verdict_class": training.get("verdict_class"),
            "arm": checkpoint.get("arm"),
            "seed": checkpoint.get("seed"),
            "source_text_count": len(texts),
            "source_text_fixture_sha256": canonical_hash(texts),
        },
        "optional_adaptive_service": optional,
        "hardware_history": {
            "source": EXP7393_PATH.as_posix(),
            "artifact": hardware,
            "original_transcript_hashes": original_hashes,
            "original_transcripts_authenticated": hardware_ok,
        },
        "gatemate_changed_state": physical,
    }
    return checks, hashes, context


def _unit_state(training: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> JsonDict:
    """Join the selected weights to their retained affine state and policy."""

    matches = [
        row
        for row in training.get("training_runs") or []
        if isinstance(row, Mapping)
        and row.get("arm") == "natural_prevalence_bernoulli_gibbs"
        and row.get("seed") == 7382001
    ]
    if len(matches) != 1:
        raise ValueError("selected Exp7385 training state is not unique")
    run = matches[0]
    state = {
        "arm": "natural_prevalence_bernoulli_gibbs",
        "seed": 7382001,
        "weights": deepcopy(checkpoint.get("weights")),
        "affine": {
            "slope": float(run.get("affine", {}).get("slope")),
            "intercept": float(run.get("affine", {}).get("intercept")),
        },
        "policy": deepcopy(run.get("selected_policy")),
        "model_version": "exp7385-natural-prevalence-bernoulli-gibbs-seed7382001",
    }
    _weight_arrays(state["weights"])
    return state


def load_fixed_fixture(root: Path) -> JsonDict:
    """Load the exact 128 texts and selected numeric Exp7385 service state."""

    training = load_json(root / EXP7385_PATH)
    checkpoint = load_json(root / CHECKPOINT_PATH)
    corpus = load_json(root / CORPUS_PATH)
    if not isinstance(corpus, list) or len(corpus) < 128:
        raise ValueError("fixed source text corpus is incomplete")
    records = [
        {
            "source_row_index": index,
            "question_id": str(row.get("question_id")),
            "step_text": str(row.get("step_text")),
        }
        for index, row in enumerate(corpus[:128])
        if isinstance(row, Mapping) and isinstance(row.get("step_text"), str)
    ]
    if len(records) != 128:
        raise ValueError("fixed source texts must contain 128 valid rows")
    texts = [row["step_text"] for row in records]
    return {
        "records": records,
        "texts": texts,
        "texts_sha256": canonical_hash(records),
        "state": _unit_state(training, checkpoint),
    }


def _weight_arrays(weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Validate and return the fixed 2-4-1 float64 numeric state."""

    if not isinstance(weights, Mapping) or set(weights) != {"w1", "b1", "w_out", "b_out"}:
        raise ValueError("weights must contain the fixed 2-4-1 fields")
    w1 = np.asarray(weights["w1"], dtype=np.float64)
    b1 = np.asarray(weights["b1"], dtype=np.float64)
    w_out = np.asarray(weights["w_out"], dtype=np.float64)
    b_out = float(weights["b_out"])
    if w1.shape != (4, 2) or b1.shape != (4,) or w_out.shape != (4,):
        raise ValueError("weights must have 2-4-1 shape")
    if not all(np.all(np.isfinite(value)) for value in (w1, b1, w_out)) or not math.isfinite(b_out):
        raise ValueError("weights must be finite")
    return w1, b1, w_out, b_out


def scalar_energy(weights: Mapping[str, Any], features: Sequence[float]) -> float:
    """Score one feature pair through NumPy operations on one row."""

    w1, b1, w_out, b_out = _weight_arrays(weights)
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("features must have shape 2 and finite values")
    linear = w1 @ values + b1
    hidden = linear / (1.0 + np.exp(-linear))
    return float(w_out @ hidden + b_out)


def vectorized_energy(weights: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    """Score a complete batch with one float64 NumPy matrix operation."""

    w1, b1, w_out, b_out = _weight_arrays(weights)
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("feature matrix must have shape batch by 2")
    linear = values @ w1.T + b1
    hidden = linear / (1.0 + np.exp(-linear))
    return hidden @ w_out + b_out


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    """Convert a finite float64 vector to probabilities without overflow."""

    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _policy_action(probability: float, policy: Mapping[str, Any], model_version: str) -> JsonDict:
    """Apply the retained typed thresholds and disabled-action safety controls."""

    result = typed_decision(
        probability,
        accept_threshold=float(policy["accept_threshold"]),
        reject_threshold=float(policy["reject_threshold"]),
        model_version=model_version,
    )
    if result["decision"] == "accept" and policy.get("accept_enabled") is not True:
        result["decision"] = "escalate"
        result["reason"] = "accept_action_uncertified_escalation"
    if result["decision"] == "reject" and policy.get("reject_enabled") is not True:
        result["decision"] = "escalate"
        result["reason"] = "reject_action_uncertified_escalation"
    return result


def _canonical_float(value: float) -> float:
    """Round only serialized output so equivalent float64 paths return equal bytes."""

    return round(float(value), 12)


def run_service(texts: Sequence[str], state: Mapping[str, Any], *, arm: str) -> JsonDict:
    """Measure one complete raw-text to serialized-return service call."""

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("texts must be a non-empty string sequence")
    weights = state.get("weights")
    _weight_arrays(weights)
    policy = state.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be present")
    model_version = str(state.get("model_version"))
    probe = PCIBProbe()
    total_started = time.perf_counter_ns()

    phase_started = time.perf_counter_ns()
    features = np.asarray(
        [
            [
                probe.compute_entity_uptake(text, ""),
                probe.compute_falsifiability_score(text, ""),
            ]
            for text in texts
        ],
        dtype=np.float64,
    )
    feature_s = (time.perf_counter_ns() - phase_started) / 1e9

    phase_started = time.perf_counter_ns()
    if arm == "scalar_numpy":
        energies = np.asarray([scalar_energy(weights, row) for row in features])
    else:
        energies = vectorized_energy(weights, features)
    affine = state.get("affine") or {}
    logits = float(affine["slope"]) * energies + float(affine["intercept"])
    probabilities = _sigmoid_array(logits)
    float(np.sum(probabilities))  # Force materialization before the timer boundary.
    scoring_s = (time.perf_counter_ns() - phase_started) / 1e9

    phase_started = time.perf_counter_ns()
    decisions = [
        _policy_action(float(probability), policy, model_version) for probability in probabilities
    ]
    policy_s = (time.perf_counter_ns() - phase_started) / 1e9

    phase_started = time.perf_counter_ns()
    journal = [
        {
            "index": index,
            "features": [_canonical_float(item) for item in features[index]],
            "energy": _canonical_float(energies[index]),
            "probability": _canonical_float(probabilities[index]),
            "action": decisions[index]["decision"],
        }
        for index in range(len(texts))
    ]
    durable_state = {
        "arm": state.get("arm"),
        "seed": state.get("seed"),
        "weights_sha256": canonical_hash(weights),
        "affine": deepcopy(dict(affine)),
        "policy": deepcopy(dict(policy)),
        "update_count": 0,
    }
    journal_bytes = json.dumps(journal, sort_keys=True, separators=(",", ":")).encode()
    state_bytes = json.dumps(durable_state, sort_keys=True, separators=(",", ":")).encode()
    serialization_s = (time.perf_counter_ns() - phase_started) / 1e9

    phase_started = time.perf_counter_ns()
    return_value = {
        "journal": journal,
        "state": durable_state,
        "actions": [row["decision"] for row in decisions],
    }
    return_bytes = json.dumps(return_value, sort_keys=True, separators=(",", ":")).encode()
    return_payload = json.loads(return_bytes)
    return_s = (time.perf_counter_ns() - phase_started) / 1e9
    complete_s = (time.perf_counter_ns() - total_started) / 1e9
    return {
        "arm": arm,
        "batch_size": len(texts),
        "features": features.tolist(),
        "energies": energies.tolist(),
        "probabilities": probabilities.tolist(),
        "actions": [row["decision"] for row in decisions],
        "return_payload": return_payload,
        "return_sha256": "sha256:" + __import__("hashlib").sha256(return_bytes).hexdigest(),
        "stage_durations_s": {
            "raw_text_to_pcib_features": feature_s,
            "numeric_energy_probability": scoring_s,
            "typed_policy": policy_s,
            "journal_state_serialization": serialization_s,
            "complete_return_materialization": return_s,
        },
        "complete_return_s": complete_s,
        "measured_bytes": {
            "raw_text_utf8": sum(len(text.encode()) for text in texts),
            "feature_float64": int(features.nbytes),
            "journal_json_utf8": len(journal_bytes),
            "state_json_utf8": len(state_bytes),
            "complete_return_json_utf8": len(return_bytes),
        },
    }


def compare_service_outputs(scalar: Mapping[str, Any], vector: Mapping[str, Any]) -> JsonDict:
    """Compare both service paths against the predeclared float64 tolerance."""

    left_energy = np.asarray(scalar.get("energies"), dtype=np.float64)
    right_energy = np.asarray(vector.get("energies"), dtype=np.float64)
    left_probability = np.asarray(scalar.get("probabilities"), dtype=np.float64)
    right_probability = np.asarray(vector.get("probabilities"), dtype=np.float64)
    same_shape = (
        left_energy.shape == right_energy.shape
        and left_probability.shape == right_probability.shape
    )
    energy_delta = float(np.max(np.abs(left_energy - right_energy))) if same_shape else math.inf
    probability_delta = (
        float(np.max(np.abs(left_probability - right_probability))) if same_shape else math.inf
    )
    actions_identical = scalar.get("actions") == vector.get("actions")
    return {
        "tolerance": FLOAT64_TOLERANCE,
        "energy_max_abs_delta": energy_delta,
        "probability_max_abs_delta": probability_delta,
        "actions_identical": actions_identical,
        "passed": same_shape
        and energy_delta <= FLOAT64_TOLERANCE
        and probability_delta <= FLOAT64_TOLERANCE
        and actions_identical,
    }


def threshold_boundary_rows(policy: Mapping[str, Any]) -> list[JsonDict]:
    """Check exact and adjacent float64 values at both action thresholds."""

    rows: list[JsonDict] = []
    for name in ("accept", "reject"):
        threshold = float(policy[f"{name}_threshold"])
        for relation, probability in (
            ("below", float(np.nextafter(threshold, -math.inf))),
            ("at", threshold),
            ("above", float(np.nextafter(threshold, math.inf))),
        ):
            scalar = _policy_action(probability, policy, "boundary-fixture")
            vector = _policy_action(float(np.asarray([probability])[0]), policy, "boundary-fixture")
            rows.append(
                {
                    "threshold": name,
                    "relation": relation,
                    "probability": probability,
                    "scalar_action": scalar["decision"],
                    "vectorized_action": vector["decision"],
                    "passed": scalar["decision"] == vector["decision"],
                }
            )
    return rows


def _row_hash(row: Mapping[str, Any]) -> str:
    """Bind one complete row while excluding its self-reference."""

    return canonical_hash({key: value for key, value in row.items() if key != "row_sha256"})


def _timing_row(
    result: Mapping[str, Any],
    *,
    batch_size: int,
    block: int,
    pair_order: str,
    execution_position: int,
    parity: Mapping[str, Any],
) -> JsonDict:
    """Reduce one service result to a complete immutable measurement row."""

    row: JsonDict = {
        "unit_id": f"batch:{batch_size}:block:{block}:arm:{result['arm']}",
        "row_type": "service_cost",
        "batch_size": batch_size,
        "block": block,
        "arm": result["arm"],
        "pair_order": pair_order,
        "execution_position": execution_position,
        "stage_durations_s": deepcopy(result["stage_durations_s"]),
        "complete_return_s": float(result["complete_return_s"]),
        "measured_bytes": deepcopy(result["measured_bytes"]),
        "return_sha256": result["return_sha256"],
        "energy_max_abs_delta": parity["energy_max_abs_delta"],
        "probability_max_abs_delta": parity["probability_max_abs_delta"],
        "actions_identical": parity["actions_identical"],
        "parity_tolerance": parity["tolerance"],
        "parity_passed": parity["passed"],
        "censored": False,
        "error": None,
    }
    row["row_sha256"] = _row_hash(row)
    return row


def measure_service_cost(
    texts: Sequence[str],
    state: Mapping[str, Any],
    *,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    blocks: int = PAIRED_BLOCKS,
    seed: int = RANDOM_SEED,
) -> list[JsonDict]:
    """Run warmed paired blocks with a fixed alternating order."""

    if blocks <= 0 or any(size <= 0 or size > len(texts) for size in batch_sizes):
        raise ValueError("benchmark batch and block limits are invalid")
    rng = np.random.default_rng(seed)
    rows: list[JsonDict] = []
    benchmark_started = time.monotonic()
    for batch_size in batch_sizes:
        batch = list(texts[:batch_size])
        print(
            f"[exp7407] phase=warmup event=before batch={batch_size} elapsed_s={time.monotonic() - benchmark_started:.3f}",
            flush=True,
        )
        for arm in ARMS:
            run_service(batch, state, arm=arm)
        print(
            f"[exp7407] phase=warmup event=after batch={batch_size} elapsed_s={time.monotonic() - benchmark_started:.3f}",
            flush=True,
        )
        scalar_first = bool(rng.integers(0, 2))
        for block in range(blocks):
            first = "scalar_numpy" if scalar_first == (block % 2 == 0) else "vectorized_numpy"
            second = "vectorized_numpy" if first == "scalar_numpy" else "scalar_numpy"
            order = (first, second)
            results: dict[str, JsonDict] = {}
            print(
                f"[exp7407] phase=benchmark event=before_pair batch={batch_size} block={block} "
                f"completed={len(rows)} elapsed_s={time.monotonic() - benchmark_started:.3f}",
                flush=True,
            )
            for arm in order:
                results[arm] = run_service(batch, state, arm=arm)
            parity = compare_service_outputs(results["scalar_numpy"], results["vectorized_numpy"])
            pair_order = f"{first}_then_{second}"
            for position, arm in enumerate(order):
                rows.append(
                    _timing_row(
                        results[arm],
                        batch_size=batch_size,
                        block=block,
                        pair_order=pair_order,
                        execution_position=position,
                        parity=parity,
                    )
                )
            print(
                f"[exp7407] phase=benchmark event=after_pair batch={batch_size} block={block} "
                f"completed={len(rows)} elapsed_s={time.monotonic() - benchmark_started:.3f}",
                flush=True,
            )
    return rows


def synthetic_service_row(
    *, batch_size: int, block: int, arm: str, scoring_s: float, complete_s: float
) -> JsonDict:
    """Build a complete timing row for deterministic reducer tests."""

    if arm not in ARMS or complete_s <= scoring_s or scoring_s <= 0:
        raise ValueError("synthetic timing values are invalid")
    remainder = complete_s - scoring_s
    stages = {
        "raw_text_to_pcib_features": remainder * 0.6,
        "numeric_energy_probability": scoring_s,
        "typed_policy": remainder * 0.1,
        "journal_state_serialization": remainder * 0.2,
        "complete_return_materialization": remainder * 0.1,
    }
    first = "scalar_numpy" if block % 2 == 0 else "vectorized_numpy"
    second = "vectorized_numpy" if first == "scalar_numpy" else "scalar_numpy"
    row: JsonDict = {
        "unit_id": f"batch:{batch_size}:block:{block}:arm:{arm}",
        "row_type": "service_cost",
        "batch_size": batch_size,
        "block": block,
        "arm": arm,
        "pair_order": f"{first}_then_{second}",
        "execution_position": int(arm == second),
        "stage_durations_s": stages,
        "complete_return_s": complete_s,
        "measured_bytes": {
            "raw_text_utf8": batch_size * 64,
            "feature_float64": batch_size * 16,
            "journal_json_utf8": batch_size * 96,
            "state_json_utf8": 256,
            "complete_return_json_utf8": batch_size * 112,
        },
        "return_sha256": "sha256:" + ("1" if arm == "scalar_numpy" else "2") * 64,
        "energy_max_abs_delta": 0.0,
        "probability_max_abs_delta": 0.0,
        "actions_identical": True,
        "parity_tolerance": FLOAT64_TOLERANCE,
        "parity_passed": True,
        "censored": False,
        "error": None,
    }
    row["row_sha256"] = _row_hash(row)
    return row


def _bootstrap_ratio(
    numerator: np.ndarray, denominator: np.ndarray, *, draws: int, seed: int
) -> JsonDict:
    """Resample paired blocks and retain the ratio-of-means interval."""

    if numerator.shape != denominator.shape or numerator.ndim != 1 or not len(numerator):
        raise ValueError("paired ratio inputs must have one common non-empty shape")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(numerator), size=(draws, len(numerator)))
    ratios = numerator[indexes].mean(axis=1) / denominator[indexes].mean(axis=1)
    return {
        "estimate": float(numerator.mean() / denominator.mean()),
        "interval_95": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))],
        "draws": draws,
        "seed": seed,
        "effective_independent_group_count": len(numerator),
    }


def summarize_cost_rows(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = RANDOM_SEED
) -> JsonDict:
    """Reduce paired scoring and complete-service ratios by batch size."""

    batch_rows: list[JsonDict] = []
    for batch_size in sorted({int(row.get("batch_size", 0)) for row in rows}):
        selected = [row for row in rows if row.get("batch_size") == batch_size]
        blocks = sorted({int(row.get("block", -1)) for row in selected})
        scalar: list[Mapping[str, Any]] = []
        vector: list[Mapping[str, Any]] = []
        for block in blocks:
            pair = [row for row in selected if row.get("block") == block]
            left = [row for row in pair if row.get("arm") == "scalar_numpy"]
            right = [row for row in pair if row.get("arm") == "vectorized_numpy"]
            if len(left) != 1 or len(right) != 1:
                raise ValueError("paired rows require exactly one row per arm and block")
            scalar.append(left[0])
            vector.append(right[0])
        scalar_scoring = np.asarray(
            [row["stage_durations_s"]["numeric_energy_probability"] for row in scalar]
        )
        vector_scoring = np.asarray(
            [row["stage_durations_s"]["numeric_energy_probability"] for row in vector]
        )
        scalar_complete = np.asarray([row["complete_return_s"] for row in scalar])
        vector_complete = np.asarray([row["complete_return_s"] for row in vector])
        scoring_ratio = _bootstrap_ratio(
            scalar_scoring, vector_scoring, draws=draws, seed=seed + batch_size
        )
        complete_ratio = _bootstrap_ratio(
            scalar_complete, vector_complete, draws=draws, seed=seed + 10_000 + batch_size
        )
        batch_rows.append(
            {
                "batch_size": batch_size,
                "paired_block_count": len(blocks),
                "scoring_ratio_scalar_over_vector": scoring_ratio,
                "complete_ratio_scalar_over_vector": complete_ratio,
                "kernel_faster": scoring_ratio["estimate"] > 1.0,
                "full_service_benefit": complete_ratio["interval_95"][0] > 1.0,
            }
        )
    if not batch_rows:
        raise ValueError("paired rows are required")
    primary = max(batch_rows, key=lambda row: row["batch_size"])
    if primary["full_service_benefit"]:
        verdict = "complete_positive_vectorized_full_service_benefit"
        verdict_class = "positive"
    elif primary["kernel_faster"]:
        verdict = "complete_null_faster_kernel_without_full_service_benefit"
        verdict_class = "null"
    else:
        verdict = "complete_null_no_vectorized_scoring_or_full_service_benefit"
        verdict_class = "null"
    return {
        "primary_batch_size": primary["batch_size"],
        "batch_rows": batch_rows,
        "honest_cost_verdict": verdict,
        "cost_verdict_class": verdict_class,
    }


def compute_amdahl(rows: Sequence[Mapping[str, Any]], *, batch_size: int, arm: str) -> JsonDict:
    """Compute measured stage fractions and bounded scoring acceleration."""

    selected = [
        row for row in rows if row.get("batch_size") == batch_size and row.get("arm") == arm
    ]
    if not selected:
        raise ValueError("rows for the selected Amdahl service are required")
    complete = float(np.mean([row["complete_return_s"] for row in selected]))
    means = {
        name: float(np.mean([row["stage_durations_s"][name] for row in selected]))
        for name in STAGE_NAMES
    }
    fractions = {name: value / complete for name, value in means.items()}
    scoring = fractions["numeric_energy_probability"]
    unaccelerated = 1.0 - scoring
    finite = 1.0 / (unaccelerated + scoring / 100.0)
    infinite = math.inf if unaccelerated <= 0 else 1.0 / unaccelerated
    necessary = unaccelerated <= 0.01
    return {
        "batch_size": batch_size,
        "arm": arm,
        "mean_complete_service_s": complete,
        "mean_stage_durations_s": means,
        "measured_stage_fractions": fractions,
        "scoring_fraction": scoring,
        "unaccelerated_fraction": unaccelerated,
        "assumed_device_scoring_rate": 100.0,
        "device_rate_is_assumption": True,
        "finite_speedup_at_assumed_100x": finite,
        "infinite_device_upper_bound": infinite,
        "hundred_x_necessary_condition": {
            "operator": "<=",
            "expected": 0.01,
            "observed": unaccelerated,
            "passed": necessary,
        },
        "recommendation": (
            "eligible_for_measured_device_trial" if necessary else "retain_cpu_or_batch_before_port"
        ),
    }


def build_board_rows(context: Mapping[str, Any]) -> list[JsonDict]:
    """Append three board dispositions without broadening old claims."""

    hardware = context.get("hardware_history", {}).get("artifact") or {}
    physical = context.get("gatemate_changed_state") or {}
    rows = placement.build_board_rows(hardware, physical)
    for row in rows:
        row["hardware_ready_score"] = 0
        row["hardware_value_score"] = 0
        row["fresh_physical_attempt"] = False
        row["row_sha256"] = _row_hash(row)
    return rows


def device_paths() -> list[JsonDict]:
    """State measured CPU work and only conditional future device paths."""

    return [
        {
            "device": "CPU",
            "role": "current PCIB counters, float64 scoring, and optional affine updates",
            "current_measured_path": True,
            "rate_class": "measured_host_service",
            "operation_performed": True,
        },
        {
            "device": "GPU",
            "role": "possible future batched dense scoring",
            "current_measured_path": False,
            "rate_class": "assumption_until_measured",
            "operation_performed": False,
        },
        {
            "device": "NPU",
            "role": "possible future batched dense scoring",
            "current_measured_path": False,
            "rate_class": "assumption_until_measured",
            "operation_performed": False,
        },
        {
            "device": "sparse proof-check FPGA",
            "role": "possible future sparse proof checking, not this dense decision head",
            "current_measured_path": False,
            "rate_class": "assumption_until_measured",
            "operation_performed": False,
        },
    ]


def _rows_valid(rows: Sequence[Any]) -> bool:
    """Require every retained row hash to bind its complete content."""

    return bool(rows) and all(
        isinstance(row, Mapping) and row.get("row_sha256") == _row_hash(row) for row in rows
    )


def _service_rows_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require all 180 paired units, stages, bytes, and parity results."""

    identities = {(row.get("batch_size"), row.get("block"), row.get("arm")) for row in rows}
    expected = {
        (batch, block, arm)
        for batch in BATCH_SIZES
        for block in range(PAIRED_BLOCKS)
        for arm in ARMS
    }
    return (
        len(rows) == 180
        and identities == expected
        and _rows_valid(rows)
        and all(row.get("parity_passed") is True for row in rows)
        and all(set(row.get("stage_durations_s") or {}) == set(STAGE_NAMES) for row in rows)
        and all(float(row.get("complete_return_s", 0.0)) > 0.0 for row in rows)
        and all(
            int((row.get("measured_bytes") or {}).get("complete_return_json_utf8", 0)) > 0
            for row in rows
        )
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the expected GateMate block and any whole-task failures separately."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    external = next(
        (row for row in failures if row.get("check") == "gatemate_changed_physical_state_receipt"),
        None,
    )
    whole = [
        row
        for row in failures
        if row.get("category")
        in {"mandatory_precondition", "completion", "safety", "required_validation"}
    ]
    return {
        "check_count": len(gates),
        "failed_count": len(failures),
        "checks": [deepcopy(dict(row)) for row in gates],
        "failures": failures,
        "external_board_block": external,
        "whole_task_blocker": whole[0] if whole else None,
        "passed_for_cpu_cost": not whole,
    }


def _required_validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require the frozen eight affected checks and every supplied terminal check."""

    affected = validation_scope.reduce_required_checks(receipts)
    terminal = [
        row
        for row in receipts
        if row.get("name")
        in {
            "declared_entrypoint_cold_replay",
            "independent_cold_recompute",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        }
    ]
    return affected["required_checks_passed"] and all(row.get("passed") is True for row in terminal)


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain each ordinary artifact field without value wrappers."""

    specific = {
        "schema": "Versioned schema; ordinary identity fields become terminal only after measured work and checks.",
        "run_date": "Use 20260919 and retain actual UTC start and end timestamps.",
        "preconditions_checked": "Record exact input hashes, eligibility, runtime, device, and entrypoint checks before dependent work.",
        "MODEL_SPECS": "Current LLM work uses model specifications; this no-LLM task uses an empty list.",
        "model_invoked": "A current attempted LLM load or generation sets this true; this task performs none.",
        "invocation_counts": "Count only current owned attempted, terminal, and in-flight LLM events.",
        "inference_substrate": "Describe current CPU and software work in one truthful string.",
        "inference_substrate_class": "Use the exact no_model_load class without runtime padding.",
        "execution_venue": "Use the closed string host.",
        "duration_s": "Measure current task duration with a monotonic clock.",
        "phase_spans": "Retain actual phase boundaries, checkpoints, and completed units.",
        "random_seed": "Freeze benchmark order and paired resampling at seed 7407307.",
        "reproducibility_checksum": "Bind current code, sources, protocol, raw rows, gates, and scores.",
        "source_artifact_hashes": "Retain exact byte hashes and absent optional inputs without importing historical counters.",
        "rows": "Retain every service arm, block, batch, cost, parity result, and board disposition.",
        "sample_size_budget": "Declare planned, attempted, completed, censored, unstarted, limits, stop rule, and paired groups.",
        "acceptance_gate_results": "Separate validation, safety, completion, efficacy, and external board checks.",
        "gate_check_summary": "Name the exact GateMate changed-state absence and any separate whole-task blocker.",
        "verifier_is_oracle": "Numerical parity is independent of labels and does not make the learned score an oracle.",
        "honest_verdict": "Completed findings start complete_; unchanged board absence remains a row-level block.",
        "verdict_class": "Use one closed terminal class without converting an efficacy null into missing work.",
        "flagged_adversarial": "Preserve critical terminal-reader findings; invalid science supplies no readiness.",
        "validation_receipts": "Retain argv, environment, required name, exit, duration, and exact log hash.",
        "repository_health": "Keep unrelated broad-suite history separate while affected failures remain disqualifying.",
        "field_principles": "Explain fields directly without wrapping their ordinary values.",
        "promotion_score": "Remain zero; this task performs no rollout, publication, or weight update.",
        "service_cost_capture_complete_score": "Equal one for complete paired host timing and parity, independent of value and GateMate state.",
        "service_cost_rows": "Retain block, batch, arm, every stage, complete wall time, measured bytes, and parity.",
        "board_rows": "Retain independent dated KV260, GateMate, and PolarFire states.",
        "hardware_ready_score": "Remain zero because this task performs no fresh physical qualification.",
        "hardware_value_score": "Remain zero because host timing is not measured hardware acceleration.",
    }
    return {
        key: specific.get(key, f"Retain the ordinary {key} evidence for exact replay.")
        for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable scientific inputs, raw rows, reductions, gates, and scores."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "random_seed",
        "source_artifact_hashes",
        "benchmark_protocol",
        "service_cost_rows",
        "service_cost_summary",
        "threshold_boundary_rows",
        "amdahl_analysis",
        "board_rows",
        "acceptance_gate_results",
        "verdict_class",
        "honest_verdict",
        "service_cost_capture_complete_score",
        "hardware_ready_score",
        "hardware_value_score",
        "promotion_score",
    )
    return canonical_hash({key: deepcopy(artifact.get(key)) for key in keys})


def _sample_budget(service_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Describe the frozen paired budget without treating rows as independent calls."""

    completed = len(service_rows)
    return {
        "planned_service_rows": 180,
        "attempted_service_rows": completed,
        "completed_service_rows": completed,
        "censored_service_rows": sum(row.get("censored") is True for row in service_rows),
        "unstarted_service_rows": max(0, 180 - completed),
        "planned_paired_blocks": 90,
        "effective_independent_group_count": len(
            {(row.get("batch_size"), row.get("block")) for row in service_rows}
        ),
        "batch_size_limits": list(BATCH_SIZES),
        "blocks_per_batch": PAIRED_BLOCKS,
        "warmup_calls": len(BATCH_SIZES) * len(ARMS),
        "stop_rule": "complete 30 paired alternating blocks for each frozen batch, then stop without duration padding",
    }


def _fixture_receipts() -> list[JsonDict]:
    """Build passing bounded receipts for pure artifact mutation tests."""

    return [
        {
            "name": name,
            "command": f"fixture {name}",
            "command_argv": ["fixture", name],
            "command_environment": {"COVERAGE_FILE": "/tmp/exp7407.coverage"},
            "scope": "fixture",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "1" * 64,
            "passed": True,
            "timed_out": False,
            "required": True,
            "command_category": "required_validation",
        }
        for name in validation_scope.REQUIRED_CHECK_NAMES
    ]


def _fixture_board_rows() -> list[JsonDict]:
    """Create the three exact terminal classes for schema mutation tests."""

    rows = [
        {
            "unit_id": "board:KV260",
            "row_type": "board_disposition",
            "board": "KV260",
            "terminal_state": "graduated_preserved",
            "last_authenticated_date": "20260915",
            "last_authenticated_venue": "kv260_fpga_fabric",
            "future_access": "ssh_only",
            "architecture_limit": "k_max<=5",
            "fpga_sampling_claimed": True,
            "hash_matched_cpu_dispatch": False,
            "new_hardware_execution_claimed": False,
            "fresh_physical_attempt": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
            "error": None,
        },
        {
            "unit_id": "board:GateMate",
            "row_type": "board_disposition",
            "board": "GateMate",
            "terminal_state": "blocked_changed_physical_state",
            "last_authenticated_date": "20260916",
            "last_authenticated_venue": "none_read_only",
            "future_access": None,
            "architecture_limit": None,
            "fpga_sampling_claimed": False,
            "hash_matched_cpu_dispatch": False,
            "new_hardware_execution_claimed": False,
            "fresh_physical_attempt": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
            "error": MISSING_RECEIPT,
        },
        {
            "unit_id": "board:PolarFire",
            "row_type": "board_disposition",
            "board": "PolarFire",
            "terminal_state": "graduated_cpu_dispatch_preserved",
            "last_authenticated_date": "20260915",
            "last_authenticated_venue": "polarfire_linux_cpu",
            "future_access": None,
            "architecture_limit": None,
            "fpga_sampling_claimed": False,
            "hash_matched_cpu_dispatch": True,
            "new_hardware_execution_claimed": False,
            "fresh_physical_attempt": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
            "error": None,
        },
    ]
    for row in rows:
        row["row_sha256"] = _row_hash(row)
    return rows


def assemble_artifact(
    *,
    service_rows: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    historical_sidecars: Sequence[Mapping[str, Any]],
    optional_adaptive_service: Mapping[str, Any],
) -> JsonDict:
    """Build one ordinary terminal artifact from raw measured evidence."""

    services = [deepcopy(dict(row)) for row in service_rows]
    boards = [deepcopy(dict(row)) for row in board_rows]
    complete = _service_rows_complete(services)
    mandatory_passed = all(
        row.get("passed") is True
        for row in preconditions
        if row.get("category") == "mandatory_precondition"
    )
    validation_passed = _required_validation_passed(validation_receipts)
    if complete:
        summary = summarize_cost_rows(services)
        amdahl = compute_amdahl(services, batch_size=max(BATCH_SIZES), arm="vectorized_numpy")
    else:
        summary = {
            "primary_batch_size": 128,
            "batch_rows": [],
            "honest_cost_verdict": "blocked_fixed_service_fixture_unavailable",
            "cost_verdict_class": "blocked",
        }
        amdahl = None
    gates = [deepcopy(dict(row)) for row in preconditions]
    gates.extend(
        [
            gate_row(
                "complete_paired_service_rows", "completion", 180, len(services), passed=complete
            ),
            gate_row(
                "float64_and_action_parity",
                "safety",
                True,
                complete and all(row.get("parity_passed") is True for row in services),
            ),
            gate_row(
                "required_affected_and_terminal_validation",
                "required_validation",
                True,
                validation_passed,
            ),
            gate_row(
                "paired_full_service_benefit",
                "scientific_efficacy",
                True,
                any(row.get("full_service_benefit") is True for row in summary["batch_rows"]),
            ),
        ]
    )
    if not mandatory_passed:
        status = "blocked"
        verdict_class = "blocked"
        honest = "blocked_fixed_service_fixture_unavailable"
    elif not complete or not validation_passed:
        status = "complete"
        verdict_class = "disqualified"
        honest = "complete_disqualified_incomplete_service_or_required_validation"
    else:
        status = "complete"
        verdict_class = str(summary["cost_verdict_class"])
        honest = str(summary["honest_cost_verdict"])
    boundaries = threshold_boundary_rows(
        {
            "accept_threshold": 0.05,
            "reject_threshold": 0.99,
            "accept_enabled": True,
            "reject_enabled": False,
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "host CPU PCIB text parsing and float64 NumPy scoring; no JAX compilation, exact solver, LLM, or board operation",
        "inference_substrate_details": {
            "device": "CPU",
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax_used": False,
            "exact_solver_used": False,
            "synchronization": "NumPy result materialized before each scoring timer ends",
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "paired_order_seed": RANDOM_SEED,
            "paired_resampling_seed": RANDOM_SEED,
        },
        "source_artifact_hashes": dict(source_hashes),
        "historical_receipt_sidecars": [deepcopy(dict(row)) for row in historical_sidecars],
        "small_ebm_training": {
            "performed": False,
            "source": "fixed clean Exp7385 numeric checkpoint",
            "llm_load_implied": False,
        },
        "benchmark_protocol": {
            "batch_sizes": list(BATCH_SIZES),
            "paired_blocks_per_batch": PAIRED_BLOCKS,
            "arms": list(ARMS),
            "order": "fixed alternating pair order after seeded initial arm",
            "warmup": "one complete service call per arm and batch before timing",
            "float64_tolerance": FLOAT64_TOLERANCE,
            "full_service_boundary": "raw text through serialized complete return",
        },
        "service_cost_rows": services,
        "service_cost_summary": summary,
        "threshold_boundary_rows": boundaries,
        "amdahl_analysis": amdahl,
        "board_rows": boards,
        "board_disposition": {
            "complete": len(boards) == 3,
            "gatemate_terminal_state": next(
                (row.get("terminal_state") for row in boards if row.get("board") == "GateMate"),
                None,
            ),
            "independent_of_cpu_cost_verdict": True,
        },
        "optional_adaptive_service": deepcopy(dict(optional_adaptive_service)),
        "device_paths": device_paths(),
        "hardware_operations": {
            "ssh": 0,
            "flash": 0,
            "install": 0,
            "purchase": 0,
            "download": 0,
            "vendor_contact": 0,
            "physical_attempt": 0,
        },
        "rows": [*services, *boards],
        "sample_size_budget": _sample_budget(services),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": any(
            row.get("name") == "adversarial_verify" and row.get("passed") is not True
            for row in validation_receipts
        ),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "required_checks_passed": validation_passed,
        "repository_health": validation_scope.build_repository_health([]),
        "promotion_score": 0,
        "service_cost_capture_complete_score": int(complete),
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_fixture_artifact() -> JsonDict:
    """Build a complete null artifact small enough for mutation-oriented tests."""

    services = [
        synthetic_service_row(
            batch_size=batch,
            block=block,
            arm=arm,
            scoring_s=0.004 if arm == "scalar_numpy" else 0.001,
            complete_s=0.010 if arm == "scalar_numpy" else 0.0105,
        )
        for batch in BATCH_SIZES
        for block in range(PAIRED_BLOCKS)
        for arm in ARMS
    ]
    preconditions = [
        gate_row("fixture", "mandatory_precondition", True, True),
        gate_row(
            "gatemate_changed_physical_state_receipt",
            "external_prerequisite",
            deepcopy(board_history.PHYSICAL_RECEIPT_CONTRACT),
            {"absence": MISSING_RECEIPT},
            passed=False,
        ),
    ]
    return assemble_artifact(
        service_rows=services,
        board_rows=_fixture_board_rows(),
        preconditions=preconditions,
        source_hashes={"fixture": "sha256:" + "2" * 64},
        validation_receipts=_fixture_receipts(),
        phase_spans=[
            {
                "phase": "fixture",
                "started_monotonic_offset_s": 0.0,
                "ended_monotonic_offset_s": 0.1,
                "duration_s": 0.1,
                "completed_units": 180,
                "checkpoint_at_utc": "2026-09-19T00:00:00+00:00",
            }
        ],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:01+00:00",
        duration_s=1.0,
        historical_sidecars=[
            {
                "path": "fixture-sidecar.json",
                "sha256": "sha256:" + "3" * 64,
                "scope": "historical_model_receipts",
            }
        ],
        optional_adaptive_service={
            "source_experiment": "Exp7399",
            "available": False,
            "eligible": False,
            "disposition": "optional_absent_does_not_block_fixed_service",
        },
    )


def validate_artifact(value: object) -> list[str]:
    """Reject altered rows, false parity, promoted hardware, and schema drift."""

    artifact = value if isinstance(value, Mapping) else {}
    services = (
        artifact.get("service_cost_rows")
        if isinstance(artifact.get("service_cost_rows"), list)
        else []
    )
    boards = artifact.get("board_rows") if isinstance(artifact.get("board_rows"), list) else []
    complete = _service_rows_complete(services)
    blocked = artifact.get("verdict_class") == "blocked"
    current_counts = artifact.get("invocation_counts")
    failures = {
        "required_fields": not REQUIRED_FIELDS.issubset(artifact),
        "identity": artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "model_declaration": artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or current_counts != ZERO_INVOCATION_COUNTS,
        "substrate": not isinstance(artifact.get("inference_substrate"), str)
        or artifact.get("inference_substrate_class") != "no_model_load"
        or artifact.get("execution_venue") != "host",
        "service_rows": (not blocked and not complete)
        or (blocked and bool(services))
        or artifact.get("service_cost_capture_complete_score") != int(complete),
        "boundary_parity": not blocked
        and (
            len(artifact.get("threshold_boundary_rows") or []) != 6
            or not all(
                row.get("passed") is True for row in artifact.get("threshold_boundary_rows") or []
            )
        ),
        "board_rows": len(boards) != 3
        or {row.get("board") for row in boards if isinstance(row, Mapping)}
        != {"KV260", "GateMate", "PolarFire"}
        or not _rows_valid(boards),
        "rows": artifact.get("rows") != [*services, *boards],
        "scores": artifact.get("promotion_score") != 0
        or artifact.get("hardware_ready_score") != 0
        or artifact.get("hardware_value_score") != 0,
        "hardware_operations": any(
            value != 0 for value in (artifact.get("hardware_operations") or {}).values()
        ),
        "verdict": artifact.get("verdict_class") not in validation_contract.CLOSED_VERDICTS
        or not str(artifact.get("honest_verdict", "")).startswith(("complete_", "blocked_")),
        "principles": set(artifact.get("field_principles") or {}) != set(artifact),
        "checksum": artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
    }
    return [name for name, failed in failures.items() if failed]


def independent_reduce(artifact: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, bool]:
    """Independently recompute raw-row equality, reductions, and source identity."""

    raw_services = raw.get("service_cost_rows") or []
    raw_boards = raw.get("board_rows") or []
    services_match = artifact.get("service_cost_rows") == raw_services
    boards_match = artifact.get("board_rows") == raw_boards
    summary_match = False
    amdahl_match = False
    if raw_services:
        summary_match = artifact.get("service_cost_summary") == summarize_cost_rows(raw_services)
        amdahl_match = artifact.get("amdahl_analysis") == compute_amdahl(
            raw_services, batch_size=128, arm="vectorized_numpy"
        )
    return {
        "service_rows_match": services_match,
        "board_rows_match": boards_match,
        "all_rows_match": artifact.get("rows") == [*raw_services, *raw_boards],
        "summary_matches_raw_rows": summary_match,
        "amdahl_matches_raw_rows": amdahl_match,
        "source_hashes_match": artifact.get("source_artifact_hashes")
        == raw.get("source_artifact_hashes"),
        "row_hashes_valid": _rows_valid(raw_services) and _rows_valid(raw_boards),
        "artifact_valid": validate_artifact(artifact) == [],
    }


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Atomically publish only a locally valid terminal artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the actual Exp7358 command plan before execution."""

    return validation_contract.build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command drift, broad pytest, and missing private parents."""

    return validation_contract.validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _span(
    phase: str, started: float, origin: float, units: int, started_at_utc: str
) -> JsonDict:  # pragma: no cover - live monotonic receipt.
    """Record one actual phase boundary without duration padding."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_monotonic_offset_s": started - origin,
        "ended_monotonic_offset_s": ended - origin,
        "duration_s": ended - started,
        "started_at_utc": started_at_utc,
        "checkpoint_at_utc": utc_now(),
        "heartbeat_times_utc": [],
        "completed_units": units,
    }


def _write_historical_sidecar(
    root: Path, path: Path, hashes: Mapping[str, str | None]
) -> JsonDict:  # pragma: no cover - live sidecar boundary.
    """Keep producer metadata outside current zero invocation counters."""

    training = load_json(root / EXP7385_PATH)
    hardware = load_json(root / EXP7393_PATH)
    payload = {
        "label": "historical_inputs_do_not_authorize_current_model_work",
        "current_MODEL_SPECS": [],
        "current_model_invoked": False,
        "current_invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "sources": [
            {
                "path": EXP7385_PATH.as_posix(),
                "sha256": hashes.get(EXP7385_PATH.as_posix()),
                "verdict_class": training.get("verdict_class"),
                "flagged_adversarial": training.get("flagged_adversarial"),
                "historical_MODEL_SPECS": training.get("MODEL_SPECS"),
                "historical_model_invoked": training.get("model_invoked"),
                "historical_invocation_counts": training.get("invocation_counts"),
            },
            {
                "path": EXP7393_PATH.as_posix(),
                "sha256": hashes.get(EXP7393_PATH.as_posix()),
                "verdict_class": hardware.get("verdict_class"),
                "flagged_adversarial": hardware.get("flagged_adversarial"),
                "historical_MODEL_SPECS": hardware.get("MODEL_SPECS"),
                "historical_model_invoked": hardware.get("model_invoked"),
                "historical_invocation_counts": hardware.get("invocation_counts"),
            },
        ],
    }
    return current_work_receipt.write_immutable_sidecar(
        path,
        scope="historical_model_receipts",
        payload=payload,
        root=root,
    )


def _terminal_commands(
    paths: ExperimentPaths,
) -> list[validation_contract.PlannedCommand]:  # pragma: no cover - subprocess plan.
    """Build declared cold replay and both unchanged strict readers."""

    python = ".venv/bin/python"
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "declared_entrypoint_cold_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--cold-replay",
                    str(paths.terminal_candidate),
                    "--raw",
                    str(paths.raw_evidence),
                ),
                "capability_end_to_end",
            ),
            "completion",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "independent_cold_recompute",
                (
                    python,
                    "-u",
                    "-c",
                    "import json,pathlib,sys;from carnot.experiment_7407_v649_service_cost import independent_reduce;"
                    "a=json.loads(pathlib.Path(sys.argv[1]).read_text());r=json.loads(pathlib.Path(sys.argv[2]).read_text());"
                    "x=independent_reduce(a,r);print(x,flush=True);raise SystemExit(not all(x.values()))",
                    str(paths.terminal_candidate),
                    str(paths.raw_evidence),
                ),
                "independent_recomputation",
            ),
            "completion",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(paths.terminal_candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(paths.terminal_candidate),
                ),
                "candidate",
            ),
            "safety",
            True,
        ),
    ]


def _entrypoint_receipt(
    root: Path, started_at_utc: str, duration_s: float
) -> JsonDict:  # pragma: no cover - live invocation receipt.
    """Retain the exact declared entrypoint and scoped environment."""

    path = root / RAW_DIR / "validation/entrypoint/declared_entrypoint_e2e.json"
    value = {
        "argv": list(getattr(sys, "orig_argv", [sys.executable, *sys.argv])),
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now(),
        "duration_s": duration_s,
        "environment": {
            key: os.environ[key]
            for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            if key in os.environ
        },
    }
    atomic_json(path, value)
    return {
        "name": "declared_entrypoint_e2e",
        "command": " ".join(value["argv"]),
        "command_argv": value["argv"],
        "command_environment": value["environment"],
        "scope": "capability_end_to_end",
        "exit_code": 0,
        "duration_s": duration_s,
        "log_path": path.relative_to(root).as_posix(),
        "log_sha256": sha256_file(path),
        "passed": True,
        "timed_out": False,
        "required": True,
        "command_category": "completion",
        "started_at_utc": started_at_utc,
        "ended_at_utc": value["completed_at_utc"],
    }


def run_experiment(
    root: Path, run_date: str, paths: ExperimentPaths
) -> JsonDict:  # pragma: no cover - exercised through the declared entrypoint.
    """Run preconditions, affected checks, timing, cold replay, and publication."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "startup", "flushed")

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "preconditions", "before")
    preconditions, hashes, context = collect_preconditions(root, paths)
    sidecar = _write_historical_sidecar(root, paths.historical_receipts, hashes)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions), phase_utc))
    mandatory_passed = all(
        row["passed"] is True
        for row in preconditions
        if row["category"] == "mandatory_precondition"
    )
    progress(run_started, "preconditions", "after", mandatory_passed=mandatory_passed)

    private_root = Path(tempfile.mkdtemp(prefix="exp7407-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "affected_validation", "before_subprocess_group", units=len(commands))
    affected = validation_contract.run_categorized_commands(
        root,
        [
            validation_contract.PlannedCommand(command, "required_validation", True)
            for command in commands
        ],
        log_dir=paths.validation_dir / "affected",
        heartbeat_s=60.0,
    )
    spans.append(_span("affected_validation", phase_started, run_started, len(affected), phase_utc))
    progress(run_started, "affected_validation", "after_subprocess_group")

    if not mandatory_passed:
        services: list[JsonDict] = []
        boards = build_board_rows(context)
    else:
        fixture = load_fixed_fixture(root)
        phase_started = time.monotonic()
        phase_utc = utc_now()
        progress(run_started, "benchmark", "before", planned_rows=180)
        services = measure_service_cost(fixture["texts"], fixture["state"])
        spans.append(_span("benchmark", phase_started, run_started, len(services), phase_utc))
        progress(run_started, "benchmark", "after", completed_rows=len(services))
        boards = build_board_rows(context)

    raw = {
        "schema": "carnot.exp7407.v649.raw_service_rows.v1",
        "service_cost_rows": services,
        "board_rows": boards,
        "source_artifact_hashes": hashes,
    }
    progress(run_started, "raw_checkpoint", "before_atomic")
    atomic_json(paths.raw_evidence, raw)
    progress(run_started, "raw_checkpoint", "after_atomic", bytes=paths.raw_evidence.stat().st_size)

    candidate = assemble_artifact(
        service_rows=services,
        board_rows=boards,
        preconditions=preconditions,
        source_hashes=hashes,
        validation_receipts=affected,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        historical_sidecars=[sidecar],
        optional_adaptive_service=context["optional_adaptive_service"],
    )
    progress(run_started, "candidate", "before_atomic")
    atomic_json(paths.terminal_candidate, candidate)
    progress(
        run_started, "candidate", "after_atomic", bytes=paths.terminal_candidate.stat().st_size
    )

    terminal_plan = _terminal_commands(paths)
    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(
        run_started,
        "terminal_validation",
        "before_subprocess_group",
        units=len(terminal_plan),
    )
    terminal = validation_contract.run_categorized_commands(
        root,
        terminal_plan,
        log_dir=paths.validation_dir / "terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal), phase_utc))
    progress(run_started, "terminal_validation", "after_subprocess_group")
    receipts = [
        *affected,
        *terminal,
        _entrypoint_receipt(root, started_at, time.monotonic() - run_started),
    ]
    final = assemble_artifact(
        service_rows=services,
        board_rows=boards,
        preconditions=preconditions,
        source_hashes=hashes,
        validation_receipts=receipts,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        historical_sidecars=[sidecar],
        optional_adaptive_service=context["optional_adaptive_service"],
    )
    progress(run_started, "terminal_write", "before_atomic", path=paths.artifact)
    write_artifact(paths.artifact, final)
    progress(run_started, "terminal_write", "after_atomic", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public experiment and private cold-replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--raw", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the thin public experiment or its fresh-process reader."""

    print("[exp7407] phase=entrypoint event=flushed", flush=True)
    args = parse_args(argv)
    if args.cold_replay is not None:
        if args.raw is None:
            raise SystemExit("--raw is required with --cold-replay")
        artifact = load_json(args.cold_replay)
        raw = load_json(args.raw)
        reduction = independent_reduce(artifact, raw)
        errors = validate_artifact(artifact)
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors) or not all(reduction.values()))
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, ExperimentPaths.defaults(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
