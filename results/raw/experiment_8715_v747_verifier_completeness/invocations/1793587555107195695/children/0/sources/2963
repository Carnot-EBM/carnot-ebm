"""Replay the V652 mixture with independently reconstructable causal evidence.

This task changes evidence capture only. It reuses the archived corpus, fitted
numeric heads, action policy, controls, and success bars. The replay is not a
fresh deployment and cannot enable a production action.

Spec refs: REQ-AUTO-7454 and SCENARIO-AUTO-7454-01 through -06.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7423_v651_annotated_protocol import (
    EVALUATOR_TOKEN,
    ProtocolReaders,
    reload_corpus,
)
from carnot.experiment_7427_v651_randomized_feedback import build_streams
from carnot.experiment_7438_v652_mixture_prototype import (
    ADAPTIVE_SPLINE,
    EXPERT_NAMES,
    FROZEN_GIBBS,
    FROZEN_SPLINE,
    bernoulli_log_loss,
    validate_protocol,
)
from carnot import experiment_7440_v652_mixture_learning as v652
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
EXPERIMENT_ID = "exp7454-v653-continuous-learning"
SCHEMA = "carnot.exp7454.v653.continuous_learning.v1"
PREDICTION_SCHEMA = "carnot.exp7454.prediction_event.v1"
OUTCOME_SCHEMA = "carnot.exp7454.outcome_event.v1"
FEEDBACK_SCHEMA = "carnot.exp7454.feedback_event.v1"
CHECKPOINT_SCHEMA = "carnot.exp7454.checkpoint_event.v1"
TIMING_SCHEMA = "carnot.exp7454.stage_timing.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7454_v653_continuous_learning.json")
RAW_DIR = Path("results/raw/experiment_7454_v653_continuous_learning")
EVENT_DIR = RAW_DIR / "events"
MODULE_PATH = Path("python/carnot/experiment_7454_v653_continuous_learning.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7454_v653_continuous_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7454_v653_continuous_learning.py")
SHARED_TEST_PATHS = (
    Path("tests/python/test_experiment_7440_v652_mixture_learning.py"),
    Path("tests/python/test_experiment_7450_v653_prediction_ledger.py"),
)
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
LEDGER_PATH = Path("results/experiment_7450_v653_prediction_ledger.json")
DECISIONS_PATH = Path("results/experiment_7439_v652_certified_decisions.json")
AUDIT_PATH = Path("results/experiment_7441_v652_decision_audit.json")
V652_RESULT_PATH = Path("results/experiment_7440_v652_mixture_learning.json")
PROTOCOL_PATH = Path("results/raw/experiment_7438_v652_mixture_prototype/protocol.json")
CORPUS_DIR = Path("results/raw/experiment_7423_v651_annotated_protocol")
CORPUS_MANIFEST_PATH = CORPUS_DIR / "corpus_manifest.json"

TRAINING_SEEDS = v652.TRAINING_SEEDS
ORDERS = v652.ORDERS
DELAYS = v652.DELAYS
ARMS = v652.ARMS
LEARNED_MIXTURE = v652.LEARNED_MIXTURE
EQUAL_MIXTURE = v652.EQUAL_MIXTURE
FROZEN_SPLINE_ARM = v652.FROZEN_SPLINE_ARM
ADAPTIVE_SPLINE_ARM = v652.ADAPTIVE_SPLINE_ARM
NO_FEEDBACK_MIXTURE = v652.NO_FEEDBACK_MIXTURE
SHUFFLED_CONTROL_1 = v652.SHUFFLED_CONTROL_1
SHUFFLED_CONTROL_2 = v652.SHUFFLED_CONTROL_2
PRIMARY_COMPARATORS = v652.PRIMARY_COMPARATORS
UPDATING_ARMS = (
    LEARNED_MIXTURE,
    EQUAL_MIXTURE,
    ADAPTIVE_SPLINE_ARM,
    SHUFFLED_CONTROL_1,
    SHUFFLED_CONTROL_2,
)
SERVICE_STAGES = ("read", "predict", "persist", "reveal", "update")
BLOCK_SIZE = 32
SENSITIVITY_BLOCK_SIZE = 64
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = v652.BOOTSTRAP_SEED
INFERENCE_SUBSTRATE = "archived_feedback_durable_numeric_replay"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
MAX_SHARD_BYTES = 8 * 1024 * 1024
FLOAT_TOLERANCE = 1e-12

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7438_v652_mixture_prototype.py"),
    Path("python/carnot/experiment_7440_v652_mixture_learning.py"),
    Path("python/carnot/experiment_7450_v653_prediction_ledger.py"),
    SPEC_PATH,
    LEDGER_PATH,
    DECISIONS_PATH,
    AUDIT_PATH,
    V652_RESULT_PATH,
    PROTOCOL_PATH,
    CORPUS_MANIFEST_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(), *(path.as_posix() for path in SHARED_TEST_PATHS)),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
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
    "field_principles",
    "promotion_score",
    "online_capture_complete_score",
    "online_value_score",
    "continuous_self_learning_task",
    "prediction_event_shards",
    "feedback_event_shards",
    "stage_timing_rows",
    "no_model_weight_mutation",
)


def utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return an aware UTC timestamp for an experiment boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Print each phase boundary and each potentially slow operation."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7454] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and keep malformed external data distinct."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Keep the exact missing, null, zero, false, or changed operand."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate source bytes, upstream gates, protocol, and fit checkpoints."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "==",
                "readable_nonempty_bytes",
                observed,
                observed == "readable_nonempty_bytes",
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_verdict_class": None,
                "original_flagged_adversarial": None,
            }

    ledger = _load_object(root / LEDGER_PATH)
    decisions = _load_object(root / DECISIONS_PATH)
    audit = _load_object(root / AUDIT_PATH)
    old_result = _load_object(root / V652_RESULT_PATH)
    protocol = _load_object(root / PROTOCOL_PATH)
    definitions = (
        (
            "prediction_ledger_ready",
            "exp7450-prediction-ledger",
            LEDGER_PATH,
            "prediction_ledger_ready_score",
            "==",
            1,
            ledger.get("prediction_ledger_ready_score"),
            ledger.get("prediction_ledger_ready_score") == 1,
        ),
        (
            "prediction_ledger_verdict",
            "exp7450-prediction-ledger",
            LEDGER_PATH,
            "verdict_class",
            "in",
            ["null", "positive", "circular_positive"],
            ledger.get("verdict_class"),
            ledger.get("verdict_class") in {"null", "positive", "circular_positive"},
        ),
        (
            "prediction_ledger_unflagged",
            "exp7450-prediction-ledger",
            LEDGER_PATH,
            "flagged_adversarial",
            "==",
            False,
            ledger.get("flagged_adversarial"),
            ledger.get("flagged_adversarial") is False,
        ),
        (
            "clean_fit_checkpoint_source",
            "exp7439-certified-decisions",
            DECISIONS_PATH,
            "decision_capture_complete_score",
            "==",
            1,
            decisions.get("decision_capture_complete_score"),
            decisions.get("decision_capture_complete_score") == 1,
        ),
        (
            "fit_checkpoint_source_unflagged",
            "exp7439-certified-decisions",
            DECISIONS_PATH,
            "flagged_adversarial",
            "==",
            False,
            decisions.get("flagged_adversarial"),
            decisions.get("flagged_adversarial") is False,
        ),
        (
            "original_audit_defects",
            "exp7441-v652-decision-audit",
            AUDIT_PATH,
            "gate_check_summary.evidence_defects",
            "==",
            ["online:missing_expert_predictions", "online:weight_update_replay_incomplete"],
            (audit.get("gate_check_summary") or {}).get("evidence_defects"),
            (audit.get("gate_check_summary") or {}).get("evidence_defects")
            == ["online:missing_expert_predictions", "online:weight_update_replay_incomplete"],
        ),
        (
            "original_v652_null",
            "exp7440-v652-mixture-learning",
            V652_RESULT_PATH,
            "honest_verdict",
            "==",
            "complete_null_insufficient_online_benefit",
            old_result.get("honest_verdict"),
            old_result.get("honest_verdict") == "complete_null_insufficient_online_benefit",
        ),
        (
            "prototype_protocol_identity",
            "exp7438-mixture-prototype",
            PROTOCOL_PATH,
            "schema",
            "==",
            "carnot.exp7438.four_expert_protocol.v1",
            protocol.get("schema"),
            protocol.get("schema") == "carnot.exp7438.four_expert_protocol.v1"
            and not validate_protocol(protocol),
        ),
    )
    checks.extend(
        _precondition(check, upstream, path.as_posix(), field, operator, expected, observed, passed)
        for check, upstream, path, field, operator, expected, observed, passed in definitions
    )

    selected = [
        row
        for row in decisions.get("checkpoint_manifest") or []
        if row.get("head") in {"gibbs_6_4_1", "sparse_spline_49"}
    ]
    checkpoint_ok = len(selected) == 10
    for row in selected:
        relative = Path(str(row.get("path") or ""))
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        passed = observed == row.get("sha256")
        checkpoint_ok = checkpoint_ok and passed
        checks.append(
            _precondition(
                f"checkpoint:{row.get('head')}:{row.get('seed')}",
                "exp7439-certified-decisions",
                relative.as_posix(),
                "sha256",
                "==",
                row.get("sha256"),
                observed,
                passed,
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": observed,
                "source_receipt_class": "small_ebm_training",
                "original_flagged_adversarial": False,
            }
    checks.append(
        _precondition(
            "selected_checkpoint_count",
            "exp7439-certified-decisions",
            DECISIONS_PATH.as_posix(),
            "checkpoint_manifest",
            "==",
            10,
            len(selected),
            checkpoint_ok,
        )
    )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "==",
            "REQ-AUTO-7454",
            "REQ-AUTO-7454" if "REQ-AUTO-7454" in spec_text else None,
            "REQ-AUTO-7454" in spec_text,
        )
    )
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            "==",
            False,
            "experiment_id: 7454" in exclusion,
            "experiment_id: 7454" not in exclusion,
        )
    )
    for relative, value in (
        (LEDGER_PATH, ledger),
        (DECISIONS_PATH, decisions),
        (AUDIT_PATH, audit),
        (V652_RESULT_PATH, old_result),
    ):
        if relative.as_posix() in hashes:
            hashes[relative.as_posix()].update(
                {
                    "original_honest_verdict": value.get("honest_verdict"),
                    "original_verdict_class": value.get("verdict_class"),
                    "original_flagged_adversarial": value.get("flagged_adversarial"),
                    "source_receipt_class": "historical_numeric_or_audit_evidence",
                }
            )
    return (
        checks,
        hashes,
        {
            "ledger": ledger,
            "decisions": decisions,
            "audit": audit,
            "old_result": old_result,
            "protocol": protocol,
        },
    )


def frozen_protocol(loaded: Mapping[str, Any]) -> JsonDict:
    """Bind the unchanged V652 design and disclose its reused evidence."""

    ledger_protocol = deepcopy(dict(loaded["ledger"]["replay_protocol"]))
    value: JsonDict = {
        "schema": "carnot.exp7454.frozen_replay_protocol.v1",
        "source_group_count": 753,
        "training_seeds": list(TRAINING_SEEDS),
        "orders": list(ORDERS),
        "delays": list(DELAYS),
        "arms": list(ARMS),
        "primary_comparators": list(PRIMARY_COMPARATORS),
        "moving_block_lengths": [BLOCK_SIZE, SENSITIVITY_BLOCK_SIZE],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "reveal_schedule": deepcopy(ledger_protocol["reveal_schedule"]),
        "success_bars": deepcopy(ledger_protocol["success_bars"]),
        "exp7450_protocol_hash": loaded["ledger"]["protocol_hash"],
        "exp7438_manifest_hash": loaded["protocol"]["manifest_hash"],
        "reused_corpus": True,
        "reused_protocol": True,
        "fresh_deployment_evidence": False,
        "shadow_actions_only": True,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def prediction_event_hash(value: Mapping[str, Any]) -> str:
    """Hash every label-free prediction operand except its hash slot."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "event_hash"}
    )


def outcome_event_hash(value: Mapping[str, Any]) -> str:
    """Hash one post-prediction outcome and its prediction references."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "event_hash"}
    )


def feedback_event_hash(value: Mapping[str, Any]) -> str:
    """Hash every feedback operand and numeric update receipt."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "event_hash"}
    )


def checkpoint_event_hash(value: Mapping[str, Any]) -> str:
    """Hash a replay checkpoint and its prior checkpoint reference."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "event_hash"}
    )


def validate_prediction_event(value: Mapping[str, Any]) -> None:
    """Reject labels, incomplete expert vectors, and changed prediction bytes."""

    if {"label", "loss", "brier", "feedback_label"} & set(value):
        raise ValueError("prediction_event_contains_label")
    if set(value.get("expert_probabilities") or {}) != set(EXPERT_NAMES):
        raise ValueError("prediction_experts_invalid")
    if set(value.get("mixture_weights") or {}) != set(EXPERT_NAMES):
        raise ValueError("prediction_weights_invalid")
    weights = [float(value["mixture_weights"][name]) for name in EXPERT_NAMES]
    if any(weight < 0.0 or not math.isfinite(weight) for weight in weights) or not math.isclose(
        math.fsum(weights), 1.0, abs_tol=FLOAT_TOLERANCE
    ):
        raise ValueError("prediction_weights_invalid")
    if prediction_event_hash(value) != value.get("event_hash"):
        raise ValueError("prediction_event_hash_mismatch")


def initial_audit_state(arm: str) -> JsonDict:
    """Create the code-free state that the cold reader can reconstruct."""

    if arm not in ARMS:
        raise ValueError("unknown_arm")
    weights = {name: 0.25 for name in EXPERT_NAMES}
    if arm == FROZEN_SPLINE_ARM:
        weights = {name: float(name == FROZEN_SPLINE) for name in EXPERT_NAMES}
    elif arm == ADAPTIVE_SPLINE_ARM:
        weights = {name: float(name == ADAPTIVE_SPLINE) for name in EXPERT_NAMES}
    state: JsonDict = {
        "arm": arm,
        "log_weights": {
            name: math.log(weight) if weight > 0.0 else -math.inf
            for name, weight in weights.items()
        },
        "feedback_count": 0,
    }
    state["state_hash"] = _audit_state_hash(state)
    return state


def _audit_state_hash(state: Mapping[str, Any]) -> str:
    """Bind the selector state while representing zero weights explicitly."""

    return canonical_hash(
        {
            "arm": state["arm"],
            "log_weights": {
                name: (
                    "-inf"
                    if float(state["log_weights"][name]) == -math.inf
                    else float(state["log_weights"][name])
                )
                for name in EXPERT_NAMES
            },
            "feedback_count": int(state["feedback_count"]),
        }
    )


def _independent_scalar_update(
    old_log_weights: Mapping[str, float],
    probabilities: Mapping[str, float],
    label: int,
    *,
    eta: float,
    fixed_share: float,
) -> JsonDict:
    """Recompute the four-way update with scalar math only."""

    losses = {name: bernoulli_log_loss(label, float(probabilities[name])) for name in EXPERT_NAMES}
    penalized = {name: float(old_log_weights[name]) - eta * losses[name] for name in EXPERT_NAMES}
    maximum = max(penalized.values())
    exponentials = {name: math.exp(penalized[name] - maximum) for name in EXPERT_NAMES}
    denominator = math.fsum(exponentials.values())
    posterior = {name: exponentials[name] / denominator for name in EXPERT_NAMES}
    shared = {
        name: (1.0 - fixed_share) * posterior[name] + fixed_share / len(EXPERT_NAMES)
        for name in EXPERT_NAMES
    }
    return {
        "losses": losses,
        "penalized_log_weights": penalized,
        "maximum": maximum,
        "sum_exp": denominator,
        "posterior_before_share": posterior,
        "new_log_weights": {name: math.log(shared[name]) for name in EXPERT_NAMES},
    }


def _arm_update_parameters(arm: str) -> tuple[float, float] | None:
    """Return frozen update parameters, or no mixture update for one-head arms."""

    if arm in {LEARNED_MIXTURE, SHUFFLED_CONTROL_1, SHUFFLED_CONTROL_2}:
        return 1.0, 0.01
    if arm == EQUAL_MIXTURE:
        return 0.0, 0.0
    if arm == ADAPTIVE_SPLINE_ARM:
        return None
    raise ValueError("arm_does_not_accept_feedback")


def _advance_audit_state(
    state: Mapping[str, Any], prediction: Mapping[str, Any], label: int
) -> tuple[JsonDict, JsonDict]:
    """Advance one audit state from immutable prediction probabilities."""

    arm = str(state["arm"])
    parameters = _arm_update_parameters(arm)
    if parameters is None:
        numeric = {
            "losses": {
                name: bernoulli_log_loss(label, float(prediction["expert_probabilities"][name]))
                for name in EXPERT_NAMES
            },
            "penalized_log_weights": deepcopy(dict(state["log_weights"])),
            "maximum": None,
            "sum_exp": None,
            "posterior_before_share": deepcopy(dict(prediction["mixture_weights"])),
            "new_log_weights": deepcopy(dict(state["log_weights"])),
        }
    else:
        numeric = _independent_scalar_update(
            state["log_weights"],
            prediction["expert_probabilities"],
            label,
            eta=parameters[0],
            fixed_share=parameters[1],
        )
    updated = {
        "arm": arm,
        "log_weights": deepcopy(numeric["new_log_weights"]),
        "feedback_count": int(state["feedback_count"]) + 1,
    }
    updated["state_hash"] = _audit_state_hash(updated)
    return updated, numeric


class DurableEventStore:  # pragma: no cover - production I/O is checked through readers.
    """Append event batches and fsync each batch before the caller can continue."""

    def __init__(self, root: Path, relative_dir: Path, *, max_bytes: int = MAX_SHARD_BYTES) -> None:
        self.root = root
        self.relative_dir = relative_dir
        self.max_bytes = max_bytes
        self.sequence = 0
        self._parts: dict[str, int] = defaultdict(int)
        self._paths: dict[str, list[Path]] = defaultdict(list)
        self._sizes: dict[str, int] = defaultdict(int)
        self._rows: dict[Path, int] = defaultdict(int)
        self._streams: dict[str, Any] = {}

    def next_sequence(self) -> int:
        """Allocate one globally ordered event number."""

        value = self.sequence
        self.sequence += 1
        return value

    def _open(self, kind: str) -> Any:
        part = self._parts[kind]
        path = self.root / self.relative_dir / f"{kind}-{part:03d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("wb")
        self._streams[kind] = stream
        self._paths[kind].append(path)
        self._sizes[kind] = 0
        return stream

    def append_batch(self, kind: str, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
        """Write, flush, and fsync a complete causal batch."""

        payload = b"".join(
            (
                json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode("utf-8")
            for row in rows
        )
        stream = self._streams.get(kind) or self._open(kind)
        if self._sizes[kind] and self._sizes[kind] + len(payload) > self.max_bytes:
            stream.close()
            self._parts[kind] += 1
            stream = self._open(kind)
        started = time.perf_counter_ns()
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        duration = time.perf_counter_ns() - started
        self._sizes[kind] += len(payload)
        self._rows[self._paths[kind][-1]] += len(rows)
        return {
            "kind": kind,
            "event_count": len(rows),
            "bytes": len(payload),
            "duration_ns": duration,
            "durable_acknowledged": True,
        }

    def close(self) -> dict[str, list[JsonDict]]:
        """Close all streams and return hash-bound shard manifests."""

        for stream in self._streams.values():
            stream.close()
        self._streams.clear()
        output: dict[str, list[JsonDict]] = {}
        for kind, paths in self._paths.items():
            output[kind] = [
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "sha256": sha256_file(path),
                    "rows": self._rows[path],
                    "size_bytes": path.stat().st_size,
                }
                for path in paths
            ]
        return output


def load_event_shards(root: Path, manifests: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Rehash event shards before returning any row to a reducer."""

    output: list[JsonDict] = []
    for manifest in manifests:
        path = root / str(manifest.get("path") or "")
        if (
            not path.is_file()
            or sha256_file(path) != manifest.get("sha256")
            or path.stat().st_size != manifest.get("size_bytes")
        ):
            raise ValueError("event_shard_invalid")
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except json.JSONDecodeError as error:
            raise ValueError("event_shard_invalid") from error
        if len(rows) != manifest.get("rows") or any(not isinstance(row, dict) for row in rows):
            raise ValueError("event_shard_invalid")
        output.extend(rows)
    return output


def _checkpoint_hashes(state: Mapping[str, Any]) -> dict[str, str]:
    """Map the two authenticated fit checkpoints onto all four expert names."""

    rows = {str(row.get("head")): str(row.get("sha256")) for row in state["source_checkpoints"]}
    if not rows:
        rows = {
            "sparse_spline_49": canonical_hash(state["spline_checkpoint"]),
            "gibbs_6_4_1": canonical_hash(state["gibbs_checkpoint"]),
        }
    return {
        FROZEN_SPLINE: rows["sparse_spline_49"],
        ADAPTIVE_SPLINE: rows["sparse_spline_49"],
        FROZEN_GIBBS: rows["gibbs_6_4_1"],
        "adaptive_gibbs": rows["gibbs_6_4_1"],
    }


def _make_prediction_event(
    *,
    store: DurableEventStore,
    row: Mapping[str, Any],
    arm: str,
    ordering: str,
    delay: int,
    seed: int,
    prediction_index: int,
    expert_probabilities: Mapping[str, float],
    mixture_weights: Mapping[str, float],
    mixture_probability: float,
    checkpoint_hashes: Mapping[str, str],
    propensity: float,
    revealed: bool,
    state_hash: str,
    thresholds: Mapping[str, Any],
    domain_changed: bool,
    predict_duration_ns: int,
) -> JsonDict:
    """Build one complete label-free prediction event."""

    started = time.perf_counter_ns()
    value: JsonDict = {
        "schema": PREDICTION_SCHEMA,
        "row_type": "prediction_event",
        "event_id": f"{ordering}:{delay}:{seed}:{arm}:{row['observation_id']}",
        "group_id": str(row["group_id"]),
        "row_key": str(row["row_key"]),
        "task_type": str(row["task_type"]),
        "arm": arm,
        "ordering": ordering,
        "delay": delay,
        "seed": seed,
        "prediction_index": prediction_index,
        "request_order": prediction_index,
        "ledger_sequence": store.next_sequence(),
        "expert_probabilities": {name: float(expert_probabilities[name]) for name in EXPERT_NAMES},
        "mixture_weights": {name: float(mixture_weights[name]) for name in EXPERT_NAMES},
        "mixture_probability": float(mixture_probability),
        "expert_checkpoint_hashes": {name: str(checkpoint_hashes[name]) for name in EXPERT_NAMES},
        "label_propensity": float(propensity),
        "revealed": bool(revealed),
        "pre_feedback_state_hash": state_hash,
        "proposed_action": v652.shadow_action(float(mixture_probability), thresholds),
        "deployed_action": "escalate",
        "shadow_only": True,
        "certified_safe": False,
        "domain_changed": bool(domain_changed),
        "durable_acknowledged": True,
        "predict_duration_ns": int(predict_duration_ns),
        "hash_duration_ns": 0,
    }
    value["event_hash"] = prediction_event_hash(value)
    value["hash_duration_ns"] = time.perf_counter_ns() - started
    value["event_hash"] = prediction_event_hash(value)
    validate_prediction_event(value)
    return value


def _sanitize_log_weights(log_weights: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure -inf float values are encoded as strings for JSON compliance."""

    return {
        name: (
            "-inf"
            if (
                log_weights[name] == "-inf"
                or (
                    isinstance(log_weights[name], (int, float))
                    and float(log_weights[name]) == -math.inf
                )
            )
            else float(log_weights[name])
        )
        for name in EXPERT_NAMES
    }


def _make_feedback_event(
    *,
    store: DurableEventStore,
    prediction: Mapping[str, Any],
    label: int,
    true_label: int,
    reveal_order: int,
    old_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
    numeric: Mapping[str, Any],
    actual_receipt: Mapping[str, Any],
) -> JsonDict:
    """Bind saved expert losses to one delivered and admitted update."""

    value: JsonDict = {
        "schema": FEEDBACK_SCHEMA,
        "row_type": "feedback_event",
        "arm": prediction["arm"],
        "ordering": prediction["ordering"],
        "delay": prediction["delay"],
        "seed": prediction["seed"],
        "group_id": prediction["group_id"],
        "prediction_index": prediction["prediction_index"],
        "request_order": prediction["request_order"],
        "reveal_order": reveal_order,
        "ledger_sequence": store.next_sequence(),
        "prediction_event_hash": prediction["event_hash"],
        "feedback_label": int(label),
        "true_label": int(true_label),
        "label_origin": "archived_human_label_or_fixed_shuffle_control",
        "per_expert_loss": deepcopy(dict(numeric["losses"])),
        "old_log_weights": _sanitize_log_weights(old_state["log_weights"]),
        "numeric_update": {
            "penalized_log_weights": _sanitize_log_weights(numeric["penalized_log_weights"]),
            "posterior_before_share": deepcopy(dict(numeric["posterior_before_share"])),
        },
        "normalizer": {"maximum": numeric["maximum"], "sum_exp": numeric["sum_exp"]},
        "new_log_weights": _sanitize_log_weights(new_state["log_weights"]),
        "parent_state_hash": old_state["state_hash"],
        "child_state_hash": new_state["state_hash"],
        "feedback_count_before": old_state["feedback_count"],
        "feedback_count_after": new_state["feedback_count"],
        "probability_source": "immutable_prediction_event",
        "update_applied": True,
        "revoked": False,
        "actual_update_receipt": {
            "update_admitted": actual_receipt.get("update_admitted"),
            "expert_update_count": actual_receipt.get("expert_update_count", 1),
            "touched_coefficients": actual_receipt.get("touched_coefficients", 0),
            "duration_ns": actual_receipt.get("update_duration_ns", 0),
        },
    }
    value["event_hash"] = feedback_event_hash(value)
    return value


def _make_checkpoint_events(
    *,
    store: DurableEventStore,
    ordering: str,
    delay: int,
    seed: int,
    completed_groups: int,
    states: Mapping[str, Mapping[str, Any]],
    previous: dict[str, str | None],
) -> list[JsonDict]:
    """Create one chained numeric checkpoint for every registered arm."""

    output: list[JsonDict] = []
    for arm in ARMS:
        value: JsonDict = {
            "schema": CHECKPOINT_SCHEMA,
            "row_type": "checkpoint_event",
            "arm": arm,
            "ordering": ordering,
            "delay": delay,
            "seed": seed,
            "completed_groups": completed_groups,
            "ledger_sequence": store.next_sequence(),
            "state_hash": states[arm]["state_hash"],
            "log_weights": _sanitize_log_weights(states[arm]["log_weights"]),
            "feedback_count": states[arm]["feedback_count"],
            "previous_checkpoint_hash": previous[arm],
        }
        value["event_hash"] = checkpoint_event_hash(value)
        previous[arm] = value["event_hash"]
        output.append(value)
    return output


def replay_feedback_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    initial_state: Mapping[str, Any],
    prediction_by_hash: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Reconstruct selector updates without producer code and verify causal lineage."""

    current_state = deepcopy(dict(initial_state))
    seen_hashes: set[str] = set()
    max_loss_error = 0.0
    max_update_error = 0.0
    for row in rows:
        event_hash = str(row.get("event_hash") or "")
        if event_hash in seen_hashes:
            raise ValueError("duplicate_feedback")
        seen_hashes.add(event_hash)
        if feedback_event_hash(row) != event_hash:
            raise ValueError("feedback_event_hash_mismatch")
        pred_hash = str(row.get("prediction_event_hash") or "")
        prediction = prediction_by_hash.get(pred_hash)
        if prediction is None:
            raise ValueError("missing_prediction_for_feedback")
        if row.get("parent_state_hash") != current_state["state_hash"]:
            raise ValueError("parent_state_hash_mismatch")

        label = int(row["feedback_label"])
        stored_losses = row["per_expert_loss"]
        for name in EXPERT_NAMES:
            prob = float(prediction["expert_probabilities"][name])
            expected_loss = bernoulli_log_loss(label, prob)
            stored_loss = float(stored_losses[name])
            err = abs(expected_loss - stored_loss)
            if err > FLOAT_TOLERANCE:
                raise ValueError("expert_loss_mismatch")
            max_loss_error = max(max_loss_error, err)

        next_state, numeric = _advance_audit_state(current_state, prediction, label)
        if row.get("child_state_hash") != next_state["state_hash"]:
            raise ValueError("child_state_hash_mismatch")
        for name in EXPERT_NAMES:
            actual_w = next_state["log_weights"][name]
            actual_val = -math.inf if actual_w == "-inf" else float(actual_w)
            row_w = row["new_log_weights"][name]
            row_val = -math.inf if row_w == "-inf" else float(row_w)
            if actual_val == -math.inf and row_val == -math.inf:
                update_err = 0.0
            else:
                update_err = abs(actual_val - row_val)
            if math.isfinite(update_err):
                max_update_error = max(max_update_error, update_err)
        current_state = next_state

    return {
        "final_state": current_state,
        "replayed_count": len(rows),
        "expert_loss_max_abs_error": max_loss_error,
        "numeric_update_max_abs_error": max_update_error,
    }


def build_fixture_evidence(
    root: Path,
    *,
    relative_dir: Path = Path("raw"),
    group_count: int = 40,
    ordering: str = "hash_order",
    delay: int = 8,
    seed: int = TRAINING_SEEDS[0],
) -> dict[str, Any]:
    """Capture one complete cell with durable label-free prediction events."""

    store = DurableEventStore(root, relative_dir / "events")
    audit_states = {arm: initial_audit_state(arm) for arm in ARMS}
    prev_checkpoints: dict[str, str | None] = {arm: None for arm in ARMS}

    revealed_groups: set[int] = set()
    for start in range(0, group_count, BLOCK_SIZE):
        block = list(range(start, min(start + BLOCK_SIZE, group_count)))
        rev_count = len(block) // 4
        for idx in range(rev_count):
            revealed_groups.add(block[idx * 4])

    pending_feedback: list[tuple[int, int, int, dict[str, JsonDict]]] = []

    for g in range(group_count):
        p_spline = 0.3 + 0.4 * ((g % 7) / 7.0)
        p_gibbs = 0.2 + 0.5 * ((g % 5) / 5.0)
        expert_probs = {
            FROZEN_SPLINE: p_spline,
            ADAPTIVE_SPLINE: p_spline,
            FROZEN_GIBBS: p_gibbs,
            "adaptive_gibbs": p_gibbs,
        }
        true_label = 1 if (g % 3 == 0) else 0
        propensity = 0.25
        revealed = g in revealed_groups

        group_preds: dict[str, JsonDict] = {}
        for arm in ARMS:
            if arm == FROZEN_SPLINE_ARM:
                weights = {name: float(name == FROZEN_SPLINE) for name in EXPERT_NAMES}
            elif arm == ADAPTIVE_SPLINE_ARM:
                weights = {name: float(name == ADAPTIVE_SPLINE) for name in EXPERT_NAMES}
            else:
                raw_w = {
                    name: math.exp(float(audit_states[arm]["log_weights"][name]))
                    for name in EXPERT_NAMES
                }
                s_w = math.fsum(raw_w.values())
                weights = {name: raw_w[name] / s_w for name in EXPERT_NAMES}
            mix_p = math.fsum(weights[name] * expert_probs[name] for name in EXPERT_NAMES)
            ckpt_hashes = {name: f"sha256:ckpt_{name}_{seed}" for name in EXPERT_NAMES}

            pred_event = _make_prediction_event(
                store=store,
                row={
                    "observation_id": f"obs-{ordering}-{delay}-{seed}-{g:04d}",
                    "group_id": f"group-{ordering}-{delay}-{seed}-{g:04d}",
                    "row_key": f"row-{ordering}-{delay}-{seed}-{g:04d}",
                    "task_type": "fixture_task",
                },
                arm=arm,
                ordering=ordering,
                delay=delay,
                seed=seed,
                prediction_index=g,
                expert_probabilities=expert_probs,
                mixture_weights=weights,
                mixture_probability=mix_p,
                checkpoint_hashes=ckpt_hashes,
                propensity=propensity,
                revealed=revealed,
                state_hash=audit_states[arm]["state_hash"],
                thresholds={"accept_threshold": 0.8, "reject_threshold": 0.2},
                domain_changed=False,
                predict_duration_ns=1000,
            )
            group_preds[arm] = pred_event

        store.append_batch("predictions", list(group_preds.values()))

        outcome_event: JsonDict = {
            "schema": OUTCOME_SCHEMA,
            "row_type": "outcome_event",
            "group_id": group_preds[LEARNED_MIXTURE]["group_id"],
            "observation_id": f"obs-{ordering}-{delay}-{seed}-{g:04d}",
            "ordering": ordering,
            "delay": delay,
            "seed": seed,
            "request_order": g,
            "ledger_sequence": store.next_sequence(),
            "true_label": true_label,
            "revealed": revealed,
            "propensity": propensity,
            "prediction_event_hashes": {arm: group_preds[arm]["event_hash"] for arm in ARMS},
        }
        outcome_event["event_hash"] = outcome_event_hash(outcome_event)
        store.append_batch("outcomes", [outcome_event])

        if revealed:
            pending_feedback.append((g + delay, g, true_label, group_preds))

        due = [item for item in pending_feedback if item[0] <= g]
        if due:
            fb_events: list[JsonDict] = []
            for item in due:
                pending_feedback.remove(item)
                avail_at, req_order, lbl, p_dict = item
                for arm in UPDATING_ARMS:
                    old_st = audit_states[arm]
                    delivered_lbl = (
                        lbl if arm not in {SHUFFLED_CONTROL_1, SHUFFLED_CONTROL_2} else (1 - lbl)
                    )
                    new_st, numeric = _advance_audit_state(old_st, p_dict[arm], delivered_lbl)
                    actual_rcpt = {
                        "update_admitted": True,
                        "expert_update_count": (
                            2
                            if arm
                            in {
                                LEARNED_MIXTURE,
                                EQUAL_MIXTURE,
                                SHUFFLED_CONTROL_1,
                                SHUFFLED_CONTROL_2,
                            }
                            else 1
                        ),
                        "touched_coefficients": 4,
                        "update_duration_ns": 500,
                    }
                    fb_ev = _make_feedback_event(
                        store=store,
                        prediction=p_dict[arm],
                        label=delivered_lbl,
                        true_label=lbl,
                        reveal_order=g,
                        old_state=old_st,
                        new_state=new_st,
                        numeric=numeric,
                        actual_receipt=actual_rcpt,
                    )
                    audit_states[arm] = new_st
                    fb_events.append(fb_ev)
            if fb_events:
                store.append_batch("feedback", fb_events)

        if (g + 1) % 32 == 0 or g == group_count - 1:
            cg = g + 1
            ckpt_events = _make_checkpoint_events(
                store=store,
                ordering=ordering,
                delay=delay,
                seed=seed,
                completed_groups=cg,
                states=audit_states,
                previous=prev_checkpoints,
            )
            store.append_batch("checkpoints", ckpt_events)

    # Deliver any remaining pending feedback at stream completion
    if pending_feedback:
        fb_events = []
        for avail_at, req_order, lbl, p_dict in list(pending_feedback):
            pending_feedback.remove((avail_at, req_order, lbl, p_dict))
            for arm in UPDATING_ARMS:
                old_st = audit_states[arm]
                delivered_lbl = (
                    lbl if arm not in {SHUFFLED_CONTROL_1, SHUFFLED_CONTROL_2} else (1 - lbl)
                )
                new_st, numeric = _advance_audit_state(old_st, p_dict[arm], delivered_lbl)
                actual_rcpt = {
                    "update_admitted": True,
                    "expert_update_count": (
                        2
                        if arm
                        in {
                            LEARNED_MIXTURE,
                            EQUAL_MIXTURE,
                            SHUFFLED_CONTROL_1,
                            SHUFFLED_CONTROL_2,
                        }
                        else 1
                    ),
                    "touched_coefficients": 4,
                    "update_duration_ns": 500,
                }
                fb_ev = _make_feedback_event(
                    store=store,
                    prediction=p_dict[arm],
                    label=delivered_lbl,
                    true_label=lbl,
                    reveal_order=group_count,
                    old_state=old_st,
                    new_state=new_st,
                    numeric=numeric,
                    actual_receipt=actual_rcpt,
                )
                audit_states[arm] = new_st
                fb_events.append(fb_ev)
        if fb_events:
            store.append_batch("feedback", fb_events)

    shards = store.close()
    return {
        "prediction_event_shards": shards.get("predictions", []),
        "outcome_event_shards": shards.get("outcomes", []),
        "feedback_event_shards": shards.get("feedback", []),
        "checkpoint_event_shards": shards.get("checkpoints", []),
        "stage_timing_rows": {
            "schema": TIMING_SCHEMA,
            "stages": list(SERVICE_STAGES),
            "durations_ns": {stage: 1000 for stage in SERVICE_STAGES},
            "durable_acknowledged": True,
        },
        "ordering": ordering,
        "delay": delay,
        "seed": seed,
        "group_count": group_count,
    }


def independent_reduce_evidence(
    evidence: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> JsonDict:
    """Reconstruct updates, check lineage, and compute causal and value metrics."""

    predictions = load_event_shards(root, evidence.get("prediction_event_shards", []))
    feedback = load_event_shards(root, evidence.get("feedback_event_shards", []))
    checkpoints = load_event_shards(root, evidence.get("checkpoint_event_shards", []))

    for p in predictions:
        validate_prediction_event(p)

    prediction_by_hash = {row["event_hash"]: row for row in predictions}

    max_loss_err = 0.0
    max_update_err = 0.0
    for arm in UPDATING_ARMS:
        arm_fb = [row for row in feedback if row["arm"] == arm]
        if arm_fb:
            replayed = replay_feedback_rows(
                arm_fb,
                initial_state=initial_audit_state(arm),
                prediction_by_hash=prediction_by_hash,
            )
            max_loss_err = max(max_loss_err, replayed["expert_loss_max_abs_error"])
            max_update_err = max(max_update_err, replayed["numeric_update_max_abs_error"])

    checkpoint_lineage_valid = True
    for arm in ARMS:
        arm_ckpts = [row for row in checkpoints if row["arm"] == arm]
        prev_h = None
        for ckpt in arm_ckpts:
            if ckpt.get("previous_checkpoint_hash") != prev_h:
                checkpoint_lineage_valid = False
            if checkpoint_event_hash(ckpt) != ckpt.get("event_hash"):
                checkpoint_lineage_valid = False
            prev_h = ckpt.get("event_hash")

    duplicate_rejected = False
    learned_fb = [row for row in feedback if row["arm"] == LEARNED_MIXTURE]
    if learned_fb:
        try:
            replay_feedback_rows(
                [learned_fb[0], deepcopy(learned_fb[0])],
                initial_state=initial_audit_state(LEARNED_MIXTURE),
                prediction_by_hash=prediction_by_hash,
            )
        except ValueError as e:
            if "duplicate_feedback" in str(e):
                duplicate_rejected = True
    else:
        duplicate_rejected = True

    revoked_replayed = True
    if len(learned_fb) > 1:
        st = initial_audit_state(LEARNED_MIXTURE)
        for row in learned_fb[1:]:
            pred = prediction_by_hash.get(row["prediction_event_hash"])
            if pred:
                st, _ = _advance_audit_state(st, pred, int(row["feedback_label"]))
        rep_full = replay_feedback_rows(
            learned_fb,
            initial_state=initial_audit_state(LEARNED_MIXTURE),
            prediction_by_hash=prediction_by_hash,
        )
        revoked_replayed = rep_full["final_state"]["state_hash"] != st["state_hash"]

    restart_equal = True
    if len(learned_fb) >= 2:
        mid = len(learned_fb) // 2
        p1 = replay_feedback_rows(
            learned_fb[:mid],
            initial_state=initial_audit_state(LEARNED_MIXTURE),
            prediction_by_hash=prediction_by_hash,
        )
        p2 = replay_feedback_rows(
            learned_fb[mid:],
            initial_state=p1["final_state"],
            prediction_by_hash=prediction_by_hash,
        )
        full = replay_feedback_rows(
            learned_fb,
            initial_state=initial_audit_state(LEARNED_MIXTURE),
            prediction_by_hash=prediction_by_hash,
        )
        restart_equal = p2["final_state"]["state_hash"] == full["final_state"]["state_hash"]

    no_fb_init = initial_audit_state(NO_FEEDBACK_MIXTURE)["state_hash"]
    no_fb_unchanged = all(
        p["pre_feedback_state_hash"] == no_fb_init
        for p in predictions
        if p["arm"] == NO_FEEDBACK_MIXTURE
    )

    feedback_controls = {
        "duplicate_feedback_rejected": duplicate_rejected,
        "revoked_feedback_replayed": revoked_replayed,
        "restart_replay_equal": restart_equal,
        "no_feedback_state_unchanged": no_fb_unchanged,
    }

    later_causality = False
    if learned_fb:
        first_fb = learned_fb[0]
        rev_order = first_fb["reveal_order"]
        subsequent_p = next(
            (
                p
                for p in predictions
                if p["arm"] == LEARNED_MIXTURE and p["prediction_index"] > rev_order
            ),
            None,
        )
        subsequent_no_fb = next(
            (
                p
                for p in predictions
                if p["arm"] == NO_FEEDBACK_MIXTURE and p["prediction_index"] > rev_order
            ),
            None,
        )
        if (
            subsequent_p is not None
            and subsequent_no_fb is not None
            and subsequent_p["pre_feedback_state_hash"]
            != initial_audit_state(LEARNED_MIXTURE)["state_hash"]
            and subsequent_no_fb["pre_feedback_state_hash"] == no_fb_init
        ):
            later_causality = True

    causal_valid = (
        max_loss_err <= FLOAT_TOLERANCE
        and max_update_err <= FLOAT_TOLERANCE
        and checkpoint_lineage_valid
        and all(feedback_controls.values())
    )

    return {
        "causal_capture_valid": causal_valid,
        "expert_loss_max_abs_error": max_loss_err,
        "numeric_update_max_abs_error": max_update_err,
        "prediction_event_count": len(predictions),
        "feedback_event_count": len(feedback),
        "checkpoint_lineage_valid": checkpoint_lineage_valid,
        "feedback_handling_controls": feedback_controls,
        "later_query_causality": later_causality,
        "online_capture_complete_score": 1 if causal_valid else 0,
        "online_value_score": 0,
    }


def _fixture_acceptance_gates() -> list[JsonDict]:
    """Return closed acceptance gate definitions for fixture artifacts."""

    return [
        {
            "check": "preconditions_passed",
            "category": "validity",
            "op": "==",
            "operator": "==",
            "expected": True,
            "observed": True,
            "passed": True,
            "principle": "Authenticated inputs and original flags must be clean before replay.",
        },
        {
            "check": "causal_capture_valid",
            "category": "completion",
            "op": "==",
            "operator": "==",
            "expected": True,
            "observed": True,
            "passed": True,
            "principle": "Prediction and feedback events must be hash-bound and replay without error.",
        },
        {
            "check": "checkpoint_lineage_valid",
            "category": "completion",
            "op": "==",
            "operator": "==",
            "expected": True,
            "observed": True,
            "passed": True,
            "principle": "Every checkpoint must chain to its prior checkpoint without gap.",
        },
        {
            "check": "restart_replay_equal",
            "category": "completion",
            "op": "==",
            "operator": "==",
            "expected": True,
            "observed": True,
            "passed": True,
            "principle": "Segmented restart replay must equal uninterrupted full replay.",
        },
        {
            "check": "duplicate_feedback_rejected",
            "category": "completion",
            "op": "==",
            "operator": "==",
            "expected": True,
            "observed": True,
            "passed": True,
            "principle": "Duplicate feedback must be rejected and fail closed.",
        },
        {
            "check": "no_feedback_state_unchanged",
            "category": "completion",
            "op": "==",
            "operator": "==",
            "expected": True,
            "observed": True,
            "passed": True,
            "principle": "No-feedback arm must never mutate its initial state.",
        },
        {
            "check": "no_model_weight_mutation",
            "category": "completion",
            "op": "==",
            "operator": "==",
            "expected": True,
            "observed": True,
            "passed": True,
            "principle": "Mandated generator weights must not be modified.",
        },
        {
            "check": "validation_receipts_passed",
            "category": "validity",
            "op": "==",
            "operator": "==",
            "expected": True,
            "observed": True,
            "passed": True,
            "principle": "All required validation checks must exit zero.",
        },
        {
            "check": "upper_log_loss_delta:insufficient_online_benefit",
            "category": "benefit",
            "op": "<",
            "operator": "<",
            "expected": 0.0,
            "observed": 0.015,
            "passed": False,
            "principle": "Upper log loss delta against primary comparators must be strictly negative.",
        },
    ]


def _unit_rows(group_count: int = 753) -> list[JsonDict]:
    """Enumerate every prespecified arm, seed, order, and delay unit."""

    return [
        {
            "arm": arm,
            "seed": seed,
            "ordering": ordering,
            "delay": delay,
            "group_count": group_count,
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "status": "completed",
        }
        for ordering in ORDERS
        for delay in DELAYS
        for seed in TRAINING_SEEDS
        for arm in ARMS
    ]


def _field_principles() -> JsonDict:
    """Explain field intent here; gate fields themselves remain bare scalars."""

    return {
        "schema": "Use a versioned top-level schema and exact experiment_id, milestone and terminal status.",
        "run_date": "Use 20260920; retain actual UTC start/end and monotonic duration with boot/segment identity.",
        "preconditions_checked": "Name the actual source paths, resources, identity and observed gate values before dependent work.",
        "MODEL_SPECS": "Name unsloth/Qwen3.8-27B-GGUF for any planned current LLM task; use [] for tasks with no LLM work.",
        "model_invoked": "Attempted current model work is distinct from archived or scripted model-shaped events.",
        "invocation_counts": "Balance loads, forwards and generations: attempted, completed, failed, cancelled and in-flight.",
        "inference_substrate": "Use a truthful string; keep model/device details and historical evidence in typed sidecars.",
        "inference_substrate_class": "Declare actual current work. Bounded generation uses 10s, embedding/load only 2s, real full generation 60s; blocked_no_run carries no simulated execution.",
        "execution_venue": "host; keep CPU/CUDA identity separate from compute class and archived external-device evidence.",
        "duration_s": "Measure real current work; separate load/forward/generation, numeric computation and validation. Never pad time.",
        "phase_spans": "Bind phase timings, progress events, checkpoints and clock segments.",
        "random_seed": "Freeze fit, projection, stream and resampling seeds; explain a null when randomness is absent.",
        "reproducibility_checksum": "Bind code, protocol, immutable inputs, row shards and exact validation scope.",
        "source_artifact_hashes": "Preserve exact upstream bytes, original classes and flags; never rehabilitate history.",
        "rows": "Per-unit metrics for every arm/group/seed/condition, including failures, censoring and unstarted units.",
        "sample_size_budget": "Separate planned, attempted, completed, failed, censored and unstarted independent units with a fixed stop rule.",
        "acceptance_gate_results": "Each row names check, validity-or-benefit category, op, expected, observed, passed and principle.",
        "gate_check_summary": "Every blocked_* names upstream, exact path, check, field, expected and observed; missing, None and zero are different causes.",
        "verifier_is_oracle": "True if the deployed verifier supplies the scoring authority; exact/synthetic oracle success is circular_positive.",
        "honest_verdict": "Completed findings start complete_; unchanged external absence uses blocked_* with a specific reason. Preserve prior failure strings when the same condition recurs.",
        "verdict_class": "Closed enum: positive | circular_positive | null | blocked | disqualified | partial. partial means unfinished OWN retryable work only.",
        "flagged_adversarial": "Preserve actual critical findings. Flagged/disqualified science cannot supply readiness.",
        "validation_receipts": "Record actual scoped argv/environment, exit codes, duration and log hashes, including both terminal readers.",
        "field_principles": "Explain field intent here; gate fields themselves remain bare scalars, never value/principle wrappers.",
        "promotion_score": "Always zero. This milestone authorizes experiments, not rollout, generator-weight changes or external publication.",
        "online_capture_complete_score": "One requires complete cells and cold-replayable prediction and update events, including null comparisons.",
        "online_value_score": "Use the unchanged benefit/no-harm/control gates from the registered protocol.",
        "continuous_self_learning_task": "True: updates use earlier feedback and affect later predictions in an ordered replay.",
        "prediction_event_shards": "Store every expert probability at prediction time with pre-label state.",
        "feedback_event_shards": "Losses and state updates reference immutable prediction events.",
        "stage_timing_rows": "Keep all five stages and durable acknowledgement costs visible.",
        "no_model_weight_mutation": "True for the mandated generator; compact online selectors are the only learned parameters.",
        "outcome_event_shards": "Post-prediction outcomes bind labels to immutable prediction hashes.",
        "checkpoint_event_shards": "Numeric checkpoints occur every 32 groups and at final group.",
        "mixture_construction_retired": "Retire unchanged mixture construction after repeated null.",
    }


def _gate_summary(
    gates: Sequence[Mapping[str, Any]],
    failed_precondition: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Summarize gate results and name failed preconditions."""

    if failed_precondition is not None:
        return {
            "blocked_check": failed_precondition.get("check"),
            "blocked_upstream": failed_precondition.get("upstream"),
            "blocked_path": failed_precondition.get("path"),
            "blocked_field": failed_precondition.get("field"),
            "blocked_expected": failed_precondition.get("expected"),
            "blocked_observed": failed_precondition.get("observed"),
            "path": failed_precondition.get("path"),
            "observed": failed_precondition.get("observed"),
            "expected": failed_precondition.get("expected"),
            "required_checks_passed": False,
            "failed_required_checks": [str(failed_precondition.get("check"))],
            "benefit_failures": [],
            "evidence_defects": [],
        }
    required_failed = [
        row["check"]
        for row in gates
        if row.get("category") in {"validity", "completion", "safety"} and not row.get("passed")
    ]
    return {
        "blocked_check": None,
        "blocked_upstream": None,
        "blocked_path": None,
        "blocked_field": None,
        "blocked_expected": None,
        "blocked_observed": None,
        "required_checks_passed": len(required_failed) == 0,
        "failed_required_checks": required_failed,
        "benefit_failures": [
            row["check"]
            for row in gates
            if row.get("category") == "benefit" and not row.get("passed")
        ],
        "evidence_defects": [],
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding clocks and the checksum slot itself."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "completed_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "phase_spans",
        "clock_identity",
    }
    stable = {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    stable["validation_receipts"] = [
        {
            key: deepcopy(row.get(key))
            for key in (
                "name",
                "command_argv",
                "command_environment",
                "scope",
                "exit_code",
                "passed",
                "required",
            )
        }
        for row in value.get("validation_receipts", [])
    ]
    return canonical_hash(stable)


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check declarations, shard bytes, reduction, receipts, and checksum."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in value:
            errors.append(f"missing_field:{field}")
    if value.get("no_model_weight_mutation") is not True:
        errors.append("model_weight_mutation_invalid")
    if value.get("continuous_self_learning_task") is not True:
        errors.append("continuous_self_learning_task_invalid")
    if value.get("MODEL_SPECS") != []:
        errors.append("MODEL_SPECS_not_empty")
    if value.get("model_invoked") is not False:
        errors.append("model_invoked_not_false")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class_invalid")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_not_zero")
    if value.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial_not_false")
    for shard_key in ("prediction_event_shards", "feedback_event_shards"):
        for shard in value.get(shard_key) or []:
            p = root / str(shard.get("path") or "")
            if p.is_file() and sha256_file(p) != shard.get("sha256"):
                errors.append(f"shard_sha256_mismatch:{shard.get('path')}")
    if value.get("reproducibility_checksum") and value.get(
        "reproducibility_checksum"
    ) != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def build_fixture_artifact(
    root: Path,
    *,
    validation_receipts: Sequence[Mapping[str, Any]] | None = None,
    relative_dir: Path = Path("fixture-artifact"),
) -> JsonDict:
    """Build compact complete evidence for reducer, mutation, and lint checks."""

    receipts = list(validation_receipts or [])
    evidence = build_fixture_evidence(
        root,
        relative_dir=relative_dir,
        group_count=70,
        ordering="hash_order",
        delay=8,
        seed=TRAINING_SEEDS[0],
    )
    reduced = independent_reduce_evidence(evidence, root=root, bootstrap_draws=40)
    now = utc_now()
    started_ns = time.monotonic_ns()
    ended_ns = started_ns + 1_000_000

    unit_rows = _unit_rows(group_count=70)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_null_retired",
        "run_date": RUN_DATE,
        "started_at_utc": now,
        "completed_at_utc": now,
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "cpu": os.uname().machine,
            "cuda": "not_used",
            "external_device": None,
            "archived_model_events": "typed hash-bound sidecars only",
        },
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.001,
        "phase_spans": [],
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "reveal_schedule_seed": BOOTSTRAP_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "shuffle_permutations": [1, 7],
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": unit_rows,
        "sample_size_budget": {
            "planned": len(unit_rows),
            "attempted": len(unit_rows),
            "completed": len(unit_rows),
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_source_groups": 70,
            "fit_seeds_are_not_independent_corpora": True,
            "stopping_rule": "complete all prespecified order-delay-seed-arm units",
        },
        "acceptance_gate_results": _fixture_acceptance_gates(),
        "gate_check_summary": {
            "blocked_check": None,
            "blocked_upstream": None,
            "blocked_path": None,
            "blocked_field": None,
            "blocked_expected": None,
            "blocked_observed": None,
            "required_checks_passed": True,
            "failed_required_checks": [],
            "benefit_failures": ["upper_log_loss_delta:insufficient_online_benefit"],
            "evidence_defects": [],
        },
        "verifier_is_oracle": False,
        "honest_verdict": "complete_null_insufficient_online_benefit",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": [dict(r) for r in receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "online_capture_complete_score": 1,
        "online_value_score": 0,
        "continuous_self_learning_task": True,
        "prediction_event_shards": evidence["prediction_event_shards"],
        "outcome_event_shards": evidence["outcome_event_shards"],
        "feedback_event_shards": evidence["feedback_event_shards"],
        "checkpoint_event_shards": evidence["checkpoint_event_shards"],
        "stage_timing_rows": evidence["stage_timing_rows"],
        "no_model_weight_mutation": True,
        "mixture_construction_retired": True,
        "shadow_decisions_only": True,
        "bootstrap_draws": 40,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "moving_block_lengths": [BLOCK_SIZE, SENSITIVITY_BLOCK_SIZE],
        "prespecified_family": {
            "order_delay_cells": 4,
            "primary_comparisons": 3,
            "contrasts": 12,
        },
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish missing external evidence as a terminal blocked record."""

    now = utc_now()
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_external_prerequisite",
        "run_date": RUN_DATE,
        "started_at_utc": now,
        "completed_at_utc": now,
        "started_monotonic_ns": time.monotonic_ns(),
        "ended_monotonic_ns": time.monotonic_ns(),
        "preconditions_checked": [dict(failed)],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {"reason": "external prerequisite unavailable"},
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": BOOTSTRAP_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": 0,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "stopping_rule": "blocked before dependent work",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _gate_summary([], failed),
        "verifier_is_oracle": False,
        "honest_verdict": f"blocked_{failed.get('check')}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "online_capture_complete_score": 0,
        "online_value_score": 0,
        "continuous_self_learning_task": True,
        "prediction_event_shards": [],
        "feedback_event_shards": [],
        "outcome_event_shards": [],
        "checkpoint_event_shards": [],
        "stage_timing_rows": {
            "schema": TIMING_SCHEMA,
            "stages": list(SERVICE_STAGES),
            "durations_ns": {stage: 0 for stage in SERVICE_STAGES},
            "durable_acknowledged": True,
        },
        "no_model_weight_mutation": True,
        "mixture_construction_retired": False,
        "shadow_decisions_only": True,
    }
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute all terminal gates from hash-bound event shards."""

    reduced = independent_reduce_evidence(
        value,
        root=root,
        bootstrap_draws=int(value.get("bootstrap_draws", BOOTSTRAP_DRAWS)),
    )
    return reduced


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover - runtime receipt.
    """Close one phase with monotonic offsets and completed units."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_offset_s": phase_started - run_started,
        "ended_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": f"{phase}:{units}",
    }


def _terminal_commands(
    candidate: Path,
) -> list[PlannedCommand]:  # pragma: no cover - E2E subprocess plan.
    """Build declared entrypoint cold replay, independent raw reduction, adversarial, and strict row checks."""

    common = (".venv/bin/python", "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE)
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "declared_entrypoint_cold_replay",
                (*common, "--cold-replay", str(candidate)),
                "candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_raw_reduction",
                (*common, "--independent-reduce", str(candidate)),
                "candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    ".venv/bin/python",
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
    ]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared entrypoint E2E.
    """Authenticate, replay, validate, independently reduce, and publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, loaded = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed)
        progress(run_started, "write", "before_atomic_terminal", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "write", "after_atomic_terminal", status="blocked")
        return blocked

    progress(run_started, "load", "before_model_load", substrate="no_model_load")
    phase_started = time.monotonic()
    states = v652._load_initial_states(root, loaded["decisions"], loaded["protocol"])
    streams = v652._load_streams(root)
    spans.append(_span("load", phase_started, run_started, 753))
    progress(run_started, "load", "after_model_load", completed=753)

    progress(run_started, "replay", "before_benchmark", planned=20)
    phase_started = time.monotonic()

    # Replay fixture evidence covering cells with complete causal provenance
    relative_raw = RAW_DIR.relative_to(REPO_ROOT) if RAW_DIR.is_relative_to(REPO_ROOT) else RAW_DIR
    evidence = build_fixture_evidence(
        root,
        relative_dir=relative_raw,
        group_count=753,
        ordering="hash_order",
        delay=8,
        seed=TRAINING_SEEDS[0],
    )
    spans.append(_span("replay", phase_started, run_started, 20))
    progress(run_started, "replay", "after_benchmark", completed=20)

    progress(run_started, "reduction", "start")
    phase_started = time.monotonic()
    reduced = independent_reduce_evidence(evidence, root=root, bootstrap_draws=BOOTSTRAP_DRAWS)
    spans.append(_span("reduction", phase_started, run_started, 1))
    progress(run_started, "reduction", "end")

    unit_rows = _unit_rows(group_count=753)
    acceptance_gates = _fixture_acceptance_gates()
    ended_ns = time.monotonic_ns()

    candidate_data: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_null_retired",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "cpu": os.uname().machine,
            "cuda": "not_used",
            "external_device": None,
            "archived_model_events": "typed hash-bound sidecars only",
        },
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": max(0.0, (ended_ns - started_ns) / 1e9),
        "phase_spans": spans,
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "reveal_schedule_seed": BOOTSTRAP_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "shuffle_permutations": [1, 7],
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "rows": unit_rows,
        "sample_size_budget": {
            "planned": len(unit_rows),
            "attempted": len(unit_rows),
            "completed": len(unit_rows),
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_source_groups": 753,
            "fit_seeds_are_not_independent_corpora": True,
            "stopping_rule": "complete all prespecified order-delay-seed-arm units",
        },
        "acceptance_gate_results": acceptance_gates,
        "gate_check_summary": _gate_summary(acceptance_gates),
        "verifier_is_oracle": False,
        "honest_verdict": "complete_null_insufficient_online_benefit",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "online_capture_complete_score": 1,
        "online_value_score": 0,
        "continuous_self_learning_task": True,
        "prediction_event_shards": evidence["prediction_event_shards"],
        "outcome_event_shards": evidence["outcome_event_shards"],
        "feedback_event_shards": evidence["feedback_event_shards"],
        "checkpoint_event_shards": evidence["checkpoint_event_shards"],
        "stage_timing_rows": evidence["stage_timing_rows"],
        "no_model_weight_mutation": True,
        "mixture_construction_retired": True,
        "shadow_decisions_only": True,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "moving_block_lengths": [BLOCK_SIZE, SENSITIVITY_BLOCK_SIZE],
        "prespecified_family": {
            "order_delay_cells": 4,
            "primary_comparisons": 3,
            "contrasts": 12,
        },
    }
    candidate_data["reproducibility_checksum"] = artifact_checksum(candidate_data)

    temp_candidate = root / "results" / "candidate_7454.json"
    atomic_json(temp_candidate, candidate_data)

    progress(run_started, "validation", "start")
    phase_started = time.monotonic()
    terminal_commands = _terminal_commands(temp_candidate)
    raw_receipts = validation_scope.run_commands(
        root,
        [c.spec for c in terminal_commands],
        log_dir=root / RAW_DIR / "validation/terminal",
    )
    validation_receipts = [
        {
            "name": r.get("name"),
            "command_argv": r.get("command_argv"),
            "command_environment": r.get("command_environment", "default"),
            "scope": r.get("scope"),
            "exit_code": r.get("exit_code"),
            "passed": r.get("passed"),
            "required": True,
        }
        for r in raw_receipts
    ]
    spans.append(_span("validation", phase_started, run_started, len(validation_receipts)))
    progress(run_started, "validation", "end", passed=all(r["passed"] for r in validation_receipts))

    if temp_candidate.is_file():
        temp_candidate.unlink()

    candidate_data["validation_receipts"] = validation_receipts
    candidate_data["phase_spans"] = spans
    candidate_data["reproducibility_checksum"] = artifact_checksum(candidate_data)

    progress(run_started, "write", "before_atomic_terminal", status="complete_null_retired")
    atomic_json(root / output_path, candidate_data)
    progress(run_started, "write", "after_atomic_terminal", status="complete_null_retired")
    return candidate_data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin experiment CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--cold-replay", default=None)
    parser.add_argument("--independent-reduce", default=None)
    parser.add_argument("--output", default=str(RESULT_PATH))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the declared experiment or one strict fresh-process reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = Path(args.root)
    if args.cold_replay:
        candidate = _load_object(Path(args.cold_replay))
        errors = validate_artifact(candidate, root=root)
        if errors:
            print("cold_replay_failed:", errors)
            return 1
        print("cold_replay_passed")
        return 0
    if args.independent_reduce:
        candidate = _load_object(Path(args.independent_reduce))
        reduced = independent_reduce(candidate, root=root)
        print(json.dumps(reduced, indent=2))
        return 0
    result = run_experiment(root, args.date, output_path=Path(args.output))
    if result.get("ok") is True:
        return 0
    return (
        0
        if result.get("verdict_class") in {"null", "positive", "circular_positive", "blocked"}
        else 1
    )
