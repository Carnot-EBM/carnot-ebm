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
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
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
    return checks, hashes, {
        "ledger": ledger,
        "decisions": decisions,
        "audit": audit,
        "old_result": old_result,
        "protocol": protocol,
    }


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

    return canonical_hash({key: deepcopy(item) for key, item in value.items() if key != "event_hash"})


def outcome_event_hash(value: Mapping[str, Any]) -> str:
    """Hash one post-prediction outcome and its prediction references."""

    return canonical_hash({key: deepcopy(item) for key, item in value.items() if key != "event_hash"})


def feedback_event_hash(value: Mapping[str, Any]) -> str:
    """Hash every feedback operand and numeric update receipt."""

    return canonical_hash({key: deepcopy(item) for key, item in value.items() if key != "event_hash"})


def checkpoint_event_hash(value: Mapping[str, Any]) -> str:
    """Hash a replay checkpoint and its prior checkpoint reference."""

    return canonical_hash({key: deepcopy(item) for key, item in value.items() if key != "event_hash"})


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
    penalized = {
        name: float(old_log_weights[name]) - eta * losses[name] for name in EXPERT_NAMES
    }
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
                json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
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
        "expert_probabilities": {
            name: float(expert_probabilities[name]) for name in EXPERT_NAMES
        },
        "mixture_weights": {name: float(mixture_weights[name]) for name in EXPERT_NAMES},
        "mixture_probability": float(mixture_probability),
        "expert_checkpoint_hashes": {
            name: str(checkpoint_hashes[name]) for name in EXPERT_NAMES
        },
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
        "old_log_weights": deepcopy(dict(old_state["log_weights"])),
        "numeric_update": {
            "penalized_log_weights": deepcopy(dict(numeric["penalized_log_weights"])),
            "posterior_before_share": deepcopy(dict(numeric["posterior_before_share"])),
        },
        "normalizer": {"maximum": numeric["maximum"], "sum_exp": numeric["sum_exp"]},
        "new_log_weights": deepcopy(dict(new_state["log_weights"])),
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
            "log_weights": deepcopy(dict(states[arm]["log_weights"])),
            "feedback_count": states[arm]["feedback_count"],
            "previous_checkpoint_hash": previous[arm],
        }
        value["event_hash"] = checkpoint_event_hash(value)
        previous[arm] = value["event_hash"]
        output.append(value)
    return output
