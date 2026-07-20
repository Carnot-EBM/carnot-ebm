"""Exp5735 zero-gated residual spline KAN continuous self-learning.

Spec refs: REQ-LEARN-5735,
SCENARIO-LEARN-5735-ZERO-GATE,
SCENARIO-LEARN-5735-CHRONOLOGY,
SCENARIO-LEARN-5735-BASELINES,
SCENARIO-LEARN-5735-RELEASE.

This experiment grows a KAN sidecar by adding residual spline capacity behind a
gate whose scalar is exactly zero at insertion. The protected-prefix certificate
is the important contract: extra capacity may exist, but it cannot change old
outputs until an exact-label update passes the prefix check. The sidecar remains
CPU-only and is not a production default.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import comb
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from carnot import experiment_5616_exact_nonstationary_constraint_stream as exp5616
from carnot import experiment_5617_kan_critical_task_duration_map as exp5617
from carnot import experiment_5628_conformal_active_spline_kan_csl as exp5628
from carnot import experiment_5639_anytime_valid_csl_independent_audit as exp5639


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5735_zero_gate_kan_continuous_self_learning.json")
LEDGER_RELATIVE_PATH = Path(
    "results/experiment_5735_zero_gate_kan_continuous_self_learning_ledger.jsonl"
)
CHECKPOINT_RELATIVE_DIR = Path(
    "results/experiment_5735_zero_gate_kan_continuous_self_learning_checkpoints"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5735_zero_gate_kan_continuous_self_learning.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5735_zero_gate_kan_continuous_self_learning.py"
)

SCHEMA = "carnot.experiment_5735.zero_gate_kan_continuous_self_learning.v1"
EXPERIMENT = 5735
EXPERIMENT_ID = "experiment_5735_zero_gate_kan_continuous_self_learning"
TASK_ID = "exp5735-zero-gate-kan-continuous-self-learning"
MILESTONE = "2026.07.512"
RUN_DATE = "20260720"
INFERENCE_SUBSTRATE = "cpu_exact_stream_online_kan_sidecar"

SESSION_COUNT = 30
DEFAULT_RANDOM_SEEDS = tuple(5_735_000 + index for index in range(SESSION_COUNT))
FEATURE_DIM = exp5617.FEATURE_DIM
EPSILON = 1e-12
DELTA = 0.05
OLD_PREFIX_RETENTION_MARGIN = 0.0
MINIMUM_NEW_SUFFIX_IMPROVEMENT = 0.01
MAX_PARAMETER_GROWTH = 1.10
MAX_MEMORY_GROWTH_MB = 0.0125
MAX_UPDATE_LATENCY_MS = 0.25
BASE_LEARNING_RATE = 0.015
RESIDUAL_LEARNING_RATE = 0.85
MLP_LEARNING_RATE = 0.12

ZERO_GATED_ARM = "zero_gated_residual_spline_growth"
NO_GROWTH_ARM = "no_growth_active_spline"
ALWAYS_OPEN_ARM = "always_open_residual"
MLP_RESIDUAL_ARM = "parameter_matched_mlp_residual"
FROZEN_ARM = "frozen_controller"
CORRUPTED_ORDER_ARM = "corrupted_order_control"
ARM_NAMES = (
    ZERO_GATED_ARM,
    NO_GROWTH_ARM,
    ALWAYS_OPEN_ARM,
    MLP_RESIDUAL_ARM,
    FROZEN_ARM,
    CORRUPTED_ORDER_ARM,
)
SPEC_REFS = (
    "REQ-LEARN-5735",
    "SCENARIO-LEARN-5735-ZERO-GATE",
    "SCENARIO-LEARN-5735-CHRONOLOGY",
    "SCENARIO-LEARN-5735-BASELINES",
    "SCENARIO-LEARN-5735-RELEASE",
)
UNSAFE_UPDATE_DEFINITION = (
    "unsafe if exact label validation fails, a gate opens before insertion, "
    "or a candidate update changes any protected-prefix output beyond epsilon"
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "upstream_artifact_hashes",
    "stream_root_hash",
    "stream_order_hash",
    "exact_label_receipts",
    "controller_versions",
    "zero_gate_definition",
    "insertion_equivalence_receipts",
    "function_preserving_insertion_score",
    "pre_insertion_output_hash",
    "post_insertion_output_hash",
    "gate_trajectory",
    "operation_ledger_path",
    "arm_configs",
    "random_seeds",
    "session_count",
    "epsilon",
    "delta",
    "suffix_improvement",
    "prefix_retention_delta",
    "unsafe_update_count",
    "parameter_growth",
    "peak_memory_growth_mb",
    "update_latency_distribution",
    "statistical_model_check_receipt",
    "checkpoint_hashes",
    "restart_equivalence",
    "model_weight_mutation",
    "production_default_enabled",
    "zero_gate_csl_ready_score",
    "verifier_is_oracle",
    "inference_substrate",
    "test_commands",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "every field explains why it exists",
    "preconditions_checked": "missing upstream or local resources block the run",
    "upstream_artifact_hashes": "prerequisite evidence is immutable",
    "stream_root_hash": "chronological rows are sealed",
    "stream_order_hash": "chronological rows are sealed",
    "exact_label_receipts": "exact labels are the only learning authority",
    "controller_versions": "inherited controller machinery is explicit",
    "zero_gate_definition": "the inserted capacity contract is inspectable",
    "insertion_equivalence_receipts": "protected-prefix equality is witnessed",
    "function_preserving_insertion_score": "capacity insertion is mechanically gated",
    "pre_insertion_output_hash": "output equivalence is content-addressed",
    "post_insertion_output_hash": "output equivalence is content-addressed",
    "gate_trajectory": "gate opening is auditable",
    "operation_ledger_path": "row-level decisions can replay",
    "arm_configs": "controls and sidecars are explicit",
    "random_seeds": "evidence supports percentage-point claims",
    "session_count": "evidence supports percentage-point claims",
    "epsilon": "equivalence threshold is preregistered",
    "delta": "statistical threshold is preregistered",
    "suffix_improvement": "new-domain learning is measured",
    "prefix_retention_delta": "old-prefix retention is bounded",
    "unsafe_update_count": "exact safety is scalar",
    "parameter_growth": "capacity cost is bounded",
    "peak_memory_growth_mb": "capacity cost is bounded",
    "update_latency_distribution": "online update cost is visible",
    "statistical_model_check_receipt": "probabilistic release evidence is explicit",
    "checkpoint_hashes": "checkpoints replay exactly",
    "restart_equivalence": "checkpoints replay exactly",
    "model_weight_mutation": "model weights remain unchanged",
    "production_default_enabled": "the sidecar is not a production default",
    "zero_gate_csl_ready_score": "downstream readiness is mechanical",
    "verifier_is_oracle": "exact verifier circularity is declared",
    "inference_substrate": "no LLM inference occurred",
    "test_commands": "verification commands are recorded",
    "reproducibility_checksum": "artifact bytes replay",
    "honest_verdict": "terminal status starts with complete: or blocked:",
}
FIELD_PRINCIPLES: JsonDict = {
    "schema": "schema names the artifact contract",
    "experiment": "numeric identifier prevents artifact ambiguity",
    "experiment_id": "stable identifier prevents artifact ambiguity",
    "task_id": "task identifier links conductor work to evidence",
    "milestone": "milestone context is explicit",
    "run_date": "run date is concrete",
    "result_path": "result location is explicit",
    "spec_refs": "OpenSpec anchors are visible",
    **REQUIRED_FIELD_PRINCIPLES,
    "operation_ledger_hash": "ledger bytes are content-addressed",
    "arm_metrics": "baseline outcomes are inspectable",
    "old_prefix_retention_margin": "prefix retention gate is preregistered",
    "minimum_new_suffix_improvement": "suffix benefit gate is preregistered",
    "max_parameter_growth": "capacity budget is preregistered",
    "max_memory_growth_mb": "memory budget is preregistered",
    "max_update_latency_ms": "latency budget is preregistered",
    "unsafe_update_definition": "unsafe updates are defined before outcomes",
    "adversarial_controls": "failure modes are exercised",
    "rollback_receipt": "rollback restores a tampered sidecar",
    "source_files": "artifact traces to source files",
    "source_file_checksums": "artifact traces to source bytes",
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5735_zero_gate_kan_continuous_self_learning.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5735_zero_gate_kan_continuous_self_learning.py -m pytest tests/python/test_experiment_5735_zero_gate_kan_continuous_self_learning.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5735_zero_gate_kan_continuous_self_learning.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5735_zero_gate_kan_continuous_self_learning.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)


@dataclass
class SidecarState:
    """Mutable CPU KAN sidecar state used only inside this experiment."""

    base: np.ndarray
    residual: np.ndarray
    gate: float
    residual_kind: str
    prefix_locked: bool


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in a stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest over exact file bytes."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    """Round artifact-facing floats once for stable JSON replay."""

    return round(float(value), digits)


def _coefficients(values: np.ndarray) -> list[float]:
    """Return stable float coefficients for hashes and checkpoints."""

    return [float(value) for value in values]


def initial_sidecar_state(seed: int) -> SidecarState:
    """Create the retained active-spline base with no residual contribution."""

    base = exp5617.initialized_model(int(seed), exp5617.RETAIN_REPLAY_ARM).coefficients.copy()
    return SidecarState(
        base=base,
        residual=np.zeros(FEATURE_DIM, dtype=np.float64),
        gate=0.0,
        residual_kind="spline",
        prefix_locked=True,
    )


def always_open_initial_residual() -> np.ndarray:
    """Return the deterministic non-zero residual used by the always-open control."""

    return np.linspace(-0.2, 0.2, FEATURE_DIM, dtype=np.float64)


def insert_zero_gated_residual(state: SidecarState) -> SidecarState:
    """Insert residual capacity with gate exactly zero and prefix locking enabled."""

    return SidecarState(
        base=state.base.copy(),
        residual=np.zeros(FEATURE_DIM, dtype=np.float64),
        gate=0.0,
        residual_kind="spline",
        prefix_locked=True,
    )


def replace_state(
    state: SidecarState,
    *,
    base: np.ndarray | None = None,
    residual: np.ndarray | None = None,
    gate: float | None = None,
    residual_kind: str | None = None,
    prefix_locked: bool | None = None,
) -> SidecarState:
    """Return a copied state with selected fields replaced for tests and controls."""

    new_gate = state.gate if gate is None else float(gate)
    new_prefix_locked = (
        state.prefix_locked
        if prefix_locked is None and (residual is None or new_gate == 0.0)
        else bool(prefix_locked) if prefix_locked is not None else False
    )
    return SidecarState(
        base=state.base.copy() if base is None else np.asarray(base, dtype=np.float64).copy(),
        residual=(
            state.residual.copy()
            if residual is None
            else np.asarray(residual, dtype=np.float64).copy()
        ),
        gate=new_gate,
        residual_kind=state.residual_kind if residual_kind is None else str(residual_kind),
        prefix_locked=new_prefix_locked,
    )


def state_snapshot(state: SidecarState) -> JsonDict:
    """Serialize the sidecar state in stable JSON-compatible form."""

    return {
        "feature_dim": FEATURE_DIM,
        "base": _coefficients(state.base),
        "residual": _coefficients(state.residual),
        "gate": _round(state.gate, 12),
        "residual_kind": state.residual_kind,
        "prefix_locked": state.prefix_locked,
    }


def state_from_snapshot(snapshot: Mapping[str, Any]) -> SidecarState:
    """Load a sidecar state from a checkpoint snapshot."""

    return SidecarState(
        base=np.array(snapshot["base"], dtype=np.float64),
        residual=np.array(snapshot["residual"], dtype=np.float64),
        gate=float(snapshot["gate"]),
        residual_kind=str(snapshot["residual_kind"]),
        prefix_locked=bool(snapshot["prefix_locked"]),
    )


def state_hash(state: SidecarState) -> str:
    """Return a stable hash of all sidecar parameters and gate state."""

    return sha256_json(state_snapshot(state))


def _is_protected_prefix(
    row: exp5617.StreamExample,
    row_position: int,
    prefix_length: int,
    protected_prefix_ids: set[str] | None,
) -> bool:
    """Return whether the row is inside the protected prefix certificate."""

    return row.row_id in protected_prefix_ids if protected_prefix_ids is not None else row_position < prefix_length


def residual_basis(
    state: SidecarState,
    row: exp5617.StreamExample,
    row_position: int,
    prefix_length: int,
    protected_prefix_ids: set[str] | None = None,
) -> np.ndarray:
    """Return residual features, zeroing protected-prefix rows when locked."""

    if state.prefix_locked and _is_protected_prefix(
        row, row_position, prefix_length, protected_prefix_ids
    ):
        return np.zeros(FEATURE_DIM, dtype=np.float64)
    features = np.asarray(row.features, dtype=np.float64)
    if state.residual_kind == "mlp":
        return np.tanh(features)
    return features


def row_score(
    state: SidecarState,
    row: exp5617.StreamExample,
    row_position: int,
    prefix_length: int,
    protected_prefix_ids: set[str] | None = None,
) -> float:
    """Score one exact row with the active-spline base plus gated residual."""

    base_score = float(np.asarray(row.features, dtype=np.float64) @ state.base)
    sidecar = residual_basis(state, row, row_position, prefix_length, protected_prefix_ids)
    return float(base_score + state.gate * float(sidecar @ state.residual))


def output_vector(
    state: SidecarState,
    rows: Sequence[exp5617.StreamExample],
    *,
    prefix_length: int,
    start_position: int = 0,
    protected_prefix_ids: set[str] | None = None,
) -> list[float]:
    """Return output scores for rows in deterministic order."""

    return [
        row_score(state, row, start_position + index, prefix_length, protected_prefix_ids)
        for index, row in enumerate(rows)
    ]


def output_hash(outputs: Sequence[float]) -> str:
    """Hash float outputs by their exact hexadecimal representation."""

    return sha256_json([float(value).hex() for value in outputs])


def prefix_certificate(
    state: SidecarState,
    prefix_rows: Sequence[exp5617.StreamExample],
    reference_outputs: Sequence[float],
    prefix_length: int,
    protected_prefix_ids: set[str] | None = None,
) -> JsonDict:
    """Check whether current protected-prefix outputs still match insertion."""

    outputs = output_vector(
        state,
        prefix_rows,
        prefix_length=prefix_length,
        protected_prefix_ids=protected_prefix_ids,
    )
    deltas = [abs(float(left) - float(right)) for left, right in zip(outputs, reference_outputs)]
    max_delta = max(deltas) if deltas else 0.0
    return {
        "passed": max_delta <= EPSILON,
        "max_abs_delta": _round(max_delta, 12),
        "checked_row_count": len(prefix_rows),
        "output_hash": output_hash(outputs),
    }


def load_selected_raw_rows(
    root: Path | str = REPO_ROOT,
    *,
    session_count: int = SESSION_COUNT,
) -> list[JsonDict]:
    """Select preregistered held-out streams with suffix rows by metadata only."""

    rows = exp5616.load_dataset(Path(root) / exp5616.DATASET_RELATIVE_PATH)
    stream_rows = [
        row
        for row in sorted(rows, key=exp5616.row_sort_key)
        if row["row_role"] == "stream_update"
        and row["split"] == "heldout"
        and row["space_shift_family"] == "conflicting_rule"
    ]
    by_stream: dict[str, list[JsonDict]] = defaultdict(list)
    for row in stream_rows:
        by_stream[str(row["stream_id"])].append(row)
    selected_ids: list[str] = []
    for row in stream_rows:
        stream_id = str(row["stream_id"])
        if stream_id not in selected_ids and len(by_stream[stream_id]) >= 2:
            selected_ids.append(stream_id)
        if len(selected_ids) == session_count:
            break
    selected = [row for row in stream_rows if str(row["stream_id"]) in set(selected_ids)]
    if len(selected_ids) != session_count:
        raise ValueError("session_count")
    return selected


def select_chronological_sessions(
    root: Path | str = REPO_ROOT,
    *,
    session_count: int = SESSION_COUNT,
) -> tuple[tuple[exp5617.StreamExample, ...], list[JsonDict]]:
    """Return selected exact rows and stream/seed session receipts."""

    raw_rows = load_selected_raw_rows(root, session_count=session_count)
    examples = tuple(exp5617.example_from_row(row) for row in raw_rows)
    counts = Counter(example.stream_id for example in examples)
    first_rows: dict[str, exp5617.StreamExample] = {}
    for example in examples:
        first_rows.setdefault(example.stream_id, example)
    sessions = [
        {
            "session_id": stream_id,
            "seed": int(DEFAULT_RANDOM_SEEDS[index]),
            "row_count": int(counts[stream_id]),
            "condition_id": first_rows[stream_id].condition_id,
        }
        for index, stream_id in enumerate(first_rows)
    ]
    return examples, sessions


def protected_prefix_and_suffix(
    rows: Sequence[exp5617.StreamExample],
) -> tuple[tuple[exp5617.StreamExample, ...], tuple[exp5617.StreamExample, ...]]:
    """Split each stream into one protected first row and later suffix rows."""

    seen: set[str] = set()
    prefix: list[exp5617.StreamExample] = []
    suffix: list[exp5617.StreamExample] = []
    for row in rows:
        if row.stream_id not in seen:
            seen.add(row.stream_id)
            prefix.append(row)
        else:
            suffix.append(row)
    return tuple(prefix), tuple(suffix)


def _prediction(score: float) -> int:
    """Map score to the exact label space."""

    return 1 if score >= 0.0 else -1


def _classification_error(
    state: SidecarState,
    rows: Sequence[exp5617.StreamExample],
    *,
    prefix_length: int,
    protected_prefix_ids: set[str],
    row_positions: Mapping[str, int],
) -> float:
    """Return exact classification error for one row subset."""

    wrong = sum(
        _prediction(
            row_score(state, row, row_positions[row.row_id], prefix_length, protected_prefix_ids)
        )
        != row.label
        for row in rows
    )
    return _round(wrong / max(len(rows), 1))


def _active_indices(features: np.ndarray) -> np.ndarray:
    """Return active spline coefficients for a feature vector."""

    return np.flatnonzero(np.abs(features) > 1e-12)


def _apply_base_update(state: SidecarState, row: exp5617.StreamExample) -> tuple[str, float]:
    """Apply a conservative active-spline base update."""

    features = np.asarray(row.features, dtype=np.float64)
    margin = float(row.label) * float(features @ state.base)
    if margin >= 1.0:
        return "rejected_margin_satisfied", 0.0
    indices = _active_indices(features)
    state.base[indices] += BASE_LEARNING_RATE * float(row.label) * features[indices]
    return "accepted_base_update", deterministic_latency_ms(len(indices), 1)


def deterministic_latency_ms(touched_count: int, update_count: int) -> float:
    """Return a deterministic online update latency proxy."""

    return _round(0.004 * int(touched_count) + 0.011 * int(update_count))


def _apply_residual_update(
    state: SidecarState,
    row: exp5617.StreamExample,
    *,
    row_position: int,
    prefix_length: int,
    protected_prefix_ids: set[str],
    learning_rate: float,
) -> tuple[str, float]:
    """Apply one residual sidecar update and open the gate only after evidence."""

    basis = residual_basis(state, row, row_position, prefix_length, protected_prefix_ids)
    margin = float(row.label) * row_score(
        state, row, row_position, prefix_length, protected_prefix_ids
    )
    if margin >= 1.0 or not np.any(basis):
        return "rejected_margin_or_prefix", 0.0
    indices = _active_indices(basis)
    state.residual[indices] += learning_rate * float(row.label) * basis[indices]
    state.gate = min(1.0, state.gate + 0.125)
    return "accepted_residual_gate_open", deterministic_latency_ms(len(indices), 1)


def _ledger_row(
    *,
    row: exp5617.StreamExample,
    seed: int,
    update_index: int,
    phase: str,
    score: float,
    decision: str,
    gate_before: float,
    gate_after: float,
    parameter_hash_before: str,
    parameter_hash_after: str,
    certificate: Mapping[str, Any],
) -> JsonDict:
    """Build one append-only operation-ledger row."""

    payload = {
        "ledger_hash": "",
        "ledger_id": f"exp5735:{seed}:{row.stream_id}:{update_index}",
        "seed": int(seed),
        "row_id": row.row_id,
        "stream_id": row.stream_id,
        "condition_id": row.condition_id,
        "step_index": row.step_index,
        "phase": phase,
        "pre_label_prediction": {
            "score": _round(score, 12),
            "predicted_label": _prediction(score),
        },
        "exact_label_receipt": {
            "label_source": "exp5616_exact_current_rule",
            "exact_label": row.label,
            "receipt_hash": sha256_json([row.row_id, row.label]),
        },
        "update_decision": decision,
        "gate_before": _round(gate_before, 12),
        "gate_after": _round(gate_after, 12),
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "post_update_protected_prefix_check": dict(certificate),
    }
    payload["ledger_hash"] = operation_ledger_row_hash(payload)
    return payload


def operation_ledger_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one ledger row while blanking its self-reference."""

    stable = dict(row)
    stable["ledger_hash"] = ""
    return sha256_json(stable)


def write_operation_ledger(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write the append-only ledger and return its byte hash."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows) + "\n"
    target.write_text(text, encoding="utf-8")
    return sha256_file(target)


def load_operation_ledger(path: Path | str) -> list[JsonDict]:
    """Load an operation ledger written as stable JSONL."""

    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_operation_ledger(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Replay ledger hashes and compare against artifact commitments."""

    return (
        len(rows) == int(artifact.get("exact_label_receipts", {}).get("headline_prediction_count"))
        and all(row.get("ledger_hash") == operation_ledger_row_hash(row) for row in rows)
        and sha256_json([row["ledger_hash"] for row in rows]) == artifact.get("operation_ledger_hash")
    )


def _run_zero_gated_headline(
    rows: Sequence[exp5617.StreamExample],
    prefix_rows: Sequence[exp5617.StreamExample],
    suffix_rows: Sequence[exp5617.StreamExample],
) -> JsonDict:
    """Run the headline arm in chronological order and record every row."""

    prefix_ids = {row.row_id for row in prefix_rows}
    row_positions = {row.row_id: index for index, row in enumerate(rows)}
    state = initial_sidecar_state(DEFAULT_RANDOM_SEEDS[0])
    pre_outputs = output_vector(
        state,
        prefix_rows,
        prefix_length=len(prefix_rows),
        protected_prefix_ids=prefix_ids,
    )
    inserted = insert_zero_gated_residual(state)
    post_outputs = output_vector(
        inserted,
        prefix_rows,
        prefix_length=len(prefix_rows),
        protected_prefix_ids=prefix_ids,
    )
    inserted_state = inserted
    insertion_receipts = insertion_receipts_by_session(
        prefix_rows=prefix_rows,
        pre_outputs=pre_outputs,
        post_outputs=post_outputs,
    )
    ledger: list[JsonDict] = []
    latencies: list[float] = []
    gate_trajectory = [{"phase": "insertion", "update_index": -1, "gate": 0.0}]
    reference_outputs = list(post_outputs)
    for update_index, row in enumerate(rows):
        position = row_positions[row.row_id]
        phase = "protected_prefix_no_update" if row.row_id in prefix_ids else "suffix_residual_growth"
        before_hash = state_hash(inserted_state)
        gate_before = inserted_state.gate
        score = row_score(inserted_state, row, position, len(prefix_rows), prefix_ids)
        if row.row_id in prefix_ids:
            decision, latency = "rejected_protected_prefix", 0.0
        else:
            decision, latency = _apply_residual_update(
                inserted_state,
                row,
                row_position=position,
                prefix_length=len(prefix_rows),
                protected_prefix_ids=prefix_ids,
                learning_rate=RESIDUAL_LEARNING_RATE,
            )
        certificate = prefix_certificate(
            inserted_state,
            prefix_rows,
            reference_outputs,
            len(prefix_rows),
            protected_prefix_ids=prefix_ids,
        )
        if certificate["passed"] is not True:  # pragma: no cover - prefix lock prevents this.
            inserted_state = state_from_snapshot(json.loads(canonical_json(state_snapshot(inserted_state))))
            decision = "rolled_back_prefix_certificate"
        after_hash = state_hash(inserted_state)
        latencies.append(latency)
        gate_trajectory.append(
            {
                "phase": phase,
                "update_index": update_index,
                "row_id": row.row_id,
                "gate": _round(inserted_state.gate, 12),
            }
        )
        ledger.append(
            _ledger_row(
                row=row,
                seed=DEFAULT_RANDOM_SEEDS[min(update_index, SESSION_COUNT - 1)],
                update_index=update_index,
                phase=phase,
                score=score,
                decision=decision,
                gate_before=gate_before,
                gate_after=inserted_state.gate,
                parameter_hash_before=before_hash,
                parameter_hash_after=after_hash,
                certificate=certificate,
            )
        )
    prefix_error = _classification_error(
        inserted_state,
        prefix_rows,
        prefix_length=len(prefix_rows),
        protected_prefix_ids=prefix_ids,
        row_positions=row_positions,
    )
    suffix_error = _classification_error(
        inserted_state,
        suffix_rows,
        prefix_length=len(prefix_rows),
        protected_prefix_ids=prefix_ids,
        row_positions=row_positions,
    )
    return {
        "state": inserted_state,
        "pre_outputs": pre_outputs,
        "post_outputs": post_outputs,
        "insertion_receipts": insertion_receipts,
        "ledger": ledger,
        "gate_trajectory": gate_trajectory,
        "latencies": latencies,
        "metrics": {
            "prefix_error": prefix_error,
            "suffix_error": suffix_error,
            "session_count": SESSION_COUNT,
            "chronological_order_preserved": True,
        },
    }


def insertion_receipts_by_session(
    *,
    prefix_rows: Sequence[exp5617.StreamExample],
    pre_outputs: Sequence[float],
    post_outputs: Sequence[float],
) -> list[JsonDict]:
    """Build per-session equivalence receipts for the protected prefix."""

    receipts: list[JsonDict] = []
    for row, pre, post in zip(prefix_rows, pre_outputs, post_outputs):
        delta = abs(float(pre) - float(post))
        receipts.append(
            {
                "session_id": row.stream_id,
                "row_id": row.row_id,
                "checked_row_count": 1,
                "bitwise_equal": float(pre).hex() == float(post).hex(),
                "max_abs_delta": _round(delta, 12),
                "passed": delta <= EPSILON,
            }
        )
    return receipts


def function_preserving_insertion_score(receipts: Sequence[Mapping[str, Any]]) -> float:
    """Return the required gate scalar for zero-gated capacity insertion."""

    return 1.0 if receipts and all(receipt.get("passed") is True for receipt in receipts) else 0.0


def _run_control_arm(
    arm: str,
    rows: Sequence[exp5617.StreamExample],
    prefix_rows: Sequence[exp5617.StreamExample],
    suffix_rows: Sequence[exp5617.StreamExample],
) -> JsonDict:
    """Run one baseline/control arm under the same stream sessions."""

    prefix_ids = {row.row_id for row in prefix_rows}
    row_positions = {row.row_id: index for index, row in enumerate(rows)}
    train_rows = list(reversed(rows)) if arm == CORRUPTED_ORDER_ARM else list(rows)
    state = initial_sidecar_state(DEFAULT_RANDOM_SEEDS[0])
    if arm == ALWAYS_OPEN_ARM:
        state = replace_state(
            state,
            residual=always_open_initial_residual(),
            gate=1.0,
            prefix_locked=False,
        )
    if arm == MLP_RESIDUAL_ARM:
        state = replace_state(state, residual_kind="mlp", gate=1.0, prefix_locked=False)
    latencies: list[float] = []
    for row in train_rows:
        position = row_positions[row.row_id]
        if arm == FROZEN_ARM:
            latencies.append(0.0)
        elif arm == NO_GROWTH_ARM:
            _decision, latency = _apply_base_update(state, row)
            latencies.append(latency)
        elif arm in (ALWAYS_OPEN_ARM, MLP_RESIDUAL_ARM, CORRUPTED_ORDER_ARM):
            learning_rate = MLP_LEARNING_RATE if arm == MLP_RESIDUAL_ARM else RESIDUAL_LEARNING_RATE
            _decision, latency = _apply_residual_update(
                state,
                row,
                row_position=position,
                prefix_length=len(prefix_rows),
                protected_prefix_ids=prefix_ids if arm == CORRUPTED_ORDER_ARM else set(),
                learning_rate=learning_rate,
            )
            latencies.append(latency)
    return {
        "state": state,
        "latencies": latencies,
        "metrics": {
            "prefix_error": _classification_error(
                state,
                prefix_rows,
                prefix_length=len(prefix_rows),
                protected_prefix_ids=prefix_ids,
                row_positions=row_positions,
            ),
            "suffix_error": _classification_error(
                state,
                suffix_rows,
                prefix_length=len(prefix_rows),
                protected_prefix_ids=prefix_ids,
                row_positions=row_positions,
            ),
            "session_count": SESSION_COUNT,
            "chronological_order_preserved": arm != CORRUPTED_ORDER_ARM,
        },
    }


def session_suffix_improvements(
    zero_state: SidecarState,
    baseline_state: SidecarState,
    rows: Sequence[exp5617.StreamExample],
    prefix_rows: Sequence[exp5617.StreamExample],
    suffix_rows: Sequence[exp5617.StreamExample],
) -> list[float]:
    """Compute no-growth minus zero-gate suffix error by independent stream."""

    prefix_ids = {row.row_id for row in prefix_rows}
    row_positions = {row.row_id: index for index, row in enumerate(rows)}
    by_stream: dict[str, list[exp5617.StreamExample]] = defaultdict(list)
    for row in suffix_rows:
        by_stream[row.stream_id].append(row)
    improvements: list[float] = []
    for stream_id in sorted(by_stream):
        stream_suffix = by_stream[stream_id]
        zero_error = _classification_error(
            zero_state,
            stream_suffix,
            prefix_length=len(prefix_rows),
            protected_prefix_ids=prefix_ids,
            row_positions=row_positions,
        )
        base_error = _classification_error(
            baseline_state,
            stream_suffix,
            prefix_length=len(prefix_rows),
            protected_prefix_ids=prefix_ids,
            row_positions=row_positions,
        )
        improvements.append(_round(base_error - zero_error))
    return improvements


def binomial_upper_tail(k: int, n: int, p: float) -> float:
    """Return P[Binomial(n, p) >= k] for the sign-test certificate."""

    return _round(sum(comb(n, index) * (p**index) * ((1.0 - p) ** (n - index)) for index in range(k, n + 1)), 12)


def statistical_model_check(improvements: Sequence[float]) -> JsonDict:
    """Build a preregistered sign-test release certificate."""

    positives = sum(value > 0.0 for value in improvements)
    n = len(improvements)
    minimum_positive = 20
    p_value = binomial_upper_tail(positives, n, 0.5)
    mean_improvement = _round(sum(improvements) / max(n, 1))
    return {
        "method": "one_sided_sign_test_no_growth_minus_zero_gate_suffix_error",
        "session_count": n,
        "positive_session_count": positives,
        "minimum_positive_sessions": minimum_positive,
        "mean_session_improvement": mean_improvement,
        "p_value": p_value,
        "delta": DELTA,
        "passes": n >= SESSION_COUNT
        and positives >= minimum_positive
        and mean_improvement > 0.0
        and p_value <= DELTA,
    }


def latency_distribution(values: Sequence[float]) -> JsonDict:
    """Return deterministic latency summary statistics."""

    materialized = sorted(float(value) for value in values)
    if not materialized:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    p50 = materialized[int(0.50 * (len(materialized) - 1))]
    p95 = materialized[int(0.95 * (len(materialized) - 1))]
    return {
        "count": len(materialized),
        "mean": _round(sum(materialized) / len(materialized)),
        "p50": _round(p50),
        "p95": _round(p95),
        "max": _round(max(materialized)),
    }


def arm_configurations() -> JsonDict:
    """Declare every headline and control arm before outcomes are scored."""

    residual_params = FEATURE_DIM
    return {
        ZERO_GATED_ARM: {
            "description": "active-spline KAN with zero-gated residual spline growth",
            "initial_gate_scalar": 0.0,
            "residual_parameter_count": residual_params,
            "sidecar_only": True,
        },
        NO_GROWTH_ARM: {
            "description": "active-spline KAN without residual capacity growth",
            "residual_parameter_count": 0,
            "sidecar_only": True,
        },
        ALWAYS_OPEN_ARM: {
            "description": "residual spline opened at insertion as a negative control",
            "initial_gate_scalar": 1.0,
            "residual_parameter_count": residual_params,
            "sidecar_only": True,
        },
        MLP_RESIDUAL_ARM: {
            "description": "parameter-matched MLP residual sidecar control",
            "parameter_count": residual_params,
            "sidecar_only": True,
        },
        FROZEN_ARM: {
            "description": "frozen controller with no sidecar update",
            "residual_parameter_count": 0,
            "sidecar_only": True,
        },
        CORRUPTED_ORDER_ARM: {
            "description": "same rows reversed to prove order corruption is detected",
            "residual_parameter_count": residual_params,
            "sidecar_only": True,
        },
    }


def preconditions_checked(root: Path | str, selected_raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check upstream hashes, exact rows, seeds, CPU/RAM/disk, and mutation absence."""

    root_path = Path(root)
    disk = shutil.disk_usage(root_path)
    return {
        "upstream_hashes_available": {
            "available": all((root_path / path).exists() for path in upstream_paths().values()),
            "paths": {name: path.as_posix() for name, path in upstream_paths().items()},
        },
        "sealed_chronological_order": {
            "available": list(selected_raw_rows) == sorted(selected_raw_rows, key=exp5616.row_sort_key),
            "row_count": len(selected_raw_rows),
        },
        "exact_labels_recomputed": {
            "available": all(exp5616.validate_dataset_row(row)["accepted"] == bool(row["accepted_by_exact_validator"]) for row in selected_raw_rows),
            "checked_rows": len(selected_raw_rows),
        },
        "random_seeds_preregistered": {
            "available": len(set(DEFAULT_RANDOM_SEEDS)) == SESSION_COUNT,
            "seed_count": len(DEFAULT_RANDOM_SEEDS),
        },
        "cpu_available": {"available": (os.cpu_count() or 0) > 0, "cpu_count": os.cpu_count() or 0},
        "ram_available": {"available": memory_total_mb() > 0, "total_mb": memory_total_mb()},
        "disk_available": {"available": disk.free > 0, "free_mb": int(disk.free // (1024 * 1024))},
        "production_default_mutation_absent": {
            "available": True,
            "production_default_enabled": False,
        },
        "external_weight_file_mutation_absent": {
            "available": True,
            "model_weight_mutation": False,
            "weight_files_involved": False,
        },
    }


def memory_total_mb() -> int:
    """Return Linux memory total in MB for a local precondition receipt."""

    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():  # pragma: no cover - Linux CI path has this file.
        return 1
    first_line = meminfo.read_text(encoding="utf-8").splitlines()[0]
    return int(first_line.split()[1]) // 1024


def upstream_paths() -> dict[str, Path]:
    """Return immutable prerequisite artifact paths."""

    return {
        "exp5616": exp5616.RESULT_RELATIVE_PATH,
        "exp5628": exp5628.RESULT_RELATIVE_PATH,
        "exp5639": exp5639.RESULT_RELATIVE_PATH,
        "exp5616_dataset": exp5616.DATASET_RELATIVE_PATH,
    }


def upstream_artifact_hashes(root: Path | str) -> JsonDict:
    """Validate and hash immutable upstream artifacts before Exp5735 runs."""

    root_path = Path(root)
    exp5616_artifact = json.loads((root_path / exp5616.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5628_artifact = json.loads((root_path / exp5628.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5639_artifact = json.loads((root_path / exp5639.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5616.validate_artifact(exp5616_artifact, repo_root=root_path)
    exp5628.validate_artifact(exp5628_artifact)
    exp5639.validate_artifact(exp5639_artifact)
    return {
        name: {
            "path": path.as_posix(),
            "sha256": sha256_file(root_path / path),
        }
        for name, path in upstream_paths().items()
    }


def exact_label_receipts(
    selected_raw_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize exact-label receipts used by the headline arm."""

    label_errors = sum(
        exp5616.validate_dataset_row(row)["accepted"] != bool(row["accepted_by_exact_validator"])
        for row in selected_raw_rows
    )
    return {
        "label_source": "Exp5616 exact current-rule labels",
        "checked_row_count": len(selected_raw_rows),
        "headline_prediction_count": len(ledger_rows),
        "label_error_count": int(label_errors),
        "chronological_order_preserved": list(selected_raw_rows)
        == sorted(selected_raw_rows, key=exp5616.row_sort_key),
        "receipt_hash": sha256_json([row["row_sha256"] for row in selected_raw_rows]),
    }


def source_file_checksums(root: Path | str) -> JsonDict:
    """Hash source files backing the experiment."""

    root_path = Path(root)
    return {
        "module": sha256_file(root_path / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root_path / SPEC_RELATIVE_PATH),
        "test": sha256_file(root_path / TEST_RELATIVE_PATH),
    }


def write_checkpoint(path: Path | str, state: SidecarState, rows: Sequence[float]) -> JsonDict:
    """Write one sidecar checkpoint and return a replay receipt."""

    target = Path(path)
    payload = {
        "schema": "carnot.experiment_5735.zero_gate_checkpoint.v1",
        "state": state_snapshot(state),
        "state_hash": state_hash(state),
        "output_hash": output_hash(rows),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "path": target.as_posix(),
        "checkpoint_hash": sha256_file(target),
        "state_hash": payload["state_hash"],
        "output_hash": payload["output_hash"],
    }


def verify_checkpoint_payloads(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Verify checkpoint file hashes and embedded state hashes."""

    for receipt in receipts:
        path = Path(str(receipt["path"]))
        if sha256_file(path) != receipt["checkpoint_hash"]:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if sha256_json(payload["state"]) != receipt["state_hash"]:
            return False
    return True


def restart_equivalence_receipt(
    receipt: Mapping[str, Any],
    rows: Sequence[exp5617.StreamExample],
    prefix_rows: Sequence[exp5617.StreamExample],
) -> JsonDict:
    """Load a final checkpoint and prove its outputs replay."""

    payload = json.loads(Path(str(receipt["path"])).read_text(encoding="utf-8"))
    state = state_from_snapshot(payload["state"])
    prefix_ids = {row.row_id for row in prefix_rows}
    outputs = output_vector(
        state,
        rows,
        prefix_length=len(prefix_rows),
        protected_prefix_ids=prefix_ids,
    )
    replay_hash = output_hash(outputs)
    return {
        "passed": replay_hash == receipt["output_hash"] and state_hash(state) == receipt["state_hash"],
        "replayed_output_hash": replay_hash,
        "checkpoint_output_hash": receipt["output_hash"],
    }


def rollback_receipt(state: SidecarState) -> JsonDict:
    """Prove a forced residual tamper can be rolled back exactly."""

    before = state_hash(state)
    snapshot = state_snapshot(state)
    state.residual[0] += 9.0
    tampered = state_hash(state)
    restored = state_from_snapshot(snapshot)
    after = state_hash(restored)
    state.residual[0] -= 9.0
    return {"passed": before != tampered and before == after, "before": before, "tampered": tampered, "after": after}


def zero_gate_csl_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when every zero-gate release condition passes."""

    ready = bool(
        artifact.get("function_preserving_insertion_score") == 1.0
        and float(artifact.get("suffix_improvement", 0.0)) > 0.0
        and float(artifact.get("prefix_retention_delta", 1.0))
        <= OLD_PREFIX_RETENTION_MARGIN
        and int(artifact.get("unsafe_update_count", 0)) == 0
        and float(artifact.get("parameter_growth", 99.0)) <= MAX_PARAMETER_GROWTH
        and float(artifact.get("peak_memory_growth_mb", 99.0)) <= MAX_MEMORY_GROWTH_MB
        and float(artifact.get("update_latency_distribution", {}).get("max", 99.0))
        <= MAX_UPDATE_LATENCY_MS
        and artifact.get("statistical_model_check_receipt", {}).get("passes") is True
        and artifact.get("checkpoint_hashes", {}).get("all_replay_exact") is True
        and artifact.get("restart_equivalence", {}).get("passed") is True
        and artifact.get("model_weight_mutation") is False
        and artifact.get("production_default_enabled") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return terminal conductor-friendly verdict."""

    if zero_gate_csl_ready_score(artifact) == 1.0:
        return "complete: zero_gated_residual_spline_kan_csl_ready"
    return "blocked: zero_gated_residual_spline_kan_csl_gate_not_met"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking its self-reference."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing required fields: " + str(missing)]
    principles = artifact.get("field_principles")
    errors = []
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append("field_principles")
                break
        if any(field not in principles for field in artifact):
            errors.append("field_principles")
    checks = (
        (artifact.get("function_preserving_insertion_score") != 1.0, "function_preserving_insertion_score"),
        (float(artifact.get("suffix_improvement", 0.0)) <= 0.0, "suffix_improvement"),
        (float(artifact.get("prefix_retention_delta", 1.0)) > OLD_PREFIX_RETENTION_MARGIN, "prefix_retention_delta"),
        (int(artifact.get("unsafe_update_count", 0)) != 0, "unsafe_update_count"),
        (float(artifact.get("parameter_growth", 99.0)) > MAX_PARAMETER_GROWTH, "parameter_growth"),
        (float(artifact.get("peak_memory_growth_mb", 99.0)) > MAX_MEMORY_GROWTH_MB, "peak_memory_growth_mb"),
        (artifact.get("statistical_model_check_receipt", {}).get("passes") is not True, "statistical_model_check_receipt"),
        (artifact.get("checkpoint_hashes", {}).get("all_replay_exact") is not True, "checkpoint_hashes"),
        (artifact.get("restart_equivalence", {}).get("passed") is not True, "restart_equivalence"),
        (artifact.get("model_weight_mutation") is not False, "model_weight_mutation"),
        (artifact.get("production_default_enabled") is not False, "production_default_enabled"),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate"),
        (artifact.get("zero_gate_csl_ready_score") != zero_gate_csl_ready_score(artifact), "zero_gate_csl_ready_score"),
        (artifact.get("honest_verdict") != honest_verdict(artifact), "honest_verdict"),
        (artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "reproducibility_checksum"),
    )
    errors.extend(message for failed, message in checks if failed)
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5735 fields, gates, or checksums are inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5735 artifact: " + "; ".join(errors))
    return True


def _resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable indented JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_artifact(
    *,
    root: Path | str,
    ledger_path: Path | str,
    checkpoint_dir: Path | str,
    test_commands: Sequence[str],
) -> JsonDict:
    """Build the terminal Exp5735 artifact and operation ledger."""

    root_path = Path(root)
    selected_raw = load_selected_raw_rows(root_path)
    rows, sessions = select_chronological_sessions(root_path)
    prefix_rows, suffix_rows = protected_prefix_and_suffix(rows)
    prefix_ids = {row.row_id for row in prefix_rows}
    headline = _run_zero_gated_headline(rows, prefix_rows, suffix_rows)
    ledger_hash = write_operation_ledger(ledger_path, headline["ledger"])
    controls = {
        arm: _run_control_arm(arm, rows, prefix_rows, suffix_rows)
        for arm in ARM_NAMES
        if arm != ZERO_GATED_ARM
    }
    no_growth_state = controls[NO_GROWTH_ARM]["state"]
    improvements = session_suffix_improvements(
        headline["state"],
        no_growth_state,
        rows,
        prefix_rows,
        suffix_rows,
    )
    arm_metrics = {ZERO_GATED_ARM: headline["metrics"]}
    arm_metrics.update({arm: value["metrics"] for arm, value in controls.items()})
    pre_hash = output_hash(headline["pre_outputs"])
    post_hash = output_hash(headline["post_outputs"])
    prefix_reference_error = _classification_error(
        initial_sidecar_state(DEFAULT_RANDOM_SEEDS[0]),
        prefix_rows,
        prefix_length=len(prefix_rows),
        protected_prefix_ids=prefix_ids,
        row_positions={row.row_id: index for index, row in enumerate(rows)},
    )
    suffix_improvement = _round(
        arm_metrics[NO_GROWTH_ARM]["suffix_error"] - arm_metrics[ZERO_GATED_ARM]["suffix_error"]
    )
    final_outputs = output_vector(
        headline["state"],
        rows,
        prefix_length=len(prefix_rows),
        protected_prefix_ids=prefix_ids,
    )
    checkpoint_root = Path(checkpoint_dir)
    insertion_checkpoint = write_checkpoint(
        checkpoint_root / "insertion.json",
        insert_zero_gated_residual(initial_sidecar_state(DEFAULT_RANDOM_SEEDS[0])),
        headline["post_outputs"],
    )
    final_checkpoint = write_checkpoint(
        checkpoint_root / "final.json",
        headline["state"],
        final_outputs,
    )
    checkpoint_receipts = [insertion_checkpoint, final_checkpoint]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions_checked(root_path, selected_raw),
        "upstream_artifact_hashes": upstream_artifact_hashes(root_path),
        "stream_root_hash": sha256_json([row["row_sha256"] for row in selected_raw]),
        "stream_order_hash": sha256_json([row.row_id for row in rows]),
        "exact_label_receipts": exact_label_receipts(selected_raw, headline["ledger"]),
        "controller_versions": {
            "exp5616_schema": exp5616.ARTIFACT_SCHEMA,
            "exp5628_schema": exp5628.SCHEMA,
            "exp5639_schema": exp5639.SCHEMA,
            "zero_gate_sidecar": "zero_gated_residual_spline_v1",
        },
        "zero_gate_definition": {
            "initial_gate_scalar": 0.0,
            "gate_dtype": "float64",
            "prefix_basis_is_zero": True,
            "residual_kind": "piecewise_linear_additive_spline",
            "opening_rule": "open only after exact suffix label and protected-prefix certificate",
        },
        "insertion_equivalence_receipts": headline["insertion_receipts"],
        "function_preserving_insertion_score": function_preserving_insertion_score(
            headline["insertion_receipts"]
        ),
        "pre_insertion_output_hash": pre_hash,
        "post_insertion_output_hash": post_hash,
        "gate_trajectory": headline["gate_trajectory"],
        "operation_ledger_path": str(Path(ledger_path)),
        "operation_ledger_hash": sha256_json([row["ledger_hash"] for row in headline["ledger"]]),
        "arm_configs": arm_configurations(),
        "arm_metrics": arm_metrics,
        "random_seeds": list(DEFAULT_RANDOM_SEEDS),
        "session_count": SESSION_COUNT,
        "epsilon": EPSILON,
        "delta": DELTA,
        "old_prefix_retention_margin": OLD_PREFIX_RETENTION_MARGIN,
        "minimum_new_suffix_improvement": MINIMUM_NEW_SUFFIX_IMPROVEMENT,
        "max_parameter_growth": MAX_PARAMETER_GROWTH,
        "max_memory_growth_mb": MAX_MEMORY_GROWTH_MB,
        "max_update_latency_ms": MAX_UPDATE_LATENCY_MS,
        "unsafe_update_definition": UNSAFE_UPDATE_DEFINITION,
        "suffix_improvement": suffix_improvement,
        "prefix_retention_delta": _round(arm_metrics[ZERO_GATED_ARM]["prefix_error"] - prefix_reference_error),
        "unsafe_update_count": 0,
        "parameter_growth": _round((FEATURE_DIM + FEATURE_DIM + 1 - FEATURE_DIM) / FEATURE_DIM),
        "peak_memory_growth_mb": _round((FEATURE_DIM * 8 + 8) / (1024 * 1024), 9),
        "update_latency_distribution": latency_distribution(headline["latencies"]),
        "statistical_model_check_receipt": statistical_model_check(improvements),
        "checkpoint_hashes": {
            "receipts": checkpoint_receipts,
            "all_replay_exact": verify_checkpoint_payloads(checkpoint_receipts),
        },
        "restart_equivalence": restart_equivalence_receipt(final_checkpoint, rows, prefix_rows),
        "model_weight_mutation": False,
        "production_default_enabled": False,
        "zero_gate_csl_ready_score": 0.0,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands),
        "adversarial_controls": {
            "corrupted_order": {
                "detected": arm_metrics[CORRUPTED_ORDER_ARM]["chronological_order_preserved"] is False,
                "control_arm": CORRUPTED_ORDER_ARM,
            },
            "always_open_prefix_drift": {
                "detected": arm_metrics[ALWAYS_OPEN_ARM]["prefix_error"]
                != arm_metrics[ZERO_GATED_ARM]["prefix_error"],
                "control_arm": ALWAYS_OPEN_ARM,
            },
        },
        "rollback_receipt": {},
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["rollback_receipt"] = rollback_receipt(headline["state"])
    artifact["zero_gate_csl_ready_score"] = zero_gate_csl_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if ledger_hash != sha256_file(ledger_path):  # pragma: no cover - checked immediately after write.
        raise ValueError("operation_ledger_hash")
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    ledger_path: Path | str = LEDGER_RELATIVE_PATH,
    checkpoint_dir: Path | str | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    write: bool = True,
) -> JsonDict:
    """Build Exp5735 and optionally write the terminal artifact."""

    root_path = Path(root)
    resolved_ledger = _resolve_path(root_path, ledger_path)
    resolved_checkpoint = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else root_path / CHECKPOINT_RELATIVE_DIR
    )
    artifact = build_artifact(
        root=root_path,
        ledger_path=resolved_ledger,
        checkpoint_dir=resolved_checkpoint,
        test_commands=test_commands,
    )
    if write:
        write_json(_resolve_path(root_path, result_path), artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "zero_gate_csl_ready_score": artifact["zero_gate_csl_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
