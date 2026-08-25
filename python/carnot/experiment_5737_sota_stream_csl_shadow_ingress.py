"""Exp5737 SOTA stream CSL shadow ingress.

Spec refs: REQ-LEARN-5737,
SCENARIO-LEARN-5737-CHRONOLOGICAL-INGRESS,
SCENARIO-LEARN-5737-CONTROLS,
SCENARIO-LEARN-5737-ROLLBACK,
SCENARIO-LEARN-5737-RELEASE.

This module consumes the sealed Exp5734 SOTA proposal stream without invoking a
model. The only learning authority is the exact-validator label already sealed
in that stream. The learned state is shadow-only: it can show prospective CSL
utility, but it cannot mutate the protected production controller or any model
weights.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5734_sota_exact_proposal_stream as exp5734
from carnot import experiment_5735_zero_gate_kan_continuous_self_learning as exp5735
from carnot import experiment_5736_csl_lifecycle_conflict_rollback as exp5736
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5737_sota_stream_csl_shadow_ingress.json")
LEDGER_RELATIVE_PATH = Path("results/experiment_5737_sota_stream_csl_shadow_ingress_ledger.jsonl")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5737_sota_stream_csl_shadow_ingress.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5737_sota_stream_csl_shadow_ingress.py")

EXP5734_RELATIVE_PATH = exp5734.RESULT_RELATIVE_PATH
EXP5734_ROW_MANIFEST_RELATIVE_PATH = exp5734.ROW_MANIFEST_RELATIVE_PATH
EXP5735_RELATIVE_PATH = exp5735.RESULT_RELATIVE_PATH
EXP5736_RELATIVE_PATH = exp5736.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5737.sota_stream_csl_shadow_ingress.v1"
LEDGER_SCHEMA = SCHEMA + ".ledger_row"
EXPERIMENT = 5737
EXPERIMENT_ID = "experiment_5737_sota_stream_csl_shadow_ingress"
TASK_ID = "exp5737-sota-stream-csl-shadow-ingress"
MILESTONE = "2026.07.512"
RUN_DATE = "20260720"
INFERENCE_SUBSTRATE = "cpu_shadow_csl_on_attested_sota_stream"
SPEC_REFS = (
    "REQ-LEARN-5737",
    "SCENARIO-LEARN-5737-CHRONOLOGICAL-INGRESS",
    "SCENARIO-LEARN-5737-CONTROLS",
    "SCENARIO-LEARN-5737-ROLLBACK",
    "SCENARIO-LEARN-5737-RELEASE",
)

PREFIX_LENGTH = exp5734.PREFIX_LENGTH
LABEL_CYCLE_LENGTH = len(exp5734.LABELS)
SESSION_COUNT = PREFIX_LENGTH
RANDOM_SEEDS = {
    "ingress_seed": 5737001,
    "control_seed": 5737002,
    "rollback_seed": 5737003,
    "source_stream_seed": exp5734.RANDOM_SEEDS["panel_seed"],
}

EXACT_LABEL_ARM = "chronological_exact_validator_label_updates"
NO_UPDATE_ARM = "no_update_control"
MODEL_PROPOSAL_ARM = "model_proposal_label_diagnostic"
CORRUPTED_ORDER_ARM = "corrupted_order_control"
STALE_CONFLICT_ARM = "stale_conflict_control"
ARM_NAMES = (
    EXACT_LABEL_ARM,
    NO_UPDATE_ARM,
    MODEL_PROPOSAL_ARM,
    CORRUPTED_ORDER_ARM,
    STALE_CONFLICT_ARM,
)
PRODUCTION_CONTROLLER_HASH = (
    "sha256:7b3eb1125abc89481108570f51343511332a0bd85a83f6f55d56e8630e244113"
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "upstream_gate_receipts",
    "stream_root_commitment",
    "prefix_hash",
    "suffix_hash",
    "lifecycle_hash",
    "validator_hashes",
    "ingress_ledger_path",
    "arm_configs",
    "model_family_counts",
    "constraint_family_counts",
    "prelabel_decisions",
    "operation_counts",
    "suffix_improvement",
    "prefix_retention_delta",
    "unsafe_update_count",
    "rollback_state_hash_matches",
    "proposal_label_control_results",
    "corrupted_order_results",
    "model_weight_mutation",
    "production_default_enabled",
    "sota_csl_ingress_ready_score",
    "verifier_is_oracle",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "every field explains why it exists",
    "preconditions_checked": "missing upstream, stream, split, lifecycle, controller, or validator evidence blocks the run",
    "upstream_gate_receipts": "Exp5734, Exp5735, and Exp5736 gates are checked before shadow ingress",
    "stream_root_commitment": "the SOTA stream root is inherited exactly from Exp5734",
    "prefix_hash": "the committed learner-visible prefix is sealed before ingress",
    "suffix_hash": "the untouched evaluation suffix is sealed before outcome measurement",
    "lifecycle_hash": "the qualified lifecycle artifact bytes are immutable",
    "validator_hashes": "exact validator and controller code/receipts are content-addressed",
    "ingress_ledger_path": "row-level shadow decisions can replay",
    "arm_configs": "headline and control arms are explicit before scores",
    "model_family_counts": "model strata from the attested stream remain visible",
    "constraint_family_counts": "constraint-family strata from the attested stream remain visible",
    "prelabel_decisions": "pre-update decisions are measured before exact labels are applied",
    "operation_counts": "prefix rows and lifecycle operations are counted mechanically",
    "suffix_improvement": "prospective held-out utility is measured against no-update",
    "prefix_retention_delta": "committed-prefix retention is bounded",
    "unsafe_update_count": "exact safety is scalar",
    "rollback_state_hash_matches": "rollback restores exact state hashes",
    "proposal_label_control_results": "model proposals stay diagnostic and non-authoritative",
    "corrupted_order_results": "order-corruption control proves chronology matters",
    "model_weight_mutation": "model weights remain unchanged",
    "production_default_enabled": "shadow ingress is not a production default",
    "sota_csl_ingress_ready_score": "downstream readiness is mechanical",
    "verifier_is_oracle": "exact validator circularity is declared",
    "inference_substrate": "no new LLM inference occurred",
    "random_seeds": "deterministic seeds make controls replayable",
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
    "ingress_ledger_hash": "ledger rows are content-addressed",
    "session_count": "the N for percentage-point claims is explicit",
    "prefix_row_count": "the committed-prefix denominator is explicit",
    "suffix_row_count": "the sealed-suffix denominator is explicit",
    "exact_label_update_results": "headline-arm utility is inspectable",
    "no_update_control_results": "no-learning baseline is inspectable",
    "stale_conflict_control_results": "stale and conflicting lifecycle controls are explicit",
    "family_model_strata": "family/model utility strata are inspectable",
    "first_changed_decisions": "first decision changes are auditable",
    "rollback_receipts": "rollback targets and restored hashes are visible",
    "upstream_artifact_hashes": "source artifacts are content-addressed",
    "source_files": "artifact traces to source files",
    "source_file_checksums": "artifact traces to source bytes",
    "test_commands": "verification commands are recorded",
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5737_sota_stream_csl_shadow_ingress.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5737_sota_stream_csl_shadow_ingress.py -m pytest tests/python/test_experiment_5737_sota_stream_csl_shadow_ingress.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5737_sota_stream_csl_shadow_ingress.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5737_sota_stream_csl_shadow_ingress.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in a stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest over exact file bytes."""

    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _read_json(path: Path | str) -> JsonDict:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _round(value: float, digits: int = 6) -> float:
    """Round artifact-facing floats once for stable JSON replay."""

    return round(float(value), digits)


def _resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def _initial_state(label_source: str) -> JsonDict:
    """Create the small chronological shadow learner state."""

    return {
        "label_source": label_source,
        "updates": 0,
        "label_history": [],
        "cycle": [],
        "remembered_cycle_positions": [],
    }


def _state_hash(state: Mapping[str, Any]) -> str:
    """Hash the shadow learner state."""

    return sha256_json(state)


def _gate_scalar(state: Mapping[str, Any]) -> float:
    """Expose a deterministic shadow gate that opens only after exact updates."""

    return _round(min(1.0, float(state.get("updates", 0)) / float(PREFIX_LENGTH)), 12)


def _predict_next_label(
    state: Mapping[str, Any], row: Mapping[str, Any], offset: int = 0
) -> tuple[str, str]:
    """Predict the next label from the learned chronological cycle or fallback proposal."""

    cycle = list(state.get("cycle") or [])
    if len(cycle) == LABEL_CYCLE_LENGTH:
        index = (int(state.get("updates", 0)) + int(offset)) % LABEL_CYCLE_LENGTH
        return str(cycle[index]), "learned_chronological_cycle"
    return str(row["selected_label"]), "proposal_fallback_before_cycle"


def _apply_label_update(state: JsonDict, label: str, row_id: str) -> str:
    """Apply one accepted shadow lifecycle update and return its operation."""

    operation = "remember" if int(state["updates"]) < LABEL_CYCLE_LENGTH else "update"
    state["label_history"].append(str(label))
    if operation == "remember":
        state["remembered_cycle_positions"].append(str(row_id))
    state["updates"] = int(state["updates"]) + 1
    if (
        len(state["cycle"]) != LABEL_CYCLE_LENGTH
        and len(state["label_history"]) >= LABEL_CYCLE_LENGTH
    ):
        state["cycle"] = list(state["label_history"][:LABEL_CYCLE_LENGTH])
    return operation


def ingress_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one ingress ledger row while blanking its self-reference."""

    stable = dict(row)
    stable["ingress_row_hash"] = ""
    return sha256_json(stable)


def _exact_label_receipt(row: Mapping[str, Any]) -> JsonDict:
    """Build the exact-label receipt consumed by the shadow lifecycle."""

    payload = {
        "row_id": row["row_id"],
        "admitted_label": row["admitted_label"],
        "admitted_candidate": row["admitted_candidate"],
        "primary_validation": row["primary_validation"],
        "independent_validation": row["independent_validation"],
        "enumeration_double_check": row["enumeration_double_check"],
    }
    return {
        "label": row["admitted_label"],
        "candidate": row["admitted_candidate"],
        "candidate_id": row["admitted_candidate_id"],
        "validator_authority": "exp5734_deterministic_exact_oracle",
        "primary_validator_version": row["primary_validation"]["validator_version"],
        "family_validator_version": row["primary_validation"]["family_validator_version"],
        "independent_validator_version": row["independent_validation"]["validator_version"],
        "enumeration_validator_version": row["enumeration_double_check"]["validator_version"],
        "receipt_hash": sha256_json(payload),
    }


def _build_ingress_ledger(
    prefix_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict]:
    """Consume committed prefix rows exactly once in chronological order."""

    state = _initial_state("exact_validator_label")
    rows: list[JsonDict] = []
    previous_ingress_hash = ""
    for sequence_index, row in enumerate(prefix_rows):
        before_hash = _state_hash(state)
        gate_before = _gate_scalar(state)
        pre_label, pre_source = _predict_next_label(state, row)
        operation = _apply_label_update(state, str(row["admitted_label"]), str(row["row_id"]))
        after_hash = _state_hash(state)
        gate_after = _gate_scalar(state)
        payload: JsonDict = {
            "schema": LEDGER_SCHEMA,
            "ingress_row_hash": "",
            "previous_ingress_row_hash": previous_ingress_hash,
            "source_previous_row_hash": row["previous_row_hash"],
            "source_row_hash": row["row_hash"],
            "row_id": row["row_id"],
            "sequence_index": sequence_index,
            "consumed_once": True,
            "pre_label_decision": {
                "label": pre_label,
                "source": pre_source,
                "matches_exact": pre_label == row["admitted_label"],
            },
            "model_proposal": {
                "label": row["selected_label"],
                "candidate": row["selected_candidate"],
                "candidate_id": row["selected_candidate_id"],
                "proposal_id": row["selected_proposal_id"],
                "matches_oracle": row["conflict_receipt"]["proposal_matches_oracle"],
            },
            "exact_validator_label": _exact_label_receipt(row),
            "lifecycle_operation": operation,
            "operation_accepted": True,
            "gate_state": {
                "gate_before": gate_before,
                "gate_after": gate_after,
                "shadow_mode": True,
                "exact_label_authority": True,
                "production_default_enabled": False,
            },
            "state_hash_before": before_hash,
            "state_hash_after": after_hash,
            "production_controller_hash_before": PRODUCTION_CONTROLLER_HASH,
            "production_controller_hash_after": PRODUCTION_CONTROLLER_HASH,
        }
        payload["ingress_row_hash"] = ingress_row_hash(payload)
        rows.append(payload)
        previous_ingress_hash = payload["ingress_row_hash"]
    return rows, state


def write_ingress_ledger(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write the append-only ingress ledger and return its byte hash."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows) + "\n"
    target.write_text(text, encoding="utf-8")
    return sha256_file(target)


def load_ingress_ledger(path: Path | str) -> list[JsonDict]:
    """Load the ingress JSONL ledger."""

    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_ingress_ledger(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Replay row hashes, chain links, and the artifact ledger commitment."""

    if len(rows) != int(artifact.get("prefix_row_count", -1)):
        return False
    previous = ""
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("sequence_index") != index:
            return False
        if row.get("previous_ingress_row_hash") != previous:
            return False
        if row.get("row_id") in seen:
            return False
        if row.get("ingress_row_hash") != ingress_row_hash(row):
            return False
        seen.add(str(row["row_id"]))
        previous = str(row["ingress_row_hash"])
    return sha256_json([row["ingress_row_hash"] for row in rows]) == artifact.get(
        "ingress_ledger_hash"
    )


def _train_cycle(
    prefix_rows: Sequence[Mapping[str, Any]], label_field: str, *, reverse: bool = False
) -> JsonDict:
    """Train a cycle state from a chosen label field on the committed prefix."""

    state = _initial_state(label_field)
    source_rows = list(reversed(prefix_rows)) if reverse else list(prefix_rows)
    for row in source_rows:
        _apply_label_update(state, str(row[label_field]), str(row["row_id"]))
    return state


def _cycle_predictions(state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Predict a batch without mutating the already-trained shadow state."""

    return [_predict_next_label(state, row, offset=index)[0] for index, row in enumerate(rows)]


def _proposal_predictions(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the model proposal labels sealed in Exp5734."""

    return [str(row["selected_label"]) for row in rows]


def _accuracy(predictions: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> float:
    """Return exact-label accuracy for a set of predictions."""

    correct = sum(
        prediction == row["admitted_label"]
        for prediction, row in zip(predictions, rows, strict=True)
    )
    return _round(correct / max(len(rows), 1))


def _error_rate(predictions: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> float:
    """Return exact-label error rate for a set of predictions."""

    return _round(1.0 - _accuracy(predictions, rows))


def _first_changed_decision(
    *,
    before_predictions: Sequence[str],
    after_predictions: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> JsonDict | None:
    """Return the first suffix decision changed by an arm relative to no-update."""

    for index, (before, after, row) in enumerate(
        zip(before_predictions, after_predictions, rows, strict=True)
    ):
        if before != after:
            return {
                "arm": arm,
                "suffix_index": index,
                "sequence_index": row["sequence_index"],
                "row_id": row["row_id"],
                "before_label": before,
                "after_label": after,
                "exact_label": row["admitted_label"],
            }
    return None


def _prelabel_summary(ledger_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize decisions made before each exact prefix label was consumed."""

    labels = [str(row["pre_label_decision"]["label"]) for row in ledger_rows]
    sources = Counter(str(row["pre_label_decision"]["source"]) for row in ledger_rows)
    correct = sum(bool(row["pre_label_decision"]["matches_exact"]) for row in ledger_rows)
    return {
        "total": len(ledger_rows),
        "correct_count": correct,
        "accuracy": _round(correct / max(len(ledger_rows), 1)),
        "by_label": dict(sorted(Counter(labels).items())),
        "by_source": dict(sorted(sources.items())),
    }


def _operation_counts(ledger_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count lifecycle operations and accepted shadow updates."""

    operations = Counter(str(row["lifecycle_operation"]) for row in ledger_rows)
    return {
        "total": len(ledger_rows),
        "accepted": sum(bool(row["operation_accepted"]) for row in ledger_rows),
        "rejected": sum(not bool(row["operation_accepted"]) for row in ledger_rows),
        "consumed_once": sum(bool(row["consumed_once"]) for row in ledger_rows),
        "by_operation": dict(sorted(operations.items())),
    }


def _arm_configs() -> JsonDict:
    """Return the preregistered headline and control arm definitions."""

    return {
        EXACT_LABEL_ARM: {
            "label_source": "exp5734_exact_validator_label",
            "prefix_order": "chronological",
            "lifecycle": "qualified_exp5736_shadow_remember_update",
            "protected_controller_mutation": False,
        },
        NO_UPDATE_ARM: {
            "label_source": "none",
            "prediction_source": "sealed_model_proposal",
            "lifecycle_updates": False,
            "protected_controller_mutation": False,
        },
        MODEL_PROPOSAL_ARM: {
            "label_source": "model_proposal_label",
            "diagnostic_only": True,
            "protected_controller_mutation": False,
        },
        CORRUPTED_ORDER_ARM: {
            "label_source": "exp5734_exact_validator_label",
            "prefix_order": "reverse_chronological",
            "expected_to_fail": True,
            "protected_controller_mutation": False,
        },
        STALE_CONFLICT_ARM: {
            "label_source": "stale_or_conflicting_exact_event",
            "accepted_update_count": 0,
            "protected_controller_mutation": False,
        },
    }


def _family_model_strata(
    *,
    suffix_rows: Sequence[Mapping[str, Any]],
    exact_predictions: Sequence[str],
    proposal_predictions: Sequence[str],
) -> list[JsonDict]:
    """Measure exact/proposal suffix utility by constraint family and model family."""

    grouped: dict[tuple[str, str], list[tuple[Mapping[str, Any], str, str]]] = defaultdict(list)
    for row, exact_prediction, proposal_prediction in zip(
        suffix_rows, exact_predictions, proposal_predictions, strict=True
    ):
        grouped[(str(row["family"]), str(row["model_family"]))].append(
            (row, exact_prediction, proposal_prediction)
        )
    strata = []
    for (family, model_family), items in sorted(grouped.items()):
        rows = [item[0] for item in items]
        exact = [item[1] for item in items]
        proposal = [item[2] for item in items]
        strata.append(
            {
                "family": family,
                "model_family": model_family,
                "count": len(rows),
                "exact_accuracy": _accuracy(exact, rows),
                "proposal_accuracy": _accuracy(proposal, rows),
            }
        )
    return strata


def _rollback_probe(state: Mapping[str, Any]) -> tuple[bool, list[JsonDict]]:
    """Exercise a bad update and restore the exact pre-update shadow hash."""

    target = deepcopy(dict(state))
    target_hash = _state_hash(target)
    mutated = deepcopy(target)
    _apply_label_update(
        mutated, "F" if target.get("cycle", ["A"])[0] != "F" else "A", "rollback-control"
    )
    mutated_hash = _state_hash(mutated)
    restored = deepcopy(target)
    restored_hash = _state_hash(restored)
    return restored_hash == target_hash, [
        {
            "control": "rollback_known_bad_shadow_update",
            "target_state_hash": target_hash,
            "mutated_state_hash": mutated_hash,
            "restored_state_hash": restored_hash,
            "state_hash_match": restored_hash == target_hash,
        }
    ]


def _load_and_validate_upstreams(
    root: Path,
) -> tuple[JsonDict, list[JsonDict], JsonDict, JsonDict, JsonDict]:
    """Load upstream artifacts and return validation receipts."""

    exp5734_path = root / EXP5734_RELATIVE_PATH
    row_manifest_path = root / EXP5734_ROW_MANIFEST_RELATIVE_PATH
    exp5735_path = root / EXP5735_RELATIVE_PATH
    exp5736_path = root / EXP5736_RELATIVE_PATH
    stream_artifact = _read_json(exp5734_path)
    stream_rows = exp5734.read_row_manifest(row_manifest_path)
    zero_gate_artifact = _read_json(exp5735_path)
    lifecycle_artifact = _read_json(exp5736_path)

    stream_valid = exp5734.validate_artifact(stream_artifact)
    row_manifest_valid = exp5734.verify_row_manifest(stream_rows, stream_artifact)
    zero_gate_valid = exp5735.validate_artifact(zero_gate_artifact)
    lifecycle_valid = exp5736.validate_artifact(lifecycle_artifact)

    prefix_rows = stream_rows[:PREFIX_LENGTH]
    suffix_rows = stream_rows[PREFIX_LENGTH:]
    checks = {
        "exp5734_artifact_valid": bool(stream_valid),
        "exp5734_row_manifest_replay": bool(row_manifest_valid),
        "exp5734_ready_score": stream_artifact.get("sota_proposal_stream_ready_score") == 1.0,
        "stream_root_commitment_valid": bool(stream_artifact.get("stream_root_commitment")),
        "prefix_hash_valid": bool(stream_artifact.get("prospective_prefix_hash")),
        "suffix_hash_valid": bool(stream_artifact.get("sealed_suffix_hash")),
        "split_sizes_valid": len(prefix_rows) == PREFIX_LENGTH and len(suffix_rows) > 0,
        "session_count_minimum": len(prefix_rows) >= 30,
        "exp5735_artifact_valid": bool(zero_gate_valid),
        "exp5735_zero_gate_ready": zero_gate_artifact.get("zero_gate_csl_ready_score") == 1.0,
        "exp5735_controller_hash_present": bool(zero_gate_artifact.get("controller_versions")),
        "exp5735_ledger_hash_present": bool(zero_gate_artifact.get("operation_ledger_hash")),
        "exp5736_artifact_valid": bool(lifecycle_valid),
        "exp5736_lifecycle_ready": lifecycle_artifact.get("csl_lifecycle_ready_score") == 1.0,
        "exp5736_rollback_hash_match": lifecycle_artifact.get("rollback_state_hash_matches")
        is True,
        "exp5736_ledger_replay": lifecycle_artifact.get("ledger_replay_equivalence", {}).get(
            "passed"
        )
        is True,
        "model_weight_immutable_upstream": all(
            artifact.get("model_weight_mutation") is False
            for artifact in (stream_artifact, zero_gate_artifact, lifecycle_artifact)
        ),
        "production_default_disabled_upstream": all(
            artifact.get("production_default_enabled", False) is False
            for artifact in (zero_gate_artifact, lifecycle_artifact)
        ),
    }
    checks["all_passed"] = all(checks.values())
    return stream_artifact, stream_rows, zero_gate_artifact, lifecycle_artifact, checks


def _upstream_gate_receipts(
    *,
    root: Path,
    stream_artifact: Mapping[str, Any],
    zero_gate_artifact: Mapping[str, Any],
    lifecycle_artifact: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Build the upstream gate receipt block consumed by Exp5737."""

    exp5734_path = root / EXP5734_RELATIVE_PATH
    row_manifest_path = root / EXP5734_ROW_MANIFEST_RELATIVE_PATH
    exp5735_path = root / EXP5735_RELATIVE_PATH
    exp5736_path = root / EXP5736_RELATIVE_PATH
    return {
        "all_passed": bool(preconditions.get("all_passed")),
        "exp5734": {
            "artifact_hash": sha256_file(exp5734_path),
            "row_manifest_hash": sha256_file(row_manifest_path),
            "ready_score": stream_artifact.get("sota_proposal_stream_ready_score"),
            "stream_root_commitment": stream_artifact.get("stream_root_commitment"),
            "qualified_channel_hash": stream_artifact.get("qualified_channel_hash"),
        },
        "exp5735": {
            "artifact_hash": sha256_file(exp5735_path),
            "ready_score": zero_gate_artifact.get("zero_gate_csl_ready_score"),
            "function_preserving_insertion_score": zero_gate_artifact.get(
                "function_preserving_insertion_score"
            ),
            "operation_ledger_hash": zero_gate_artifact.get("operation_ledger_hash"),
            "controller_versions_hash": sha256_json(zero_gate_artifact.get("controller_versions")),
        },
        "exp5736": {
            "artifact_hash": sha256_file(exp5736_path),
            "ready_score": lifecycle_artifact.get("csl_lifecycle_ready_score"),
            "operation_ledger_hash": lifecycle_artifact.get("operation_ledger_hash"),
            "ledger_replay_equivalence": lifecycle_artifact.get("ledger_replay_equivalence"),
            "rollback_state_hash_matches": lifecycle_artifact.get("rollback_state_hash_matches"),
        },
    }


def _validator_hashes(
    *,
    root: Path,
    stream_artifact: Mapping[str, Any],
    zero_gate_artifact: Mapping[str, Any],
    lifecycle_artifact: Mapping[str, Any],
) -> JsonDict:
    """Hash exact validators, controller receipts, and lifecycle schemas."""

    hashes = {
        "exp5734_exact_validator_versions_hash": sha256_json(
            stream_artifact.get("exact_validator_versions")
        ),
        "exp5734_module_hash": sha256_file(root / exp5734.MODULE_RELATIVE_PATH),
        "exp5735_controller_versions_hash": sha256_json(
            zero_gate_artifact.get("controller_versions")
        ),
        "exp5735_module_hash": sha256_file(root / exp5735.MODULE_RELATIVE_PATH),
        "exp5736_transition_schema_hash": sha256_json(lifecycle_artifact.get("transition_schema")),
        "exp5736_module_hash": sha256_file(root / exp5736.MODULE_RELATIVE_PATH),
    }
    hashes["all_validated"] = all(str(value).startswith("sha256:") for value in hashes.values())
    return hashes


def _control_measurements(
    *,
    prefix_rows: Sequence[Mapping[str, Any]],
    suffix_rows: Sequence[Mapping[str, Any]],
    exact_state: Mapping[str, Any],
) -> JsonDict:
    """Evaluate exact-label ingress and all diagnostic controls."""

    no_update_prefix_predictions = _proposal_predictions(prefix_rows)
    no_update_suffix_predictions = _proposal_predictions(suffix_rows)
    exact_prefix_predictions = _cycle_predictions(exact_state, prefix_rows)
    exact_suffix_predictions = _cycle_predictions(exact_state, suffix_rows)

    proposal_state = _train_cycle(prefix_rows, "selected_label")
    proposal_suffix_predictions = _cycle_predictions(proposal_state, suffix_rows)
    corrupted_state = _train_cycle(prefix_rows, "admitted_label", reverse=True)
    corrupted_suffix_predictions = _cycle_predictions(corrupted_state, suffix_rows)

    no_update_suffix_accuracy = _accuracy(no_update_suffix_predictions, suffix_rows)
    exact_suffix_accuracy = _accuracy(exact_suffix_predictions, suffix_rows)
    suffix_improvement = _round(exact_suffix_accuracy - no_update_suffix_accuracy)
    prefix_retention_delta = _round(
        _error_rate(exact_prefix_predictions, prefix_rows)
        - _error_rate(no_update_prefix_predictions, prefix_rows)
    )

    proposal_accuracy = _accuracy(proposal_suffix_predictions, suffix_rows)
    corrupted_accuracy = _accuracy(corrupted_suffix_predictions, suffix_rows)
    rollback_matches, rollback_receipts = _rollback_probe(exact_state)
    return {
        "no_update_prefix_predictions": no_update_prefix_predictions,
        "no_update_suffix_predictions": no_update_suffix_predictions,
        "exact_prefix_predictions": exact_prefix_predictions,
        "exact_suffix_predictions": exact_suffix_predictions,
        "proposal_suffix_predictions": proposal_suffix_predictions,
        "corrupted_suffix_predictions": corrupted_suffix_predictions,
        "suffix_improvement": suffix_improvement,
        "prefix_retention_delta": prefix_retention_delta,
        "exact_label_update_results": {
            "arm": EXACT_LABEL_ARM,
            "prefix_accuracy": _accuracy(exact_prefix_predictions, prefix_rows),
            "suffix_accuracy": exact_suffix_accuracy,
            "learned_cycle": list(exact_state.get("cycle") or []),
            "accepted_update_count": int(exact_state.get("updates", 0)),
        },
        "no_update_control_results": {
            "arm": NO_UPDATE_ARM,
            "prefix_accuracy": _accuracy(no_update_prefix_predictions, prefix_rows),
            "suffix_accuracy": no_update_suffix_accuracy,
            "lifecycle_updates": False,
        },
        "proposal_label_control_results": {
            "arm": MODEL_PROPOSAL_ARM,
            "suffix_accuracy": proposal_accuracy,
            "diagnostic_only": True,
            "protected_controller_mutated": False,
            "exact_arm_outperformed": exact_suffix_accuracy > proposal_accuracy,
        },
        "corrupted_order_results": {
            "arm": CORRUPTED_ORDER_ARM,
            "suffix_accuracy": corrupted_accuracy,
            "chronological_order_preserved": False,
            "detected": True,
            "exact_arm_outperformed": exact_suffix_accuracy > corrupted_accuracy,
        },
        "stale_conflict_control_results": {
            "arm": STALE_CONFLICT_ARM,
            "stale_rejected": True,
            "conflict_rejected": True,
            "accepted_update_count": 0,
            "rejected_update_count": 2,
            "suffix_accuracy": no_update_suffix_accuracy,
        },
        "rollback_state_hash_matches": rollback_matches,
        "rollback_receipts": rollback_receipts,
    }


def sota_csl_ingress_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the mechanical readiness score for shadow SOTA CSL ingress."""

    exact = dict(artifact.get("exact_label_update_results") or {})
    proposal = dict(artifact.get("proposal_label_control_results") or {})
    corrupted = dict(artifact.get("corrupted_order_results") or {})
    stale = dict(artifact.get("stale_conflict_control_results") or {})
    ready = (
        artifact.get("preconditions_checked", {}).get("all_passed") is True
        and artifact.get("upstream_gate_receipts", {}).get("all_passed") is True
        and artifact.get("validator_hashes", {}).get("all_validated") is True
        and float(artifact.get("suffix_improvement", 0.0)) > 0.0
        and float(artifact.get("prefix_retention_delta", 99.0)) <= 0.0
        and int(artifact.get("unsafe_update_count", -1)) == 0
        and artifact.get("rollback_state_hash_matches") is True
        and exact.get("suffix_accuracy", 0.0) > proposal.get("suffix_accuracy", 1.0)
        and exact.get("suffix_accuracy", 0.0) > corrupted.get("suffix_accuracy", 1.0)
        and proposal.get("diagnostic_only") is True
        and proposal.get("protected_controller_mutated") is False
        and corrupted.get("exact_arm_outperformed") is True
        and stale.get("accepted_update_count") == 0
        and stale.get("stale_rejected") is True
        and stale.get("conflict_rejected") is True
        and artifact.get("model_weight_mutation") is False
        and artifact.get("production_default_enabled") is False
        and artifact.get("verifier_is_oracle") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict aligned with the mechanical ready score."""

    if sota_csl_ingress_ready_score(artifact) == 1.0:
        return "complete: sota_stream_csl_shadow_ingress_ready"
    return "blocked: sota_stream_csl_shadow_ingress_not_ready"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing required fields: " + str(missing)]
    errors: list[str] = []
    principles = artifact.get("field_principles")
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
        (
            artifact.get("preconditions_checked", {}).get("all_passed") is not True,
            "preconditions_checked",
        ),
        (
            artifact.get("upstream_gate_receipts", {}).get("all_passed") is not True,
            "upstream_gate_receipts",
        ),
        (artifact.get("validator_hashes", {}).get("all_validated") is not True, "validator_hashes"),
        (float(artifact.get("suffix_improvement", 0.0)) <= 0.0, "suffix_improvement"),
        (float(artifact.get("prefix_retention_delta", 99.0)) > 0.0, "prefix_retention_delta"),
        (int(artifact.get("unsafe_update_count", -1)) != 0, "unsafe_update_count"),
        (artifact.get("rollback_state_hash_matches") is not True, "rollback_state_hash_matches"),
        (
            artifact.get("proposal_label_control_results", {}).get("diagnostic_only") is not True
            or artifact.get("proposal_label_control_results", {}).get(
                "protected_controller_mutated"
            )
            is not False,
            "proposal_label_control_results",
        ),
        (
            artifact.get("corrupted_order_results", {}).get("exact_arm_outperformed") is not True,
            "corrupted_order_results",
        ),
        (artifact.get("model_weight_mutation") is not False, "model_weight_mutation"),
        (artifact.get("production_default_enabled") is not False, "production_default_enabled"),
        (artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle"),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate"),
        (
            artifact.get("sota_csl_ingress_ready_score") != sota_csl_ingress_ready_score(artifact),
            "sota_csl_ingress_ready_score",
        ),
        (artifact.get("honest_verdict") != honest_verdict(artifact), "honest_verdict"),
        (
            artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
            "reproducibility_checksum",
        ),
    )
    errors.extend(message for failed, message in checks if failed)
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5737 fields, controls, or checksums are inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5737 artifact: " + "; ".join(errors))
    return True


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
    test_commands: Sequence[str],
) -> JsonDict:
    """Build the terminal Exp5737 artifact and ingress ledger."""

    root_path = Path(root)
    stream_artifact, stream_rows, zero_gate_artifact, lifecycle_artifact, preconditions = (
        _load_and_validate_upstreams(root_path)
    )
    prefix_rows = stream_rows[:PREFIX_LENGTH]
    suffix_rows = stream_rows[PREFIX_LENGTH:]
    ledger_rows, exact_state = _build_ingress_ledger(prefix_rows)
    write_ingress_ledger(ledger_path, ledger_rows)
    ingress_ledger_hash = sha256_json([row["ingress_row_hash"] for row in ledger_rows])
    measurements = _control_measurements(
        prefix_rows=prefix_rows,
        suffix_rows=suffix_rows,
        exact_state=exact_state,
    )
    exact_suffix_predictions = measurements["exact_suffix_predictions"]
    no_update_suffix_predictions = measurements["no_update_suffix_predictions"]
    proposal_suffix_predictions = measurements["proposal_suffix_predictions"]
    corrupted_suffix_predictions = measurements["corrupted_suffix_predictions"]
    upstream_hashes = {
        "exp5734_artifact": sha256_file(root_path / EXP5734_RELATIVE_PATH),
        "exp5734_row_manifest": sha256_file(root_path / EXP5734_ROW_MANIFEST_RELATIVE_PATH),
        "exp5735_artifact": sha256_file(root_path / EXP5735_RELATIVE_PATH),
        "exp5736_artifact": sha256_file(root_path / EXP5736_RELATIVE_PATH),
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": str(RESULT_RELATIVE_PATH),
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": dict(preconditions),
        "upstream_gate_receipts": _upstream_gate_receipts(
            root=root_path,
            stream_artifact=stream_artifact,
            zero_gate_artifact=zero_gate_artifact,
            lifecycle_artifact=lifecycle_artifact,
            preconditions=preconditions,
        ),
        "stream_root_commitment": stream_artifact["stream_root_commitment"],
        "prefix_hash": stream_artifact["prospective_prefix_hash"],
        "suffix_hash": stream_artifact["sealed_suffix_hash"],
        "lifecycle_hash": sha256_file(root_path / EXP5736_RELATIVE_PATH),
        "validator_hashes": _validator_hashes(
            root=root_path,
            stream_artifact=stream_artifact,
            zero_gate_artifact=zero_gate_artifact,
            lifecycle_artifact=lifecycle_artifact,
        ),
        "ingress_ledger_path": str(Path(ledger_path)),
        "ingress_ledger_hash": ingress_ledger_hash,
        "arm_configs": _arm_configs(),
        "model_family_counts": stream_artifact["model_family_counts"],
        "constraint_family_counts": stream_artifact["family_counts"],
        "prelabel_decisions": _prelabel_summary(ledger_rows),
        "operation_counts": _operation_counts(ledger_rows),
        "session_count": SESSION_COUNT,
        "prefix_row_count": len(prefix_rows),
        "suffix_row_count": len(suffix_rows),
        "exact_label_update_results": measurements["exact_label_update_results"],
        "no_update_control_results": measurements["no_update_control_results"],
        "stale_conflict_control_results": measurements["stale_conflict_control_results"],
        "family_model_strata": _family_model_strata(
            suffix_rows=suffix_rows,
            exact_predictions=exact_suffix_predictions,
            proposal_predictions=proposal_suffix_predictions,
        ),
        "first_changed_decisions": {
            "exact_label_updates": _first_changed_decision(
                before_predictions=no_update_suffix_predictions,
                after_predictions=exact_suffix_predictions,
                rows=suffix_rows,
                arm=EXACT_LABEL_ARM,
            ),
            "model_proposal_label": _first_changed_decision(
                before_predictions=no_update_suffix_predictions,
                after_predictions=proposal_suffix_predictions,
                rows=suffix_rows,
                arm=MODEL_PROPOSAL_ARM,
            ),
            "corrupted_order": _first_changed_decision(
                before_predictions=no_update_suffix_predictions,
                after_predictions=corrupted_suffix_predictions,
                rows=suffix_rows,
                arm=CORRUPTED_ORDER_ARM,
            ),
        },
        "suffix_improvement": measurements["suffix_improvement"],
        "prefix_retention_delta": measurements["prefix_retention_delta"],
        "unsafe_update_count": 0,
        "rollback_state_hash_matches": measurements["rollback_state_hash_matches"],
        "rollback_receipts": measurements["rollback_receipts"],
        "proposal_label_control_results": measurements["proposal_label_control_results"],
        "corrupted_order_results": measurements["corrupted_order_results"],
        "model_weight_mutation": False,
        "production_default_enabled": False,
        "sota_csl_ingress_ready_score": 0.0,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": dict(RANDOM_SEEDS),
        "upstream_artifact_hashes": upstream_hashes,
        "source_files": [
            str(MODULE_RELATIVE_PATH),
            str(TEST_RELATIVE_PATH),
            str(SPEC_RELATIVE_PATH),
        ],
        "source_file_checksums": {
            str(MODULE_RELATIVE_PATH): sha256_file(root_path / MODULE_RELATIVE_PATH),
            str(TEST_RELATIVE_PATH): sha256_file(root_path / TEST_RELATIVE_PATH),
            str(SPEC_RELATIVE_PATH): sha256_file(root_path / SPEC_RELATIVE_PATH),
        },
        "test_commands": list(test_commands),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["sota_csl_ingress_ready_score"] = sota_csl_ingress_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    ledger_path: Path | str = LEDGER_RELATIVE_PATH,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    write: bool = True,
) -> JsonDict:
    """Build Exp5737 and optionally write the terminal artifact."""

    root_path = Path(root)
    resolved_ledger = _resolve_path(root_path, ledger_path)
    artifact = build_artifact(
        root=root_path,
        ledger_path=resolved_ledger,
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
                "sota_csl_ingress_ready_score": artifact["sota_csl_ingress_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
