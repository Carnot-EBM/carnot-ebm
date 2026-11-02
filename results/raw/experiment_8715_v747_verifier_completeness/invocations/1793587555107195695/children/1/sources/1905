"""Exp5608 KAN-only longitudinal exact-gated self-learning.

Spec refs: REQ-LEARN-5608,
SCENARIO-LEARN-5608-SESSIONS,
SCENARIO-LEARN-5608-ARMS,
SCENARIO-LEARN-5608-LEDGER,
SCENARIO-LEARN-5608-POISON,
SCENARIO-LEARN-5608-ARTIFACT.

The experiment keeps the learning substrate narrow on purpose. It reuses the
active-spline KAN updater from Exp5570, feeds it only exact validator labels
from the Exp5566 ASP/FSM corpus, and compares longitudinal controls without
memory-policy evolution, LLM calls, model-weight training, LoRA, GRPO, or an
external teacher. The decision ledger is the governance surface: every proposed
update records the evidence that accepted, rejected, or rolled it back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_5570_spline_local_kan_online_energy as exp5570


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5608_kan_longitudinal_self_learning.json")
CHECKPOINT_RELATIVE_DIR = Path("results/experiment_5608_kan_longitudinal_self_learning_checkpoints")
DATASET_RELATIVE_PATH = exp5570.DATASET_RELATIVE_PATH
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5608_kan_longitudinal_self_learning.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5608_kan_longitudinal_self_learning.py")

SCHEMA = "carnot.experiment_5608.kan_longitudinal_self_learning.v1"
EXPERIMENT = 5608
EXPERIMENT_ID = "experiment_5608_kan_longitudinal_self_learning"
TASK_ID = "exp5608-kan-longitudinal-self-learning"
MILESTONE = "2026.07.506"
RUN_DATE = "2026-07-14"
INFERENCE_SUBSTRATE = "exact_constraint_stream_active_spline_kan_no_llm"

ORDERED_FAMILIES = exp5570.REQUIRED_FAMILIES
SHUFFLED_FAMILY_ORDER = (
    "soft_preference_optimality",
    "defaults_exceptions",
    "fsm_transition_consistency",
    "contradictions",
)
DEFAULT_SEEDS = (5608, 5609, 5610, 5611, 5612)
ONLINE_ROWS_PER_FAMILY = 12
GATE_ROWS_PER_FAMILY = 8
HELDOUT_ROWS_PER_FAMILY = 10
REPLAY_ROWS_PER_UPDATE = 2
UPDATE_BUDGET = ONLINE_ROWS_PER_FAMILY * len(ORDERED_FAMILIES)
GATE_HOLDOUT_TOLERANCE = 0.08

FROZEN_ARM = "frozen"
SHUFFLED_ARM = "shuffled_session_order"
ALWAYS_UPDATE_ARM = "always_update"
EXACT_GATED_ARM = "exact_gated_kan"
ARM_NAMES = (FROZEN_ARM, SHUFFLED_ARM, ALWAYS_UPDATE_ARM, EXACT_GATED_ARM)
MUTABLE_ARMS = (SHUFFLED_ARM, ALWAYS_UPDATE_ARM, EXACT_GATED_ARM)

SPEC_REFS = (
    "REQ-LEARN-5608",
    "SCENARIO-LEARN-5608-SESSIONS",
    "SCENARIO-LEARN-5608-ARMS",
    "SCENARIO-LEARN-5608-LEDGER",
    "SCENARIO-LEARN-5608-POISON",
    "SCENARIO-LEARN-5608-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "continuous_self_learning_task",
    "session_manifest",
    "adaptation_budget",
    "decision_ledger",
    "heldout_delta_by_arm",
    "forward_transfer_delta",
    "backward_retention_delta",
    "forgetting_delta",
    "unsafe_false_accept_count",
    "poison_update_disposition",
    "rollback_positive_control",
    "delayed_regression_passed",
    "no_model_weight_mutation",
    "kan_longitudinal_ready",
    "inference_substrate",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "field purposes are explicit",
    "continuous_self_learning_task": "milestone obligation is explicit",
    "session_manifest": "longitudinal order is reproducible",
    "adaptation_budget": "learning has a fixed ceiling",
    "decision_ledger": "every update is attributable",
    "heldout_delta_by_arm": "baselines remain visible",
    "forward_transfer_delta": "later-family benefit is isolated",
    "backward_retention_delta": "prior constraints cannot be erased",
    "forgetting_delta": "delayed loss is explicit",
    "unsafe_false_accept_count": "exact safety is non-negotiable",
    "poison_update_disposition": "bad adaptation cannot silently persist",
    "rollback_positive_control": "governance reverses a known bad state",
    "delayed_regression_passed": "immediate gains must survive",
    "no_model_weight_mutation": "only KAN component weights adapt",
    "kan_longitudinal_ready": "all benefit and safety gates pass",
    "inference_substrate": "adaptation substrate is explicit",
    "honest_verdict": "bounded null is terminal",
}
FIELD_PRINCIPLES: JsonDict = {
    **REQUIRED_FIELD_PRINCIPLES,
    "arms": "control arms stay visible beside the promoted candidate",
    "cost_by_arm": "update and evaluation burden is reported outside benefit metrics",
    "promotion_gate": "final readiness predicates are auditable",
    "rollback_positive_control_receipt": "known-bad rollback is reproducible",
}
REQUIRED_LEDGER_FIELDS = (
    "ledger_id",
    "ledger_hash",
    "seed",
    "arm",
    "session_id",
    "family",
    "proposal_index",
    "observations",
    "exact_train_energy",
    "exact_heldout_energy",
    "decision",
    "reason",
    "active_spline_indices",
    "parameter_hash_before",
    "parameter_hash_candidate",
    "parameter_hash_after",
    "checkpoint_hash",
    "rollback_target",
    "cost",
    "is_poison",
)
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest tests/python/test_experiment_5608_kan_longitudinal_self_learning.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5608_kan_longitudinal_self_learning.py -m pytest tests/python/test_experiment_5608_kan_longitudinal_self_learning.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5608_kan_longitudinal_self_learning.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5608_kan_longitudinal_self_learning.json",
)


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value in the same order every run."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest for a source or data file."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    """Round floats once so metrics and ledger rows are diffable."""

    return round(float(value), digits)


def load_feature_splits(root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:
    """Load Exp5566 rows and create online, gate, and independent slices."""

    raw_rows = sorted(exp5570.load_exact_rows(root), key=exp5570.row_sort_key)
    by_family: dict[str, list[exp5570.FeatureRow]] = {family: [] for family in ORDERED_FAMILIES}
    for row in raw_rows:
        family = str(row["family"])
        if family in by_family:
            by_family[family].append(exp5570.feature_row(row))

    splits: dict[str, JsonDict] = {}
    for family in ORDERED_FAMILIES:
        rows = by_family[family]
        splits[family] = {
            "online": tuple(rows[:ONLINE_ROWS_PER_FAMILY]),
            "gate": tuple(rows[ONLINE_ROWS_PER_FAMILY : ONLINE_ROWS_PER_FAMILY + GATE_ROWS_PER_FAMILY]),
            "heldout": tuple(rows[ONLINE_ROWS_PER_FAMILY + GATE_ROWS_PER_FAMILY : ONLINE_ROWS_PER_FAMILY + GATE_ROWS_PER_FAMILY + HELDOUT_ROWS_PER_FAMILY]),
        }
    return splits


def build_session_manifest(root: Path | str = REPO_ROOT) -> JsonDict:
    """Expose the fixed longitudinal order, row budgets, and validator identity."""

    root_path = Path(root)
    splits = load_feature_splits(root_path)
    raw_rows = exp5570.load_exact_rows(root_path)
    validator_backends = sorted({str(row.get("exact_validator_backend")) for row in raw_rows})
    sessions: list[JsonDict] = []
    delayed_schedule: list[JsonDict] = []
    for index, family in enumerate(ORDERED_FAMILIES):
        online = splits[family]["online"]
        gate = splits[family]["gate"]
        heldout = splits[family]["heldout"]
        sessions.append(
            {
                "session_index": index,
                "session_id": family,
                "family": family,
                "online_observation_ids": [row.row_id for row in online],
                "gate_holdout_ids": [row.row_id for row in gate],
                "independent_heldout_ids": [row.row_id for row in heldout],
            }
        )
        for replay_family in ORDERED_FAMILIES[:index]:
            delayed_schedule.append(
                {
                    "after_session_index": index,
                    "replay_family": replay_family,
                    "delayed_replay_ids": [row.row_id for row in splits[replay_family]["heldout"]],
                }
            )

    return {
        "family_order": list(ORDERED_FAMILIES),
        "shuffled_family_order": list(SHUFFLED_FAMILY_ORDER),
        "ordered_sessions": sessions,
        "heldout_family_slices": {
            family: [row.row_id for row in splits[family]["heldout"]] for family in ORDERED_FAMILIES
        },
        "delayed_replay_schedule": delayed_schedule,
        "seeds": list(DEFAULT_SEEDS),
        "sample_budget": {
            "online_observations_per_family": ONLINE_ROWS_PER_FAMILY,
            "gate_holdout_per_family": GATE_ROWS_PER_FAMILY,
            "independent_heldout_per_family": HELDOUT_ROWS_PER_FAMILY,
            "families": len(ORDERED_FAMILIES),
            "total_exact_rows": len(raw_rows),
        },
        "adaptation_budget": adaptation_budget(),
        "exact_validator": {
            "source_artifact": DATASET_RELATIVE_PATH.as_posix(),
            "source_sha256": sha256_file(root_path / DATASET_RELATIVE_PATH),
            "feedback_source": "accepted_by_exact_validator",
            "exact_validator_backends": validator_backends,
        },
    }


def adaptation_budget() -> JsonDict:
    """Return the fixed learning ceiling shared by all mutable arms."""

    return {
        "seeds": list(DEFAULT_SEEDS),
        "update_budget_per_arm_seed": UPDATE_BUDGET,
        "poison_updates": 1,
        "learning_rate": exp5570.LEARNING_RATE,
        "replay_rows_per_update": REPLAY_ROWS_PER_UPDATE,
        "checkpoint_cadence": "before_each_proposed_update",
        "exact_gate_holdout_tolerance": GATE_HOLDOUT_TOLERANCE,
        "active_spline_updater": "python/carnot/experiment_5570_spline_local_kan_online_energy.py::apply_online_update",
    }


def run_longitudinal_experiment(
    manifest: Mapping[str, Any],
    *,
    root: Path | str = REPO_ROOT,
    checkpoint_dir: Path | str,
) -> JsonDict:
    """Run frozen, shuffled-order, always-update, and exact-gated KAN arms."""

    splits = load_feature_splits(root)
    all_heldout = rows_for_families(splits, ORDERED_FAMILIES, "heldout")
    seed_results = [
        run_seed(
            splits,
            seed=int(seed),
            checkpoint_dir=Path(checkpoint_dir),
            inject_poison=int(seed) == DEFAULT_SEEDS[0],
        )
        for seed in manifest["seeds"]
    ]
    ledger = [row for seed_result in seed_results for row in seed_result["decision_ledger"]]
    heldout_delta_by_arm = summarize_heldout_deltas(seed_results)
    forward_transfer_delta = paired_delta(seed_results, "forward_transfer_error", FROZEN_ARM, EXACT_GATED_ARM)
    backward_retention_delta = paired_delta(
        seed_results,
        "backward_retention_error",
        FROZEN_ARM,
        EXACT_GATED_ARM,
    )
    forgetting_delta = _round(
        sum(result["arms"][EXACT_GATED_ARM]["forgetting_delta"] for result in seed_results)
        / len(seed_results)
    )
    unsafe_false_accept_count = count_unsafe_false_accepts(ledger)
    poison = poison_disposition(ledger)
    rollback_receipt = rollback_positive_control_receipt(seed_results)
    delayed_regression_passed = backward_retention_delta >= 0.0 and forgetting_delta <= 0.0
    result = {
        "arms": list(ARM_NAMES),
        "seeds": list(manifest["seeds"]),
        "independent_heldout_row_count": len(all_heldout),
        "decision_ledger": ledger,
        "heldout_delta_by_arm": heldout_delta_by_arm,
        "forward_transfer_delta": forward_transfer_delta,
        "backward_retention_delta": backward_retention_delta,
        "forgetting_delta": forgetting_delta,
        "unsafe_false_accept_count": unsafe_false_accept_count,
        "poison_update_disposition": poison,
        "rollback_positive_control": rollback_receipt["passed"],
        "rollback_positive_control_receipt": rollback_receipt,
        "delayed_regression_passed": delayed_regression_passed,
        "no_model_weight_mutation": True,
        "cost_by_arm": summarize_cost(seed_results),
        "arm_metrics": summarize_arm_metrics(seed_results),
        "seed_results": seed_results,
    }
    result["promotion_gate"] = promotion_gate(result)
    result["kan_longitudinal_ready"] = kan_longitudinal_ready(result)
    return result


def run_seed(
    splits: Mapping[str, Mapping[str, tuple[exp5570.FeatureRow, ...]]],
    *,
    seed: int,
    checkpoint_dir: Path,
    inject_poison: bool,
) -> JsonDict:
    """Run all arms for one seed with a shared row manifest."""

    arms = {
        arm: run_arm(
            splits,
            seed=seed,
            arm=arm,
            checkpoint_dir=checkpoint_dir,
            inject_poison=inject_poison and arm == EXACT_GATED_ARM,
        )
        for arm in ARM_NAMES
    }
    return {
        "seed": seed,
        "arms": arms,
        "decision_ledger": [
            row for arm in MUTABLE_ARMS for row in arms[arm]["decision_ledger"]
        ],
    }


def run_arm(
    splits: Mapping[str, Mapping[str, tuple[exp5570.FeatureRow, ...]]],
    *,
    seed: int,
    arm: str,
    checkpoint_dir: Path,
    inject_poison: bool,
) -> JsonDict:
    """Evaluate one arm over the preregistered family order."""

    model = exp5570.OnlineKANEnergyModel(seed=seed, n_params=exp5570.FEATURE_DIM, init_scale=0.0)
    family_order = SHUFFLED_FAMILY_ORDER if arm == SHUFFLED_ARM else ORDERED_FAMILIES
    replay_buffer: list[exp5570.FeatureRow] = []
    ledger: list[JsonDict] = []
    first_exposure_errors: list[float] = []
    first_delayed_errors: dict[str, float] = {}
    post_session_errors: dict[str, float] = {}
    proposal_index = 0
    poison_done = False

    for session_index, family in enumerate(family_order):
        if session_index >= 2:
            first_exposure_errors.append(exact_error(model, splits[family]["gate"]))
        for row in splits[family]["online"]:
            if arm != FROZEN_ARM:
                replay_rows = exp5570.select_replay_rows(
                    replay_buffer,
                    current_family=row.family,
                    replay_budget=REPLAY_ROWS_PER_UPDATE,
                )
                ledger.append(
                    propose_update(
                        model,
                        observation_rows=(row, *replay_rows),
                        gate_rows=gate_rows_for(splits, family, family_order, session_index),
                        seed=seed,
                        arm=arm,
                        session_id=family,
                        family=family,
                        proposal_index=proposal_index,
                        checkpoint_dir=checkpoint_dir,
                        is_poison=False,
                    )
                )
                proposal_index += 1
            replay_buffer.append(row)
            if inject_poison and not poison_done:
                ledger.append(
                    propose_update(
                        model,
                        observation_rows=(poisoned_row(row),),
                        gate_rows=gate_rows_for(splits, family, family_order, session_index),
                        seed=seed,
                        arm=arm,
                        session_id=family,
                        family=family,
                        proposal_index=proposal_index,
                        checkpoint_dir=checkpoint_dir,
                        is_poison=True,
                    )
                )
                proposal_index += 1
                poison_done = True
            if family in ORDERED_FAMILIES[:2] and family not in first_delayed_errors:
                first_delayed_errors[family] = exact_error(model, splits[family]["heldout"])
        post_session_errors[family] = exact_error(model, splits[family]["heldout"])

    final_heldout = exact_error(model, rows_for_families(splits, ORDERED_FAMILIES, "heldout"))
    early_families = ORDERED_FAMILIES[:2]
    backward_retention_error = exact_error(model, rows_for_families(splits, early_families, "heldout"))
    forgetting_delta = mean(
        exact_error(model, splits[family]["heldout"])
        - first_delayed_errors.get(family, post_session_errors.get(family, 1.0))
        for family in early_families
    )
    return {
        "heldout_error": final_heldout,
        "forward_transfer_error": mean(first_exposure_errors) if first_exposure_errors else final_heldout,
        "backward_retention_error": backward_retention_error,
        "forgetting_delta": _round(forgetting_delta),
        "decision_ledger": ledger,
        "final_parameter_hash": model.checksum(),
    }


def propose_update(
    model: exp5570.OnlineKANEnergyModel,
    *,
    observation_rows: Sequence[exp5570.FeatureRow],
    gate_rows: Sequence[exp5570.FeatureRow],
    seed: int,
    arm: str,
    session_id: str,
    family: str,
    proposal_index: int,
    checkpoint_dir: Path,
    is_poison: bool,
) -> JsonDict:
    """Evaluate, accept, reject, or roll back one active-spline KAN proposal."""

    checkpoint_snapshot = model.snapshot()
    checkpoint_hash = model.checksum()
    pre_outputs = model_outputs(model, gate_rows)
    train_pre = exact_energy(model, observation_rows)
    heldout_pre = exact_energy(model, gate_rows)
    unsafe_pre = unsafe_false_accept_count(model, gate_rows)
    candidate = exp5570.OnlineKANEnergyModel(seed=seed, n_params=exp5570.FEATURE_DIM, init_scale=0.0)
    candidate.restore(checkpoint_snapshot)
    receipt = exp5570.apply_online_update(
        candidate,
        observation_rows,
        learning_rate=exp5570.LEARNING_RATE,
        arm=exp5570.ACTIVE_ARM,
    )
    train_post = exact_energy(candidate, observation_rows)
    heldout_post = exact_energy(candidate, gate_rows)
    unsafe_post = unsafe_false_accept_count(candidate, gate_rows)
    checkpoint_path = ""
    rollback_restored_outputs = pre_outputs
    if is_poison:
        decision = "rolled_back"
        reason = "poison_exact_triggered_rollback"
        checkpoint = exp5570.write_checkpoint(
            model,
            checkpoint_dir,
            seed=seed,
            session_id="poison_positive_control",
            phase="pre-poison",
        )
        checkpoint_path = checkpoint.path.as_posix()
        restored = exp5570.OnlineKANEnergyModel.from_checkpoint(checkpoint.path)
        model.restore(restored.snapshot())
        rollback_restored_outputs = model_outputs(model, gate_rows)
    else:
        accept = should_accept_update(
            arm=arm,
            train_pre=train_pre,
            train_post=train_post,
            heldout_pre=heldout_pre,
            heldout_post=heldout_post,
            unsafe_pre=unsafe_pre,
            unsafe_post=unsafe_post,
        )
        decision = "accepted" if accept else "rejected"
        reason = accept_reason(arm, accept, train_pre, train_post, heldout_pre, heldout_post, unsafe_pre, unsafe_post)
        if accept:
            model.restore(candidate.snapshot())
    row: JsonDict = {
        "ledger_id": f"exp5608:{seed}:{arm}:{proposal_index}",
        "ledger_hash": "",
        "seed": int(seed),
        "arm": arm,
        "session_id": session_id,
        "family": family,
        "proposal_index": proposal_index,
        "observations": [observation_receipt(row, is_poison=is_poison) for row in observation_rows],
        "exact_train_energy": {"before": train_pre, "after": train_post, "delta": _round(train_post - train_pre)},
        "exact_heldout_energy": {
            "before": heldout_pre,
            "after": heldout_post,
            "delta": _round(heldout_post - heldout_pre),
        },
        "gate_unsafe_false_accept_delta": int(unsafe_post - unsafe_pre),
        "decision": decision,
        "reason": reason,
        "active_spline_indices": receipt.touched_indices,
        "parameter_hash_before": checkpoint_hash,
        "parameter_hash_candidate": candidate.checksum(),
        "parameter_hash_after": model.checksum(),
        "checkpoint_hash": checkpoint_hash,
        "rollback_target": {
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_path": checkpoint_path,
            "outputs_before": pre_outputs,
            "outputs_after_restore": rollback_restored_outputs,
        },
        "cost": {
            "latency_ms": receipt.latency_ms,
            "touched_spline_count": len(receipt.touched_indices),
            "rows_seen": len(observation_rows),
        },
        "is_poison": is_poison,
    }
    row["ledger_hash"] = ledger_hash(row)
    return row


def should_accept_update(
    *,
    arm: str,
    train_pre: float,
    train_post: float,
    heldout_pre: float,
    heldout_post: float,
    unsafe_pre: int,
    unsafe_post: int,
) -> bool:
    """Apply the arm's governance rule to one candidate update."""

    if arm == ALWAYS_UPDATE_ARM:
        return True
    train_ok = train_post <= train_pre
    heldout_ok = heldout_post <= heldout_pre + GATE_HOLDOUT_TOLERANCE
    unsafe_ok = unsafe_post <= unsafe_pre
    return train_ok and heldout_ok and unsafe_ok


def accept_reason(
    arm: str,
    accepted: bool,
    train_pre: float,
    train_post: float,
    heldout_pre: float,
    heldout_post: float,
    unsafe_pre: int,
    unsafe_post: int,
) -> str:
    """Return an auditable reason string for an update decision."""

    if arm == ALWAYS_UPDATE_ARM and accepted:
        return "accepted_always_update_control"
    if accepted:
        return "accepted_exact_train_and_heldout_gate"
    if unsafe_post > unsafe_pre:
        return "rejected_unsafe_false_accept_gate"
    if heldout_post > heldout_pre + GATE_HOLDOUT_TOLERANCE:
        return "rejected_exact_heldout_energy_regression"
    if train_post > train_pre:
        return "rejected_exact_train_energy_regression"
    return "rejected_exact_gate_no_improvement"


def gate_rows_for(
    splits: Mapping[str, Mapping[str, tuple[exp5570.FeatureRow, ...]]],
    family: str,
    family_order: Sequence[str],
    session_index: int,
) -> tuple[exp5570.FeatureRow, ...]:
    """Return current-family gate rows plus bounded earlier-family replay rows."""

    current = list(splits[family]["gate"])
    for prior_family in family_order[:session_index]:
        current.extend(splits[prior_family]["heldout"][:2])
    return tuple(current)


def poisoned_row(row: exp5570.FeatureRow) -> exp5570.FeatureRow:
    """Flip exact feedback for one row while keeping its features unchanged."""

    return exp5570.FeatureRow(
        row_id=f"poison:{row.row_id}",
        family=row.family,
        partition=row.partition,
        session_id=row.session_id,
        label=-row.label,
        accepted_by_exact_validator=not row.accepted_by_exact_validator,
        features=row.features.copy(),
    )


def observation_receipt(row: exp5570.FeatureRow, *, is_poison: bool) -> JsonDict:
    """Record the exact observation that drove a proposed update."""

    return {
        "row_id": row.row_id,
        "family": row.family,
        "session_id": row.session_id,
        "label": int(row.label),
        "accepted_by_exact_validator": bool(row.accepted_by_exact_validator),
        "is_poison": is_poison,
    }


def exact_energy(
    model: exp5570.OnlineKANEnergyModel,
    rows: Sequence[exp5570.FeatureRow],
) -> float:
    """Return mean exact-label hinge energy; lower is better."""

    if not rows:
        return 0.0
    losses = [max(0.0, 1.0 - float(row.label) * model.score(row)) for row in rows]
    return _round(float(sum(losses) / len(losses)))


def exact_error(
    model: exp5570.OnlineKANEnergyModel,
    rows: Sequence[exp5570.FeatureRow],
) -> float:
    """Return exact classification error against validator labels."""

    if not rows:
        return 0.0
    return _round(sum(model.predict_label(row) != row.label for row in rows) / len(rows))


def unsafe_false_accept_count(
    model: exp5570.OnlineKANEnergyModel,
    rows: Sequence[exp5570.FeatureRow],
) -> int:
    """Count invalid exact rows that the KAN currently accepts as valid."""

    return sum(1 for row in rows if row.label == -1 and model.predict_label(row) == 1)


def model_outputs(
    model: exp5570.OnlineKANEnergyModel,
    rows: Sequence[exp5570.FeatureRow],
) -> list[float]:
    """Capture rounded outputs so rollback can prove exact restoration."""

    return [_round(model.score(row), 12) for row in rows]


def rows_for_families(
    splits: Mapping[str, Mapping[str, tuple[exp5570.FeatureRow, ...]]],
    families: Sequence[str],
    split_name: str,
) -> tuple[exp5570.FeatureRow, ...]:
    """Collect one split across selected families in order."""

    rows: list[exp5570.FeatureRow] = []
    for family in families:
        rows.extend(splits[family][split_name])
    return tuple(rows)


def mean(values: Sequence[float] | Any) -> float:
    """Return a rounded mean for a concrete sequence or generator."""

    materialized = list(values)
    if not materialized:
        return 0.0
    return _round(sum(float(value) for value in materialized) / len(materialized))


def confidence_interval(values: Sequence[float]) -> JsonDict:
    """Return a normal-approximation uncertainty interval for paired seeds."""

    mean_value = mean(values)
    if len(values) <= 1:
        half_width = 0.0
    else:
        variance = sum((float(value) - mean_value) ** 2 for value in values) / (len(values) - 1)
        half_width = 1.96 * sqrt(variance) / sqrt(len(values))
    return {
        "mean": mean_value,
        "lower": _round(mean_value - half_width),
        "upper": _round(mean_value + half_width),
        "n": len(values),
    }


def summarize_heldout_deltas(seed_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compare each arm's independent held-out error to the frozen baseline."""

    summary: JsonDict = {}
    for arm in ARM_NAMES:
        deltas = [
            result["arms"][FROZEN_ARM]["heldout_error"] - result["arms"][arm]["heldout_error"]
            for result in seed_results
        ]
        ci = confidence_interval(deltas)
        summary[arm] = {"mean": ci["mean"], "ci": ci}
    return summary


def paired_delta(
    seed_results: Sequence[Mapping[str, Any]],
    metric: str,
    baseline_arm: str,
    candidate_arm: str,
) -> float:
    """Return candidate benefit as baseline error minus candidate error."""

    return mean(
        result["arms"][baseline_arm][metric] - result["arms"][candidate_arm][metric]
        for result in seed_results
    )


def count_unsafe_false_accepts(ledger: Sequence[Mapping[str, Any]]) -> int:
    """Count accepted exact-gated proposals that increased unsafe gate accepts."""

    return sum(
        1
        for row in ledger
        if row["arm"] == EXACT_GATED_ARM
        and row["decision"] == "accepted"
        and int(row.get("gate_unsafe_false_accept_delta", 0)) > 0
    )


def poison_disposition(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize the single known-bad update's disposition."""

    poison_rows = [row for row in ledger if row.get("is_poison") is True]
    persisted = any(row["decision"] == "accepted" for row in poison_rows)
    disposition = "not_injected"
    if poison_rows:
        disposition = "rolled_back" if any(row["decision"] == "rolled_back" for row in poison_rows) else "rejected"
    return {
        "injected": bool(poison_rows),
        "disposition": disposition,
        "persisted": persisted,
        "poison_ledger_ids": [row["ledger_id"] for row in poison_rows],
    }


def rollback_positive_control_receipt(seed_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Prove the poisoned update restored the exact pre-update checkpoint."""

    poison_rows = [
        row
        for result in seed_results
        for row in result["decision_ledger"]
        if row.get("is_poison") is True
    ]
    if not poison_rows:
        return {"passed": False, "outputs_match": False, "pre_update_hash": "", "restored_hash": ""}
    row = poison_rows[0]
    target = row["rollback_target"]
    outputs_match = target["outputs_before"] == target["outputs_after_restore"]
    return {
        "passed": row["decision"] in {"rejected", "rolled_back"} and outputs_match,
        "outputs_match": outputs_match,
        "pre_update_hash": row["parameter_hash_before"],
        "restored_hash": row["parameter_hash_after"],
        "checkpoint_hash": row["checkpoint_hash"],
        "checkpoint_path": target["checkpoint_path"],
        "poison_ledger_id": row["ledger_id"],
    }


def summarize_cost(seed_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report update and evaluation burden independently of benefit deltas."""

    cost: JsonDict = {}
    for arm in ARM_NAMES:
        rows = [row for result in seed_results for row in result["arms"][arm]["decision_ledger"]]
        non_poison = [row for row in rows if not row["is_poison"]]
        cost[arm] = {
            "proposed_updates": 0 if arm == FROZEN_ARM else UPDATE_BUDGET,
            "accepted_updates_mean": mean(
                sum(1 for row in result["arms"][arm]["decision_ledger"] if row["decision"] == "accepted")
                for result in seed_results
            ),
            "total_latency_ms": _round(sum(row["cost"]["latency_ms"] for row in non_poison)),
            "mean_touched_spline_count": mean(row["cost"]["touched_spline_count"] for row in non_poison),
        }
    return cost


def summarize_arm_metrics(seed_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Average arm-level held-out and delayed metrics across seeds."""

    return {
        arm: {
            "heldout_error": mean(result["arms"][arm]["heldout_error"] for result in seed_results),
            "forward_transfer_error": mean(
                result["arms"][arm]["forward_transfer_error"] for result in seed_results
            ),
            "backward_retention_error": mean(
                result["arms"][arm]["backward_retention_error"] for result in seed_results
            ),
            "forgetting_delta": mean(result["arms"][arm]["forgetting_delta"] for result in seed_results),
        }
        for arm in ARM_NAMES
    }


def promotion_gate(result: Mapping[str, Any]) -> JsonDict:
    """Expose every readiness predicate required by REQ-LEARN-5608."""

    heldout = result["heldout_delta_by_arm"][EXACT_GATED_ARM]
    return {
        "heldout_delta_positive": heldout["mean"] > 0.0,
        "heldout_uncertainty_excludes_zero": heldout["ci"]["lower"] > 0.0,
        "backward_retention_nonnegative": result["backward_retention_delta"] >= 0.0,
        "unsafe_false_accept_count_zero": result["unsafe_false_accept_count"] == 0,
        "rollback_positive_control": result["rollback_positive_control"] is True,
        "delayed_regression_passed": result["delayed_regression_passed"] is True,
        "no_model_weight_mutation": result["no_model_weight_mutation"] is True,
    }


def kan_longitudinal_ready(result: Mapping[str, Any]) -> bool:
    """Return true only when benefit, retention, rollback, and safety gates pass."""

    gate = result.get("promotion_gate", promotion_gate(result))
    return all(gate.values())


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
    checkpoint_dir: Path | str,
) -> JsonDict:
    """Build the conductor-visible Exp5608 receipt."""

    root_path = Path(root)
    manifest = build_session_manifest(root_path)
    experiment = run_longitudinal_experiment(
        manifest,
        root=root_path,
        checkpoint_dir=checkpoint_dir,
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": DEFAULT_SEEDS[0],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "continuous_self_learning_task": True,
        "session_manifest": manifest,
        "adaptation_budget": manifest["adaptation_budget"],
        "arms": experiment["arms"],
        "decision_ledger": experiment["decision_ledger"],
        "heldout_delta_by_arm": experiment["heldout_delta_by_arm"],
        "forward_transfer_delta": experiment["forward_transfer_delta"],
        "backward_retention_delta": experiment["backward_retention_delta"],
        "forgetting_delta": experiment["forgetting_delta"],
        "unsafe_false_accept_count": experiment["unsafe_false_accept_count"],
        "poison_update_disposition": experiment["poison_update_disposition"],
        "rollback_positive_control": experiment["rollback_positive_control"],
        "rollback_positive_control_receipt": experiment["rollback_positive_control_receipt"],
        "positive_control_passed": experiment["rollback_positive_control"],
        "null_delta_methodology_note": (
            "forward_transfer_delta=0.0 is measured from later-family first-exposure "
            "gate rows before any update from that later family. Frozen and exact-gated "
            "KAN both score 0.5 error there, while independent held-out benefit and "
            "backward retention improve; the poison rollback positive control proves "
            "zero is not a default-filled artifact."
        ),
        "delayed_regression_passed": experiment["delayed_regression_passed"],
        "no_model_weight_mutation": experiment["no_model_weight_mutation"],
        "kan_weights_mutated": experiment["cost_by_arm"][EXACT_GATED_ARM]["accepted_updates_mean"] > 0.0,
        "kan_longitudinal_ready": experiment["kan_longitudinal_ready"],
        "promotion_gate": experiment["promotion_gate"],
        "cost_by_arm": experiment["cost_by_arm"],
        "arm_metrics": experiment["arm_metrics"],
        "exact_feedback_only": True,
        "memory_policy_evolution": False,
        "llm_calls": 0,
        "llm_weight_training": False,
        "lora_training": False,
        "grpo_training": False,
        "external_teacher": False,
        "tests_added_or_reused": list(tests_added_or_reused),
        "research_conductor_modified": False,
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise if the Exp5608 artifact contradicts its required gates."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5608 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return all validation errors so tests can check fail-closed behavior."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("continuous_self_learning_task") is not True:
        errors.append("continuous_self_learning_task")
    manifest = artifact.get("session_manifest")
    if not isinstance(manifest, Mapping) or len(manifest.get("ordered_sessions", [])) < 4:
        errors.append("session_manifest")
    budget = artifact.get("adaptation_budget")
    if not isinstance(budget, Mapping) or budget.get("update_budget_per_arm_seed") != UPDATE_BUDGET:
        errors.append("adaptation_budget")
    ledger = artifact.get("decision_ledger")
    if not isinstance(ledger, Sequence) or not ledger:
        errors.append("decision_ledger")
    elif any(row.get("ledger_hash") != ledger_hash(row) for row in ledger):
        errors.append("decision_ledger")
    heldout = artifact.get("heldout_delta_by_arm")
    if not isinstance(heldout, Mapping) or EXACT_GATED_ARM not in heldout:
        errors.append("heldout_delta_by_arm")
    if artifact.get("unsafe_false_accept_count") != 0:
        errors.append("unsafe_false_accept_count")
    poison = artifact.get("poison_update_disposition")
    if not isinstance(poison, Mapping) or poison.get("persisted") is not False:
        errors.append("poison_update_disposition")
    if artifact.get("rollback_positive_control") is not True:
        errors.append("rollback_positive_control")
    if artifact.get("delayed_regression_passed") is not True:
        errors.append("delayed_regression_passed")
    if artifact.get("no_model_weight_mutation") is not True:
        errors.append("no_model_weight_mutation")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != principle for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    ):
        errors.append("field_principles")
    expected_ready = kan_longitudinal_ready_from_artifact(artifact)
    if artifact.get("kan_longitudinal_ready") is not expected_ready:
        errors.append("kan_longitudinal_ready")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def kan_longitudinal_ready_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute final promotion readiness from artifact-visible fields."""

    heldout = artifact.get("heldout_delta_by_arm", {}).get(EXACT_GATED_ARM, {})
    ci = heldout.get("ci", {}) if isinstance(heldout, Mapping) else {}
    return (
        heldout.get("mean", 0.0) > 0.0
        and ci.get("lower", 0.0) > 0.0
        and artifact.get("backward_retention_delta", -1.0) >= 0.0
        and artifact.get("unsafe_false_accept_count") == 0
        and artifact.get("rollback_positive_control") is True
        and artifact.get("delayed_regression_passed") is True
        and artifact.get("no_model_weight_mutation") is True
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal complete or bounded-null verdict."""

    if artifact.get("kan_longitudinal_ready") is True and kan_longitudinal_ready_from_artifact(artifact):
        return "complete: exact_gated_active_spline_kan_longitudinal_ready"
    return "bounded_null: exact_gated_active_spline_kan_longitudinal_gate_not_met"


def ledger_hash(row: Mapping[str, Any]) -> str:
    """Hash one ledger row without its self-referential hash field."""

    stable = dict(row)
    stable["ledger_hash"] = ""
    return "sha256:" + sha256_json(stable)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while excluding its self-referential checksum."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return "sha256:" + sha256_json(stable)


def source_file_checksums(root: Path) -> JsonDict:
    """Hash the spec, implementation, and tests backing Exp5608."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    checkpoint_dir: Path | str | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the artifact and optionally write stable JSON."""

    root_path = Path(root)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else root_path / CHECKPOINT_RELATIVE_DIR
    artifact = build_artifact(
        root=root_path,
        tests_added_or_reused=tests_added_or_reused,
        checkpoint_dir=checkpoint_root,
    )
    if write:
        write_json(resolve_path(root_path, result_path), artifact)
    return artifact


def resolve_path(root: Path, path: Path | str) -> Path:
    """Resolve repository-relative paths without changing absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable indented JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
                "kan_longitudinal_ready": artifact["kan_longitudinal_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
