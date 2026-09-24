"""Qualify a fail-closed world-model verifier on the real E3 policy seam.

REQ-ARC-WMTE-7580. This module contains the reusable integrity guard and the
CPU-only experiment runner. The guard is opt-in. It can reject an engine that
the current path selected, but it cannot admit a rejected engine or lower the
exact held-out threshold.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.agentic.arc_executable_world_model import Transition, WorldModelVerifier
from carnot.agentic.arc_world_model_trust_energy import _split_prefix_heldout
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.experiment_7303_validation_scope import (
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)

EXPERIMENT_ID = 7580
MILESTONE = "2026.09.662"
RUN_DATE = "20260924"
REQUIREMENT_ID = "REQ-ARC-WMTE-7580"
SCHEMA = "carnot.experiment_7580_v662_arc_verifier_support.v1"
GUARD_ENV = "CARNOT_ARC_VERIFIER_INTEGRITY_GUARD"
EXACT_ACCEPTANCE_THRESHOLD = 1.0
MIN_DISTINCT_HELDOUT = 2
MODEL_SPECS: list[dict[str, Any]] = []

RESULT_REL = Path("results/experiment_7580_v662_arc_verifier_support.json")
RAW_REL = Path("results/raw/experiment_7580_v662_arc_verifier_support")
LIVE_PROTOCOL_REL = RAW_REL / "live_panel_protocol.json"
PANELS = {"A": ("su15", "sp80", "ft09"), "B": ("sb26", "g50t", "dc22")}
PANEL_GAMES = tuple(game for games in PANELS.values() for game in games)
PANEL_SEEDS = (7582001, 7582002)
PANEL_ARMS = ("current_verifier", "integrity_guard")

FIELD_PRINCIPLES = {
    "honest_verdict": "A complete prefix closes execution; it does not establish benefit.",
    "verdict_class": "A closed class keeps fixture success distinct from live benefit.",
    "flagged_adversarial": "Flagged evidence cannot open readiness.",
    "gate_check_summary": "A blocked result names the exact missing upstream value.",
    "acceptance_gate_results": "Validity, readiness, and benefit remain separate.",
    "rows": "Raw per-unit numerators and denominators make pooled values reproducible.",
    "inference_substrate_class": "No model call means no live inference claim.",
    "MODEL_SPECS": "Current model identity is empty because this task loads no model.",
    "invocation_counts": "Current loads, forwards, generations, and tokens stay separate.",
    "duration_s": "Monotonic current work excludes inherited timing and artificial sleeps.",
    "source_artifact_hashes": "Exact source bytes bind every conclusion.",
    "validation_receipts": "Commands, worktree, exits, and log hashes bind each check.",
    "field_principles": "One-line reasons keep fields auditable outside this prompt.",
    "verifier_is_oracle": "Label-reading fixture controls cannot prove oracle-distinct value.",
    "verifier_support_ready_score": "Readiness needs reproduced faults, a positive control, and E3 rejection.",
    "live_panel_protocol_path": "The later live comparison must not change panels or budgets.",
    "support_rows": "Every attempted transition retains split, exception, and purity evidence.",
    "solve_provenance": "Development fixtures receive no game-level solve credit.",
    "production_defaults_changed": "The research guard remains opt-in and reject-only.",
}


def progress(started: float, phase: str, event: str, **detail: Any) -> None:
    """Print one flushed boundary with monotonic elapsed time."""

    fields = " ".join(f"{key}={value}" for key, value in sorted(detail.items()))
    suffix = f" {fields}" if fields else ""
    print(
        f"[exp7580] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_hash(value: Any) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transition_id(row: Any) -> str:
    """Hash the full observed transition, including its answer."""

    return canonical_hash(
        {
            "grid": np.asarray(row.grid),
            "action": int(row.action),
            "data": deepcopy(row.data),
            "next_grid": np.asarray(row.next_grid),
            "level_before": int(row.level_before),
            "level_after": int(row.level_after),
        }
    )


def transition_input_id(row: Any) -> str:
    """Hash only what an engine receives so contradictory answers share one ID."""

    return canonical_hash(
        {"grid": np.asarray(row.grid), "action": int(row.action), "data": deepcopy(row.data)}
    )


def current_prefix_heldout(rows: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    return _split_prefix_heldout(list(rows))


def integrity_guard_enabled() -> bool:
    """Only the exact value ``1`` arms the research guard."""

    return os.environ.get(GUARD_ENV) == "1"


def _prediction(engine: Callable[..., Any], row: Any) -> tuple[np.ndarray | None, str]:
    try:
        raw = engine(np.asarray(row.grid).copy(), int(row.action), deepcopy(row.data))
        array = np.asarray(raw)
        if array.dtype.kind not in "iub" or array.ndim != 2:
            return None, f"OutputTypeError:{array.dtype.str}:{array.ndim}d"
        return array.astype(np.int64), ""
    except BaseException as exc:  # generated code can raise SystemExit or KeyboardInterrupt
        return None, f"{type(exc).__name__}:{str(exc)[:120]}"


def _same(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    return bool(
        left is not None
        and right is not None
        and left.shape == right.shape
        and np.array_equal(left, right)
    )


def _support_failures(
    heldout: Sequence[Any], prompt_ids: set[str], refactor_ids: set[str]
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    heldout_ids = [transition_id(row) for row in heldout]
    failures: list[str] = []
    overlap = sorted(set(heldout_ids) & (prompt_ids | refactor_ids))
    if overlap:
        failures.append("heldout_overlaps_prompt_or_refactor")
    if len(set(heldout_ids)) < len(heldout_ids):
        failures.append("duplicate_heldout_transition_ids")
    if len(set(heldout_ids)) < MIN_DISTINCT_HELDOUT:
        failures.append("fewer_than_two_distinct_heldout_transitions")
    answers: dict[str, set[str]] = defaultdict(set)
    for row in heldout:
        answers[transition_input_id(row)].add(canonical_hash(np.asarray(row.next_grid)))
    contradictions = {
        input_id: sorted(values) for input_id, values in answers.items() if len(values) > 1
    }
    if contradictions:
        failures.append("contradictory_same_input_transitions")
    return failures, overlap, contradictions


def verify_integrity(
    transitions: Sequence[Any],
    engine_factory: Callable[[], Callable[..., Any]] | None,
    *,
    heldout_rows: Sequence[Any] | None = None,
    prompt_transition_ids: Sequence[str] | None = None,
    refactor_transition_ids: Sequence[str] | None = None,
    exact_threshold: float = EXACT_ACCEPTANCE_THRESHOLD,
) -> dict[str, Any]:
    """Score one candidate without dropping exceptions or sharing engine state.

    The first prediction from a fresh instance is the scored prediction. The
    same instance repeats the input once, then a second fresh instance predicts
    it again. Any drift is an integrity failure even when the first answer was
    correct by chance.
    """

    rows = list(transitions)
    if heldout_rows is None:
        _prefix, selected = current_prefix_heldout(rows)
        heldout = list(selected)
    else:
        heldout = list(heldout_rows)
    prompt_ids = set(prompt_transition_ids or ())
    refactor_ids = set(refactor_transition_ids or ())
    failures, overlap, contradictions = _support_failures(heldout, prompt_ids, refactor_ids)
    if not callable(engine_factory):
        failures.append("missing_fresh_engine_factory")

    exact_numerator = 0
    fidelity_sum = 0.0
    changing_denominator = 0
    noop_denominator = 0
    noop_hallucinations = 0
    exception_count = 0
    exception_kinds: Counter[str] = Counter()
    purity_passed = True
    support_rows: list[dict[str, Any]] = []
    for index, row in enumerate(heldout):
        first = repeat = fresh = None
        first_error = repeat_error = fresh_error = ""
        if callable(engine_factory):
            try:
                engine = engine_factory()
            except BaseException as exc:
                first_error = f"Factory{type(exc).__name__}:{str(exc)[:120]}"
            else:
                first, first_error = _prediction(engine, row)
                repeat, repeat_error = _prediction(engine, row)
            try:
                second_engine = engine_factory()
            except BaseException as exc:
                fresh_error = f"Factory{type(exc).__name__}:{str(exc)[:120]}"
            else:
                fresh, fresh_error = _prediction(second_engine, row)
        else:
            first_error = repeat_error = fresh_error = "missing_fresh_engine_factory"

        repeated_equal = not first_error and not repeat_error and _same(first, repeat)
        fresh_equal = not first_error and not fresh_error and _same(first, fresh)
        row_pure = bool(repeated_equal and fresh_equal)
        purity_passed = purity_passed and row_pure
        if first_error:
            exception_count += 1
            exception_kinds[first_error.split(":", 1)[0]] += 1

        origin = np.asarray(row.grid)
        expected = np.asarray(row.next_grid)
        shape_ok = first is not None and first.shape == expected.shape
        exact = bool(shape_ok and np.array_equal(first, expected))
        exact_numerator += int(exact)
        changed = not np.array_equal(origin, expected)
        fidelity: float | None = None
        hallucinated: bool | None = None
        if changed:
            changing_denominator += 1
            fidelity = 0.0
            if shape_ok:
                wrote = first != origin
                union = (origin != expected) | wrote
                fidelity = float(((first == expected) & union).sum() / union.sum())
            fidelity_sum += fidelity
        else:
            noop_denominator += 1
            hallucinated = not exact
            noop_hallucinations += int(hallucinated)
        support_rows.append(
            {
                "index": index,
                "transition_id": transition_id(row),
                "input_id": transition_input_id(row),
                "split_membership": "heldout",
                "changing": bool(changed),
                "exact": exact,
                "change_fidelity": fidelity,
                "noop_hallucinated": hallucinated,
                "exception": first_error or None,
                "repeat_exception": repeat_error or None,
                "fresh_exception": fresh_error or None,
                "repeat_same_instance_equal": repeated_equal,
                "fresh_instance_equal": fresh_equal,
                "purity_passed": row_pure,
            }
        )

    exact_denominator = len(heldout)
    exact_accuracy = exact_numerator / exact_denominator if exact_denominator else 0.0
    change_fidelity = fidelity_sum / changing_denominator if changing_denominator else 0.0
    noop_rate = noop_hallucinations / noop_denominator if noop_denominator else 0.0
    if failures:
        reason = "insufficient_support"
    elif exception_count:
        reason = "engine_exception"
    elif not purity_passed:
        reason = "impure_engine"
    elif exact_accuracy < float(exact_threshold):
        reason = "exact_below_threshold"
    else:
        reason = "accepted"
    return {
        "accepted": reason == "accepted",
        "reason": reason,
        "exact_threshold": float(exact_threshold),
        "exact_numerator": exact_numerator,
        "exact_denominator": exact_denominator,
        "exact_accuracy": exact_accuracy,
        "change_fidelity_numerator": fidelity_sum,
        "changing_denominator": changing_denominator,
        "change_fidelity": change_fidelity,
        "noop_hallucination_numerator": noop_hallucinations,
        "noop_denominator": noop_denominator,
        "noop_hallucination_rate": noop_rate,
        "exception_count": exception_count,
        "exception_kinds": dict(exception_kinds),
        "purity_passed": purity_passed,
        "distinct_heldout_transition_count": len({transition_id(row) for row in heldout}),
        "support_failures": failures,
        "overlap_transition_ids": overlap,
        "contradictory_inputs": contradictions,
        "split_hash": canonical_hash(
            {
                "prompt": sorted(prompt_ids),
                "refactor": sorted(refactor_ids),
                "heldout": [transition_id(row) for row in heldout],
            }
        ),
        "effective_flags": {GUARD_ENV: True, "exact_threshold": exact_threshold},
        "support_rows": support_rows,
    }


def apply_e3_integrity_guard(
    attempt: dict[str, Any],
    outcome: Any,
    *,
    game: str,
    transitions: Sequence[Any],
    load_engine: Callable[[str], tuple[Callable[..., Any], Any]],
    proposal_transitions: Sequence[Any] | None = None,
) -> Any:
    """Post-qualify the candidate selected by E3 without widening acceptance."""

    if not integrity_guard_enabled():
        return outcome
    rows = list(transitions)
    if proposal_transitions is None:
        from carnot.agentic.arc_llm_reinduction import _proposal_prefix

        prompt_rows = list(_proposal_prefix(rows))
    else:
        prompt_rows = list(proposal_transitions)
    _prefix, heldout = current_prefix_heldout(rows)
    refactor_rows: list[Any] = []
    if int(getattr(outcome, "refinement_rounds_used", 0)) > 1:
        from carnot.agentic.arc_world_model_trust_energy import (
            cegis_accept_split_enabled,
            split_refinement_acceptance,
        )

        refactor_rows = (
            list(split_refinement_acceptance(rows).refinable)
            if cegis_accept_split_enabled()
            else rows
        )
    selected_name = str(getattr(outcome, "selected_candidate_name", ""))
    factory: Callable[[], Callable[..., Any]] | None = None
    if selected_name == "loaded_world_model.py":
        factory = lambda: load_engine(str(game))[0]
    receipt = verify_integrity(
        rows,
        factory,
        heldout_rows=heldout,
        prompt_transition_ids=[transition_id(row) for row in prompt_rows],
        refactor_transition_ids=[transition_id(row) for row in refactor_rows],
    )
    receipt["selected_candidate_name"] = selected_name
    receipt["reject_only"] = True
    attempt["verifier_integrity_guard"] = receipt
    if not receipt["accepted"]:
        outcome.planned = False
        outcome.plan = []
        outcome.accepted_by_heldout_verifier = False
        outcome.skipped = f"verifier_integrity_{receipt['reason']}"
    return outcome


def _fixture_row(value: int, *, step: int = 1, changed: bool = True) -> Transition:
    grid = np.asarray([[value]], dtype=np.int64)
    next_grid = np.asarray([[value + (step if changed else 0)]], dtype=np.int64)
    return Transition(grid, 1, None, next_grid, 0, 0)


def _source_factory(source: str) -> Callable[[], Callable[..., Any]]:
    """Build a new generated-code namespace for every requested engine instance."""

    def build() -> Callable[..., Any]:
        namespace: dict[str, Any] = {"np": np}
        exec(compile(source, "<exp7580-generated-engine>", "exec"), namespace)
        return namespace["engine"]

    return build


def _current_score(rows: Sequence[Any], engine: Callable[..., Any]) -> dict[str, Any]:
    result = WorldModelVerifier(list(rows)).score(engine)
    return {
        "n": result.n,
        "n_correct": result.n_correct,
        "accuracy": result.accuracy,
        "n_changing": result.n_changing,
        "change_fidelity": result.change_fidelity,
        "n_noop": result.n_noop,
        "noop_hallucination_rate": result.noop_hallucination_rate,
        "n_engine_called": result.n_engine_called,
        "n_engine_raised": result.n_engine_raised,
        "engine_raise_kinds": dict(result.engine_raise_kinds),
    }


def _measurement_row(
    unit: str,
    kind: str,
    current: Mapping[str, Any],
    guard: Mapping[str, Any],
    *,
    expected_guard_accept: bool,
    reproduced_or_control_passed: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "unit": unit,
        "arm": "current_verifier_vs_integrity_guard",
        "kind": kind,
        "seed": 7580,
        "metric_direction": "reject_adversarial_accept_independent_positive",
        "censored": False,
        "provenance": "independently_built_fixture",
        "current": dict(current),
        "guard_accepted": bool(guard.get("accepted")),
        "guard_reason": guard.get("reason"),
        "guard_exact_numerator": guard.get("exact_numerator"),
        "guard_exact_denominator": guard.get("exact_denominator"),
        "guard_changing_denominator": guard.get("changing_denominator"),
        "guard_noop_denominator": guard.get("noop_denominator"),
        "guard_exception_count": guard.get("exception_count"),
        "guard_purity_passed": guard.get("purity_passed"),
        "guard_split_hash": guard.get("split_hash"),
        "expected_guard_accept": expected_guard_accept,
        "reproduced_or_control_passed": reproduced_or_control_passed,
        "support_rows": deepcopy(list(guard.get("support_rows") or [])),
        **dict(extra or {}),
    }


def measure_fixture_support() -> dict[str, Any]:
    """Reproduce the three B2 defects and retain two fail-closed controls."""

    measured: list[dict[str, Any]] = []

    exception_rows = [_fixture_row(i) for i in range(8)]
    exception_source = """\
def engine(grid, action, data=None):
    if int(grid[0, 0]) < 7:
        raise RuntimeError("fixture")
    return grid + 1
"""
    exception_factory = _source_factory(exception_source)
    exception_current = _current_score(exception_rows, exception_factory())
    exception_guard = verify_integrity(
        exception_rows, exception_factory, heldout_rows=exception_rows
    )
    measured.append(
        _measurement_row(
            "exception_denominator_loss",
            "defect_reproduction",
            exception_current,
            exception_guard,
            expected_guard_accept=False,
            reproduced_or_control_passed=bool(
                exception_current["n_engine_raised"] == 7
                and exception_current["n_changing"] == 1
                and exception_current["change_fidelity"] == 1.0
                and exception_guard["changing_denominator"] == 8
                and exception_guard["accepted"] is False
            ),
        )
    )

    stateful_rows = [_fixture_row(base, step=step) for base, step in ((0, 1), (10, 2), (20, 3))]
    stateful_source = """\
counter = 0
def engine(grid, action, data=None):
    global counter
    counter += 1
    return grid + counter
"""
    stateful_factory = _source_factory(stateful_source)
    chronological = _current_score(stateful_rows, stateful_factory())
    reversed_score = _current_score(list(reversed(stateful_rows)), stateful_factory())
    stateful_guard = verify_integrity(stateful_rows, stateful_factory, heldout_rows=stateful_rows)
    measured.append(
        _measurement_row(
            "stateful_order_dependence",
            "defect_reproduction",
            chronological,
            stateful_guard,
            expected_guard_accept=False,
            reproduced_or_control_passed=bool(
                chronological["accuracy"] == 1.0
                and reversed_score["accuracy"] < 1.0
                and stateful_guard["reason"] == "impure_engine"
            ),
            extra={"current_reversed": reversed_score},
        )
    )

    overlap_row = _fixture_row(40)
    prefix, heldout = current_prefix_heldout([overlap_row])
    honest_factory = _source_factory("def engine(grid, action, data=None):\n    return grid + 1\n")
    overlap_guard = verify_integrity(
        [overlap_row],
        honest_factory,
        heldout_rows=heldout,
        prompt_transition_ids=[transition_id(row) for row in prefix],
    )
    overlap_current = {
        "prefix": _current_score(prefix, honest_factory()),
        "heldout": _current_score(heldout, honest_factory()),
        "same_object": prefix[0] is heldout[0],
    }
    measured.append(
        _measurement_row(
            "one_row_overlap",
            "defect_reproduction",
            overlap_current,
            overlap_guard,
            expected_guard_accept=False,
            reproduced_or_control_passed=bool(
                overlap_current["same_object"]
                and overlap_current["prefix"]["accuracy"] == 1.0
                and overlap_current["heldout"]["accuracy"] == 1.0
                and overlap_guard["reason"] == "insufficient_support"
            ),
        )
    )

    prompt_rows = [_fixture_row(50), _fixture_row(51)]
    positive_rows = [_fixture_row(60), _fixture_row(70)]
    positive_guard = verify_integrity(
        [*prompt_rows, *positive_rows],
        honest_factory,
        heldout_rows=positive_rows,
        prompt_transition_ids=[transition_id(row) for row in prompt_rows],
    )
    measured.append(
        _measurement_row(
            "independent_positive",
            "positive_control",
            _current_score(positive_rows, honest_factory()),
            positive_guard,
            expected_guard_accept=True,
            reproduced_or_control_passed=positive_guard["accepted"] is True,
        )
    )

    contradictory = [
        Transition(np.asarray([[80]]), 1, None, np.asarray([[81]]), 0, 0),
        Transition(np.asarray([[80]]), 1, None, np.asarray([[82]]), 0, 0),
    ]
    contradiction_guard = verify_integrity(
        contradictory, honest_factory, heldout_rows=contradictory
    )
    measured.append(
        _measurement_row(
            "contradictory_same_input",
            "adversarial_control",
            _current_score(contradictory, honest_factory()),
            contradiction_guard,
            expected_guard_accept=False,
            reproduced_or_control_passed=bool(
                contradiction_guard["reason"] == "insufficient_support"
                and "contradictory_same_input_transitions"
                in contradiction_guard["support_failures"]
            ),
        )
    )
    reduction = independent_reduce(measured)
    return {
        "rows": measured,
        "support_rows": [
            {"unit": row["unit"], **support} for row in measured for support in row["support_rows"]
        ],
        "independent_reduction": reduction,
        "verifier_support_ready_score": reduction["verifier_support_ready_score"],
    }


def independent_reduce(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce fixture rows without trusting their stored readiness score."""

    by_name = {str(row.get("unit")): row for row in rows}
    required = {
        "exception_denominator_loss",
        "stateful_order_dependence",
        "one_row_overlap",
        "independent_positive",
        "contradictory_same_input",
    }
    complete = set(by_name) == required
    expected = all(
        bool(row.get("guard_accepted")) is bool(row.get("expected_guard_accept"))
        and row.get("reproduced_or_control_passed") is True
        for row in by_name.values()
    )
    exception = by_name.get("exception_denominator_loss", {})
    positive = by_name.get("independent_positive", {})
    denominators = (
        exception.get("guard_exact_denominator") == 8
        and exception.get("guard_changing_denominator") == 8
        and exception.get("guard_exception_count") == 7
        and positive.get("guard_exact_denominator") == 2
    )
    ready = bool(complete and expected and denominators)
    return {
        "fixture_set_complete": complete,
        "expected_contrasts_passed": expected,
        "denominator_contract_passed": denominators,
        "verifier_support_ready_score": int(ready),
    }


def _registry_games(path: Path) -> dict[str, Mapping[str, Any]]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = value.get("games") if isinstance(value, Mapping) else value
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("game")): row for row in rows if isinstance(row, Mapping) and row.get("game")
    }


def build_live_panel_protocol(repo_root: Path) -> dict[str, Any]:
    """Freeze the downstream live A/B without running a model or game."""

    registry_path = Path(repo_root) / "ops/arc_solve_registry.yaml"
    games = _registry_games(registry_path)
    rows = []
    for panel, panel_games in PANELS.items():
        for game in panel_games:
            known = games.get(game)
            for seed in PANEL_SEEDS:
                for arm in PANEL_ARMS:
                    rows.append(
                        {
                            "panel": panel,
                            "game": game,
                            "seed": seed,
                            "arm": arm,
                            "registry_prechecked": known is not None,
                            "known_full_game_clear": bool(
                                known and known.get("full_game_clear") is True
                            ),
                            "solve_credit_allowed": False,
                            "censored": False,
                            "provenance": "frozen_protocol_unstarted",
                        }
                    )
    return {
        "schema": "carnot.exp7580.live_panel_protocol.v1",
        "panels": {key: list(value) for key, value in PANELS.items()},
        "seeds": list(PANEL_SEEDS),
        "arms": list(PANEL_ARMS),
        "max_actions_per_episode": 600,
        "max_inductions_per_episode": 1,
        "plan_outcome_window_actions": 32,
        "exact_acceptance_threshold": EXACT_ACCEPTANCE_THRESHOLD,
        "generator_settings_identical_between_arms": True,
        "adapter_withholding": {
            "game_adapters": True,
            "stored_engines": True,
            "stored_solutions": True,
            "banked_trajectories": True,
        },
        "source_derived_masks_allowed": False,
        "source_derived_models_allowed": False,
        "registry_path": "ops/arc_solve_registry.yaml",
        "registry_sha256": sha256_file(registry_path),
        "rows": rows,
    }


def validate_live_panel_protocol(protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if protocol.get("panels") != {key: list(value) for key, value in PANELS.items()}:
        errors.append("panels")
    if protocol.get("seeds") != list(PANEL_SEEDS):
        errors.append("seeds")
    if protocol.get("arms") != list(PANEL_ARMS):
        errors.append("arms")
    if protocol.get("max_actions_per_episode") != 600:
        errors.append("max_actions_per_episode")
    if protocol.get("max_inductions_per_episode") != 1:
        errors.append("max_inductions_per_episode")
    if protocol.get("plan_outcome_window_actions") != 32:
        errors.append("plan_outcome_window_actions")
    rows = protocol.get("rows")
    if not isinstance(rows, list) or len(rows) != 24:
        errors.append("rows")
    elif any(
        row.get("registry_prechecked") is not True or row.get("solve_credit_allowed") is not False
        for row in rows
    ):
        errors.append("registry_or_solve_credit")
    if protocol.get("source_derived_masks_allowed") is not False:
        errors.append("source_derived_masks_allowed")
    if protocol.get("source_derived_models_allowed") is not False:
        errors.append("source_derived_models_allowed")
    return errors


@contextlib.contextmanager
def _temporary_env(changes: Mapping[str, str | None]):
    previous = {key: os.environ.get(key) for key in changes}
    try:
        for key, value in changes.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _ScriptedFrame:
    def __init__(self, grid: np.ndarray) -> None:
        self.frame = [grid.tolist()]
        self.levels_completed = 0
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


def _drive_e3_disabled(guard_value: str | None, n_actions: int) -> dict[str, Any]:
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    random.seed(7580)
    np.random.seed(7580)
    with _temporary_env(
        {
            GUARD_ENV: guard_value,
            "CARNOT_ARC_DISABLE_INDUCTION": "1",
            "CARNOT_ARC_ACTION_PROVENANCE": None,
        }
    ):
        policy = E3AgentPolicy("zz99", proposer=None, explore_budget=100)
        frames: list[_ScriptedFrame] = []
        latest = None
        actions = []
        for index in range(n_actions):
            action = policy.next_move(frames, latest)
            actions.append(_jsonable(action))
            rng = np.random.RandomState(index + 1)
            latest = _ScriptedFrame(rng.randint(0, 4, size=(8, 8)))
            frames.append(latest)
        return {
            "actions": actions,
            "model_calls": 0,
            "provenance": policy.action_provenance(),
            "environment_work": {"frames": len(frames), "transitions": len(policy.transitions)},
            "rng_state": canonical_hash(
                {"python": repr(random.getstate()), "numpy": repr(np.random.get_state())}
            ),
        }


def run_disabled_e3_parity(n_actions: int = 6) -> dict[str, Any]:
    """Compare unset and explicit-off guard states through real policy calls."""

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        unset = _drive_e3_disabled(None, n_actions)
        explicit_off = _drive_e3_disabled("0", n_actions)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    result = {
        "actions_equal": unset["actions"] == explicit_off["actions"],
        "calls_equal": unset["model_calls"] == explicit_off["model_calls"],
        "provenance_equal": unset["provenance"] == explicit_off["provenance"],
        "environment_work_equal": unset["environment_work"] == explicit_off["environment_work"],
        "rng_state_equal": unset["rng_state"] == explicit_off["rng_state"],
        "current_model_calls": unset["model_calls"] + explicit_off["model_calls"],
        "unset": unset,
        "explicit_off": explicit_off,
    }
    result["passed"] = all(
        result[key]
        for key in (
            "actions_equal",
            "calls_equal",
            "provenance_equal",
            "environment_work_equal",
            "rng_state_equal",
        )
    )
    return result


def measure_e3_call_site_rejection() -> dict[str, Any]:
    """Drive the actual E3 guard method with one generated mutable engine."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    rows = [_fixture_row(index) for index in range(6)]
    source = """\
counter = 0
def engine(grid, action, data=None):
    global counter
    counter += 1
    return grid + counter
"""
    factory = _source_factory(source)
    outcome = SimpleNamespace(
        planned=True,
        plan=[{"action": 1}],
        engine=factory(),
        selected_candidate_name="loaded_world_model.py",
        accepted_by_heldout_verifier=True,
        refinement_rounds_used=1,
        skipped="",
    )
    attempt: dict[str, Any] = {}
    with _temporary_env({GUARD_ENV: "1", "CARNOT_ARC_DISABLE_INDUCTION": "1"}):
        policy = E3AgentPolicy("zz99", proposer=SimpleNamespace(), explore_budget=100)
        policy._apply_world_model_integrity_guard(
            attempt,
            outcome,
            game="zz99",
            transitions=rows,
            load_engine=lambda game: (factory(), None),
        )
    receipt = dict(attempt.get("verifier_integrity_guard") or {})
    passed = bool(
        receipt.get("accepted") is False
        and receipt.get("reason") == "impure_engine"
        and outcome.planned is False
        and outcome.plan == []
        and outcome.accepted_by_heldout_verifier is False
    )
    return {
        "passed": passed,
        "outcome": {
            "planned": outcome.planned,
            "plan": outcome.plan,
            "accepted_by_heldout_verifier": outcome.accepted_by_heldout_verifier,
            "skipped": outcome.skipped,
        },
        "receipt": receipt,
        "generated_source_sha256": canonical_hash(source),
    }


def _artifact_rows(fixtures: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in fixtures.get("rows") or []:
        current = comparison.get("current") or {}
        rows.append(
            {
                "unit": comparison.get("unit"),
                "arm": "current_verifier",
                "seed": comparison.get("seed"),
                "metric_direction": comparison.get("metric_direction"),
                "numerator": current.get("n_correct"),
                "denominator": current.get("n"),
                "accepted": (
                    float(current.get("accuracy", 0.0)) >= EXACT_ACCEPTANCE_THRESHOLD
                    if isinstance(current.get("accuracy"), (int, float))
                    else None
                ),
                "censored": False,
                "provenance": comparison.get("provenance"),
            }
        )
        rows.append(
            {
                "unit": comparison.get("unit"),
                "arm": "integrity_guard",
                "seed": comparison.get("seed"),
                "metric_direction": comparison.get("metric_direction"),
                "numerator": comparison.get("guard_exact_numerator"),
                "denominator": comparison.get("guard_exact_denominator"),
                "accepted": comparison.get("guard_accepted"),
                "censored": False,
                "provenance": comparison.get("provenance"),
            }
        )
    return rows


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "experiment_id": artifact.get("experiment_id"),
            "run_date": artifact.get("run_date"),
            "fixture_comparisons": artifact.get("fixture_comparisons"),
            "live_panel_protocol": artifact.get("live_panel_protocol"),
            "source_artifact_hashes": artifact.get("source_artifact_hashes"),
            "effective_flags": artifact.get("effective_flags"),
        }
    )


def build_artifact(
    *,
    repo_root: Path,
    run_date: str,
    duration_s: float,
    fixtures: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    validation_receipts: Sequence[Mapping[str, Any]],
    e2e_receipts: Sequence[Mapping[str, Any]],
    terminal_receipts: Sequence[Mapping[str, Any]],
    parity: Mapping[str, Any] | None = None,
    e3_rejection: Mapping[str, Any] | None = None,
    flagged_adversarial: bool = False,
) -> dict[str, Any]:
    parity_row = dict(parity or {"passed": True})
    e3_row = dict(e3_rejection or {"passed": True})
    validation_passed = all(row.get("passed") is True for row in validation_receipts)
    e2e_passed = all(row.get("passed") is True for row in e2e_receipts)
    terminal_passed = all(row.get("passed") is True for row in terminal_receipts)
    fixture_ready = fixtures.get("verifier_support_ready_score") == 1
    ready = bool(
        fixture_ready
        and parity_row.get("passed") is True
        and e3_row.get("passed") is True
        and validation_passed
        and e2e_passed
        and terminal_passed
        and not flagged_adversarial
    )
    verdict_class = "circular_positive" if ready else "disqualified"
    verdict = (
        "complete_circular_positive_fixture_guard_support_ready_no_live_benefit_claim"
        if ready
        else "complete_disqualified_required_validation_failure"
    )
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "gate_check_summary": [],
        "acceptance_gate_results": {
            "validity": bool(fixture_ready and parity_row.get("passed") and e3_row.get("passed")),
            "readiness": ready,
            "benefit": False,
            "benefit_reason": "No live panel or model call ran; fixture success is circular.",
        },
        "rows": _artifact_rows(fixtures),
        "fixture_comparisons": deepcopy(list(fixtures.get("rows") or [])),
        "support_rows": deepcopy(list(fixtures.get("support_rows") or [])),
        "independent_reduction": deepcopy(dict(fixtures.get("independent_reduction") or {})),
        "e3_disabled_parity": parity_row,
        "e3_call_site_rejection": e3_row,
        "inference_substrate": "deterministic_cpu_verifier_fixtures_and_protocol_freeze",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            **ZERO_INVOCATION_COUNTS,
            "forward_calls_attempted": 0,
            "forward_calls_completed": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "duration_s": float(duration_s),
        "source_artifact_hashes": dict(source_hashes),
        "validation_receipts": [dict(row) for row in validation_receipts],
        "e2e_receipts": [dict(row) for row in e2e_receipts],
        "terminal_validation_receipts": [dict(row) for row in terminal_receipts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "verifier_is_oracle": True,
        "verifier_support_ready_score": int(ready),
        "live_panel_protocol_path": LIVE_PROTOCOL_REL.as_posix(),
        "live_panel_protocol": deepcopy(dict(protocol)),
        "solve_provenance": "development_proxy",
        "production_defaults_changed": False,
        "effective_flags": {
            "guard_env": GUARD_ENV,
            "guard_default": False,
            "exact_acceptance_threshold": EXACT_ACCEPTANCE_THRESHOLD,
            "hud_mask_added": False,
            "source_dynamics_added": False,
        },
        "random_seed": 7580,
        "methodology_note": (
            "Exact 1.0 is expected on the code-defined positive fixture. It is a circular "
            "plumbing control, not a model-capability or live-benefit result."
        ),
        "prior_verdict_retirement": {
            "experiment_id": 6015,
            "scope": "Do not repeat HUD-mask-only A/B; verifier integrity uses no HUD treatment.",
            "scientific_hypothesis_retired": False,
        },
        "repository_root": str(Path(repo_root).resolve()),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    """Build a complete blocked record without substitute measurement evidence."""

    summary = {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "honest_verdict": f"complete_blocked_{check}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": summary,
        "acceptance_gate_results": {"validity": False, "readiness": False, "benefit": False},
        "rows": [],
        "support_rows": [],
        "inference_substrate": "precondition_check_only",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            **ZERO_INVOCATION_COUNTS,
            "forward_calls_attempted": 0,
            "forward_calls_completed": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "duration_s": float(duration_s),
        "source_artifact_hashes": {},
        "validation_receipts": [],
        "e2e_receipts": [],
        "terminal_validation_receipts": [],
        "field_principles": dict(FIELD_PRINCIPLES),
        "verifier_is_oracle": True,
        "verifier_support_ready_score": 0,
        "live_panel_protocol_path": LIVE_PROTOCOL_REL.as_posix(),
        "solve_provenance": "development_proxy",
        "production_defaults_changed": False,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check schema, raw reduction, support, and zero-call declarations."""

    errors: list[str] = []
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if artifact.get("production_defaults_changed") is not False:
        errors.append("production_defaults_changed")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_declaration")
    counts = artifact.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(value != 0 for value in counts.values()):
        errors.append("invocation_counts")
    if artifact.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if set(artifact.get("field_principles") or {}) != set(FIELD_PRINCIPLES):
        errors.append("field_principles")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if not isinstance(summary, Mapping) or set(summary) != required:
            errors.append("gate_check_summary")
        if artifact.get("verifier_support_ready_score") != 0:
            errors.append("blocked_ready_score")
        return errors

    protocol = artifact.get("live_panel_protocol")
    if not isinstance(protocol, Mapping):
        errors.append("live_panel_protocol")
    else:
        errors.extend(
            f"live_panel_protocol:{error}" for error in validate_live_panel_protocol(protocol)
        )
    fixture_rows = artifact.get("fixture_comparisons")
    if not isinstance(fixture_rows, list):
        errors.append("fixture_comparisons")
    else:
        reduced = independent_reduce(fixture_rows)
        if reduced != artifact.get("independent_reduction"):
            errors.append("independent_reduction")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 10:
        errors.append("rows")
    elif Counter(str(row.get("arm")) for row in rows) != Counter(
        {"current_verifier": 5, "integrity_guard": 5}
    ):
        errors.append("rows_per_arm")
    if not isinstance(artifact.get("support_rows"), list) or not artifact.get("support_rows"):
        errors.append("support_rows")
    flags = artifact.get("effective_flags")
    if not isinstance(flags, Mapping) or flags.get("guard_default") is not False:
        errors.append("effective_flags")
    ready = artifact.get("verifier_support_ready_score")
    if ready not in (0, 1):
        errors.append("verifier_support_ready_score")
    if ready == 1:
        if artifact.get("verdict_class") != "circular_positive":
            errors.append("ready_verdict_class")
        if artifact.get("flagged_adversarial") is not False:
            errors.append("ready_flagged")
        if artifact.get("e3_disabled_parity", {}).get("passed") is not True:
            errors.append("e3_disabled_parity")
        if artifact.get("e3_call_site_rejection", {}).get("passed") is not True:
            errors.append("e3_call_site_rejection")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


AFFECTED_MANIFEST = {
    "tests": ("tests/python/test_experiment_7580_v662_arc_verifier_support.py",),
    "changed_modules": ("python/carnot/experiment_7580_v662_arc_verifier_support.py",),
    "static_paths": (
        "scripts/experiments/experiment_7580_v662_arc_verifier_support.py",
        "python/carnot/agentic/arc_competition_agent.py",
    ),
    "spec": "openspec/capabilities/arc-world-model-trust-energy/spec.md",
}

PREREQUISITE_PATHS = (
    "CLAUDE.md",
    "CODEX.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "ops/arc_solve_registry.yaml",
    "docs/research-notes/b2-positive-control-2026-09-23.md",
    "results/experiment_6015_wm_hud_mask_change_gate_four_arm_live.json",
    "python/carnot/agentic/arc_executable_world_model.py",
    "python/carnot/agentic/arc_competition_agent.py",
    AFFECTED_MANIFEST["spec"],
)


def collect_preconditions(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Authenticate every external input before fixture measurement."""

    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for label in PREREQUISITE_PATHS:
        path = root / label
        exists = path.is_file()
        rows.append(
            {
                "check": "file_exists",
                "upstream": "worktree",
                "path": label,
                "field": "is_file",
                "op": "==",
                "expected": True,
                "observed": exists,
                "passed": exists,
            }
        )
        if exists:
            hashes[label] = sha256_file(path)
    spec_path = root / AFFECTED_MANIFEST["spec"]
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    requirement_present = REQUIREMENT_ID in spec_text
    rows.append(
        {
            "check": "matching_requirement",
            "upstream": "OpenSpec",
            "path": AFFECTED_MANIFEST["spec"],
            "field": REQUIREMENT_ID,
            "op": "contains",
            "expected": True,
            "observed": requirement_present,
            "passed": requirement_present,
        }
    )
    registry = (
        _registry_games(root / "ops/arc_solve_registry.yaml")
        if (root / "ops/arc_solve_registry.yaml").is_file()
        else {}
    )
    observed_games = sorted(game for game in PANEL_GAMES if game in registry)
    rows.append(
        {
            "check": "panel_registry_precheck",
            "upstream": "arc_solve_registry",
            "path": "ops/arc_solve_registry.yaml",
            "field": "panel_games",
            "op": "==",
            "expected": sorted(PANEL_GAMES),
            "observed": observed_games,
            "passed": observed_games == sorted(PANEL_GAMES),
        }
    )
    note = root / "docs/research-notes/b2-positive-control-2026-09-23.md"
    note_text = note.read_text(encoding="utf-8") if note.is_file() else ""
    leads = all(
        phrase in note_text
        for phrase in (
            "Rows where the engine raises are dropped",
            "gate does not enforce engine purity",
            "measures nothing out of sample",
            "exp6015 already ran the mask live",
        )
    )
    rows.append(
        {
            "check": "b2_leads_present",
            "upstream": "b2_positive_control_note",
            "path": str(note.relative_to(root)),
            "field": "four_named_leads",
            "op": "==",
            "expected": True,
            "observed": leads,
            "passed": leads,
        }
    )
    return rows, hashes


def _e2e_commands(root: Path, private: Path) -> list[Any]:  # pragma: no cover - declarations
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    targets = {
        "e2e_009": ("tests/python/test_arc_induction_state_persistence.py",),
        "e2e_010": ("tests/python/test_arc_tool_grammar_transport.py",),
        "e2e_011": ("tests/python/test_arc_decision_telemetry.py",),
        "e2e_012": (
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7492_e6_timed_cost_profile.py",
            "tests/python/test_arc_decision_telemetry.py",
        ),
        "e2e_013": (
            "tests/python/test_arc_decision_telemetry.py",
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
            "tests/python/test_semif_arc_readout_eval.py",
        ),
    }
    commands = [
        CommandSpec(
            name,
            (pytest, *common, f"--basetemp={private / name}", *paths, "-q"),
            name.replace("_", "-").upper(),
            900.0,
        )
        for name, paths in targets.items()
    ]
    commands.append(
        CommandSpec(
            "llm_off_environment_smoke",
            (
                "/usr/bin/env",
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private / "r11l-smoke.json"),
            ),
            "private LLM-off real-environment smoke",
            300.0,
        )
    )
    return commands


def _terminal_commands(root: Path, candidate: Path) -> list[Any]:  # pragma: no cover
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    python = str(root / ".venv/bin/python")
    wrapper = str(root / "scripts/experiments/experiment_7580_v662_arc_verifier_support.py")
    common = ("--root", str(root), "--date", RUN_DATE)
    return [
        CommandSpec(
            "declared_entrypoint",
            (python, "-u", wrapper, *common, "--validate", str(candidate)),
            "declared read-only entrypoint",
            300.0,
        ),
        CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, *common, "--cold-replay", str(candidate)),
            "fresh-process fixture replay",
            300.0,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, *common, "--independent-reduce", str(candidate)),
            "independent raw-row reduction",
            300.0,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact measured candidate",
            300.0,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact measured candidate",
            300.0,
        ),
    ]


def _prepare_command_parent(command: Any) -> None:  # pragma: no cover - entrypoint I/O
    for argument in command.argv:
        if str(argument).startswith("--basetemp="):
            Path(str(argument).split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        if str(argument).startswith("--data-file="):
            Path(str(argument).split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    if "--output" in command.argv:
        index = command.argv.index("--output")
        Path(command.argv[index + 1]).parent.mkdir(parents=True, exist_ok=True)


def _run_prepared(
    root: Path,
    commands: Sequence[Any],
    *,
    log_dir: Path,
) -> list[dict[str, Any]]:  # pragma: no cover - entrypoint subprocesses
    receipts: list[dict[str, Any]] = []
    for command in commands:
        _prepare_command_parent(command)
        current = run_commands(root, [command], log_dir=log_dir / command.name)
        for row in current:
            row["worktree"] = str(root.resolve())
        receipts.extend(current)
    return receipts


def cold_replay(path: Path) -> list[str]:
    """Rebuild deterministic fixtures and compare them with stored raw rows."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_artifact(artifact)
    fresh = measure_fixture_support()
    if fresh["rows"] != artifact.get("fixture_comparisons"):
        errors.append("cold_fixture_rows_mismatch")
    if fresh["independent_reduction"] != artifact.get("independent_reduction"):
        errors.append("cold_reduction_mismatch")
    return list(dict.fromkeys(errors))


def independent_replay(path: Path) -> list[str]:
    """Recompute the readiness reducer from only terminal comparative rows."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = artifact.get("fixture_comparisons")
    if not isinstance(rows, list):
        return ["fixture_comparisons"]
    reduced = independent_reduce(rows)
    return [] if reduced == artifact.get("independent_reduction") else ["independent_reduction"]


def _source_hashes(root: Path, prerequisite_hashes: Mapping[str, str]) -> dict[str, str]:
    hashes = dict(prerequisite_hashes)
    for label in (
        AFFECTED_MANIFEST["changed_modules"][0],
        AFFECTED_MANIFEST["tests"][0],
        *AFFECTED_MANIFEST["static_paths"],
    ):
        path = root / label
        if path.is_file():
            hashes[label] = sha256_file(path)
    return hashes


def run_experiment(  # pragma: no cover - exercised by the declared capability entrypoint
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_REL,
) -> dict[str, Any]:
    """Measure fixtures, run bounded checks, and publish the terminal artifact."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = Path(repo_root).resolve()
    started = time.monotonic()
    progress(started, "preconditions", "start", root=root)
    preconditions, prerequisite_hashes = collect_preconditions(root)
    failed_preconditions = [row for row in preconditions if row.get("passed") is not True]
    progress(
        started,
        "preconditions",
        "end",
        passed=not failed_preconditions,
        checks=len(preconditions),
    )
    if failed_preconditions:
        failure = failed_preconditions[0]
        blocked = blocked_artifact(
            run_date=run_date,
            duration_s=time.monotonic() - started,
            check=str(failure["check"]),
            upstream=str(failure["upstream"]),
            path=str(failure["path"]),
            field=str(failure["field"]),
            expected=failure["expected"],
            observed=failure["observed"],
        )
        blocked["preconditions_checked"] = preconditions
        progress(started, "publish", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        progress(started, "publish", "after_atomic_blocked", path=output_path)
        return blocked

    raw_dir = root / RAW_REL
    raw_dir.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="exp7580-", dir="/tmp"))

    progress(started, "measurement", "start")
    fixtures = measure_fixture_support()
    parity = run_disabled_e3_parity()
    e3_rejection = measure_e3_call_site_rejection()
    protocol = build_live_panel_protocol(root)
    protocol_errors = validate_live_panel_protocol(protocol)
    atomic_json(root / LIVE_PROTOCOL_REL, protocol)
    progress(
        started,
        "measurement",
        "end",
        fixtures=fixtures["verifier_support_ready_score"],
        parity=parity["passed"],
        e3_rejection=e3_rejection["passed"],
        protocol_errors=len(protocol_errors),
    )

    progress(started, "scoped_validation", "before_subprocesses")
    basetemp = private / "validation-basetemp"
    coverage_file = private / "coverage" / ".coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    scoped_commands = build_scoped_commands(
        root,
        AFFECTED_MANIFEST["tests"],
        AFFECTED_MANIFEST["changed_modules"],
        static_paths=AFFECTED_MANIFEST["static_paths"],
        basetemp=basetemp,
        coverage_file=coverage_file,
    )
    validation_receipts = _run_prepared(
        root, scoped_commands, log_dir=raw_dir / "validation/scoped"
    )
    validation_reduction = reduce_required_checks(validation_receipts)
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        passed=validation_reduction["required_checks_passed"],
    )

    progress(started, "arc_e2e", "before_subprocesses")
    e2e_receipts = _run_prepared(
        root, _e2e_commands(root, private / "e2e"), log_dir=raw_dir / "validation/e2e"
    )
    progress(
        started,
        "arc_e2e",
        "after_subprocesses",
        passed=all(row.get("passed") is True for row in e2e_receipts),
    )

    hashes = _source_hashes(root, prerequisite_hashes)
    hashes[LIVE_PROTOCOL_REL.as_posix()] = sha256_file(root / LIVE_PROTOCOL_REL)
    candidate = build_artifact(
        repo_root=root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        fixtures=fixtures,
        protocol=protocol,
        source_hashes=hashes,
        validation_receipts=validation_receipts,
        e2e_receipts=e2e_receipts,
        terminal_receipts=[],
        parity=parity,
        e3_rejection=e3_rejection,
        flagged_adversarial=False,
    )
    candidate["preconditions_checked"] = preconditions
    candidate["affected_file_validation_manifest"] = deepcopy(AFFECTED_MANIFEST)
    candidate["validation_reduction"] = validation_reduction
    candidate["repository_health"] = {
        "status": "not_assessed_unscoped_forbidden",
        "pre_existing_repository_debt": [],
        "affects_required_checks": False,
    }
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    progress(started, "terminal_validation", "before_subprocesses")
    terminal_receipts = _run_prepared(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    terminal_passed = all(row.get("passed") is True for row in terminal_receipts)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_receipts)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    final = build_artifact(
        repo_root=root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        fixtures=fixtures,
        protocol=protocol,
        source_hashes=hashes,
        validation_receipts=validation_receipts,
        e2e_receipts=e2e_receipts,
        terminal_receipts=terminal_receipts,
        parity=parity,
        e3_rejection=e3_rejection,
        flagged_adversarial=critical or not terminal_passed,
    )
    final["preconditions_checked"] = preconditions
    final["affected_file_validation_manifest"] = deepcopy(AFFECTED_MANIFEST)
    final["validation_reduction"] = validation_reduction
    final["repository_health"] = candidate["repository_health"]
    errors = validate_artifact(final)
    if protocol_errors:
        errors.extend(f"protocol:{error}" for error in protocol_errors)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{list(dict.fromkeys(errors))}")
    progress(started, "publish", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    progress(
        started,
        "publish",
        "after_atomic_terminal",
        verdict=final["honest_verdict"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_REL)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate", type=Path)
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.validate:
        errors = validate_artifact(json.loads(args.validate.read_text(encoding="utf-8")))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.cold_replay:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce:
        errors = independent_replay(args.independent_reduce)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, output_path=args.output)
    return 0
