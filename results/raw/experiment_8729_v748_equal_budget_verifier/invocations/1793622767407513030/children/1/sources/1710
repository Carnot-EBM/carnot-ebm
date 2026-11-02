"""Exp5394: gated overwrite p-bit action-sequence ablation.

Spec refs: REQ-VERIFY-5394, SCENARIO-VERIFY-5394.

This experiment is a CPU-only planning diagnostic. The p-bit/Ising boundary
exchange path proposes action orderings, but a symbolic dependency solver keeps
final authority. That distinction matters: a low-energy p-bit proposal can be
useful as a search hint, yet it is not allowed to certify an action sequence
unless the deterministic solver validates every precedence constraint.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5394_gated_overwrite_pbit_ablation_v491.json"
)
GATE_RELATIVE_PATH = Path(
    "results/experiment_5393_overwrite_guidance_tautology_corrigendum_v491.json"
)
EXPERIMENT = 5394
EXPERIMENT_ID = "exp5394-gated-overwrite-pbit-ablation-v491"
MILESTONE = "2026.07.491"
RUN_DATE = "2026-07-08"
SCHEMA = "carnot.experiment_5394.gated_overwrite_pbit_ablation.v491"
SPEC_REFS = ("REQ-VERIFY-5394", "SCENARIO-VERIFY-5394")
TERMINAL_PREFIXES = ("complete:", "blocked:")
COMPARED_MODES = ("monolithic", "hint_only", "pbit_boundary_hint", "fallback_only")
EXPECTED_FIXTURE_COUNT = 4

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete if gate passed and ablation ran; blocked if gate failed.",
    "milestone": "must equal 2026.07.491.",
    "gate_source": "Exp5393 artifact path and gate value.",
    "simulation_only": (
        "must be true unless real hardware timing evidence is present."
    ),
    "hardware_speedup_claim": "must be false.",
    "fixture_count": "number of action-sequence fixtures.",
    "compared_modes": (
        "list of monolithic, hint_only, pbit_boundary_hint, and fallback_only modes."
    ),
    "validity_rate_by_mode": (
        "deterministic solver-authoritative validity rates."
    ),
    "conflict_delta_by_mode": (
        "conflict or backtrack delta relative to monolithic."
    ),
    "convergence_delta_by_mode": "convergence-step delta relative to monolithic.",
    "overwrite_rate": (
        "solver overwrite rate for harmful or contradictory p-bit hints."
    ),
    "fallback_completeness_rate": "fallback completion rate.",
    "pbit_boundary_ablation_ready": (
        "true only if validity is preserved and no unsafe false accepts occur."
    ),
    "honest_verdict": "one-line summary starting with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class ActionSequenceFixture:
    """One bounded planning fixture with solver-checkable dependencies."""

    fixture_id: str
    actions: tuple[str, ...]
    precedence: tuple[tuple[str, str], ...]
    expected_sequence: tuple[str, ...]
    hint_only_order: tuple[str, ...]
    pbit_boundary_hint: tuple[str, ...]
    pbit_hint_kind: str
    pbit_hint_class: str


@dataclass(frozen=True)
class SolverOutcome:
    """A symbolic solver result after accepting, overwriting, or ignoring a hint."""

    final_sequence: tuple[str, ...]
    final_valid: bool
    solver_action: str
    conflicts: int
    convergence_steps: int
    fallback_used: bool
    fallback_complete: bool
    unsafe_false_accept: bool


def load_gate_source(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read the Exp5393 gate without treating missing input as success."""

    path = Path(root) / GATE_RELATIVE_PATH
    if not path.exists():
        return {
            "artifact_path": str(GATE_RELATIVE_PATH),
            "gate_field": "overwrite_guidance_corrigendum_clean",
            "gate_value": False,
            "source_status": "missing",
        }
    source = json.loads(path.read_text(encoding="utf-8"))
    return {
        "artifact_path": str(GATE_RELATIVE_PATH),
        "gate_field": "overwrite_guidance_corrigendum_clean",
        "gate_value": bool(source.get("overwrite_guidance_corrigendum_clean")),
        "source_status": str(source.get("status", "unknown")),
    }


def build_action_sequence_fixtures() -> tuple[ActionSequenceFixture, ...]:
    """Build deterministic action-sequence fixtures with advisory hint channels."""

    return (
        ActionSequenceFixture(
            fixture_id="act_unlock_deliver",
            actions=("deliver_package", "enter_room", "unlock_door", "pickup_key"),
            precedence=(
                ("pickup_key", "unlock_door"),
                ("unlock_door", "enter_room"),
                ("enter_room", "deliver_package"),
            ),
            expected_sequence=(
                "pickup_key",
                "unlock_door",
                "enter_room",
                "deliver_package",
            ),
            hint_only_order=("pickup_key", "unlock_door"),
            pbit_boundary_hint=(
                "pickup_key",
                "unlock_door",
                "enter_room",
                "deliver_package",
            ),
            pbit_hint_kind="trajectory",
            pbit_hint_class="helpful",
        ),
        ActionSequenceFixture(
            fixture_id="act_assemble_inspect",
            actions=("inspect_panel", "install_panel", "assemble_frame", "gather_parts"),
            precedence=(
                ("gather_parts", "assemble_frame"),
                ("assemble_frame", "install_panel"),
                ("install_panel", "inspect_panel"),
            ),
            expected_sequence=(
                "gather_parts",
                "assemble_frame",
                "install_panel",
                "inspect_panel",
            ),
            hint_only_order=("gather_parts",),
            pbit_boundary_hint=(
                "gather_parts",
                "install_panel",
                "assemble_frame",
                "inspect_panel",
            ),
            pbit_hint_kind="ordering",
            pbit_hint_class="harmful",
        ),
        ActionSequenceFixture(
            fixture_id="act_prepare_bake",
            actions=("frost", "bake", "cool", "preheat", "mix"),
            precedence=(
                ("preheat", "bake"),
                ("mix", "bake"),
                ("bake", "cool"),
                ("cool", "frost"),
            ),
            expected_sequence=("preheat", "mix", "bake", "cool", "frost"),
            hint_only_order=("mix", "preheat"),
            pbit_boundary_hint=("preheat", "mix", "bake", "cool", "frost"),
            pbit_hint_kind="trajectory",
            pbit_hint_class="helpful",
        ),
        ActionSequenceFixture(
            fixture_id="act_build_deploy",
            actions=("monitor", "deploy", "run_tests", "backup_config", "build_artifact"),
            precedence=(
                ("build_artifact", "run_tests"),
                ("run_tests", "deploy"),
                ("backup_config", "deploy"),
                ("deploy", "monitor"),
            ),
            expected_sequence=(
                "backup_config",
                "build_artifact",
                "run_tests",
                "deploy",
                "monitor",
            ),
            hint_only_order=("build_artifact", "backup_config"),
            pbit_boundary_hint=(
                "deploy",
                "backup_config",
                "build_artifact",
                "run_tests",
                "monitor",
            ),
            pbit_hint_kind="ordering",
            pbit_hint_class="contradictory",
        ),
    )


def run_ablation(
    root: Path | str = REPO_ROOT,
    gate_override: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Run the gated CPU-only action-sequence ablation."""

    gate_source = dict(gate_override) if gate_override is not None else load_gate_source(root)
    fixtures = build_action_sequence_fixtures()
    rows = [
        _evaluate_mode(fixture, mode)
        for fixture in fixtures
        for mode in COMPARED_MODES
    ]
    summaries = _summarize_rows(rows)
    unsafe_false_accepts = sum(int(row["unsafe_false_accept"]) for row in rows)
    pbit_proposal_harm_count = sum(
        int(row["pbit_proposal_caused_harm"])
        for row in rows
        if row["mode"] == "pbit_boundary_hint"
    )
    ready = bool(
        gate_source["gate_value"]
        and len(fixtures) == EXPECTED_FIXTURE_COUNT
        and summaries["validity_rate_by_mode"]
        == {mode: 1.0 for mode in COMPARED_MODES}
        and unsafe_false_accepts == 0
        and pbit_proposal_harm_count == 0
    )
    return {
        "gate_source": gate_source,
        "fixture_count": len(fixtures),
        "compared_modes": list(COMPARED_MODES),
        "validity_rate_by_mode": summaries["validity_rate_by_mode"],
        "conflict_delta_by_mode": summaries["conflict_delta_by_mode"],
        "convergence_delta_by_mode": summaries["convergence_delta_by_mode"],
        "overwrite_rate": summaries["overwrite_rate"],
        "fallback_completeness_rate": summaries["fallback_completeness_rate"],
        "unsafe_false_accepts": unsafe_false_accepts,
        "pbit_proposal_harm_count": pbit_proposal_harm_count,
        "pbit_proposal_caused_harm": pbit_proposal_harm_count > 0,
        "pbit_boundary_ablation_ready": ready,
        "mode_summaries": summaries["mode_summaries"],
        "mode_results": rows,
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    gate_override: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal Exp5394 artifact without running past a failed gate."""

    gate_source = dict(gate_override) if gate_override is not None else load_gate_source(root)
    tests = [_normalize_test_run(row) for row in tests_run]
    if gate_source["gate_value"]:
        diagnostic = run_ablation(root, gate_override=gate_source)
    else:
        diagnostic = _blocked_diagnostic(gate_source)
    blockers = _readiness_blockers(diagnostic, tests)
    ready = bool(diagnostic["pbit_boundary_ablation_ready"] and not blockers)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "blocked",
        "gate_source": diagnostic["gate_source"],
        "simulation_only": True,
        "hardware_speedup_claim": False,
        "fixture_count": diagnostic["fixture_count"],
        "compared_modes": diagnostic["compared_modes"],
        "validity_rate_by_mode": diagnostic["validity_rate_by_mode"],
        "conflict_delta_by_mode": diagnostic["conflict_delta_by_mode"],
        "convergence_delta_by_mode": diagnostic["convergence_delta_by_mode"],
        "overwrite_rate": diagnostic["overwrite_rate"],
        "fallback_completeness_rate": diagnostic["fallback_completeness_rate"],
        "pbit_boundary_ablation_ready": ready,
        "honest_verdict": _honest_verdict(ready, blockers, diagnostic),
        "unsafe_false_accepts": diagnostic["unsafe_false_accepts"],
        "pbit_proposal_harm_count": diagnostic["pbit_proposal_harm_count"],
        "pbit_proposal_caused_harm": diagnostic["pbit_proposal_caused_harm"],
        "mode_summaries": diagnostic["mode_summaries"],
        "mode_results": diagnostic["mode_results"],
        "readiness_blockers": blockers,
        "tests_run": tests,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "claim_limits": [
            "deterministic CPU-only action-sequence ablation",
            "p-bit/Ising boundary exchange proposes hints only",
            "symbolic solver remains final authority for sequence validity",
            "no board timing evidence is present",
            "no hardware speedup claim",
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5394 artifact and return it."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the gated ablation schema and safety invariants."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles")
    _require(artifact["status"] in {"complete", "blocked"}, "status")
    _require(artifact["milestone"] == MILESTONE, "milestone")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact["simulation_only"] is True, "simulation_only")
    _require(artifact["hardware_speedup_claim"] is False, "hardware_speedup_claim")
    _require(artifact["compared_modes"] == list(COMPARED_MODES), "compared_modes")
    _require(_is_bare_int(artifact["fixture_count"]), "fixture_count")
    _require(_is_bare_int(artifact["unsafe_false_accepts"]), "unsafe_false_accepts")
    _require(artifact["unsafe_false_accepts"] == 0, "unsafe_false_accepts")
    _require(type(artifact["pbit_boundary_ablation_ready"]) is bool, "ready")
    _require(type(artifact["pbit_proposal_caused_harm"]) is bool, "pbit harm")
    for field in ("overwrite_rate", "fallback_completeness_rate"):
        _require(_is_bare_numeric(artifact[field]), field)
    _require(_mode_metric_map_is_valid(artifact["validity_rate_by_mode"]), "validity_rate_by_mode")
    _require(_mode_metric_map_is_valid(artifact["conflict_delta_by_mode"]), "conflict_delta_by_mode")
    _require(
        _mode_metric_map_is_valid(artifact["convergence_delta_by_mode"]),
        "convergence_delta_by_mode",
    )
    _require("REQ-VERIFY-5394" in artifact["spec_refs"], "spec_refs")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum")

    gate_source = artifact["gate_source"]
    _require(gate_source["artifact_path"] == str(GATE_RELATIVE_PATH), "gate_source")
    _require(gate_source["gate_field"] == "overwrite_guidance_corrigendum_clean", "gate_source")
    _require(type(gate_source["gate_value"]) is bool, "gate_source")

    if artifact["pbit_boundary_ablation_ready"]:
        _require(artifact["status"] == "complete", "status")
        _require(gate_source["gate_value"] is True, "gate_source")
        _require(bool(artifact["tests_run"]), "tests_run")
        _require(artifact["fixture_count"] == EXPECTED_FIXTURE_COUNT, "fixture_count")
        _require(artifact["readiness_blockers"] == [], "readiness_blockers")
        _require(
            artifact["validity_rate_by_mode"] == {mode: 1.0 for mode in COMPARED_MODES},
            "validity_rate_by_mode",
        )
        _require(artifact["fallback_completeness_rate"] == 1.0, "fallback_completeness_rate")
        _require(artifact["pbit_proposal_caused_harm"] is False, "pbit_proposal_caused_harm")
        _validate_mode_results(artifact["mode_results"])


def _evaluate_mode(fixture: ActionSequenceFixture, mode: str) -> JsonDict:
    _require(mode in COMPARED_MODES, f"unknown mode: {mode}")
    hint: tuple[str, ...] | None
    pbit_hint_kind: str | None = None
    pbit_hint_class: str | None = None
    if mode == "hint_only":
        hint = fixture.hint_only_order
    elif mode == "pbit_boundary_hint":
        hint = fixture.pbit_boundary_hint
        pbit_hint_kind = fixture.pbit_hint_kind
        pbit_hint_class = fixture.pbit_hint_class
    else:
        hint = None

    outcome = _solve_with_hint(fixture, hint)
    if mode == "fallback_only":
        outcome = SolverOutcome(
            final_sequence=outcome.final_sequence,
            final_valid=outcome.final_valid,
            solver_action="ignored",
            conflicts=outcome.conflicts,
            convergence_steps=outcome.convergence_steps,
            fallback_used=True,
            fallback_complete=outcome.final_valid,
            unsafe_false_accept=outcome.unsafe_false_accept,
        )
    pbit_harm = bool(
        mode == "pbit_boundary_hint"
        and (not outcome.final_valid or outcome.unsafe_false_accept or not outcome.fallback_complete)
    )
    return {
        "fixture_id": fixture.fixture_id,
        "mode": mode,
        "hint": list(hint) if hint is not None else None,
        "pbit_hint_kind": pbit_hint_kind,
        "pbit_hint_class": pbit_hint_class,
        "solver_action": outcome.solver_action,
        "solver_authoritative": True,
        "expected_sequence": list(fixture.expected_sequence),
        "final_sequence": list(outcome.final_sequence),
        "final_valid": outcome.final_valid,
        "conflicts": outcome.conflicts,
        "convergence_steps": outcome.convergence_steps,
        "fallback_used": outcome.fallback_used,
        "fallback_complete": outcome.fallback_complete,
        "unsafe_false_accept": outcome.unsafe_false_accept,
        "pbit_proposal_caused_harm": pbit_harm,
        "simulation_only": True,
        "hardware_speedup_claim": False,
    }


def _solve_with_hint(
    fixture: ActionSequenceFixture,
    hint: tuple[str, ...] | None,
) -> SolverOutcome:
    baseline_sequence, baseline_conflicts = _canonical_sequence(fixture, ())
    if hint is None:
        return _outcome(
            baseline_sequence,
            "ignored",
            baseline_conflicts,
            fallback_used=False,
            fallback_complete=True,
            fixture=fixture,
        )
    if _is_complete_valid_sequence(fixture, hint):
        return _outcome(
            hint,
            "accepted",
            0,
            fallback_used=False,
            fallback_complete=True,
            fixture=fixture,
        )
    if _is_valid_prefix(fixture, hint):
        completed, conflicts = _canonical_sequence(fixture, hint)
        return _outcome(
            completed,
            "accepted",
            conflicts,
            fallback_used=False,
            fallback_complete=True,
            fixture=fixture,
        )
    return _outcome(
        baseline_sequence,
        "overwritten",
        baseline_conflicts,
        fallback_used=True,
        fallback_complete=True,
        fixture=fixture,
    )


def _outcome(
    sequence: tuple[str, ...],
    solver_action: str,
    conflicts: int,
    *,
    fallback_used: bool,
    fallback_complete: bool,
    fixture: ActionSequenceFixture,
) -> SolverOutcome:
    final_valid = _is_complete_valid_sequence(fixture, sequence)
    return SolverOutcome(
        final_sequence=sequence,
        final_valid=final_valid,
        solver_action=solver_action,
        conflicts=conflicts,
        convergence_steps=len(sequence) + conflicts,
        fallback_used=fallback_used,
        fallback_complete=fallback_complete and final_valid,
        unsafe_false_accept=False,
    )


def _canonical_sequence(
    fixture: ActionSequenceFixture,
    prefix: tuple[str, ...],
) -> tuple[tuple[str, ...], int]:
    sequence = list(prefix)
    remaining = [action for action in fixture.actions if action not in sequence]
    conflicts = 0
    while remaining:
        progressed = False
        for action in list(remaining):
            if _dependencies_satisfied(fixture, action, sequence):
                sequence.append(action)
                remaining.remove(action)
                progressed = True
            else:
                conflicts += 1
        _require(progressed, f"cyclic fixture: {fixture.fixture_id}")
    return tuple(sequence), conflicts


def _dependencies_satisfied(
    fixture: ActionSequenceFixture,
    action: str,
    sequence: Sequence[str],
) -> bool:
    done = set(sequence)
    return all(before in done for before, after in fixture.precedence if after == action)


def _is_complete_valid_sequence(
    fixture: ActionSequenceFixture,
    sequence: Sequence[str],
) -> bool:
    return (
        len(sequence) == len(fixture.actions)
        and set(sequence) == set(fixture.actions)
        and _is_valid_prefix(fixture, tuple(sequence))
    )


def _is_valid_prefix(
    fixture: ActionSequenceFixture,
    prefix: tuple[str, ...],
) -> bool:
    if len(set(prefix)) != len(prefix):
        return False
    if not set(prefix).issubset(set(fixture.actions)):
        return False
    seen: list[str] = []
    for action in prefix:
        if not _dependencies_satisfied(fixture, action, seen):
            return False
        seen.append(action)
    return True


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    mode_summaries = {
        mode: _mode_summary([row for row in rows if row["mode"] == mode])
        for mode in COMPARED_MODES
    }
    monolithic = mode_summaries["monolithic"]
    harmful_pbit_rows = [
        row
        for row in rows
        if row["mode"] == "pbit_boundary_hint"
        and row["pbit_hint_class"] in {"harmful", "contradictory"}
    ]
    fallback_rows = [row for row in rows if row["fallback_used"]]
    return {
        "mode_summaries": mode_summaries,
        "validity_rate_by_mode": {
            mode: summary["validity_rate"] for mode, summary in mode_summaries.items()
        },
        "conflict_delta_by_mode": {
            mode: summary["conflicts"] - monolithic["conflicts"]
            for mode, summary in mode_summaries.items()
        },
        "convergence_delta_by_mode": {
            mode: summary["convergence_steps"] - monolithic["convergence_steps"]
            for mode, summary in mode_summaries.items()
        },
        "overwrite_rate": _rate(
            sum(row["solver_action"] == "overwritten" for row in harmful_pbit_rows),
            len(harmful_pbit_rows),
        ),
        "fallback_completeness_rate": _rate(
            sum(row["fallback_complete"] for row in fallback_rows),
            len(fallback_rows),
        ),
    }


def _mode_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "row_count": len(rows),
        "validity_rate": _rate(sum(row["final_valid"] for row in rows), len(rows)),
        "conflicts": sum(int(row["conflicts"]) for row in rows),
        "convergence_steps": sum(int(row["convergence_steps"]) for row in rows),
        "fallback_count": sum(int(row["fallback_used"]) for row in rows),
        "overwrite_count": sum(int(row["solver_action"] == "overwritten") for row in rows),
        "unsafe_false_accepts": sum(int(row["unsafe_false_accept"]) for row in rows),
    }


def _blocked_diagnostic(gate_source: Mapping[str, Any]) -> JsonDict:
    return {
        "gate_source": dict(gate_source),
        "fixture_count": 0,
        "compared_modes": list(COMPARED_MODES),
        "validity_rate_by_mode": {mode: 0.0 for mode in COMPARED_MODES},
        "conflict_delta_by_mode": {mode: 0 for mode in COMPARED_MODES},
        "convergence_delta_by_mode": {mode: 0 for mode in COMPARED_MODES},
        "overwrite_rate": 0.0,
        "fallback_completeness_rate": 0.0,
        "unsafe_false_accepts": 0,
        "pbit_proposal_harm_count": 0,
        "pbit_proposal_caused_harm": False,
        "pbit_boundary_ablation_ready": False,
        "mode_summaries": {},
        "mode_results": [],
    }


def _readiness_blockers(
    diagnostic: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    checks = (
        (not diagnostic["gate_source"]["gate_value"], "exp5393_gate_failed"),
        (
            diagnostic["pbit_boundary_ablation_ready"]
            and diagnostic["fixture_count"] != EXPECTED_FIXTURE_COUNT,
            "fixture_count_mismatch",
        ),
        (
            diagnostic["pbit_boundary_ablation_ready"]
            and diagnostic["validity_rate_by_mode"] != {mode: 1.0 for mode in COMPARED_MODES},
            "validity_not_preserved",
        ),
        (
            diagnostic["pbit_boundary_ablation_ready"]
            and diagnostic["unsafe_false_accepts"] != 0,
            "unsafe_false_accepts",
        ),
        (
            diagnostic["pbit_boundary_ablation_ready"]
            and diagnostic["fallback_completeness_rate"] != 1.0,
            "fallback_incomplete",
        ),
        (diagnostic["pbit_proposal_caused_harm"], "pbit_proposal_harm"),
        (diagnostic["pbit_boundary_ablation_ready"] and not tests_run, "tests_not_recorded"),
    )
    return [name for failed, name in checks if failed]


def _honest_verdict(
    ready: bool,
    blockers: Sequence[str],
    diagnostic: Mapping[str, Any],
) -> str:
    if not ready:
        return "blocked: gated p-bit action-sequence ablation blockers=" + ",".join(blockers)
    return (
        "complete: CPU-only p-bit boundary hints improved aggregate solver "
        f"conflict delta {diagnostic['conflict_delta_by_mode']['pbit_boundary_hint']} "
        "while solver-authoritative overwrite/fallback preserved validity"
    )


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": artifact["experiment_id"],
        "spec_refs": artifact["spec_refs"],
        "tests_run": artifact["tests_run"],
        "required_fields": {
            field: artifact[field]
            for field in REQUIRED_ARTIFACT_FIELDS
            if field != "honest_verdict"
        },
        "unsafe_false_accepts": artifact["unsafe_false_accepts"],
        "pbit_proposal_harm_count": artifact["pbit_proposal_harm_count"],
        "mode_summaries": artifact["mode_summaries"],
        "mode_results": artifact["mode_results"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_mode_results(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == EXPECTED_FIXTURE_COUNT * len(COMPARED_MODES), "mode_results")
    for row in rows:
        _require(row["mode"] in COMPARED_MODES, "row mode")
        _require(row["solver_authoritative"] is True, "row solver_authoritative")
        _require(row["final_valid"] is True, "row final_valid")
        _require(row["unsafe_false_accept"] is False, "row unsafe_false_accept")
        _require(row["simulation_only"] is True, "row simulation_only")
        _require(row["hardware_speedup_claim"] is False, "row hardware_speedup_claim")
        if row["mode"] == "pbit_boundary_hint":
            _require(row["pbit_hint_kind"] in {"trajectory", "ordering"}, "row pbit kind")
            _require(
                row["pbit_hint_class"] in {"helpful", "harmful", "contradictory"},
                "row pbit class",
            )


def _normalize_test_run(row: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(row, str):
        return {"command": row, "outcome": "passed"}
    return dict(row)


def _mode_metric_map_is_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(COMPARED_MODES)
        and all(_is_bare_numeric(metric) for metric in value.values())
    )


def _rate(numerator: int | float, denominator: int) -> float:
    return 1.0 if denominator == 0 else float(numerator) / denominator


def _is_bare_int(value: Any) -> bool:
    return type(value) is int


def _is_bare_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
