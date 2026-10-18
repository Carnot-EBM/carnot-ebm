"""Audit the first fully frozen live ARC engine after Experiment 6968.

Spec refs: REQ-ARC-WMTE-6981 and SCENARIO-ARC-WMTE-6981-*.

The audit reads only archived run evidence. Missing prompt or transition bytes
produce a blocked result because a later reconstruction could hide leakage.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_6968_arc_post_refit_induction_audit as prior
from carnot.agentic.arc_producer_evidence import (
    EVIDENCE_MANIFEST_SCHEMA,
    read_evidence_manifest,
)


JsonDict = dict[str, Any]

EXPERIMENT_ID = 6981
RANDOM_SEED = 6_981_202_609_04
BOOTSTRAP_RESAMPLES = 10_000
INFERENCE_SUBSTRATE = "fresh_process_replay_of_frozen_live_agent_engine_and_live_path_fixture"
SOURCE_REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SOURCE_REPO_ROOT
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
PRIOR_ARTIFACT_PATH = Path("results/experiment_6968_arc_post_refit_induction_audit.json")
MANIFEST_PATH = Path("results/arc_e3/r11l/attempts/manifest.jsonl")
SCORER_PATH = Path("scripts/arc_e3_induced_model_quality.py")
LIVE_POLICY_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
WORLD_MODEL_PATH = Path("python/carnot/agentic/arc_executable_world_model.py")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
OUTPUT_PATH = Path("results/experiment_6981_arc_live_engine_generalization_audit.json")
SOURCE_NAMES = ("run", "engine", "prompt", "transitions", "environment", "scorer", "live_policy")
CONTROL_NAMES = (
    "identity",
    "constant_delta",
    "nearest_shown_delta",
    "row_table_memorization",
)

canonical_json = prior.canonical_json
sha256_bytes = prior.sha256_bytes
sha256_path = prior.sha256_path
execute_engine_fresh = prior.execute_engine_fresh
score_controls = prior.score_controls

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "selected_run_provenance",
    "engine_hash",
    "prompt_hash",
    "transition_hash",
    "environment_hash",
    "scorer_hash",
    "live_policy_hash",
    "rows",
    "per_transition_rows",
    "split_rows",
    "purity_rows",
    "engine_execution_rows",
    "control_rows",
    "paired_control_delta_rows",
    "memorization_signature_rows",
    "live_path_trace_rows",
    "live_influence_fixture_rows",
    "first_step_candidate_rows",
    "first_step_ranking_rows",
    "prefix_exact_accuracy",
    "heldout_exact_accuracy",
    "heldout_changing_accuracy",
    "heldout_noop_accuracy",
    "generalization_gap",
    "live_path_reachable_score",
    "arc_engine_audit_complete_score",
    "arc_generalization_positive_score",
    "solve_claimed",
    "level_claimed",
    "registry_updated",
    "submitted_to_leaderboard",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A stated reason makes each evidence field reviewable.",
    "preconditions_checked": "Preflight results stop missing evidence from becoming measured data.",
    "inference_substrate": "The substrate separates frozen replay from model generation.",
    "duration_s": "Measured wall time exposes whether the audit executed.",
    "source_artifact_hashes": "Content hashes bind every input to exact bytes.",
    "selected_run_provenance": "Chronological provenance prevents outcome-based run selection.",
    "engine_hash": "The engine digest identifies the exact executed program.",
    "prompt_hash": "The prompt digest fixes what the generator could observe.",
    "transition_hash": "The transition digest fixes row values and order.",
    "environment_hash": "The environment digest fixes the recorded runtime context.",
    "scorer_hash": "The scorer digest fixes metric definitions.",
    "live_policy_hash": "The policy digest fixes the production route under test.",
    "rows": "All manifest rows expose selection and rejection decisions.",
    "per_transition_rows": "Per-row evidence permits independent metric recomputation.",
    "split_rows": "Explicit memberships make the holdout boundary reproducible.",
    "purity_rows": "Purity checks expose prompt and feedback leakage.",
    "engine_execution_rows": "Restricted child receipts prove actual frozen-code execution.",
    "control_rows": "Matched controls bound gains from no-op or copied deltas.",
    "paired_control_delta_rows": "Paired intervals retain the transition as the uncertainty unit.",
    "memorization_signature_rows": "Memorization signals separate lookup from transfer.",
    "live_path_trace_rows": "Trace rows bind the factory, policy, loader, router, planner, and action.",
    "live_influence_fixture_rows": "An action difference proves behavioral reachability.",
    "first_step_candidate_rows": "Candidate rows expose every prediction used for ranking.",
    "first_step_ranking_rows": "Frozen orders compare engine foresight with the shipped baseline.",
    "prefix_exact_accuracy": "Prefix fit is a transcription check, not generalization evidence.",
    "heldout_exact_accuracy": "Held-out exact accuracy measures whole-state transfer.",
    "heldout_changing_accuracy": "Changing accuracy prevents no-op prevalence from inflating credit.",
    "heldout_noop_accuracy": "No-op accuracy measures hallucinated state changes.",
    "generalization_gap": "The prefix-minus-tail gap exposes fit collapse.",
    "live_path_reachable_score": "Credit requires the exact engine to change a production-path action.",
    "arc_engine_audit_complete_score": "Completion requires every source and output row to terminate.",
    "arc_generalization_positive_score": "Positive credit requires pure superiority and live influence.",
    "solve_claimed": "Transition prediction does not prove a solve.",
    "level_claimed": "First-step foresight does not prove level completion.",
    "registry_updated": "A read-only audit must leave the solve registry unchanged.",
    "submitted_to_leaderboard": "Local replay does not imply external submission.",
    "random_seed": "A fixed seed makes paired resampling reproducible.",
    "reproducibility_checksum": "A timing-free checksum detects scientific-content drift.",
    "gate_check_summary": "Blocked checks state expected and observed values for repair.",
    "verifier_is_oracle": "False distinguishes prediction scoring from environment control.",
    "verdict_class": "A closed class prevents a blocked result from reading as positive.",
    "honest_verdict": "A terminal prefix makes the result machine-classifiable.",
}


def _read_json_object(path: Path | None) -> JsonDict | None:
    """Read an immutable JSON object and treat malformed bytes as unavailable."""

    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _gate(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record both sides of one precondition so a failure is actionable."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": expected == observed if passed is None else bool(passed),
    }


def _failed_gates(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project failed checks into the required blocked-result schema."""

    return [
        {
            "failed_check": row.get("check"),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in rows
        if row.get("passed") is not True
    ]


def _safe_source(repo_root: Path, raw_path: Any) -> Path | None:
    """Resolve a recorded source only when it stays below the selected checkout."""

    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        return None
    candidate = (repo_root / raw_path).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return candidate


def _prior_cutoff(repo_root: Path) -> tuple[str | None, JsonDict | None]:
    """Read the immutable engine timestamp chosen by the prior audit."""

    document = _read_json_object(repo_root / PRIOR_ARTIFACT_PATH)
    provenance = document.get("selected_run_provenance", {}) if document else {}
    selected = provenance.get("selected", {}) if isinstance(provenance, Mapping) else {}
    cutoff = selected.get("engine_emitted_at") if isinstance(selected, Mapping) else None
    return (str(cutoff) if cutoff else None), document


def _source_record(row: Mapping[str, Any], name: str) -> JsonDict:
    """Return one manifest source record, including legacy engine-file rows."""

    if row.get("schema") == EVIDENCE_MANIFEST_SCHEMA and name == "run":
        path = row.get("manifest_row_path")
        return {
            "path": str(Path("results/arc_e3") / str(path)) if isinstance(path, str) else None,
            "sha256": row.get("manifest_row_sha256"),
        }
    value = row.get(name)
    if isinstance(value, Mapping):
        path = value.get("path")
        if row.get("schema") == EVIDENCE_MANIFEST_SCHEMA and isinstance(path, str):
            path = str(Path("results/arc_e3") / path)
        return {"path": path, "sha256": value.get("sha256")}
    if name == "engine" and isinstance(row.get("file"), str):
        return {
            "path": str(MANIFEST_PATH.parent / str(row["file"])),
            "sha256": None,
        }
    return {"path": None, "sha256": None}


def _verify_source(
    repo_root: Path, name: str, record: Mapping[str, Any]
) -> tuple[JsonDict, Path | None]:
    """Verify one recorded source path and full SHA-256 digest."""

    path = _safe_source(repo_root, record.get("path"))
    expected = record.get("sha256")
    observed = sha256_path(path) if path is not None else None
    valid_expected = (
        isinstance(expected, str) and expected.startswith("sha256:") and len(expected) == 71
    )
    stated_expectation = expected if valid_expected else "recorded_sha256"
    check = _gate(
        f"{name}_hash",
        stated_expectation,
        observed,
        passed=bool(valid_expected and observed == expected),
    )
    return check, path


def _run_matches_engine(document: JsonDict | None, engine_hash: str | None) -> bool:
    """Require one completed E3 r11l row to name the selected engine digest."""

    if document is None or document.get("complete") is not True or document.get("policy") != "e3":
        return False
    games = document.get("per_game", [])
    return any(
        isinstance(row, Mapping)
        and row.get("game") == "r11l"
        and row.get("engine_sha256") == engine_hash
        for row in games
        if isinstance(games, list)
    )


def discover_post_6968_rows(repo_root: Path) -> tuple[list[JsonDict], str | None]:
    """List all later manifest rows and verify eligibility without using scores."""

    cutoff, _ = _prior_cutoff(repo_root)
    manifest = repo_root / MANIFEST_PATH
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], cutoff
    prospective = read_evidence_manifest(repo_root / "results" / "arc_e3", "r11l")
    candidates: list[JsonDict] = []
    for manifest_index, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, Mapping):
            continue
        ts = str(raw.get("ts") or "")
        if not cutoff or not ts or ts <= cutoff:
            continue
        records = {name: _source_record(raw, name) for name in SOURCE_NAMES}
        checks: list[JsonDict] = []
        paths: dict[str, Path | None] = {}
        for name, record in records.items():
            check, path = _verify_source(repo_root, name, record)
            checks.append(check)
            paths[name] = path
        engine_hash = checks[SOURCE_NAMES.index("engine")]["observed_value"]
        engine_prefix = str(engine_hash or "").removeprefix("sha256:")[:16]
        checks.append(_gate("engine_name_hash_prefix", raw.get("sha256_16"), engine_prefix))
        run_document = _read_json_object(paths["run"])
        checks.append(
            _gate(
                "completed_live_run_engine_binding",
                True,
                _run_matches_engine(run_document, engine_hash),
            )
        )
        if raw.get("schema") == EVIDENCE_MANIFEST_SCHEMA:
            envelope_validation = (
                prospective[manifest_index] if manifest_index < len(prospective) else None
            )
            checks.append(
                _gate(
                    "producer_evidence_envelope_eligible",
                    True,
                    bool(envelope_validation and envelope_validation.get("eligible")),
                )
            )
        failed = _failed_gates(checks)
        candidates.append(
            {
                "manifest_index": manifest_index,
                "ts": ts,
                "sha256_16": raw.get("sha256_16"),
                "score_ignored_for_selection": raw.get("score"),
                "sources": records,
                "source_checks": checks,
                "eligible": not failed,
                "rejection_reasons": failed,
            }
        )
    return candidates, cutoff


def select_earliest_eligible(rows: Sequence[Mapping[str, Any]]) -> tuple[JsonDict | None, str]:
    """Select the first eligible row by timestamp and manifest order only."""

    eligible = [dict(row) for row in rows if row.get("eligible") is True]
    if not eligible:
        return None, "no_eligible_post_exp6968_manifest_row"
    eligible.sort(key=lambda row: (str(row.get("ts")), int(row.get("manifest_index", -1))))
    return eligible[0], "earliest_eligible_post_exp6968_manifest_timestamp"


def _mean_boolean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    """Average measured boolean rows and keep an empty channel explicit."""

    values = [float(bool(row[field])) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def _paired_control_deltas_all(
    engine_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = RANDOM_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[JsonDict]:
    """Compute paired intervals for every preregistered control, not a chosen winner."""

    engine = {
        str(row.get("transition_id")): bool(row.get("changing_transition_correct"))
        for row in engine_rows
        if row.get("changing_transition_correct") is not None
    }
    if not engine:
        return []
    output: list[JsonDict] = []
    ids = sorted(engine)
    for control_index, control in enumerate(CONTROL_NAMES):
        controls = {
            str(row.get("transition_id")): bool(row.get("changing_transition_correct"))
            for row in control_rows
            if row.get("control") == control and row.get("changing_transition_correct") is not None
        }
        if any(row_id not in controls for row_id in ids):
            continue
        deltas = np.asarray(
            [float(engine[row_id]) - float(controls[row_id]) for row_id in ids], dtype=float
        )
        rng = np.random.default_rng(seed + control_index)
        samples = rng.choice(deltas, size=(resamples, len(deltas)), replace=True).mean(axis=1)
        low, high = np.percentile(samples, [2.5, 97.5]).tolist()
        control_accuracy = float(np.mean([controls[row_id] for row_id in ids]))
        for index, row_id in enumerate(ids):
            output.append(
                {
                    "transition_id": row_id,
                    "control": control,
                    "engine_correct": engine[row_id],
                    "control_correct": controls[row_id],
                    "paired_delta": float(deltas[index]),
                    "mean_paired_delta": float(deltas.mean()),
                    "control_changing_accuracy": control_accuracy,
                    "interval_low": float(low),
                    "interval_high": float(high),
                    "interval_level": 0.95,
                    "cluster_unit": "transition",
                    "bootstrap_resamples": resamples,
                    "random_seed": seed + control_index,
                    "terminal": True,
                }
            )
    return output


def rank_first_step_groups(
    engine_path: Path, groups: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Rank current-state candidates only by predicted changed-cell count."""

    candidate_rows: list[JsonDict] = []
    ranking_rows: list[JsonDict] = []
    for group in groups:
        group_id = str(group.get("group_id"))
        grid = group.get("current_grid")
        candidates = group.get("candidates", [])
        replay_rows = [
            {
                "transition_id": f"{group_id}::{candidate.get('candidate_id')}",
                "index": index,
                "grid": grid,
                "next_grid": grid,
                "action": candidate.get("action"),
                "data": candidate.get("data"),
                "level_before": 0,
                "level_after": 0,
            }
            for index, candidate in enumerate(candidates if isinstance(candidates, list) else [])
            if isinstance(candidate, Mapping)
        ]
        scored, _ = execute_engine_fresh(engine_path, replay_rows)
        sortable: list[tuple[bool, int, str]] = []
        for candidate, score in zip(candidates, scored, strict=True):
            candidate_id = str(candidate.get("candidate_id"))
            changed = score.get("predicted_changed_cell_count")
            exception = score.get("exception")
            candidate_rows.append(
                {
                    "group_id": group_id,
                    "candidate_id": candidate_id,
                    "action": candidate.get("action"),
                    "data": candidate.get("data"),
                    "predicted_changed_cell_count": changed,
                    "exception": exception,
                    "latency_s": score.get("latency_s"),
                    "realized_first_step_value": candidate.get("realized_first_step_value"),
                    "terminal": score.get("terminal") is True,
                }
            )
            sortable.append((exception is not None, -int(changed or 0), candidate_id))
        engine_order = [candidate_id for _, _, candidate_id in sorted(sortable)]
        baseline = [str(value) for value in group.get("shipped_baseline_order", [])]
        ranking_rows.append(
            {
                "group_id": group_id,
                "engine_order": engine_order,
                "shipped_baseline_order": baseline,
                "engine_top_candidate": engine_order[0] if engine_order else None,
                "shipped_baseline_top_candidate": baseline[0] if baseline else None,
                "ranking_changed": bool(
                    engine_order and baseline and engine_order[0] != baseline[0]
                ),
                "ranking_inputs": [
                    "current_grid",
                    "action",
                    "data",
                    "frozen_engine_prediction",
                ],
                "future_frames_used_for_ranking": False,
                "terminal": bool(engine_order)
                and len(engine_order) == len(baseline)
                and all(row["terminal"] for row in candidate_rows if row["group_id"] == group_id),
            }
        )
    return candidate_rows, ranking_rows


def _source_trace(symbol: str, value: Any, called: bool) -> JsonDict:
    """Describe one production symbol and whether the fixture reached it."""

    path = inspect.getsourcefile(value)
    try:
        line = inspect.getsourcelines(value)[1]
    except (OSError, TypeError):
        line = None
    return {
        "symbol": symbol,
        "path": str(Path(path).resolve().relative_to(SOURCE_REPO_ROOT)) if path else None,
        "line": line,
        "called_by_fixture": called,
        "terminal": True,
    }


def _trace_and_run_live_influence_fixture_in_process(
    engine_path: Path, expected_engine_hash: str | None, *, restrict_engine: bool = False
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Use the production factory, loader, router, planner, and action method once."""

    from carnot.agentic import arc_competition_agent as live
    from carnot.agentic import arc_executable_world_model as world

    called = {
        name: False
        for name in (
            "make_carnot_agent",
            "E3AgentPolicy",
            "load_engine",
            "_world_model_candidates",
            "plan_in_model",
            "next_move",
        )
    }

    class FixtureBase:
        """Supply only the game identifier required by the production factory."""

        def __init__(self) -> None:
            self.game_id = "r11l"

    fixture: JsonDict = {
        "factory_constructed_e3_policy": False,
        "loaded_engine_hash": None,
        "selected_candidate_name": None,
        "engine_first_action": None,
        "no_engine_first_action": None,
        "reachable": False,
        "exception": None,
        "terminal": True,
    }
    try:
        agent_class = live.make_carnot_agent(FixtureBase, cascade=True, proposer=object())
        called["make_carnot_agent"] = True
        agent = agent_class()
        policy = agent._policy
        control_agent = agent_class()
        called["E3AgentPolicy"] = isinstance(policy, live.E3AgentPolicy)
        fixture["factory_constructed_e3_policy"] = called["E3AgentPolicy"]
        with tempfile.TemporaryDirectory(prefix="carnot-exp6981-live-") as temporary:
            root = Path(temporary)
            target = root / "r11l" / "world_model.py"
            target.parent.mkdir(parents=True)
            shutil.copyfile(engine_path, target)
            if restrict_engine:
                sys.addaudithook(prior._deny_generated_side_effects)
            old_root = world.E3_DIR
            world.E3_DIR = root
            try:
                engine, is_done = world.load_engine("r11l")
                called["load_engine"] = True
            finally:
                world.E3_DIR = old_root
        fixture["loaded_engine_hash"] = sha256_path(engine_path)
        candidates = policy._world_model_candidates(engine, is_done)
        called["_world_model_candidates"] = True
        selected = candidates[0] if candidates else None
        fixture["selected_candidate_name"] = selected.name if selected else None
        start = np.zeros((2, 2), dtype=np.int64)
        plan = (
            world.plan_in_model(selected.engine, selected.is_level_complete, start)
            if selected is not None
            else None
        )
        called["plan_in_model"] = True
        policy.phase = "execute"
        policy.plan = list(plan or [])
        policy.pi = 0
        engine_action = policy.next_move([], None)
        called["next_move"] = True
        control_policy = control_agent._policy
        control_policy.phase = "execute"
        control_policy.plan = []
        control_policy.pi = 0
        control_action = control_policy.next_move([], None)
        fixture["engine_first_action"] = list(engine_action)
        fixture["no_engine_first_action"] = list(control_action)
        fixture["reachable"] = bool(
            expected_engine_hash
            and fixture["loaded_engine_hash"] == expected_engine_hash
            and selected is not None
            and plan
            and engine_action != control_action
        )
    except Exception as error:  # noqa: BLE001 - a failed fixture is measured reachability.
        fixture["exception"] = f"{type(error).__name__}: {str(error)[:240]}"
    symbols = (
        ("make_carnot_agent", live.make_carnot_agent),
        ("E3AgentPolicy", live.E3AgentPolicy),
        ("load_engine", world.load_engine),
        ("_world_model_candidates", live.E3AgentPolicy._world_model_candidates),
        ("plan_in_model", world.plan_in_model),
        ("next_move", live.E3AgentPolicy.next_move),
    )
    trace = [_source_trace(name, value, called[name]) for name, value in symbols]
    return trace, [fixture]


def trace_and_run_live_influence_fixture(
    engine_path: Path, expected_engine_hash: str | None, *, timeout_s: float = 30.0
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Run the production live-path fixture in a bounded disposable process."""

    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6981_arc_live_engine_generalization_audit",
        "--live-fixture-worker",
        str(engine_path.resolve()),
        str(expected_engine_hash or ""),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(SOURCE_REPO_ROOT / "python"),
        }
    )
    symbols = (
        "make_carnot_agent",
        "E3AgentPolicy",
        "load_engine",
        "_world_model_candidates",
        "plan_in_model",
        "next_move",
    )
    with tempfile.TemporaryDirectory(prefix="carnot-exp6981-parent-") as temporary:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=temporary,
                env=environment,
                timeout=timeout_s,
                check=False,
            )
            payload = json.loads(result.stdout.splitlines()[-1]) if result.stdout else {}
            if result.returncode == 0 and isinstance(payload, Mapping):
                return list(payload.get("trace", [])), list(payload.get("fixture", []))
            error = f"worker_exit_{result.returncode}: {result.stderr[-240:]}"
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as caught:
            error = f"{type(caught).__name__}: {str(caught)[:240]}"
    trace = [
        {
            "symbol": symbol,
            "path": None,
            "line": None,
            "called_by_fixture": False,
            "terminal": True,
        }
        for symbol in symbols
    ]
    fixture = {
        "factory_constructed_e3_policy": False,
        "loaded_engine_hash": None,
        "selected_candidate_name": None,
        "engine_first_action": None,
        "no_engine_first_action": None,
        "reachable": False,
        "exception": error,
        "terminal": True,
    }
    return trace, [fixture]


def _empty_artifact(repo_root: Path) -> JsonDict:
    """Create the complete schema before any external check can fail."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {
            "prior_audit": {
                "path": str(PRIOR_ARTIFACT_PATH),
                "sha256": sha256_path(repo_root / PRIOR_ARTIFACT_PATH),
            },
            "attempt_manifest": {
                "path": str(MANIFEST_PATH),
                "sha256": sha256_path(repo_root / MANIFEST_PATH),
            },
            "solve_registry": {
                "path": str(REGISTRY_PATH),
                "sha256": sha256_path(repo_root / REGISTRY_PATH),
            },
        },
        "selected_run_provenance": {},
        "engine_hash": None,
        "prompt_hash": None,
        "transition_hash": None,
        "environment_hash": None,
        "scorer_hash": None,
        "live_policy_hash": None,
        "rows": [],
        "per_transition_rows": [],
        "split_rows": [],
        "purity_rows": [],
        "engine_execution_rows": [],
        "control_rows": [],
        "paired_control_delta_rows": [],
        "memorization_signature_rows": [],
        "live_path_trace_rows": [],
        "live_influence_fixture_rows": [],
        "first_step_candidate_rows": [],
        "first_step_ranking_rows": [],
        "prefix_exact_accuracy": None,
        "heldout_exact_accuracy": None,
        "heldout_changing_accuracy": None,
        "heldout_noop_accuracy": None,
        "generalization_gap": None,
        "live_path_reachable_score": 0,
        "arc_engine_audit_complete_score": 0,
        "arc_generalization_positive_score": 0,
        "solve_claimed": False,
        "level_claimed": False,
        "registry_updated": False,
        "submitted_to_leaderboard": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "gate_check_summary": [],
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_arc_live_engine_generalization_audit",
    }


def _stable_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content without wall time, process IDs, or row latency."""

    value = deepcopy(dict(artifact))
    value.pop("duration_s", None)
    value.pop("reproducibility_checksum", None)
    for rows_name in ("engine_execution_rows", "first_step_candidate_rows"):
        for row in value.get(rows_name, []):
            if isinstance(row, dict):
                row.pop("latency_s", None)
                row.pop("parent_pid", None)
                row.pop("worker_pid", None)
    return sha256_bytes(canonical_json(value))


def _finish_blocked(artifact: JsonDict, checks: list[JsonDict], started: float) -> JsonDict:
    """Finish a blocked result with no partial measurements."""

    artifact["preconditions_checked"] = checks
    artifact["gate_check_summary"] = _failed_gates(checks)
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = _stable_checksum(artifact)
    return artifact


def _selected_paths(repo_root: Path, selected: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve already-verified selected source paths for later reads."""

    paths: dict[str, Path] = {}
    sources = selected.get("sources", {})
    for name in SOURCE_NAMES:
        record = sources.get(name, {}) if isinstance(sources, Mapping) else {}
        path = _safe_source(repo_root, record.get("path") if isinstance(record, Mapping) else None)
        if path is not None:
            paths[name] = path
    return paths


def _purity_checks(
    transition: Mapping[str, Any], prompt: Mapping[str, Any]
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    """Rebuild the prefix and reject every forbidden information channel."""

    rows = transition.get("rows", [])
    prompt_ids = transition.get("prompt_row_ids", [])
    repair_ids = transition.get("repair_feedback_row_ids", [])
    split, per_row, split_failures = prior.rebuild_split(
        rows if isinstance(rows, list) else [],
        prompt_row_ids=prompt_ids if isinstance(prompt_ids, list) else [],
        repair_feedback_row_ids=repair_ids if isinstance(repair_ids, list) else [],
    )
    checks = [
        _gate(
            "prompt_membership_matches_transition_source",
            prompt_ids,
            prompt.get("included_transition_ids"),
        ),
        _gate(
            "repair_feedback_membership_matches_transition_source",
            repair_ids,
            prompt.get("repair_feedback_transition_ids"),
        ),
        _gate("prompt_used_game_source", False, prompt.get("used_game_source")),
        _gate("prompt_used_hand_derived_rules", False, prompt.get("used_hand_derived_rules")),
        _gate(
            "heldout_revealed_before_generation",
            False,
            prompt.get("heldout_revealed_before_generation"),
        ),
        _gate("split_reconstruction_failures", [], split_failures),
    ]
    heldout = set(split["heldout_row_ids"])
    prompt_visible = set(str(value) for value in prompt.get("included_transition_ids", []))
    feedback_visible = set(str(value) for value in prompt.get("repair_feedback_transition_ids", []))
    checks.extend(
        [
            _gate("heldout_rows_in_prompt", [], sorted(heldout & prompt_visible)),
            _gate("heldout_rows_in_repair_feedback", [], sorted(heldout & feedback_visible)),
        ]
    )
    purity = [
        {
            **row,
            "game_source_visible": bool(prompt.get("used_game_source")),
            "hand_derived_rules_visible": bool(prompt.get("used_hand_derived_rules")),
            "terminal": True,
        }
        for row in per_row
    ]
    return split, purity, checks


def build_artifact(*, date: str, repo_root: Path = REPO_ROOT) -> JsonDict:
    """Build a positive, null, or blocked result from immutable local evidence."""

    del date
    started = time.monotonic()
    artifact = _empty_artifact(repo_root)
    registry_hash_before = sha256_path(repo_root / REGISTRY_PATH)
    rows, cutoff = discover_post_6968_rows(repo_root)
    selected, reason = select_earliest_eligible(rows)
    artifact["rows"] = rows
    artifact["selected_run_provenance"] = {
        "selection_reason": reason,
        "post_exp6968_cutoff": cutoff,
        "candidate_count": len(rows),
        "eligible_count": sum(row.get("eligible") is True for row in rows),
        "selected": selected,
    }
    checks = [
        _gate("prior_exp6968_artifact", True, (repo_root / PRIOR_ARTIFACT_PATH).is_file()),
        _gate("post_exp6968_cutoff_recorded", True, cutoff is not None),
        _gate("attempt_manifest", True, (repo_root / MANIFEST_PATH).is_file()),
        _gate("post_exp6968_manifest_rows", ">=1", len(rows), passed=len(rows) >= 1),
        _gate(
            "eligible_post_exp6968_live_engine",
            ">=1",
            sum(row.get("eligible") is True for row in rows),
            passed=selected is not None,
        ),
    ]
    if selected is None:
        if rows:
            checks.extend(dict(row) for row in rows[0].get("source_checks", []))
        return _finish_blocked(artifact, checks, started)
    checks.extend(dict(row) for row in selected.get("source_checks", []))
    paths = _selected_paths(repo_root, selected)
    prompt = _read_json_object(paths.get("prompt"))
    transitions = _read_json_object(paths.get("transitions"))
    checks.extend(
        [
            _gate("immutable_prompt_source", True, prompt is not None),
            _gate("immutable_transition_source", True, transitions is not None),
        ]
    )
    if prompt is None or transitions is None:
        return _finish_blocked(artifact, checks, started)
    source_hashes = {
        name: {
            "path": str(selected["sources"][name]["path"]),
            "sha256": sha256_path(paths[name]),
        }
        for name in SOURCE_NAMES
    }
    artifact["source_artifact_hashes"].update(source_hashes)
    artifact["engine_hash"] = source_hashes["engine"]["sha256"]
    artifact["prompt_hash"] = source_hashes["prompt"]["sha256"]
    artifact["transition_hash"] = source_hashes["transitions"]["sha256"]
    artifact["environment_hash"] = source_hashes["environment"]["sha256"]
    artifact["scorer_hash"] = source_hashes["scorer"]["sha256"]
    artifact["live_policy_hash"] = source_hashes["live_policy"]["sha256"]
    checks.extend(
        [
            _gate(
                "transition_engine_hash", artifact["engine_hash"], transitions.get("engine_sha256")
            ),
            _gate(
                "transition_prompt_hash", artifact["prompt_hash"], transitions.get("prompt_sha256")
            ),
            _gate(
                "transition_environment_hash",
                artifact["environment_hash"],
                transitions.get("environment_sha256"),
            ),
            _gate(
                "transition_scorer_hash", artifact["scorer_hash"], transitions.get("scorer_sha256")
            ),
            _gate(
                "transition_live_policy_hash",
                artifact["live_policy_hash"],
                transitions.get("live_policy_sha256"),
            ),
            _gate(
                "transition_source_schema",
                "carnot.arc_live_engine_audit_source.v1",
                transitions.get("schema"),
            ),
        ]
    )
    source_rows = transitions.get("rows", [])
    groups = transitions.get("first_step_candidate_groups", [])
    checks.extend(
        [
            _gate(
                "source_transition_rows",
                ">=1",
                len(source_rows) if isinstance(source_rows, list) else 0,
                passed=bool(source_rows) if isinstance(source_rows, list) else False,
            ),
            _gate(
                "recorded_first_step_candidate_groups",
                ">=1",
                len(groups) if isinstance(groups, list) else 0,
                passed=bool(groups) if isinstance(groups, list) else False,
            ),
        ]
    )
    try:
        for row in source_rows if isinstance(source_rows, list) else []:
            prior._array(row.get("grid"))
            prior._array(row.get("next_grid"))
    except (AttributeError, TypeError, ValueError) as error:
        checks.append(
            _gate("source_transition_schema", "valid_2d_grids", f"{type(error).__name__}: {error}")
        )
    if _failed_gates(checks):
        return _finish_blocked(artifact, checks, started)
    split, purity, purity_checks = _purity_checks(transitions, prompt)
    artifact["split_rows"] = [split]
    artifact["purity_rows"] = purity
    checks.extend(purity_checks)
    if _failed_gates(purity_checks):
        checks.append(_gate("split_purity", True, False))
        return _finish_blocked(artifact, checks, started)
    checks.append(_gate("split_purity", True, True))
    shown_ids = set(split["shown_row_ids"])
    normalized = []
    for source_row in sorted(source_rows, key=lambda row: int(row.get("index", -1))):
        row = dict(source_row)
        row["split"] = "shown_prefix" if str(row.get("transition_id")) in shown_ids else "heldout"
        row["row_sha256"] = sha256_bytes(canonical_json(source_row))
        normalized.append(row)
    artifact["per_transition_rows"] = normalized
    engine_rows, execution_receipts = execute_engine_fresh(paths["engine"], normalized)
    for score, source_row in zip(engine_rows, normalized, strict=True):
        score["split"] = source_row["split"]
    receipt = execution_receipts[0]
    artifact["engine_execution_rows"] = [
        {
            **row,
            "fresh_process": receipt["fresh_process"],
            "restricted_process": receipt["restricted_process"],
            "write_guard_enabled": receipt["write_guard_enabled"],
            "network_guard_enabled": receipt["network_guard_enabled"],
            "parent_pid": receipt["parent_pid"],
            "worker_pid": receipt["worker_pid"],
        }
        for row in engine_rows
    ]
    shown = [row for row in normalized if row["split"] == "shown_prefix"]
    heldout = [row for row in normalized if row["split"] == "heldout"]
    heldout_ids = {str(row["transition_id"]) for row in heldout}
    artifact["control_rows"] = score_controls(shown, heldout)
    shown_scores = [row for row in engine_rows if row["split"] == "shown_prefix"]
    heldout_scores = [row for row in engine_rows if str(row["transition_id"]) in heldout_ids]
    artifact["prefix_exact_accuracy"] = _mean_boolean(shown_scores, "exact_transition_correct")
    artifact["heldout_exact_accuracy"] = _mean_boolean(heldout_scores, "exact_transition_correct")
    artifact["heldout_changing_accuracy"] = _mean_boolean(
        heldout_scores, "changing_transition_correct"
    )
    artifact["heldout_noop_accuracy"] = _mean_boolean(heldout_scores, "noop_correct")
    if (
        artifact["prefix_exact_accuracy"] is not None
        and artifact["heldout_exact_accuracy"] is not None
    ):
        artifact["generalization_gap"] = (
            artifact["prefix_exact_accuracy"] - artifact["heldout_exact_accuracy"]
        )
    artifact["paired_control_delta_rows"] = _paired_control_deltas_all(
        heldout_scores, artifact["control_rows"]
    )
    artifact["memorization_signature_rows"] = prior.detect_memorization(
        paths["engine"].read_text(encoding="utf-8"), engine_rows, artifact["control_rows"]
    )
    trace, influence = trace_and_run_live_influence_fixture(
        paths["engine"], artifact["engine_hash"]
    )
    artifact["live_path_trace_rows"] = trace
    artifact["live_influence_fixture_rows"] = influence
    artifact["live_path_reachable_score"] = int(influence[0].get("reachable") is True)
    candidate_rows, ranking_rows = rank_first_step_groups(paths["engine"], groups)
    artifact["first_step_candidate_rows"] = candidate_rows
    artifact["first_step_ranking_rows"] = ranking_rows
    terminal = bool(
        len(engine_rows) == len(normalized)
        and all(row.get("terminal") is True for row in engine_rows)
        and execution_receipts[0].get("terminal") is True
        and len(artifact["control_rows"]) == len(CONTROL_NAMES) * len(heldout)
        and all(row.get("terminal") is True for row in artifact["control_rows"])
        and artifact["paired_control_delta_rows"]
        and all(row.get("terminal") is True for row in artifact["paired_control_delta_rows"])
        and all(row.get("terminal") is True for row in trace)
        and all(row.get("terminal") is True for row in influence)
        and candidate_rows
        and all(row.get("terminal") is True for row in candidate_rows)
        and ranking_rows
        and all(row.get("terminal") is True for row in ranking_rows)
    )
    artifact["arc_engine_audit_complete_score"] = int(terminal)
    interval_summaries = {
        row["control"]: row
        for row in artifact["paired_control_delta_rows"]
        if row["transition_id"] == sorted(heldout_ids)[0]
    }
    strongest_control_accuracy = max(
        (row["control_changing_accuracy"] for row in interval_summaries.values()), default=1.0
    )
    positive = bool(
        terminal
        and artifact["live_path_reachable_score"] == 1
        and artifact["heldout_changing_accuracy"] is not None
        and artifact["heldout_changing_accuracy"] > strongest_control_accuracy
        and len(interval_summaries) == len(CONTROL_NAMES)
        and all(row["interval_low"] > 0 for row in interval_summaries.values())
        and all(row.get("pure") is True for row in purity)
    )
    artifact["arc_generalization_positive_score"] = int(positive)
    artifact["verdict_class"] = "positive" if positive else "null"
    artifact["honest_verdict"] = (
        "positive_arc_live_engine_generalizes_and_influences_first_action"
        if positive
        else "null_arc_live_engine_does_not_clear_generalization_and_reachability_gate"
    )
    checks.append(
        _gate(
            "solve_registry_unchanged", registry_hash_before, sha256_path(repo_root / REGISTRY_PATH)
        )
    )
    artifact["preconditions_checked"] = checks
    artifact["gate_check_summary"] = []
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = _stable_checksum(artifact)
    return artifact


def write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the artifact atomically so readers never observe partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only audit and write one terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260904")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    artifact = build_artifact(date=args.date, repo_root=REPO_ROOT)
    write_json_atomic(output, artifact)
    print(f"wrote {output}")
    print(artifact["honest_verdict"])
    return 1 if artifact["verdict_class"] == "blocked" else 0


def _live_fixture_worker(argv: Sequence[str]) -> int:  # pragma: no cover - child protocol.
    """Execute the live fixture after installing the generated-code effect guard."""

    trace, fixture = _trace_and_run_live_influence_fixture_in_process(
        Path(argv[1]), argv[2] or None, restrict_engine=True
    )
    print(json.dumps({"trace": trace, "fixture": fixture}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess protocol.
    if len(sys.argv) >= 2 and sys.argv[1] == "--live-fixture-worker":
        raise SystemExit(_live_fixture_worker(sys.argv[1:]))
    raise SystemExit(main())
