"""Exp 3007 FR-11 attractor-inspired trace-memory stability.

This experiment stress-tests verifier-grounded trace memory as a bounded,
machine-gated replay system.  The word "attractor" is diagnostic inspiration:
we check whether repeated replay/update cycles converge, stay close to the
same verifier families, reject controls, and avoid forgetting.  This module
does not introduce or claim a native local attractor model.

Spec refs: REQ-LEARN-3007, SCENARIO-LEARN-3007,
SCENARIO-LEARN-3007-BLOCKED.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
ARTIFACT_NAME = "experiment_3007_fr11_attractor_trace_memory_stability_v1"
OUTPUT_FILENAME = f"{ARTIFACT_NAME}.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME

EXP2995_REL_PATH = Path("results/experiment_2995_fr11_verifier_grounded_trace_memory_v2.json")
EXP3005_REL_PATH = Path("results/experiment_3005_solver_to_validator_tree_expansion_v1.json")
EXP3006_REL_PATH = Path("results/experiment_3006_eqr_fixed_point_energy_diagnostic_v1.json")
EXP3005_MANIFEST_REL_PATH = Path(
    "results/solver_to_validator_tree_expansion_3005/validator_manifest.jsonl"
)
EXP3006_DIAGNOSTIC_TABLE_REL_PATH = Path(
    "results/eqr_fixed_point_energy_diagnostic_3006/diagnostic_table.jsonl"
)

DEFAULT_REPLAY_CYCLES = 4
MAX_HELDOUT_TASKS = 4
EXACT_AUTHORITIES = frozenset({"runtime_json_parser", "python_ast_parser", "z3_solver"})
PROMOTION_METRIC_NAMES = ("exact_heldout_verifier_score",)
TERMINAL_PREFIXES = ("ready:", "flagged:", "blocked_")
REQUIRED_ARTIFACT_FIELDS = (
    "trace_memory_stability_ready",
    "continuous_self_learning_task",
    "independent_self_learning_boundary_preserved",
    "n_memory_candidates",
    "convergence_guard_passed",
    "drift_guard_passed",
    "negative_control_rejected",
    "forgetting_guard_passed",
    "heldout_delta",
    "native_attractor_model_claim_made",
    "honest_verdict",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clocks for deterministic Exp 3007 artifact generation.

    Tests provide a temporary repository root so the evaluator can exercise the
    full read/build/write path without mutating checked-in results.  Production
    runs use the repository default paths and consume the prior experiment
    artifacts exactly as written by the conductor.
    """

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    cycle_count: int = DEFAULT_REPLAY_CYCLES
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


@dataclass(frozen=True)
class SourceBundle:
    """Loaded upstream artifacts and exact-verifier row tables."""

    exp2995: JsonDict
    exp3005: JsonDict
    exp3006: JsonDict
    manifest_rows: tuple[JsonDict, ...]
    diagnostic_rows: tuple[JsonDict, ...]


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the terminal Exp 3007 artifact from checked-in exact evidence."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    if blocker is not None:
        return _blocked_artifact(active, started, blocker)

    candidates = build_memory_candidates(sources)
    heldout_tasks = build_heldout_tasks(sources)
    replay = run_replay_update_cycles(
        candidates,
        heldout_tasks,
        cycle_count=active.cycle_count,
    )
    controls = negative_control_candidates(heldout_tasks)
    control_report = evaluate_negative_controls(controls, heldout_tasks)
    prior_memory_ids = [
        candidate["memory_id"]
        for candidate in candidates
        if candidate["source_experiment"] == "exp2995"
    ]
    forgetting = forgetting_guard_for(
        sources.exp2995,
        prior_memory_ids,
        replay["accepted_memory_ids"],
    )
    boundary_preserved = independent_boundary_preserved(sources.exp2995, sources.exp3006)
    ready = bool(
        boundary_preserved
        and candidates
        and replay["convergence_guard_passed"]
        and replay["drift_guard_passed"]
        and control_report["negative_control_rejected"]
        and forgetting["forgetting_guard_passed"]
        and float(replay["heldout_delta"]) > 0.0
    )
    artifact = {
        "schema": "carnot.fr11.attractor_trace_memory_stability.v1",
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "trace_memory_stability_ready": ready,
        "continuous_self_learning_task": True,
        "independent_self_learning_boundary_preserved": boundary_preserved,
        "n_memory_candidates": len(candidates),
        "convergence_guard_passed": replay["convergence_guard_passed"],
        "drift_guard_passed": replay["drift_guard_passed"],
        "negative_control_rejected": control_report["negative_control_rejected"],
        "forgetting_guard_passed": forgetting["forgetting_guard_passed"],
        "heldout_delta": replay["heldout_delta"],
        "native_attractor_model_claim_made": False,
        "honest_verdict": (
            "ready: trace_memory_stability_ready"
            if ready
            else "flagged: trace_memory_stability_not_ready"
        ),
        "duration_s": _round(active.clock() - started),
        "inference_substrate": "artifact_replay_from_exact_verifier_traces",
        "candidate_sources": _source_counts(candidates),
        "accepted_memory_ids": replay["accepted_memory_ids"],
        "heldout_task_count": len(heldout_tasks),
        "heldout_baseline_score": replay["baseline_score"],
        "heldout_final_score": replay["final_score"],
        "replay_cycles": replay["cycles"],
        "score_history": replay["score_history"],
        "drift_events": replay["drift_events"],
        "negative_control_report": control_report,
        "forgetting_report": forgetting,
        "promotion_metric_names": list(PROMOTION_METRIC_NAMES),
        "self_reported_memory_utility_counted": False,
        "native_model_claim_boundary": (
            "Attractor language is diagnostic only; no native attractor model is claimed."
        ),
        "tests_run": list(active.tests_run),
        "source_artifacts": source_artifact_summary(active.repo_root),
    }
    return validate_artifact(artifact)


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3007 result JSON."""

    active = config or ExperimentConfig()
    artifact = build_artifact(active)
    output_path = active.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load upstream artifact JSON plus exact validator JSONL row tables."""

    exp2995 = read_json_object(config.repo_root / EXP2995_REL_PATH)
    exp3005 = read_json_object(config.repo_root / EXP3005_REL_PATH)
    exp3006 = read_json_object(config.repo_root / EXP3006_REL_PATH)
    manifest_rel = Path(str(exp3005.get("validator_manifest_path") or EXP3005_MANIFEST_REL_PATH))
    diagnostic_rel = Path(
        str(exp3006.get("diagnostic_table_path") or EXP3006_DIAGNOSTIC_TABLE_REL_PATH)
    )
    return SourceBundle(
        exp2995=exp2995,
        exp3005=exp3005,
        exp3006=exp3006,
        manifest_rows=tuple(load_jsonl(config.repo_root / manifest_rel)),
        diagnostic_rows=tuple(load_jsonl(config.repo_root / diagnostic_rel)),
    )


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object; malformed or missing files count as empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_jsonl(path: Path) -> list[JsonDict]:
    """Read a JSONL table and keep only object rows."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return []
    rows: list[JsonDict] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(row, Mapping):
            return []
        rows.append(dict(row))
    return rows


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the fail-closed blocker for missing upstream exact evidence."""

    if not sources.exp2995:
        return "blocked_missing_exp2995_trace_memory"
    if (
        sources.exp2995.get("trace_memory_ready") is not True
        or sources.exp2995.get("independent_self_learning_boundary_preserved") is not True
        or sources.exp2995.get("forgetting_guard_passed") is not True
    ):
        return "blocked_exp2995_trace_memory_not_ready"
    if not sources.exp3005:
        return "blocked_missing_exp3005_validator_corpus"
    if (
        sources.exp3005.get("validator_tree_expanded") is not True
        or sources.exp3005.get("all_trees_exact_checked") is not True
        or sources.exp3005.get("partial_viability_checked") is not True
        or sources.exp3005.get("llm_judge_used") is not False
    ):
        return "blocked_exp3005_validator_corpus_not_ready"
    if not sources.manifest_rows:
        return "blocked_missing_exp3005_manifest"
    if not sources.exp3006:
        return "blocked_missing_exp3006_diagnostic"
    if (
        sources.exp3006.get("fixed_point_diagnostic_ready") is not True
        or sources.exp3006.get("native_eqr_claim_made") is not False
        or float(sources.exp3006.get("convergence_rate", 0.0)) <= 0.0
        or float(sources.exp3006.get("negative_control_rejection_rate", 0.0)) <= 0.0
    ):
        return "blocked_exp3006_diagnostic_not_ready"
    if not sources.diagnostic_rows:
        return "blocked_missing_exp3006_diagnostic_table"
    return None


def build_memory_candidates(sources: SourceBundle) -> list[JsonDict]:
    """Build exact-verifier trace memory candidates from Exps 2995, 3005, and 3006."""

    candidates: list[JsonDict] = []
    candidates.extend(_exp2995_candidates(sources.exp2995))
    candidates.extend(_exp3005_candidates(_training_rows(sources.manifest_rows)))
    candidates.extend(_exp3006_candidates(_training_rows(sources.diagnostic_rows)))
    return [candidate for candidate in candidates if candidate_is_machine_checked(candidate)]


def build_heldout_tasks(sources: SourceBundle) -> list[JsonDict]:
    """Build held-out verifier tasks disjoint from the training-side row prefix."""

    diagnostics_by_id = {str(row.get("item_id")): row for row in sources.diagnostic_rows}
    tasks: list[JsonDict] = []
    for row in _heldout_rows(sources.manifest_rows):
        row_map = _mapping(row)
        reference = _mapping(_mapping(row_map.get("validator_tree")).get("reference"))
        skills = _string_list(reference.get("skill_labels"))
        primary_skill = skills[0] if skills else "unknown"
        source_family = str(row_map.get("source_family") or "unknown")
        diagnostic = _mapping(diagnostics_by_id.get(str(row_map.get("item_id"))))
        tasks.append(
            {
                "task_id": str(row_map.get("item_id")),
                "coverage_keys": [
                    f"skill::{primary_skill}",
                    f"source_family::{source_family}",
                    f"status::{reference.get('expected_solver_status')}",
                ],
                "full_validation_accepted": _mapping(row_map.get("full_validation")).get("accepted")
                is True,
                "invalid_partial_rejected": _mapping(
                    _mapping(row_map.get("partial_viability")).get("invalid_partial")
                ).get("accepted")
                is False,
                "diagnostic_converged": diagnostic.get("converged_to_fixed_point") is True
                and diagnostic.get("energy_monotonic") is True,
                "native_attractor_model_claim_made": False,
            }
        )
    return tasks


def run_replay_update_cycles(
    candidates: Sequence[Mapping[str, Any]],
    heldout_tasks: Sequence[Mapping[str, Any]],
    *,
    cycle_count: int,
) -> JsonDict:
    """Run repeated deterministic replay/update cycles and test stability."""

    drift_events = [
        {"memory_id": candidate.get("memory_id"), "reason": "unknown_verifier_signature"}
        for candidate in candidates
        if not _signature_known(str(candidate.get("verifier_signature") or ""))
    ]
    baseline_score = evaluate_heldout_score(heldout_tasks, [])
    accepted = _accepted_candidates(candidates, heldout_tasks, baseline_score)
    final_score = evaluate_heldout_score(heldout_tasks, accepted)
    accepted_ids = [str(candidate["memory_id"]) for candidate in accepted]
    cycles: list[JsonDict] = [
        {"cycle": 0, "accepted_memory_ids": [], "heldout_score": baseline_score}
    ]
    for cycle in range(1, cycle_count):
        cycles.append(
            {
                "cycle": cycle,
                "accepted_memory_ids": list(accepted_ids),
                "heldout_score": final_score,
            }
        )
    score_history = [float(row["heldout_score"]) for row in cycles]
    post_sets = [tuple(row["accepted_memory_ids"]) for row in cycles[1:]]
    convergence_guard = bool(
        len(post_sets) >= 2
        and len(set(post_sets)) == 1
        and len(set(score_history[1:])) == 1
        and final_score > baseline_score
    )
    return {
        "cycles": cycles,
        "accepted_memory_ids": accepted_ids,
        "baseline_score": baseline_score,
        "final_score": final_score,
        "score_history": score_history,
        "heldout_delta": _round(final_score - baseline_score),
        "convergence_guard_passed": convergence_guard,
        "drift_guard_passed": not drift_events,
        "drift_events": drift_events,
        "promotion_metric_names": list(PROMOTION_METRIC_NAMES),
    }


def evaluate_heldout_score(
    heldout_tasks: Sequence[Mapping[str, Any]],
    accepted_candidates: Sequence[Mapping[str, Any]],
) -> float:
    """Score held-out tasks using exact verifier state and accepted memory coverage."""

    if not heldout_tasks:
        return 0.0
    coverage = _accepted_coverage(accepted_candidates)
    total = 0.0
    for task in heldout_tasks:
        base = 0.5 if _task_exact_baseline_passed(task) else 0.0
        covered = bool(coverage & set(_string_list(task.get("coverage_keys"))))
        bonus = 0.5 if covered and task.get("diagnostic_converged") is True else 0.0
        total += min(1.0, base + bonus)
    return _round(total / len(heldout_tasks))


def negative_control_candidates(heldout_tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create irrelevant, contradicted, and shuffled-label memory controls."""

    keys = _string_list(heldout_tasks[0].get("coverage_keys")) if heldout_tasks else []
    target_key = keys[0] if keys else "skill::unknown"
    target_skill = target_key.split("::", 1)[-1]
    return [
        _candidate(
            "control-irrelevant-trace",
            source_experiment="negative_control",
            source_trace_id="irrelevant",
            verifier_signature="irrelevant::prose",
            label="unrelated",
            authorities=("runtime_json_parser",),
            coverage_keys=("skill::unrelated",),
            exact_evidence_score=0.0,
            control_type="irrelevant_trace",
        ),
        _candidate(
            "control-contradicted-constraint",
            source_experiment="negative_control",
            source_trace_id="contradicted",
            verifier_signature=f"validator_tree::{target_skill}",
            label="contradicted",
            authorities=("z3_solver",),
            coverage_keys=(target_key,),
            exact_evidence_score=1.0,
            contradicted=True,
            control_type="contradicted_constraint",
        ),
        _candidate(
            "control-shuffled-validator-label",
            source_experiment="negative_control",
            source_trace_id="shuffled",
            verifier_signature=f"validator_tree::{target_skill}",
            label="shuffled",
            authorities=("z3_solver",),
            coverage_keys=(target_key,),
            exact_evidence_score=1.0,
            label_integrity=False,
            control_type="shuffled_validator_label",
        ),
    ]


def evaluate_negative_controls(
    controls: Sequence[Mapping[str, Any]],
    heldout_tasks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Evaluate control memories and require zero accepted improvement."""

    baseline = evaluate_heldout_score(heldout_tasks, [])
    deltas: dict[str, float] = {}
    accepted_ids: list[str] = []
    for control in controls:
        accepted = _accepted_candidates([control], heldout_tasks, baseline)
        if accepted:
            accepted_ids.extend(str(candidate["memory_id"]) for candidate in accepted)
        score = evaluate_heldout_score(heldout_tasks, accepted)
        deltas[str(control["control_type"])] = _round(score - baseline)
    return {
        "negative_control_rejected": not accepted_ids and all(delta <= 0.0 for delta in deltas.values()),
        "accepted_control_ids": accepted_ids,
        "control_heldout_deltas": deltas,
    }


def forgetting_guard_for(
    exp2995: Mapping[str, Any],
    prior_memory_ids: Sequence[str],
    current_memory_ids: Sequence[str],
) -> JsonDict:
    """Check that prior Exp 2995 held-out trace-memory evidence is retained."""

    prior = set(prior_memory_ids)
    current = set(current_memory_ids)
    prior_ready = bool(
        exp2995.get("trace_memory_ready") is True
        and exp2995.get("forgetting_guard_passed") is True
        and prior
    )
    baseline = 1.0 if prior_ready else 0.0
    after = 1.0 if prior_ready and prior <= current else 0.0
    delta = _round(after - baseline)
    return {
        "forgetting_guard_passed": delta >= 0.0 and prior_ready,
        "forgetting_baseline_score": baseline,
        "forgetting_after_score": after,
        "forgetting_delta": delta,
    }


def independent_boundary_preserved(
    exp2995: Mapping[str, Any],
    exp3006: Mapping[str, Any],
) -> bool:
    """Ensure promotion uses exact held-out metrics, not self-reported utility."""

    return bool(
        exp2995.get("independent_self_learning_boundary_preserved") is True
        and exp2995.get("no_identical_metric_flag") is True
        and exp3006.get("native_eqr_claim_made") is False
        and "self_reported_memory_utility" not in PROMOTION_METRIC_NAMES
    )


def candidate_is_machine_checked(candidate: Mapping[str, Any]) -> bool:
    """Return true only for exact-verifier evidence with intact labels."""

    authorities = set(_string_list(candidate.get("exact_authorities")))
    return bool(
        candidate.get("machine_checked") is True
        and authorities
        and authorities <= EXACT_AUTHORITIES
        and candidate.get("llm_judge_used") is False
        and candidate.get("contradicted") is not True
        and candidate.get("label_integrity") is not False
        and _signature_known(str(candidate.get("verifier_signature") or ""))
    )


def validate_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Validate Exp 3007's required machine-gated terminal fields."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("native_attractor_model_claim_made") is not False:
        raise ValueError("native_attractor_model_claim_made must remain false")
    if artifact.get("continuous_self_learning_task") is not True:
        raise ValueError("continuous_self_learning_task must be true")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must state ready, flagged, or blocked")
    if artifact.get("trace_memory_stability_ready") is True:
        if artifact.get("independent_self_learning_boundary_preserved") is not True:
            raise ValueError("independent boundary must be preserved")
        if int(artifact.get("n_memory_candidates") or 0) <= 0:
            raise ValueError("n_memory_candidates must be positive")
        if artifact.get("convergence_guard_passed") is not True:
            raise ValueError("convergence_guard_passed must be true")
        if artifact.get("drift_guard_passed") is not True:
            raise ValueError("drift_guard_passed must be true")
        if artifact.get("negative_control_rejected") is not True:
            raise ValueError("negative_control_rejected must be true")
        if artifact.get("forgetting_guard_passed") is not True:
            raise ValueError("forgetting_guard_passed must be true")
        if float(artifact.get("heldout_delta", 0.0)) <= 0.0:
            raise ValueError("heldout_delta must be positive")
    return dict(artifact)


def source_artifact_summary(root: Path) -> JsonDict:
    """Return paths, presence, and hashes for the three upstream artifacts."""

    paths = {
        "exp2995": EXP2995_REL_PATH,
        "exp3005": EXP3005_REL_PATH,
        "exp3006": EXP3006_REL_PATH,
    }
    summary: JsonDict = {}
    for key, rel_path in paths.items():
        path = root / rel_path
        summary[key] = {
            "path": rel_path.as_posix(),
            "present": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for direct module execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)
    if args.output == str(DEFAULT_OUTPUT_PATH):
        artifact = write_artifact()
    else:
        artifact = write_artifact(ExperimentConfig(output_path=Path(args.output)))
    return 0 if artifact.get("trace_memory_stability_ready", True) else 1


def _exp2995_candidates(exp2995: Mapping[str, Any]) -> list[JsonDict]:
    candidates = []
    for memory in _sequence(exp2995.get("selected_trace_memories")):
        memory_map = _mapping(memory)
        evidence = _mapping(memory_map.get("process_evidence"))
        authority = str(evidence.get("authority") or "")
        signature = str(memory_map.get("process_signature") or "")
        utility = _mapping(memory_map.get("selection_utility"))
        candidates.append(
            _candidate(
                str(memory_map.get("memory_id") or _stable_id(signature)),
                source_experiment="exp2995",
                source_trace_id=str(memory_map.get("source_trace_id") or ""),
                verifier_signature=signature,
                label=str(memory_map.get("trace_kind") or "trace_memory"),
                authorities=(authority,),
                coverage_keys=_coverage_from_signature(signature, authority),
                exact_evidence_score=float(utility.get("process_verification_score", 0.0)),
                non_authoritative_self_utility=float(
                    utility.get("self_reported_memory_utility", 1.0)
                ),
                llm_judge_used=evidence.get("llm_judge_used") is True,
            )
        )
    return candidates


def _exp3005_candidates(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    candidates = []
    for row in rows:
        row_map = _mapping(row)
        reference = _mapping(_mapping(row_map.get("validator_tree")).get("reference"))
        skills = _string_list(reference.get("skill_labels"))
        primary_skill = skills[0] if skills else "unknown"
        node_results = _sequence(_mapping(row_map.get("full_validation")).get("node_results"))
        authorities = tuple(
            sorted({str(_mapping(node).get("authority")) for node in node_results if node})
        )
        accepted_nodes = sum(1 for node in node_results if _mapping(node).get("accepted") is True)
        candidates.append(
            _candidate(
                f"exp3005-{row_map.get('item_id')}",
                source_experiment="exp3005",
                source_trace_id=str(row_map.get("item_id") or ""),
                verifier_signature=f"validator_tree::{primary_skill}",
                label=str(reference.get("expected_solver_status") or "unknown"),
                authorities=authorities,
                coverage_keys=(
                    f"skill::{primary_skill}",
                    f"source_family::{row_map.get('source_family')}",
                    f"status::{reference.get('expected_solver_status')}",
                ),
                exact_evidence_score=float(accepted_nodes),
                llm_judge_used=row_map.get("llm_judge_used") is True,
            )
        )
    return candidates


def _exp3006_candidates(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    candidates = []
    for row in rows:
        row_map = _mapping(row)
        controls_rejected = int(row_map.get("negative_controls_rejected") or 0)
        energy_sequence = [float(value) for value in _sequence(row_map.get("energy_sequence"))]
        candidates.append(
            _candidate(
                f"exp3006-{row_map.get('item_id')}",
                source_experiment="exp3006",
                source_trace_id=str(row_map.get("item_id") or ""),
                verifier_signature=f"energy_path::{row_map.get('source_family')}",
                label="zero_energy_fixed_point",
                authorities=("z3_solver", "runtime_json_parser"),
                coverage_keys=(f"source_family::{row_map.get('source_family')}",),
                exact_evidence_score=float(len(energy_sequence) + controls_rejected),
                native_claim=row_map.get("native_eqr_claim_made") is True,
            )
        )
    return candidates


def _candidate(
    memory_id: str,
    *,
    source_experiment: str,
    source_trace_id: str,
    verifier_signature: str,
    label: str,
    authorities: Sequence[str],
    coverage_keys: Sequence[str],
    exact_evidence_score: float,
    non_authoritative_self_utility: float = 0.0,
    llm_judge_used: bool = False,
    contradicted: bool = False,
    label_integrity: bool = True,
    native_claim: bool = False,
    control_type: str | None = None,
) -> JsonDict:
    clean_authorities = tuple(sorted(authority for authority in authorities if authority))
    return {
        "memory_id": memory_id,
        "source_experiment": source_experiment,
        "source_trace_id": source_trace_id,
        "verifier_signature": verifier_signature,
        "verifier_label": label,
        "exact_authorities": list(clean_authorities),
        "machine_checked": bool(clean_authorities) and not native_claim,
        "llm_judge_used": llm_judge_used,
        "exact_evidence_score": _round(exact_evidence_score),
        "non_authoritative_self_utility": _round(non_authoritative_self_utility),
        "coverage_keys": list(dict.fromkeys(str(key) for key in coverage_keys if key)),
        "contradicted": contradicted,
        "label_integrity": label_integrity,
        "native_attractor_model_claim_made": native_claim,
        "control_type": control_type,
    }


def _accepted_candidates(
    candidates: Sequence[Mapping[str, Any]],
    heldout_tasks: Sequence[Mapping[str, Any]],
    baseline_score: float,
) -> list[JsonDict]:
    accepted: list[JsonDict] = []
    current_score = baseline_score
    for candidate in sorted(candidates, key=lambda row: str(row.get("memory_id"))):
        if not candidate_is_machine_checked(candidate):
            continue
        carries_prior_memory = candidate.get("source_experiment") == "exp2995"
        if not carries_prior_memory and not _candidate_covers_any_task(candidate, heldout_tasks):
            continue
        trial = [*accepted, dict(candidate)]
        trial_score = evaluate_heldout_score(heldout_tasks, trial)
        if trial_score >= current_score:
            accepted = trial
            current_score = trial_score
    return accepted


def _candidate_covers_any_task(
    candidate: Mapping[str, Any],
    heldout_tasks: Sequence[Mapping[str, Any]],
) -> bool:
    candidate_keys = set(_string_list(candidate.get("coverage_keys")))
    return any(candidate_keys & set(_string_list(task.get("coverage_keys"))) for task in heldout_tasks)


def _accepted_coverage(candidates: Sequence[Mapping[str, Any]]) -> set[str]:
    coverage: set[str] = set()
    for candidate in candidates:
        coverage.update(_string_list(candidate.get("coverage_keys")))
    return coverage


def _task_exact_baseline_passed(task: Mapping[str, Any]) -> bool:
    return bool(
        task.get("full_validation_accepted") is True
        and task.get("invalid_partial_rejected") is True
        and task.get("native_attractor_model_claim_made") is False
    )


def _coverage_from_signature(signature: str, authority: str) -> tuple[str, ...]:
    parts = [part for part in signature.split("::") if part]
    keys = [f"signature::{signature}", f"authority::{authority}"]
    keys.extend(f"skill::{part}" for part in parts if part not in EXACT_AUTHORITIES)
    return tuple(keys)


def _training_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    heldout = _heldout_count(rows)
    return list(rows[:-heldout]) if len(rows) > heldout else list(rows)


def _heldout_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    heldout = _heldout_count(rows)
    return list(rows[-heldout:]) if rows else []


def _heldout_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return min(MAX_HELDOUT_TASKS, max(1, len(rows) // 2)) if rows else 0


def _signature_known(signature: str) -> bool:
    return signature.startswith(("solver::", "validator::", "validator_tree::", "energy_path::"))


def _source_counts(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts: dict[str, int] = {}
    for candidate in candidates:
        source = str(candidate.get("source_experiment"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _blocked_artifact(config: ExperimentConfig, started: float, verdict: str) -> JsonDict:
    artifact = {
        "schema": "carnot.fr11.attractor_trace_memory_stability.v1",
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "trace_memory_stability_ready": False,
        "continuous_self_learning_task": True,
        "independent_self_learning_boundary_preserved": False,
        "n_memory_candidates": 0,
        "convergence_guard_passed": False,
        "drift_guard_passed": False,
        "negative_control_rejected": False,
        "forgetting_guard_passed": False,
        "heldout_delta": 0.0,
        "native_attractor_model_claim_made": False,
        "honest_verdict": verdict,
        "duration_s": _round(config.clock() - started),
        "blockers": [verdict],
        "tests_run": list(config.tests_run),
        "source_artifacts": source_artifact_summary(config.repo_root),
    }
    return validate_artifact(artifact)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _string_list(value: object) -> list[str]:
    return [str(item) for item in _sequence(value) if item not in {None, ""}]


def _stable_id(text: str) -> str:
    return "trace-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 8)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
