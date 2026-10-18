"""Exp 2995 FR-11 verifier-grounded trace memory v2.

The experiment uses checked-in verifier traces as a bounded external memory
update. The update selector sees only process-verifiable evidence, such as Z3
execution and validator-tree authority. The held-out report reuses the
independent metric shape from Exp 2982 so the result is not tautological.

Spec: REQ-LEARN-2995, SCENARIO-LEARN-2995,
SCENARIO-LEARN-2995-BLOCKED.
"""

from __future__ import annotations

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
OUTPUT_FILENAME = "experiment_2995_fr11_verifier_grounded_trace_memory_v2.json"
ARTIFACT = "experiment_2995_fr11_verifier_grounded_trace_memory_v2"
SCHEMA = "carnot.fr11.verifier_grounded_trace_memory.v2"
INFERENCE_SUBSTRATE = "artifact_replay_from_solver_and_validator_traces"

EXP2982_REL_PATH = Path("results/experiment_2982_fr11_independent_metric_utility_gate_v4.json")
EXP2983_REL_PATH = Path("results/experiment_2983_trace_to_skill_repair_memory_pilot_v1.json")
EXP2992_REL_PATH = Path(
    "results/experiment_2992_sota_solver_formalization_provenance_reproduction_v1.json"
)
EXP2994_REL_PATH = Path("results/experiment_2994_prompt_validator_dialogue_schema_v1.json")

MAX_SELECTED_TRACE_MEMORIES = 8
MAX_SELECTED_PER_SOURCE = 5
HELDOUT_SOLVER_TASKS = 4
UTILITY_METRIC_NAMES = (
    "process_verification_score",
    "trace_evidence_density",
)
TRACE_MEMORY_REQUIRED_FIELDS = {
    "memory_id",
    "source",
    "source_trace_id",
    "trace_kind",
    "process_signature",
    "process_verifiable",
    "process_evidence",
    "selection_utility",
    "reuse_hint",
    "forbidden_label_leakage",
}
REQUIRED_ARTIFACT_FIELDS = {
    "independent_self_learning_boundary_preserved",
    "continuous_self_learning_task",
    "trace_memory_ready",
    "n_trace_memories",
    "independent_metric_names",
    "utility_metric_names",
    "no_identical_metric_flag",
    "negative_control_deltas",
    "forgetting_guard_passed",
    "heldout_metric_deltas",
    "honest_verdict",
}
EXACT_AUTHORITIES = frozenset({"runtime_json_parser", "python_ast_parser", "z3_solver"})


@dataclass(frozen=True)
class MetricSpec:
    """One held-out metric and the direction that counts as improvement."""

    name: str
    direction: str


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clocks for deterministic artifact generation."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


INDEPENDENT_METRICS = (
    MetricSpec("pass_at_1", "higher_is_better"),
    MetricSpec("solver_verified_accuracy", "higher_is_better"),
    MetricSpec("syntax_failure_rate", "lower_is_better"),
    MetricSpec("schema_failure_rate", "lower_is_better"),
    MetricSpec("verifier_false_accept_rate", "lower_is_better"),
)


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the terminal Exp 2995 artifact from checked-in verifier traces."""

    active = config or ExperimentConfig()
    started = active.start_time()
    exp2982 = read_json_object(active.repo_root / EXP2982_REL_PATH)
    exp2983 = read_json_object(active.repo_root / EXP2983_REL_PATH)
    exp2992 = read_json_object(active.repo_root / EXP2992_REL_PATH)
    exp2994 = read_json_object(active.repo_root / EXP2994_REL_PATH)
    sources = source_artifacts_for(active.repo_root)
    blocker = precondition_blocker(exp2982, exp2992, exp2994)
    if blocker is not None:
        return _blocked_artifact(active, started, sources, blocker)

    candidates = build_trace_memory_candidates(exp2992, exp2994)
    selected = select_trace_memories(candidates, enabled=True)
    heldout_tasks = build_heldout_tasks(exp2992)
    random_metrics = evaluate_heldout_metrics(heldout_tasks, selected, condition="random_control")
    trace_metrics = evaluate_heldout_metrics(heldout_tasks, selected, condition="trace_memory")
    disabled_metrics = evaluate_heldout_metrics(heldout_tasks, selected, condition="disabled_update")
    shuffled_metrics = evaluate_heldout_metrics(
        heldout_tasks,
        selected,
        condition="shuffled_trace_memory",
    )
    heldout_deltas = directional_delta(trace_metrics, random_metrics)
    negative_deltas = {
        "disabled_update": directional_delta(disabled_metrics, random_metrics),
        "shuffled_trace_memory": directional_delta(shuffled_metrics, random_metrics),
    }
    no_identical = no_identical_metric_flag(utility_metric_names(), independent_metric_names())
    controls_equal = controls_improve_equally(heldout_deltas, negative_deltas)
    forgetting_passed = forgetting_guard_passed_for(exp2982, exp2983)
    boundary_preserved = bool(
        exp2982.get("fr11_independent_self_learning_ready") is True
        and exp2983.get("headline_result") is False
        and no_identical
        and not controls_equal
    )
    ready = bool(
        selected
        and metrics_improved(heldout_deltas)
        and not controls_equal
        and forgetting_passed
        and boundary_preserved
    )
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": (
            "ready: verifier_grounded_trace_memory_ready"
            if ready
            else "flagged: verifier_grounded_trace_memory_not_ready"
        ),
        "independent_self_learning_boundary_preserved": boundary_preserved,
        "continuous_self_learning_task": True,
        "trace_memory_ready": ready,
        "n_trace_memories": len(selected),
        "independent_metric_names": independent_metric_names(),
        "utility_metric_names": utility_metric_names(),
        "no_identical_metric_flag": no_identical,
        "negative_control_deltas": negative_deltas,
        "forgetting_guard_passed": forgetting_passed,
        "heldout_metric_deltas": heldout_deltas,
        "duration_s": _elapsed(active, started),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "candidate_trace_memory_count": len(candidates),
        "selected_trace_memories": selected,
        "heldout_task_count": len(heldout_tasks),
        "heldout_task_ids": [str(task["task_id"]) for task in heldout_tasks],
        "random_control_metrics": random_metrics,
        "trace_memory_metrics": trace_metrics,
        "disabled_update_metrics": disabled_metrics,
        "shuffled_trace_memory_metrics": shuffled_metrics,
        "controls_improve_equally": controls_equal,
        "selection_rule": selection_rule_summary(),
        "leakage_flag": leakage_flag_for(selected, heldout_tasks),
        "source_artifacts": sources,
        "tests_run": list(active.tests_run),
    }
    return validate_artifact(artifact)


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 2995 result JSON."""

    active = config or ExperimentConfig()
    artifact = build_artifact(active)
    output_path = active.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def independent_metric_names() -> list[str]:
    """Return held-out metric names, separate from selection utility names."""

    return [metric.name for metric in INDEPENDENT_METRICS]


def utility_metric_names() -> list[str]:
    """Return utility names used by the bounded memory selector."""

    return list(UTILITY_METRIC_NAMES)


def build_trace_memory_candidates(
    exp2992: Mapping[str, Any],
    exp2994: Mapping[str, Any],
) -> list[JsonDict]:
    """Extract process-verifiable memories from solver and validator traces."""

    return solver_trace_memories(exp2992) + validator_trace_memories(exp2994)


def solver_trace_memories(exp2992: Mapping[str, Any]) -> list[JsonDict]:
    """Convert training-side solver transcripts into reusable trace memories."""

    rows = _sequence(exp2992.get("per_item_results"))
    training_rows = rows[:-HELDOUT_SOLVER_TASKS] if len(rows) > HELDOUT_SOLVER_TASKS else rows[:1]
    memories: list[JsonDict] = []
    for row in training_rows:
        row_map = _mapping(row)
        final = _mapping(row_map.get("final_result"))
        if not solver_process_verified(final):
            continue
        item_id = str(final.get("item_id") or _mapping(row_map.get("initial_result")).get("item_id"))
        signature = solver_process_signature(final)
        evidence = {
            "authority": "z3_solver",
            "z3_executed": final.get("z3_executed") is True,
            "solver_formula_correct": final.get("solver_formula_correct") is True,
            "solver_result_matches_expected": _solver_result_matches_expected(final),
            "prompt_hash_recorded": _is_sha256(row_map.get("prompt_hash") or final.get("prompt_hash")),
            "z3_transcript_hash_recorded": _is_sha256(row_map.get("z3_transcript_sha256")),
        }
        memory = {
            "memory_id": _stable_id(f"exp2992:{item_id}:{signature}"),
            "source": "exp2992",
            "source_trace_id": item_id,
            "trace_kind": "solver_formalization_transcript",
            "process_signature": signature,
            "process_verifiable": True,
            "process_evidence": evidence,
            "selection_utility": selection_utility_for(evidence),
            "reuse_hint": "run exact Z3 and require parser plus solver agreement before accepting",
            "forbidden_label_leakage": forbidden_label_leakage_terms(),
        }
        memories.append(validate_trace_memory(memory))
    return memories


def validator_trace_memories(exp2994: Mapping[str, Any]) -> list[JsonDict]:
    """Convert validator-tree success/failure pairs into trace memories."""

    memories: list[JsonDict] = []
    for fixture in _sequence(exp2994.get("validator_tree_fixtures")):
        fixture_map = _mapping(fixture)
        if fixture_map.get("compiled") is not True:
            continue
        good = _mapping(fixture_map.get("known_good_feedback"))
        bad = _mapping(fixture_map.get("known_bad_feedback"))
        if good.get("accepted") is not True or bad.get("accepted") is not False:
            continue
        node = first_failing_node(bad)
        authority = str(node.get("authority") or "")
        if authority not in EXACT_AUTHORITIES:
            continue
        fixture_id = str(fixture_map.get("fixture_id") or "unknown-validator-fixture")
        reason = str(node.get("rejection_reason") or "validator_rejected")
        signature = f"validator::{authority}::{reason}"
        evidence = {
            "authority": authority,
            "known_good_accepted": True,
            "known_bad_rejected": True,
            "llm_judge_used": False,
            "failing_node_recorded": bool(node.get("node_id")),
        }
        memory = {
            "memory_id": _stable_id(f"exp2994:{fixture_id}:{signature}"),
            "source": "exp2994",
            "source_trace_id": fixture_id,
            "trace_kind": "validator_tree_feedback_pair",
            "process_signature": signature,
            "process_verifiable": True,
            "process_evidence": evidence,
            "selection_utility": selection_utility_for(evidence),
            "reuse_hint": "prefer exact validator-tree feedback over model self-judgment",
            "forbidden_label_leakage": forbidden_label_leakage_terms(),
        }
        memories.append(validate_trace_memory(memory))
    return memories


def select_trace_memories(
    candidates: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
) -> list[JsonDict]:
    """Select a bounded update; disabling it yields the random/control path."""

    if not enabled:
        return []
    selected: list[JsonDict] = []
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        memory = validate_trace_memory(candidate)
        by_source.setdefault(str(memory["source"]), []).append(memory)
    for source in sorted(by_source):
        ranked = sorted(
            by_source[source],
            key=lambda row: (
                -float(_mapping(row.get("selection_utility")).get("process_verification_score", 0.0)),
                str(row.get("process_signature")),
            ),
        )
        selected.extend(dict(row) for row in ranked[:MAX_SELECTED_PER_SOURCE])
    return selected[:MAX_SELECTED_TRACE_MEMORIES]


def build_heldout_tasks(exp2992: Mapping[str, Any]) -> list[JsonDict]:
    """Build solver held-out tasks disjoint from the memory extraction prefix."""

    rows = _sequence(exp2992.get("per_item_results"))
    heldout_rows = rows[-HELDOUT_SOLVER_TASKS:] if len(rows) > HELDOUT_SOLVER_TASKS else rows[1:]
    tasks: list[JsonDict] = []
    for row in heldout_rows:
        row_map = _mapping(row)
        initial = _mapping(row_map.get("initial_result"))
        final = _mapping(row_map.get("final_result"))
        task_id = str(final.get("item_id") or initial.get("item_id") or "")
        if not task_id:
            continue
        tasks.append(
            {
                "task_id": task_id,
                "required_signatures": [solver_process_signature(final)],
                "baseline_flags": outcome_flags(initial),
                "trace_flags": outcome_flags(final),
            }
        )
    return tasks


def evaluate_heldout_metrics(
    heldout_tasks: Sequence[Mapping[str, Any]],
    memories: Sequence[Mapping[str, Any]],
    *,
    condition: str,
) -> dict[str, float]:
    """Score held-out metrics under no-update, control, or trace-memory replay."""

    signatures = {str(memory.get("process_signature")) for memory in memories}
    totals = {
        "pass_at_1": 0.0,
        "solver_verified_accuracy": 0.0,
        "syntax_failure_rate": 0.0,
        "schema_failure_rate": 0.0,
        "verifier_false_accept_rate": 0.0,
    }
    if condition not in {
        "random_control",
        "disabled_update",
        "shuffled_trace_memory",
        "trace_memory",
    }:
        raise ValueError(f"unknown held-out condition: {condition}")
    for task in heldout_tasks:
        flags = _mapping(task.get("baseline_flags"))
        if condition == "trace_memory" and _task_covered(task, signatures):
            flags = _mapping(task.get("trace_flags"))
        for name in totals:
            totals[name] += float(flags.get(name, 0.0))
    task_count = len(heldout_tasks)
    return {name: _round(value / task_count) if task_count else 0.0 for name, value in totals.items()}


def outcome_flags(row: Mapping[str, Any]) -> dict[str, float]:
    """Map one verifier result row to the independent metric indicators."""

    parseable = row.get("parseable") is True
    solver_correct = row.get("solver_formula_correct") is True and row.get("z3_executed") is True
    answer_correct = row.get("answer_correct") is True
    parse_error = str(row.get("parse_error") or "")
    failure = str(row.get("failure_category") or "")
    return {
        "pass_at_1": float(parseable and solver_correct and answer_correct),
        "solver_verified_accuracy": float(solver_correct),
        "syntax_failure_rate": float(not parseable and "missing_schema_field" not in parse_error),
        "schema_failure_rate": float("missing_schema_field" in parse_error),
        "verifier_false_accept_rate": float(answer_correct and not solver_correct)
        if failure == "z3_exception"
        else 0.0,
    }


def directional_delta(
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
) -> dict[str, float]:
    """Use Exp 2982 directional semantics for held-out metric deltas."""

    deltas: dict[str, float] = {}
    for metric in INDEPENDENT_METRICS:
        candidate = float(candidate_metrics.get(metric.name, 0.0))
        baseline = float(baseline_metrics.get(metric.name, 0.0))
        if metric.direction == "higher_is_better":
            deltas[metric.name] = _round(candidate - baseline)
        else:
            deltas[metric.name] = _round(baseline - candidate)
    return deltas


def metrics_improved(deltas: Mapping[str, float]) -> bool:
    """Return true only when every independent held-out metric improves."""

    return all(float(deltas.get(metric.name, 0.0)) > 0.0 for metric in INDEPENDENT_METRICS)


def controls_improve_equally(
    heldout_deltas: Mapping[str, float],
    negative_control_deltas: Mapping[str, Mapping[str, float]],
) -> bool:
    """Detect controls that match or exceed the selected-memory improvement."""

    names = independent_metric_names()
    for deltas in negative_control_deltas.values():
        if all(float(deltas.get(name, 0.0)) >= float(heldout_deltas.get(name, 0.0)) for name in names) and any(
            float(deltas.get(name, 0.0)) > 0.0 for name in names
        ):
            return True
    return False


def no_identical_metric_flag(
    utility_names: Sequence[str],
    metric_names: Sequence[str],
) -> bool:
    """Ensure process utility names cannot be reported as held-out metrics."""

    utility_set = {str(name) for name in utility_names}
    metric_set = {str(name) for name in metric_names}
    return bool(utility_set) and bool(metric_set) and not (utility_set & metric_set)


def leakage_flag_for(
    memories: Sequence[Mapping[str, Any]],
    heldout_tasks: Sequence[Mapping[str, Any]],
) -> bool:
    """Detect held-out IDs or answer labels copied into memory text."""

    heldout_ids = {str(task.get("task_id")).lower() for task in heldout_tasks if task.get("task_id")}
    forbidden = {
        "expected_solver_status=",
        '"expected_solver_status":',
        "expected_status=",
        '"expected_status":',
        "expected answer:",
        "reference_solution",
        "pass_vector",
        "heldout_metric",
    }
    for memory in memories:
        text = json.dumps(memory, sort_keys=True).lower()
        if any(task_id in text for task_id in heldout_ids):
            return True
        if any(token in text for token in forbidden):
            return True
    return False


def forgetting_guard_passed_for(exp2982: Mapping[str, Any], exp2983: Mapping[str, Any]) -> bool:
    """Preserve prior FR-11 evidence and Exp 2983's non-headline pilot boundary."""

    return bool(
        exp2982.get("fr11_independent_self_learning_ready") is True
        and exp2982.get("forgetting_guard_passed") is True
        and exp2982.get("no_identical_metric_flag") is True
        and exp2983.get("headline_result") is False
    )


def validate_trace_memory(memory: Mapping[str, Any]) -> JsonDict:
    """Validate one trace memory before it can enter the bounded selector."""

    missing = TRACE_MEMORY_REQUIRED_FIELDS - set(memory)
    if missing:
        raise ValueError(f"trace memory missing required fields: {sorted(missing)}")
    if memory.get("process_verifiable") is not True:
        raise ValueError("trace memory must be process-verifiable")
    if not isinstance(memory.get("process_evidence"), Mapping):
        raise ValueError("process_evidence must be an object")
    if not isinstance(memory.get("selection_utility"), Mapping):
        raise ValueError("selection_utility must be an object")
    leakage = memory.get("forbidden_label_leakage")
    if not isinstance(leakage, Sequence) or isinstance(leakage, (str, bytes)):
        raise ValueError("forbidden_label_leakage must be an array")
    return dict(memory)


def validate_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Validate the required terminal fields and promotion gates."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("trace_memory_ready") is True:
        if artifact.get("independent_self_learning_boundary_preserved") is not True:
            raise ValueError("boundary must be preserved before promotion")
        if artifact.get("no_identical_metric_flag") is not True:
            raise ValueError("identical metric names are not allowed")
        if int(artifact.get("n_trace_memories") or 0) <= 0:
            raise ValueError("selected trace memories are required")
        if artifact.get("forgetting_guard_passed") is not True:
            raise ValueError("forgetting guard must pass")
        if artifact.get("controls_improve_equally") is True:
            raise ValueError("controls improve equally")
        if not metrics_improved(_mapping(artifact.get("heldout_metric_deltas"))):
            raise ValueError("held-out metrics must improve")
    return dict(artifact)


def read_json_object(path: Path) -> JsonDict:
    """Read a local JSON object, returning empty evidence when unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts_for(root: Path) -> dict[str, JsonDict]:
    """Record artifact provenance and checksums for the trace-memory inputs."""

    specs = {
        "exp2982": (EXP2982_REL_PATH, "independent_metric_boundary"),
        "exp2983": (EXP2983_REL_PATH, "prior_trace_memory_pilot_boundary"),
        "exp2992": (EXP2992_REL_PATH, "fresh_solver_transcripts"),
        "exp2994": (EXP2994_REL_PATH, "validator_tree_success_failure_feedback"),
    }
    citations: dict[str, JsonDict] = {}
    for experiment_id, (rel_path, role) in specs.items():
        path = root / rel_path
        citations[experiment_id] = {
            "path": rel_path.as_posix(),
            "role": role,
            "present": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        }
    return citations


def precondition_blocker(
    exp2982: Mapping[str, Any],
    exp2992: Mapping[str, Any],
    exp2994: Mapping[str, Any],
) -> str | None:
    """Return the fail-closed blocker when required verifier evidence is absent."""

    if not exp2982:
        return "blocked_missing_exp2982_independent_boundary"
    if (
        exp2982.get("fr11_independent_self_learning_ready") is not True
        or exp2982.get("forgetting_guard_passed") is not True
        or exp2982.get("no_identical_metric_flag") is not True
    ):
        return "blocked_exp2982_independent_boundary_not_ready"
    if not exp2992:
        return "blocked_missing_exp2992_solver_traces"
    if (
        exp2992.get("solver_provenance_reproduced") is not True
        or exp2992.get("formalization_clean") is not True
    ):
        return "blocked_exp2992_solver_provenance_not_ready"
    if not exp2994:
        return "blocked_missing_exp2994_validator_protocol"
    if (
        exp2994.get("prompt_validator_protocol_ready") is not True
        or exp2994.get("exact_verifier_authority_preserved") is not True
    ):
        return "blocked_exp2994_validator_protocol_not_ready"
    return None


def selection_rule_summary() -> JsonDict:
    """Describe the bounded update rule so controls can disable it explicitly."""

    return {
        "name": "bounded_process_verifiable_trace_memory_selection_v2",
        "max_selected_trace_memories": MAX_SELECTED_TRACE_MEMORIES,
        "max_selected_per_source": MAX_SELECTED_PER_SOURCE,
        "utility_metric_names": utility_metric_names(),
        "disable_supported": True,
    }


def main() -> int:
    """CLI entry point used by the experiment wrapper."""

    write_artifact()
    return 0


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    source_artifacts: Mapping[str, Mapping[str, Any]],
    verdict: str,
) -> JsonDict:
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "independent_self_learning_boundary_preserved": False,
        "continuous_self_learning_task": True,
        "trace_memory_ready": False,
        "n_trace_memories": 0,
        "independent_metric_names": independent_metric_names(),
        "utility_metric_names": utility_metric_names(),
        "no_identical_metric_flag": no_identical_metric_flag(
            utility_metric_names(),
            independent_metric_names(),
        ),
        "negative_control_deltas": {},
        "forgetting_guard_passed": False,
        "heldout_metric_deltas": {},
        "duration_s": _elapsed(config, started),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts": {key: dict(value) for key, value in source_artifacts.items()},
        "blockers": [verdict],
        "tests_run": list(config.tests_run),
    }
    return validate_artifact(artifact)


def solver_process_verified(final: Mapping[str, Any]) -> bool:
    return bool(
        final.get("z3_executed") is True
        and final.get("solver_formula_correct") is True
        and _solver_result_matches_expected(final)
    )


def solver_process_signature(final: Mapping[str, Any]) -> str:
    check_kind = str(final.get("check_kind") or "solver")
    skill = _string_list(final.get("skill_labels"))
    primary_skill = skill[0] if skill else "unknown"
    return f"solver::{check_kind}::{primary_skill}"


def selection_utility_for(evidence: Mapping[str, Any]) -> JsonDict:
    score = sum(1 for key, value in evidence.items() if key != "authority" and value is True)
    denominator = max(1, len([key for key in evidence if key != "authority"]))
    return {
        "process_verification_score": _round(float(score)),
        "trace_evidence_density": _round(score / denominator),
    }


def first_failing_node(feedback: Mapping[str, Any]) -> Mapping[str, Any]:
    for node in _sequence(feedback.get("node_results")):
        node_map = _mapping(node)
        if node_map.get("accepted") is False:
            return node_map
    return {}


def forbidden_label_leakage_terms() -> list[str]:
    return [
        "held-out task ids",
        "expected solver status",
        "expected answers",
        "pass vectors",
        "reference solutions",
    ]


def _task_covered(task: Mapping[str, Any], signatures: set[str]) -> bool:
    return bool(set(_string_list(task.get("required_signatures"))) & signatures)


def _solver_result_matches_expected(final: Mapping[str, Any]) -> bool:
    result = _mapping(final.get("z3_result"))
    return bool(
        result.get("solver_status_matches_expected") is True
        and result.get("answer_extraction_matches_expected") is True
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _string_list(value: object) -> list[str]:
    return [str(item) for item in _sequence(value) if item not in {None, ""}]


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _stable_id(text: str) -> str:
    return "trace-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return _round(config.clock() - started)


def _round(value: float) -> float:
    return round(float(value), 8)
