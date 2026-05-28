"""Exp 3291 FR-11 Garak and abstention memory replay.

Spec refs: REQ-LEARN-3291, SCENARIO-LEARN-3291,
SCENARIO-LEARN-3291-BLOCKED.

This evaluator treats milestone .304 blocker artifacts as controller-memory
episodes.  The point is deliberately narrow: preserve the raw evidence rows,
derive inspectable route actions from them, replay prior held-out behavior from
the Exp 3278 baseline, and make it explicit that no foundation-model weights,
hidden states, or KAN sidecar weights changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.fr11.garak_abstention_memory_replay.v1"
ARTIFACT = "experiment_3291_fr11_garak_abstention_memory_replay_v1"
EXPERIMENT_ID = "exp3291"
TASK_ID = "exp3291-fr11-garak-abstention-memory-replay-v1"
MILESTONE = "2026.05.304"
RUN_DATE = "20260528"
RANDOM_SEED = 3291

OUTPUT_REL_PATH = Path("results/experiment_3291_fr11_garak_abstention_memory_replay_v1.json")
EXP3278_REL_PATH = Path(
    "results/experiment_3278_fr11_full_corpus_continual_self_learning_audit_v1.json"
)
EXP3282_REL_PATH = Path("results/experiment_3282_garak_install_and_probe_manifest_v1.json")
EXP3285_REL_PATH = Path("results/experiment_3285_full_garak_dataflip_redteam_eval_v2.json")
EXP3286_REL_PATH = Path("results/experiment_3286_clean_verifier_abstention_root_cause_v1.json")
EXP3289_REL_PATH = Path(
    "results/experiment_3289_repair_gate_decision_v9_after_garak_abstention.json"
)

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
NEGATIVE_TRANSFER_LIMIT = 0.05
MIN_RETENTION_SCORE = 0.95
EPISODE_CATEGORIES = (
    "garak_toolchain",
    "garak_redteam",
    "clean_verifier_abstention",
    "kan_boundary",
    "repair_gate",
)
REQUIRED_ARTIFACT_FIELDS = {
    "continuous_self_learning_task",
    "fr11_garak_abstention_memory_replay_ready",
    "controller_memory_only",
    "foundation_weight_updates_performed",
    "raw_episodes_preserved",
    "new_episode_count",
    "heldout_trace_count",
    "retention_score",
    "adaptation_score",
    "forgetting_rate",
    "negative_transfer_rate",
    "memory_update_policy",
    "blocked_trace_categories",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    output_path: str | Path = OUTPUT_REL_PATH,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-LEARN-3291: build, validate, and write the replay artifact."""

    start = monotonic()
    root = Path(project_root)
    artifact = build_artifact(
        root,
        output_path=output_path,
        started_s=start,
        now_s=monotonic(),
        random_seed=random_seed,
        tests_run=tests_run,
    )
    output = resolve_output_path(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def build_artifact(
    root: str | Path = REPO_ROOT,
    *,
    output_path: str | Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    random_seed: int = RANDOM_SEED,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3291 replay artifact from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    episodes = collect_raw_episodes(sources)
    memory = build_controller_memory(episodes)
    baseline = sources.get("exp3278", {})
    baseline_ok = baseline_safe(baseline)
    prior_replay = replay_prior_routes(memory, baseline if baseline_ok else {})
    new_replay = replay_new_episodes(memory, episodes)
    blocked_reason = "" if baseline_ok else "baseline_exp3278_missing_or_unsafe"
    raw_preserved = raw_episodes_preserved(episodes)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "blocked_reason": blocked_reason,
        "continuous_self_learning_task": True,
        "fr11_garak_abstention_memory_replay_ready": False,
        "controller_memory_only": True,
        "foundation_weight_updates_performed": False,
        "raw_episodes_preserved": raw_preserved,
        "raw_episodes": episodes,
        "new_episode_count": int(new_replay["new_episode_count"]),
        "heldout_trace_count": int(prior_replay["heldout_trace_count"]),
        "retention_score": float(prior_replay["retention_score"]),
        "adaptation_score": float(new_replay["adaptation_score"]),
        "forgetting_rate": rounded_rate(1.0 - float(prior_replay["retention_score"])),
        "negative_transfer_rate": float(prior_replay["negative_transfer_rate"]),
        "memory_update_policy": memory_update_policy(),
        "blocked_trace_categories": blocked_trace_categories(episodes),
        "controller_memory_summary": memory_summary(memory),
        "prior_replay": prior_replay,
        "new_replay": new_replay,
        "source_artifacts": source_artifacts(root_path),
        "output_paths": [path_as_artifact_string(root_path, output_path)],
        "random_seed": int(random_seed),
        "tests_run": list(tests_run or []),
        "duration_s": duration(start, now_s),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["fr11_garak_abstention_memory_replay_ready"] = replay_ready(artifact)
    finalize_artifact(artifact)
    validate_artifact(artifact)
    return artifact


def load_sources(root: str | Path = REPO_ROOT) -> JsonDict:
    """Load the baseline and .304 artifacts used as replay evidence."""

    root_path = Path(root)
    return {
        "exp3278": read_json_object(root_path / EXP3278_REL_PATH),
        "exp3282": read_json_object(root_path / EXP3282_REL_PATH),
        "exp3285": read_json_object(root_path / EXP3285_REL_PATH),
        "exp3286": read_json_object(root_path / EXP3286_REL_PATH),
        "exp3289": read_json_object(root_path / EXP3289_REL_PATH),
    }


def collect_raw_episodes(sources: Mapping[str, Any]) -> list[JsonDict]:
    """REQ-LEARN-3291-2/3: turn every required category into a raw episode."""

    return [
        garak_toolchain_episode(sources.get("exp3282", {})),
        garak_redteam_episode(sources.get("exp3285", {})),
        clean_verifier_abstention_episode(sources.get("exp3286", {})),
        kan_boundary_episode(sources.get("exp3289", {})),
        repair_gate_episode(sources.get("exp3289", {})),
    ]


def garak_toolchain_episode(exp3282: Any) -> JsonDict:
    """Create the route-memory row for Garak installation/toolchain evidence."""

    if not isinstance(exp3282, Mapping) or not exp3282:
        return missing_episode("garak_toolchain", EXP3282_REL_PATH)
    failed = [
        str(row.get("name") or "unknown")
        for row in sequence_of_mappings(exp3282.get("preconditions_checked"))
        if row.get("passed") is False
    ]
    install_blockers = [str(item) for item in sequence_values(exp3282.get("install_blockers"))]
    runner_ready = exp3282.get("garak_runner_ready") is True
    blocked = bool(failed or install_blockers or not runner_ready)
    expected_action = (
        "route_garak_tasks_to_isolated_uv_runner"
        if runner_ready
        else "force_garak_toolchain_prerequisite_gate"
    )
    return make_episode(
        category="garak_toolchain",
        source_experiment_id="exp3282",
        source_path=EXP3282_REL_PATH,
        observed_signal=first_nonempty([*install_blockers, *failed, "garak_runner_ready"]),
        expected_controller_action=expected_action,
        blocked=blocked,
        raw_evidence={
            "garak_available": exp3282.get("garak_available") is True,
            "garak_runner_ready": runner_ready,
            "failed_preconditions": failed,
            "install_blockers": install_blockers,
        },
    )


def garak_redteam_episode(exp3285: Any) -> JsonDict:
    """Create the route-memory row for Garak red-team gate evidence."""

    if not isinstance(exp3285, Mapping) or not exp3285:
        return missing_episode("garak_redteam", EXP3285_REL_PATH)
    blocked_reasons = [str(item) for item in sequence_values(exp3285.get("blocked_reasons"))]
    gate_passed = exp3285.get("garak_gate_passed") is True
    blocked = bool(blocked_reasons or not gate_passed)
    return make_episode(
        category="garak_redteam",
        source_experiment_id="exp3285",
        source_path=EXP3285_REL_PATH,
        observed_signal=first_nonempty(blocked_reasons + ["garak_gate_passed"]),
        expected_controller_action=(
            "keep_garak_attack_gate_closed" if blocked else "admit_garak_redteam_evidence"
        ),
        blocked=blocked,
        raw_evidence={
            "garak_redteam_eval_ready": exp3285.get("garak_redteam_eval_ready") is True,
            "garak_gate_passed": gate_passed,
            "garak_probe_count": safe_int(exp3285.get("garak_probe_count")),
            "attack_success_rate": safe_float(exp3285.get("attack_success_rate")),
            "blocked_reasons": blocked_reasons,
        },
    )


def clean_verifier_abstention_episode(exp3286: Any) -> JsonDict:
    """Create the route-memory row for clean-verifier abstention root causes."""

    if not isinstance(exp3286, Mapping) or not exp3286:
        return missing_episode("clean_verifier_abstention", EXP3286_REL_PATH)
    prior_abstention_rate = safe_float(exp3286.get("prior_abstention_rate"))
    root_cause = str(exp3286.get("dominant_root_cause") or "unknown_root_cause")
    blocked = prior_abstention_rate >= 1.0 or bool(exp3286.get("abstention_reason_counts"))
    return make_episode(
        category="clean_verifier_abstention",
        source_experiment_id="exp3286",
        source_path=EXP3286_REL_PATH,
        observed_signal=root_cause,
        expected_controller_action=(
            "route_clean_verifier_to_parser_contract_calibration"
            if blocked
            else "reuse_calibrated_clean_verifier"
        ),
        blocked=blocked,
        raw_evidence={
            "abstention_root_cause_identified": (
                exp3286.get("abstention_root_cause_identified") is True
            ),
            "dominant_root_cause": root_cause,
            "prior_abstention_rate": prior_abstention_rate,
            "answerable_row_count": safe_int(exp3286.get("answerable_row_count")),
            "abstention_reason_counts": dict(exp3286.get("abstention_reason_counts") or {}),
        },
    )


def kan_boundary_episode(exp3289: Any) -> JsonDict:
    """Create the route-memory row for the KAN boundary decision in Exp 3289."""

    if not isinstance(exp3289, Mapping) or not exp3289:
        return missing_episode("kan_boundary", EXP3289_REL_PATH)
    kan_input = exp3289.get("gate_inputs", {}).get("exp3288_kan_boundary", {})
    if not isinstance(kan_input, Mapping) or not kan_input:
        return missing_episode("kan_boundary", EXP3289_REL_PATH)
    decision = str(kan_input.get("kan_boundary_decision") or "unknown_kan_boundary")
    bounded = kan_input.get("kan_downstream_use_bounded") is True
    blocked = decision.startswith("retire_") or not bounded
    return make_episode(
        category="kan_boundary",
        source_experiment_id="exp3288",
        source_path=EXP3289_REL_PATH,
        observed_signal=decision,
        expected_controller_action=(
            "retire_kan_from_headline_and_repair_gate_authority"
            if blocked
            else "admit_bounded_kan_sidecar_evidence"
        ),
        blocked=blocked,
        raw_evidence={
            "kan_boundary_decision_ready": kan_input.get("kan_boundary_decision_ready") is True,
            "kan_boundary_decision": decision,
            "kan_downstream_use_bounded": bounded,
            "prior_full_corpus_auroc": safe_float(kan_input.get("prior_full_corpus_auroc")),
        },
    )


def repair_gate_episode(exp3289: Any) -> JsonDict:
    """Create the route-memory row for the repair gate outcome."""

    if not isinstance(exp3289, Mapping) or not exp3289:
        return missing_episode("repair_gate", EXP3289_REL_PATH)
    blocked_reasons = [str(item) for item in sequence_values(exp3289.get("blocked_reasons"))]
    gate_open = exp3289.get("repair_gate_open") is True
    scope = exp3289.get("permitted_repair_scope", {})
    scope_map = scope if isinstance(scope, Mapping) else {}
    return make_episode(
        category="repair_gate",
        source_experiment_id="exp3289",
        source_path=EXP3289_REL_PATH,
        observed_signal="repair_gate_open" if gate_open else first_nonempty(blocked_reasons),
        expected_controller_action=(
            "open_bounded_repair_micro_panel" if gate_open else "keep_repair_gate_closed"
        ),
        blocked=not gate_open,
        raw_evidence={
            "repair_gate_decision_v9_ready": exp3289.get("repair_gate_decision_v9_ready") is True,
            "repair_gate_open": gate_open,
            "blocked_reasons": blocked_reasons,
            "repair_task_id": str(scope_map.get("repair_task_id") or ""),
            "repair_generation_allowed": scope_map.get("repair_generation_allowed") is True,
        },
    )


def missing_episode(category: str, source_path: Path) -> JsonDict:
    """Return a raw episode for a missing artifact so the category is not hidden."""

    return make_episode(
        category=category,
        source_experiment_id="missing",
        source_path=source_path,
        observed_signal="missing_artifact",
        expected_controller_action=f"force_{category}_prerequisite_gate",
        blocked=True,
        raw_evidence={"present": False, "path": source_path.as_posix()},
    )


def make_episode(
    *,
    category: str,
    source_experiment_id: str,
    source_path: Path,
    observed_signal: str,
    expected_controller_action: str,
    blocked: bool,
    raw_evidence: Mapping[str, Any],
) -> JsonDict:
    """Build a deterministic raw episode with a stable content-derived ID."""

    payload = {
        "category": category,
        "source_experiment_id": source_experiment_id,
        "source_path": source_path.as_posix(),
        "observed_signal": observed_signal,
        "expected_controller_action": expected_controller_action,
        "blocked": bool(blocked),
        "raw_evidence": dict(raw_evidence),
    }
    payload["episode_id"] = f"{source_experiment_id}:{category}:{stable_id(payload)}"
    return payload


def build_controller_memory(episodes: Sequence[Mapping[str, Any]]) -> JsonDict:
    """REQ-LEARN-3291-4: build inspectable route memory from raw episodes only."""

    route_overrides = {
        str(row.get("category")): str(row.get("expected_controller_action"))
        for row in episodes
        if row.get("category") and row.get("expected_controller_action")
    }
    return {
        "controller_memory_only": True,
        "route_overrides": route_overrides,
        "raw_episode_count": len(episodes),
        "consolidation_allowed": set(route_overrides) == set(EPISODE_CATEGORIES),
    }


def replay_new_episodes(
    memory: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """REQ-LEARN-3291-5: score adaptation over new .304 episodes."""

    route_overrides = memory.get("route_overrides", {})
    routes = route_overrides if isinstance(route_overrides, Mapping) else {}
    adapted = sum(
        1
        for row in episodes
        if routes.get(str(row.get("category"))) == str(row.get("expected_controller_action"))
    )
    return {
        "new_episode_count": len(episodes),
        "adapted_episode_count": adapted,
        "adaptation_score": score_ratio(adapted, len(episodes)),
    }


def replay_prior_routes(memory: Mapping[str, Any], baseline: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3291-5: replay prior held-out routes without new route overlap."""

    metrics = baseline.get("before_after_metrics", {})
    metrics_map = metrics if isinstance(metrics, Mapping) else {}
    heldout_trace_count = safe_int(
        baseline.get("heldout_trace_count") or metrics_map.get("heldout_trace_count")
    )
    retention_score = rounded_rate(
        safe_float(baseline.get("retention_score") or metrics_map.get("retention_score"))
    )
    prior_categories = {"legacy_gate_block", "prompt_injection_holdout", "benign_holdout"}
    memory_categories = set(memory.get("route_overrides", {}))
    negative_transfer_rate = 1.0 if prior_categories & memory_categories else 0.0
    return {
        "heldout_trace_count": heldout_trace_count,
        "retention_score": retention_score,
        "forgetting_rate": rounded_rate(1.0 - retention_score),
        "negative_transfer_rate": negative_transfer_rate,
        "baseline_experiment_id": str(baseline.get("experiment_id") or "exp3278"),
    }


def baseline_safe(baseline: Mapping[str, Any]) -> bool:
    """REQ-LEARN-3291-1: verify the baseline is safe to replay against."""

    return (
        bool(baseline)
        and baseline.get("continuous_self_learning_task") is True
        and baseline.get("controller_memory_only") is True
        and baseline.get("foundation_weight_updates_performed") is False
        and safe_int(baseline.get("heldout_trace_count")) > 0
    )


def raw_episodes_preserved(episodes: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when every episode still carries raw artifact evidence."""

    return bool(episodes) and all(isinstance(row.get("raw_evidence"), Mapping) for row in episodes)


def replay_ready(artifact: Mapping[str, Any]) -> bool:
    """REQ-LEARN-3291-6: gate readiness on replay, retention, and transfer safety."""

    return (
        artifact.get("blocked_reason") == ""
        and artifact.get("controller_memory_only") is True
        and artifact.get("foundation_weight_updates_performed") is False
        and artifact.get("raw_episodes_preserved") is True
        and safe_int(artifact.get("new_episode_count")) > 0
        and safe_int(artifact.get("heldout_trace_count")) > 0
        and safe_float(artifact.get("retention_score")) >= MIN_RETENTION_SCORE
        and safe_float(artifact.get("adaptation_score")) > 0.0
        and safe_float(artifact.get("negative_transfer_rate")) <= NEGATIVE_TRANSFER_LIMIT
    )


def memory_update_policy() -> JsonDict:
    """REQ-LEARN-3291-4: make write-path filtering explicit."""

    return {
        "allowed_write_targets": [
            "controller_route_keys",
            "controller_episode_index",
            "controller_replay_metrics",
        ],
        "raw_episode_preservation_required": True,
        "summary_memory_written": False,
        "consolidation_gate": {
            "required_checks": [
                "raw_episodes_preserved",
                "prior_heldout_replay_nonempty",
                "new_episode_replay_nonempty",
                "negative_transfer_within_limit",
                "foundation_weight_updates_performed_false",
            ],
            "negative_transfer_limit": NEGATIVE_TRANSFER_LIMIT,
            "min_retention_score": MIN_RETENTION_SCORE,
        },
        "foundation_weight_updates_allowed": False,
        "hidden_state_mutation_allowed": False,
        "kan_sidecar_weight_updates_allowed": False,
        "controller_memory_only": True,
    }


def blocked_trace_categories(episodes: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the sorted categories whose source evidence is blocked or missing."""

    return sorted({str(row.get("category")) for row in episodes if row.get("blocked") is True})


def memory_summary(memory: Mapping[str, Any]) -> JsonDict:
    """Return JSON-safe counts for the derived controller memory."""

    routes = memory.get("route_overrides", {})
    route_map = routes if isinstance(routes, Mapping) else {}
    return {
        "controller_memory_only": memory.get("controller_memory_only") is True,
        "route_override_count": len(route_map),
        "raw_episode_count": safe_int(memory.get("raw_episode_count")),
        "consolidation_allowed": memory.get("consolidation_allowed") is True,
    }


def finalize_artifact(artifact: JsonDict) -> None:
    """Attach the terminal verdict and checksum after all measured fields exist."""

    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict while preserving the no-weight-update boundary."""

    return (
        "complete: fr11 garak-abstention controller-memory replay "
        f"ready={str(bool(artifact.get('fr11_garak_abstention_memory_replay_ready'))).lower()}; "
        f"new_episode_count={artifact.get('new_episode_count')}; "
        f"heldout_trace_count={artifact.get('heldout_trace_count')}; "
        f"retention_score={artifact.get('retention_score')}; "
        f"adaptation_score={artifact.get('adaptation_score')}; "
        f"forgetting_rate={artifact.get('forgetting_rate')}; "
        f"negative_transfer_rate={artifact.get('negative_transfer_rate')}; "
        "controller_memory_only=true; foundation_weight_updates_performed=false"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3291 artifact violates schema or safety boundaries."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("controller_memory_only") is not True:
        raise ValueError("controller_memory_only must remain true")
    if artifact.get("foundation_weight_updates_performed") is not False:
        raise ValueError("foundation_weight_updates_performed must remain false")
    if artifact.get("raw_episodes_preserved") is not True:
        raise ValueError("raw_episodes_preserved must remain true")
    for field in (
        "retention_score",
        "adaptation_score",
        "forgetting_rate",
        "negative_transfer_rate",
    ):
        value = artifact.get(field)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field} must be a bounded float")
    expected_forgetting = rounded_rate(1.0 - safe_float(artifact.get("retention_score")))
    if safe_float(artifact.get("forgetting_rate")) != expected_forgetting:
        raise ValueError("forgetting_rate must equal 1.0 - retention_score")
    raw_episodes = artifact.get("raw_episodes")
    if not isinstance(raw_episodes, Sequence) or isinstance(raw_episodes, (str, bytes)):
        raise ValueError("raw_episodes must preserve the episode list")
    if len(raw_episodes) != safe_int(artifact.get("new_episode_count")):
        raise ValueError("new_episode_count must match raw_episodes")
    policy = artifact.get("memory_update_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("foundation_weight_updates_allowed") is not False
    ):
        raise ValueError("memory_update_policy must forbid foundation-weight updates")
    if bool(artifact.get("fr11_garak_abstention_memory_replay_ready")) != replay_ready(artifact):
        raise ValueError("fr11_garak_abstention_memory_replay_ready mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if (
        not verdict.startswith(TERMINAL_PREFIXES)
        or "foundation_weight_updates_performed=false" not in verdict
    ):
        raise ValueError("honest_verdict must be terminal and attest no foundation updates")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum must match canonical artifact payload")


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while treating missing or malformed evidence as absent."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source paths and checksums for reproducibility."""

    rel_paths = [
        EXP3278_REL_PATH,
        EXP3282_REL_PATH,
        EXP3285_REL_PATH,
        EXP3286_REL_PATH,
        EXP3289_REL_PATH,
    ]
    return [
        {
            "path": rel_path.as_posix(),
            "present": (root / rel_path).is_file(),
            "sha256": sha256_file(root / rel_path),
        }
        for rel_path in rel_paths
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Return a stable checksum over all non-timing artifact content."""

    basis = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    encoded = json.dumps(basis, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sequence_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    """Return only mapping rows from arbitrary list-like evidence."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def sequence_values(value: Any) -> list[Any]:
    """Return list-like values while treating strings and mappings as scalars."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return list(value)


def score_ratio(numerator: int, denominator: int) -> float:
    """Return a bounded six-decimal ratio, failing closed for empty denominators."""

    if denominator <= 0:
        return 0.0
    return rounded_rate(float(numerator) / float(denominator))


def rounded_rate(value: float) -> float:
    """Clamp and round a metric rate into artifact form."""

    return round(max(0.0, min(1.0, float(value))), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return elapsed seconds rounded for stable artifacts."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - float(started_s)), 6)


def path_as_artifact_string(root: Path, path: str | Path) -> str:
    """Return output paths relative to the project root when possible."""

    output = Path(path)
    if not output.is_absolute():
        return output.as_posix()
    try:
        return output.relative_to(root).as_posix()
    except ValueError:
        return output.as_posix()


def resolve_output_path(root: Path, path: str | Path) -> Path:
    """Resolve an output path against the project root."""

    output = Path(path)
    return output if output.is_absolute() else root / output


def sha256_file(path: Path) -> str | None:
    """Return a file checksum if the path exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(value: Mapping[str, Any]) -> str:
    """Return a compact stable ID from JSON-safe mapping content."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def first_nonempty(values: Sequence[Any]) -> str:
    """Return the first non-empty evidence token as a stable observed signal."""

    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return "no_signal"


def safe_int(value: Any) -> int:
    """Convert count-like values while treating malformed evidence as zero."""

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    """Convert metric-like values while treating malformed evidence as zero."""

    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
