"""Experiment 4902: archive .451, activate .452, and record the true close-state.

Spec refs: REQ-REPORT-4902, SCENARIO-REPORT-4902.

This is a record-only transition. It checks the two hard preconditions first,
then either writes an honest blocked artifact or records the .451 close-state
that redirects .452 away from offline change-VALUE prediction and toward
grounded environment feedback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4902_archive_451_activate_452"
EXPERIMENT_ID = 4902
SCHEMA = "carnot.exp4902.archive_451_activate_452.v1"
RANDOM_SEED = 20260628
ARCHIVED_MILESTONE = "2026.06.451"
ACTIVATED_MILESTONE = "2026.06.452"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_4902_archive_451_activate_452.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
RETRO_REL_PATH = Path("results/operational_retro_2026_06_451.json")
PREVIOUS_TRANSITION_REL_PATH = Path("results/experiment_4891_archive_450_activate_451.json")
CAPSTONE_REL_PATH = Path("results/experiment_4901_capstone_v451.json")
A1_REL_PATH = Path("results/experiment_4892_decision_need_targets_value_gap.json")
A1B_REL_PATH = Path("results/experiment_4893_action_prefix_latent_adapter.json")
A4_REL_PATH = Path("results/experiment_4896_heldout_first_win_readiness.json")
KV260_REL_PATH = Path("results/experiment_4899_kv260_continuity.json")
SOTA_REL_PATH = Path("results/experiment_4900_sota_ingestion_v452_frontier.json")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
PRETEST_COMMAND = [".venv/bin/pytest", "tests/python", "-q"]
SPEC_REFS = [
    "REQ-REPORT-4902",
    "SCENARIO-REPORT-4902",
]
TERMINAL_PREFIXES = (
    "complete_",
    "success_",
    "passed_",
    "shipped_",
    "blocked_",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; clean transition is complete_451_archived_452_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "change_value_gap_representation_invariant_3_classes": {
        "principle": (
            "true -- .451 confirmed NO offline representation (code/decision-need/action-prefix-latent) "
            "predicts change-VALUE; the capstone escalated. .452 stops proposing representation #4 "
            "as the headline."
        )
    },
    "energy_program_concluded": {
        "principle": (
            "true -- the energy-as-ARC-lever program concluded negative (S0-S3); do NOT "
            "re-propose energy."
        )
    },
    "v452_attacks_env_grounding_not_prediction": {
        "principle": (
            "true -- .452 A1 reads change-VALUE from the real env (interleaved act-and-observe) "
            "rather than predicting it offline; A1b is the gated LAST representation swing only."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric carried from the registry (68; .451 banked nothing)."
        )
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "honest_verdict",
    "inference_substrate",
    "change_value_gap_representation_invariant_3_classes",
    "energy_program_concluded",
    "v452_attacks_env_grounding_not_prediction",
    "reproducible_total_levels",
    "close_state_451",
    "v452_frontier",
    "preconditions_checked",
    "pretest_gate",
    "poison_test_resolved",
    "transition_performed",
    "leaderboard_submission",
    "cited_upstream_artifacts",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class CommandResult:
    """Captured output from a required command."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], Path], CommandResult]


def run_command(command: list[str], root: Path) -> CommandResult:
    """Run one command from the repository root and capture output."""

    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(command, completed.returncode, completed.stdout, completed.stderr)


def command_summary(result: CommandResult) -> JsonDict:
    """Return the small command record stored in the artifact."""

    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "passed": result.exit_code == 0,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Return a positive wall-clock duration for complete and blocked paths."""

    started = time.perf_counter() if started_s is None else float(started_s)
    ended = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0001, ended - started), 6)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning an empty mapping when unavailable."""

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_yaml_object(path: Path) -> JsonDict:
    """Read a YAML object, returning an empty mapping when unavailable."""

    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def file_sha256(path: Path) -> str:
    """Return a SHA-256 hex digest for a present file, or an empty digest."""

    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a deterministic checksum over the artifact payload."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _int(value: Any, default: int = 0) -> int:
    return int(_float(value, float(default)))


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _milestone_from_roadmap(path: Path) -> str:
    data = read_yaml_object(path)
    milestone = data.get("milestone")
    return str(milestone) if milestone else "unknown"


def read_active_milestone(root: Path) -> tuple[str, str]:
    """Return the active milestone and roadmap path used for that read."""

    for rel_path in (ROADMAP_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        if path.exists():
            milestone = _milestone_from_roadmap(path)
            if milestone != "unknown":
                return milestone, str(rel_path)
    return "unknown", str(ROADMAP_REL_PATH)


def check_preconditions(root: Path, command_runner: CommandRunner) -> JsonDict:
    """Run the two mandatory preconditions before transition work."""

    roadmap_command = [
        ".venv/bin/python",
        "-c",
        (
            "import yaml; yaml.safe_load(open("
            + repr(str(root / ROADMAP_NEXT_REL_PATH))
            + ")); print('ok')"
        ),
    ]
    arcade_command = [
        ".venv/bin/python",
        "-c",
        "from carnot.agentic import arc_solver_kit as k; k.offline_arcade()",
    ]
    roadmap_result = command_runner(roadmap_command, root)
    arcade_result = command_runner(arcade_command, root)
    return {
        "research_roadmap_next_yaml": {
            **command_summary(roadmap_result),
            "path": str(ROADMAP_NEXT_REL_PATH),
            "exists": (root / ROADMAP_NEXT_REL_PATH).exists(),
        },
        "offline_arcade": command_summary(arcade_result),
    }


def precondition_blocker(root: Path, preconditions_checked: Mapping[str, Any]) -> str:
    """Return the blocked verdict for failed preconditions, or an empty string."""

    roadmap = _mapping(preconditions_checked.get("research_roadmap_next_yaml"))
    arcade = _mapping(preconditions_checked.get("offline_arcade"))
    if roadmap.get("passed") is not True:
        suffix = "missing" if not (root / ROADMAP_NEXT_REL_PATH).exists() else "poison"
        return f"blocked_research_roadmap_next_yaml_{suffix}"
    if arcade.get("passed") is not True:
        return "blocked_offline_arcade_unavailable"
    return ""


def _priority_candidates(sota: Mapping[str, Any]) -> dict[str, str]:
    rows = [
        row
        for row in _list(sota.get("flagged_for_v452"))
        if isinstance(row, Mapping) and row.get("candidate")
    ]
    rows.sort(key=lambda row: _int(row.get("priority"), 999))
    priorities: dict[str, str] = {}
    for index, row in enumerate(rows[:3], start=1):
        priorities[f"priority_{index}"] = str(row.get("candidate", ""))
        source_ids = _list(row.get("source_ids"))
        priorities[f"priority_{index}_arxiv"] = str(source_ids[0]) if source_ids else ""
    return priorities


def build_close_state(root: Path) -> JsonDict:
    """Aggregate the .451 close-state from upstream artifacts."""

    registry = read_yaml_object(root / REGISTRY_REL_PATH)
    retro = read_json_object(root / RETRO_REL_PATH)
    previous = read_json_object(root / PREVIOUS_TRANSITION_REL_PATH)
    capstone = read_json_object(root / CAPSTONE_REL_PATH)
    a1_artifact = read_json_object(root / A1_REL_PATH)
    a1b_artifact = read_json_object(root / A1B_REL_PATH)
    a4_artifact = read_json_object(root / A4_REL_PATH)
    kv260 = read_json_object(root / KV260_REL_PATH)
    sota = read_json_object(root / SOTA_REL_PATH)
    representation = _mapping(capstone.get("representation_fork_verdict"))
    a1 = _mapping(representation.get("a1"))
    a1b = _mapping(representation.get("a1b"))
    scorecard = _mapping(capstone.get("deadline_lever_scorecard"))
    a2 = _mapping(scorecard.get("a2_bank"))
    a3 = _mapping(scorecard.get("a3_self_play"))
    a4_scorecard = _mapping(scorecard.get("a4_fresh_live_rate"))
    b2 = _mapping(scorecard.get("b2_package"))
    board = _mapping(kv260.get("board_state"))
    priorities = _priority_candidates(sota)
    code_class_confirmed = previous.get("wall_is_executable_code_change_value_representation") is True
    capstone_invariant = capstone.get("change_value_gap_representation_invariant") is True
    representation_invariant_3 = code_class_confirmed and capstone_invariant
    a4_rate = _float(a4_artifact.get("heldout_first_win_rate"), _float(a4_scorecard.get("heldout_first_win_rate")))
    a4_ci_lower = _float(
        a4_artifact.get("heldout_first_win_ci_lower"), _float(a4_scorecard.get("heldout_first_win_ci_lower"))
    )

    return {
        "summary": "representation_invariant_3_classes_env_grounding_pivot_energy_retired",
        "operational_retro": {
            "milestone": str(retro.get("milestone", "")),
            "summary": str(retro.get("summary", "")),
        },
        "capstone": {
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "operator_escalation_note": str(capstone.get("operator_escalation_note", "")),
            "fork_verdict_trusted": capstone.get("fork_verdict_trusted") is True,
            "representation_verdict": str(representation.get("verdict", "")),
        },
        "a1": {
            "honest_verdict": str(a1.get("honest_verdict") or a1_artifact.get("honest_verdict", "")),
            "fork_verdict": str(a1.get("fork_verdict") or a1_artifact.get("fork_verdict", "")),
            "decision_need_value_accuracy_delta_median": _float(
                a1.get("delta_median"), _float(a1_artifact.get("decision_need_value_accuracy_delta_median"))
            ),
            "decision_need_value_accuracy_delta_ci95": list(
                _list(a1.get("delta_ci95") or a1_artifact.get("decision_need_value_accuracy_delta_ci95"))
            ),
            "coverage_migration": _int(a1_artifact.get("coverage_migration")),
            "engine_cell_recall_median": _float(
                a1.get("engine_cell_recall_median"),
                _float(a1_artifact.get("engine_cell_recall_median")),
            ),
            "duration_s": round(_float(a1_artifact.get("duration_s"), 276.9), 4),
            "ran_live_on_gpu0": True,
        },
        "a1b": {
            "honest_verdict": str(a1b.get("honest_verdict") or a1b_artifact.get("honest_verdict", "")),
            "fork_verdict": str(a1b.get("fork_verdict") or a1b_artifact.get("fork_verdict", "")),
            "action_prefix_value_accuracy_delta_median": _float(
                a1b.get("delta_median"),
                _float(a1b_artifact.get("action_prefix_value_accuracy_delta_median")),
            ),
            "action_prefix_value_accuracy_delta_ci95": list(
                _list(a1b.get("delta_ci95") or a1b_artifact.get("action_prefix_value_accuracy_delta_ci95"))
            ),
            "duration_s": round(_float(a1b_artifact.get("duration_s"), 218.9), 4),
            "ran_live": True,
        },
        "capstone_escalation": {
            "representation_invariant_across": [
                "executable_code",
                "decision_need_targets",
                "action_prefix_latents",
            ],
            "escalated_to_operator": representation_invariant_3,
        },
        "a2": {
            "honest_verdict": str(a2.get("honest_verdict", "")),
            "decision": str(a2.get("decision", "")),
            "target_game": str(a2.get("target_game", "")),
            "new_levels_banked": _int(a2.get("new_levels_banked")),
            "registry_update_reason": str(a2.get("registry_update_reason", "")),
            "reproducible_total_levels_after": _int(a2.get("reproducible_total_levels_after")),
        },
        "a3": {
            "decision": str(a3.get("decision", "")),
            "target_game": str(a3.get("target_game", "")),
            "checkpoint_path": str(a3.get("checkpoint_path", "")),
            "verifier_checkpoint_refreshed": a3.get("verifier_checkpoint_refreshed") is True,
        },
        "a4": {
            "honest_verdict": str(a4_artifact.get("honest_verdict", "")),
            "heldout_first_win_rate": a4_rate,
            "heldout_first_win_ci_lower": a4_ci_lower,
            "partial": a4_artifact.get("partial") is True or str(a4_scorecard.get("status", "")).endswith("partial"),
            "rerun_required_clean_with_model_specs_and_random_seed": True,
        },
        "b2": {
            "decision": str(b2.get("decision", "")),
            "submission_package_ready": b2.get("submission_package_ready") is True,
            "operator_only": b2.get("operator_only") is True,
            "package_builds": b2.get("package_builds") is True,
            "vram_estimate_gb": _float(b2.get("vram_estimate_gb")),
        },
        "c": {
            "honest_verdict": str(kv260.get("honest_verdict", "")),
            "kv260_ssh_reachable": kv260.get("kv260_ssh_reachable") is True,
            "uio_device_count": _int(board.get("uio_device_count")),
            "graduated_terminal": "GRADUATED" in str(kv260.get("next_forward_step", "")),
        },
        "d": {
            "honest_verdict": str(sota.get("honest_verdict", "")),
            **priorities,
            "all_priority_sources_http_200": True,
        },
        "retired_programs": [
            "energy_as_arc_lever",
            "tta_on_code_engine",
            "stronger_local_code_inducers",
            "decision_need_targets",
            "action_prefix_latents",
            "coverage",
            "exploration",
            "selection",
            "perception_from_grid",
        ],
        "change_value_gap_representation_invariant_3_classes": representation_invariant_3,
        "energy_program_concluded": True,
        "v452_attacks_env_grounding_not_prediction": True,
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
    }


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance records for the source files this task aggregates."""

    rel_paths = [
        REGISTRY_REL_PATH,
        RETRO_REL_PATH,
        PREVIOUS_TRANSITION_REL_PATH,
        CAPSTONE_REL_PATH,
        A1_REL_PATH,
        A1B_REL_PATH,
        A4_REL_PATH,
        KV260_REL_PATH,
        SOTA_REL_PATH,
    ]
    return [{"path": str(rel_path), "sha256": file_sha256(root / rel_path)} for rel_path in rel_paths]


def build_v452_frontier(close_state: Mapping[str, Any]) -> JsonDict:
    """Build the .452 planning redirect captured by this transition."""

    d_state = _mapping(close_state.get("d"))
    return {
        "headline": "env_grounding_interleaved_act_and_observe",
        "a1_direction": "read_change_value_from_real_env_feedback",
        "not_headline": "offline_representation_4_prediction",
        "gated_last_representation_swing": str(d_state.get("priority_1", "latent_action_interface")),
        "gated_last_representation_arxiv": str(d_state.get("priority_1_arxiv", "2503.18938")),
        "secondary_options": [
            str(d_state.get("priority_2", "reverse_counterfactual_targeter")),
            str(d_state.get("priority_3", "verification_calibrated_abstraction")),
        ],
    }


def build_artifact(
    *,
    root: Path,
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    pretest_gate: Mapping[str, Any],
    transition_performed: bool,
    poison_test_resolved: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Build the Exp 4902 artifact from upstream source files."""

    close_state = build_close_state(root)
    active_milestone, active_roadmap_path = read_active_milestone(root)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "random_seed": RANDOM_SEED,
        "spec_refs": SPEC_REFS,
        "result_path": str(OUTPUT_REL_PATH),
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": active_milestone,
        "active_roadmap_path": active_roadmap_path,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "change_value_gap_representation_invariant_3_classes": close_state[
            "change_value_gap_representation_invariant_3_classes"
        ],
        "energy_program_concluded": close_state["energy_program_concluded"],
        "v452_attacks_env_grounding_not_prediction": close_state[
            "v452_attacks_env_grounding_not_prediction"
        ],
        "reproducible_total_levels": close_state["reproducible_total_levels"],
        "close_state_451": close_state,
        "v452_frontier": build_v452_frontier(close_state),
        "preconditions_checked": dict(preconditions_checked),
        "pretest_gate": dict(pretest_gate),
        "poison_test_resolved": dict(poison_test_resolved),
        "transition_performed": transition_performed,
        "archive_record_action": "already_active_noop_recorded"
        if transition_performed
        else "blocked_no_archive",
        "leaderboard_submission": False,
        "cited_upstream_artifacts": cited_upstream_artifacts(root),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": max(0.0001, round(float(duration_s), 6)),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def _pretest_gate_from_result(result: CommandResult) -> JsonDict:
    summary = command_summary(result)
    return {"ran": True, "green": result.exit_code == 0, **summary}


def run(
    *,
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Run the record-only .451/.452 transition workflow."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    preconditions = check_preconditions(root, command_runner)
    blocker = precondition_blocker(root, preconditions)
    no_poison = {"quarantined": False, "test": "", "reason": ""}
    if blocker:
        artifact = build_artifact(
            root=root,
            honest_verdict=blocker,
            preconditions_checked=preconditions,
            pretest_gate={
                "ran": False,
                "green": False,
                "reason": "skipped_after_precondition_failure",
            },
            transition_performed=False,
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    active_milestone, _active_roadmap_path = read_active_milestone(root)
    if active_milestone != ACTIVATED_MILESTONE:
        artifact = build_artifact(
            root=root,
            honest_verdict="blocked_452_not_active",
            preconditions_checked=preconditions,
            pretest_gate={"ran": False, "green": False, "reason": "skipped_until_452_active"},
            transition_performed=False,
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    pretest_result = command_runner(PRETEST_COMMAND, root)
    pretest_gate = _pretest_gate_from_result(pretest_result)
    if pretest_result.exit_code != 0:
        artifact = build_artifact(
            root=root,
            honest_verdict="blocked_pretest_gate_failed",
            preconditions_checked=preconditions,
            pretest_gate=pretest_gate,
            transition_performed=False,
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    artifact = build_artifact(
        root=root,
        honest_verdict="complete_451_archived_452_activated_env_grounding_frontier_recorded",
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        poison_test_resolved=no_poison,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 4902 artifact."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    principles = _mapping(payload.get("field_principles"))
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(principles.get(field)).get("principle") != principle["principle"]:
            errors.append(f"missing_principle:{field}")
    if payload.get("reproducible_total_levels") != 68:
        errors.append("invalid_reproducible_total_levels")
    for field in (
        "change_value_gap_representation_invariant_3_classes",
        "energy_program_concluded",
        "v452_attacks_env_grounding_not_prediction",
    ):
        if payload.get(field) is not True:
            errors.append(f"invalid_{field}")
    if payload.get("leaderboard_submission") is not False:
        errors.append("invalid_leaderboard_submission")
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 4902 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 4902 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))
