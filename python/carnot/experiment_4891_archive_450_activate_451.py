"""Experiment 4891: archive .450, activate .451, and record the value wall.

Spec refs: REQ-REPORT-4891, SCENARIO-REPORT-4891.

This is a record-only transition. The close-state matters more than the
mechanical archive action: Exp 4882 made the executable-code change-VALUE wall
trustworthy, Exp 4883 did not establish METHOD_IS_CEILING because it was
fabrication-flagged, and .451 must test alternative world-model
representations instead of re-proposing energy or same-code-engine stages.
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
EXPERIMENT = "experiment_4891_archive_450_activate_451"
EXPERIMENT_ID = 4891
SCHEMA = "carnot.exp4891.archive_450_activate_451.v1"
RANDOM_SEED = 20260628
ARCHIVED_MILESTONE = "2026.06.450"
ACTIVATED_MILESTONE = "2026.06.451"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_4891_archive_450_activate_451.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
RETRO_REL_PATH = Path("results/operational_retro_2026_06_450.json")
A1_REL_PATH = Path("results/experiment_4882_ttt_dynamics_value_gap.json")
A1B_REL_PATH = Path("results/experiment_4883_inducer_ceiling_ab.json")
AUDIT_REL_PATH = Path("results/experiment_4887_value_gap_inducer_audit.json")
A2_REL_PATH = Path("results/experiment_4884_levelup_attempt.json")
A3_REL_PATH = Path("results/experiment_4885_self_play_verifier_checkpoint.json")
A4_REL_PATH = Path("results/experiment_4886_heldout_first_win_readiness.json")
B2_REL_PATH = Path("results/experiment_4888_submission_package_harden.json")
C_REL_PATH = Path("results/experiment_4889_kv260_continuity.json")
D_REL_PATH = Path("results/experiment_4890_sota_ingestion_v451_frontier.json")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
PRETEST_COMMAND = [".venv/bin/pytest", "tests/python", "-q"]
SPEC_REFS = [
    "REQ-REPORT-4891",
    "SCENARIO-REPORT-4891",
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
        "principle": "terminal prefix; clean transition is complete_450_archived_451_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "a1_inducer_ceiling_hard_trustworthy": {
        "principle": (
            "true -- the .450 A1 fork (INDUCER_CEILING_HARD) was B1-audited genuinely "
            "diagnostic (non-degenerate graded control); the executable-code engine learns "
            "change-LOCATION but not change-VALUE."
        )
    },
    "a1b_was_fabrication_flagged_non_test": {
        "principle": (
            "true -- the .450 A1b inducer A/B was DURATION_TOO_SHORT (13.7s); the "
            "METHOD_IS_CEILING attribution is NOT established and .451 tests it via "
            "alternative representations."
        )
    },
    "wall_is_executable_code_change_value_representation": {
        "principle": (
            "true -- three attempts on the code representation (free-form, structured, TTA) "
            "all hit the same change-VALUE ceiling; .451 changes the representation class."
        )
    },
    "energy_program_concluded": {
        "principle": (
            "true -- the energy-as-ARC-lever program is concluded (negative); do NOT "
            "re-propose energy stages."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric carried from the registry (68 after "
            "g50t L2), not re-counted."
        )
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "honest_verdict",
    "inference_substrate",
    "a1_inducer_ceiling_hard_trustworthy",
    "a1b_was_fabrication_flagged_non_test",
    "wall_is_executable_code_change_value_representation",
    "energy_program_concluded",
    "reproducible_total_levels",
    "a450_close_state",
    "preconditions_checked",
    "pretest_gate",
    "transition_performed",
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
    """Run one command from the repository root and capture its output."""

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
    """Return a positive wall-clock duration for both complete and blocked paths."""

    started = time.perf_counter() if started_s is None else float(started_s)
    ended = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0001, ended - started), 6)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning an empty mapping when the file is absent."""

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_yaml_object(path: Path) -> JsonDict:
    """Read a YAML object, returning an empty mapping when the file is absent."""

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


def _milestone_from_text(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def read_active_milestone(root: Path) -> tuple[str, str]:
    """Return the active milestone and roadmap path used for that read."""

    for rel_path in (ROADMAP_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        if path.exists():
            milestone = _milestone_from_text(path.read_text(encoding="utf-8"))
            if milestone != "unknown":
                return milestone, str(rel_path)
    return "unknown", str(ROADMAP_REL_PATH)


def check_preconditions(root: Path, command_runner: CommandRunner) -> JsonDict:
    """Run the two mandatory preconditions before any transition work."""

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


def _energy_program_concluded(root: Path, retro: Mapping[str, Any]) -> bool:
    texts = [str(retro.get("summary", ""))]
    for rel_path in (ROADMAP_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        texts.append(path.read_text(encoding="utf-8") if path.exists() else "")
    joined = "\n".join(texts)
    return (
        "Energy CONCLUDED" in joined
        or "Energy-as-ARC-lever program CONCLUDED" in joined
        or "energy-as-ARC-lever program is concluded" in joined
    )


def _duration_too_short(a1b: Mapping[str, Any], audit: Mapping[str, Any]) -> bool:
    flags = _list(_mapping(audit.get("a1b_adversarial_result")).get("flags"))
    kinds = [str(_mapping(flag).get("kind", "")) for flag in flags]
    reasons = [str(reason) for reason in _list(audit.get("a1b_failure_reasons"))]
    return (
        "DURATION_TOO_SHORT" in kinds
        or "a1b_duration_below_live_floor" in reasons
        or (_float(a1b.get("duration_s")) < 60.0 and a1b.get("flagged_adversarial") is True)
    )


def _priority_candidates(sota: Mapping[str, Any]) -> dict[str, str]:
    rows = [
        row
        for row in _list(sota.get("flagged_for_v451"))
        if isinstance(row, Mapping) and row.get("candidate")
    ]
    rows.sort(key=lambda row: _int(row.get("priority"), 999))
    return {
        f"priority_{index}": str(row.get("candidate", ""))
        for index, row in enumerate(rows[:3], start=1)
    }


def build_close_state(root: Path) -> JsonDict:
    """Aggregate the .450 close-state from upstream JSON and registry files."""

    registry = read_yaml_object(root / REGISTRY_REL_PATH)
    retro = read_json_object(root / RETRO_REL_PATH)
    a1 = read_json_object(root / A1_REL_PATH)
    a1b = read_json_object(root / A1B_REL_PATH)
    audit = read_json_object(root / AUDIT_REL_PATH)
    a2 = read_json_object(root / A2_REL_PATH)
    a3 = read_json_object(root / A3_REL_PATH)
    a4 = read_json_object(root / A4_REL_PATH)
    b2 = read_json_object(root / B2_REL_PATH)
    c = read_json_object(root / C_REL_PATH)
    d = read_json_object(root / D_REL_PATH)
    c_board = _mapping(c.get("board_state"))
    a1_trustworthy = (
        a1.get("honest_verdict") == "complete_ttt_dynamics_no_value_lift_INDUCER_CEILING_HARD"
        and a1.get("fork_verdict") == "INDUCER_CEILING_HARD"
        and audit.get("a1_genuinely_diagnostic") is True
        and audit.get("a1_positive_control_non_degenerate_confirmed") is True
        and a1.get("positive_control_non_degenerate") is True
        and _float(a1.get("engine_cell_recall_median")) > 0.0
        and _float(a1.get("tta_changed_cell_value_accuracy_delta_median")) <= 0.0
    )
    duration_too_short = _duration_too_short(a1b, audit)
    a1b_non_test = (
        a1b.get("flagged_adversarial") is True
        and audit.get("a1b_ab_trustworthy") is False
        and duration_too_short
    )
    energy_concluded = _energy_program_concluded(root, retro)
    priority = _priority_candidates(d)

    return {
        "summary": (
            "a1_inducer_ceiling_hard_trustworthy_a1b_non_test_"
            "executable_code_change_value_wall_energy_concluded"
        ),
        "operational_retro": {
            "honest_verdict": str(retro.get("honest_verdict", "")),
            "milestone": str(retro.get("milestone", "")),
            "summary": str(retro.get("summary", "")),
        },
        "a1": {
            "honest_verdict": str(a1.get("honest_verdict", "")),
            "fork_verdict": str(a1.get("fork_verdict", "")),
            "generator_backend": str(a1.get("generator_backend", "")),
            "duration_s": round(_float(a1.get("duration_s")), 4),
            "inference_substrate": str(a1.get("inference_substrate", "")),
            "n_games_measured": _int(a1.get("n_games_measured")),
            "positive_control_game": str(a1.get("positive_control_game", "")),
            "positive_control_non_degenerate": a1.get("positive_control_non_degenerate") is True,
            "b1_audited_genuinely_diagnostic": audit.get("a1_genuinely_diagnostic") is True,
            "engine_cell_recall_median": _float(a1.get("engine_cell_recall_median")),
            "tta_changed_cell_value_accuracy_delta_median": _float(
                a1.get("tta_changed_cell_value_accuracy_delta_median")
            ),
            "tta_value_accuracy_delta_ci95": list(_list(a1.get("tta_value_accuracy_delta_ci95"))),
            "coverage_migration_count": _int(a1.get("coverage_migration_count")),
        },
        "a1b": {
            "honest_verdict": str(a1b.get("honest_verdict", "")),
            "flagged_adversarial": a1b.get("flagged_adversarial") is True,
            "duration_s": round(_float(a1b.get("duration_s")), 4),
            "duration_too_short_flagged": duration_too_short,
            "inducer_ceiling_attribution": str(a1b.get("inducer_ceiling_attribution", "")),
            "a1b_ab_trustworthy": audit.get("a1b_ab_trustworthy") is True,
            "method_is_ceiling_established": False,
            "failure_reasons": list(_list(audit.get("a1b_failure_reasons"))),
        },
        "a2": {
            "honest_verdict": str(a2.get("honest_verdict", "")),
            "target_game": str(a2.get("target_game", "")),
            "new_levels_banked": _int(a2.get("new_levels_banked")),
            "reproducible_total_levels_before": _int(a2.get("reproducible_total_levels_before")),
            "reproducible_total_levels_after": _int(a2.get("reproducible_total_levels_after")),
            "solve_provenance": str(a2.get("solve_provenance", "")),
        },
        "a3": {
            "honest_verdict": str(a3.get("honest_verdict", "")),
            "target_game": str(a3.get("target_game", "")),
            "verifier_checkpoint_refreshed": a3.get("verifier_checkpoint_refreshed") is True,
        },
        "a4": {
            "honest_verdict": str(a4.get("honest_verdict", "")),
            "heldout_first_win_rate": _float(a4.get("heldout_first_win_rate")),
            "heldout_first_win_ci_lower": _float(a4.get("heldout_first_win_ci_lower")),
            "live_agent_ran": a4.get("live_agent_ran") is True,
            "flagged_adversarial": a4.get("flagged_adversarial") is True,
        },
        "b2": {
            "honest_verdict": str(b2.get("honest_verdict", "")),
            "submission_package_ready": b2.get("submission_package_ready") is True,
            "operator_only": b2.get("operator_only") is True,
            "vram_estimate_gb": _float(b2.get("vram_estimate_gb")),
        },
        "c": {
            "honest_verdict": str(c.get("honest_verdict", "")),
            "kv260_ssh_reachable": c.get("kv260_ssh_reachable") is True,
            "uio_device_count": _int(c_board.get("uio_device_count")),
        },
        "d": {
            "honest_verdict": str(d.get("honest_verdict", "")),
            "aimed_at_fork_verdict": str(d.get("aimed_at_fork_verdict", "")),
            **priority,
        },
        "energy_program_concluded": energy_concluded,
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "a1_inducer_ceiling_hard_trustworthy": a1_trustworthy,
        "a1b_was_fabrication_flagged_non_test": a1b_non_test,
        "wall_is_executable_code_change_value_representation": (
            a1_trustworthy and a1b_non_test and energy_concluded
        ),
    }


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance records for the source files this task aggregates."""

    rel_paths = [
        REGISTRY_REL_PATH,
        RETRO_REL_PATH,
        A1_REL_PATH,
        A1B_REL_PATH,
        AUDIT_REL_PATH,
        A2_REL_PATH,
        A3_REL_PATH,
        A4_REL_PATH,
        B2_REL_PATH,
        C_REL_PATH,
        D_REL_PATH,
    ]
    return [{"path": str(rel_path), "sha256": file_sha256(root / rel_path)} for rel_path in rel_paths]


def build_artifact(
    *,
    root: Path,
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    pretest_gate: Mapping[str, Any],
    transition_performed: bool,
    duration_s: float,
) -> JsonDict:
    """Build the Exp 4891 artifact from upstream source files."""

    close_state = build_close_state(root)
    active_milestone, active_roadmap_path = read_active_milestone(root)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "random_seed": RANDOM_SEED,
        "spec_refs": SPEC_REFS,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": active_milestone,
        "active_roadmap_path": active_roadmap_path,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "a1_inducer_ceiling_hard_trustworthy": close_state[
            "a1_inducer_ceiling_hard_trustworthy"
        ],
        "a1b_was_fabrication_flagged_non_test": close_state[
            "a1b_was_fabrication_flagged_non_test"
        ],
        "wall_is_executable_code_change_value_representation": close_state[
            "wall_is_executable_code_change_value_representation"
        ],
        "energy_program_concluded": close_state["energy_program_concluded"],
        "reproducible_total_levels": close_state["reproducible_total_levels"],
        "a450_close_state": close_state,
        "preconditions_checked": dict(preconditions_checked),
        "pretest_gate": dict(pretest_gate),
        "transition_performed": transition_performed,
        "archive_record_action": "already_active_noop_recorded"
        if transition_performed
        else "blocked_no_archive",
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
    """Run the record-only .450/.451 transition workflow."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    preconditions = check_preconditions(root, command_runner)
    blocker = precondition_blocker(root, preconditions)
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
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    active_milestone, _active_roadmap_path = read_active_milestone(root)
    if active_milestone != ACTIVATED_MILESTONE:
        artifact = build_artifact(
            root=root,
            honest_verdict="blocked_451_not_active",
            preconditions_checked=preconditions,
            pretest_gate={"ran": False, "green": False, "reason": "skipped_until_451_active"},
            transition_performed=False,
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
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    artifact = build_artifact(
        root=root,
        honest_verdict="complete_450_archived_451_activated_value_representation_wall_recorded",
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 4891 artifact."""

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
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")
    for field in (
        "a1_inducer_ceiling_hard_trustworthy",
        "a1b_was_fabrication_flagged_non_test",
        "wall_is_executable_code_change_value_representation",
        "energy_program_concluded",
    ):
        if payload.get(field) is not True:
            errors.append(f"invalid_{field}")
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 4891 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 4891 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))
