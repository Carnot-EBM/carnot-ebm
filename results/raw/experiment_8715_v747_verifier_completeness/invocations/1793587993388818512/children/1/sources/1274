"""Experiment 4881: archive .449, activate .450, and record the corrected fork state.

Spec refs: REQ-REPORT-4881, SCENARIO-REPORT-4881.

This is a record-only transition. The important point is preserving the .449
truth: the A1 fork probe ran live, but its positive control failed because the
exact-match metric was degenerate. The corrigendum is the usable signal:
change-location is learnable, change-value is still the residual.
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
EXPERIMENT = "experiment_4881_archive_449_activate_450"
EXPERIMENT_ID = 4881
SCHEMA = "carnot.exp4881.archive_449_activate_450.v1"
RANDOM_SEED = 20260627
ARCHIVED_MILESTONE = "2026.06.449"
ACTIVATED_MILESTONE = "2026.06.450"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_4881_archive_449_activate_450.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4880_capstone_v449.json")
CORRIGENDUM_REL_PATH = Path("results/arc_fork_probe_accuracy_corrigendum.json")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
PRETEST_COMMAND = [".venv/bin/pytest", "tests/python", "-q"]
SPEC_REFS = [
    "REQ-REPORT-4881",
    "SCENARIO-REPORT-4881",
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
        "principle": "terminal prefix; clean transition is complete_449_archived_450_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "a449_fork_untrusted_non_test": {
        "principle": (
            "true -- the .449 A1 fork probe ran live on GPU-0 but its positive control failed on the "
            "degenerate exact-match metric (B1 b1_trusted=false); the fork is NOT trustworthy."
        )
    },
    "corrigendum_change_location_learnable": {
        "principle": (
            "true -- the induced engine predicts change-LOCATION (cell_recall <=0.88) but change-VALUE "
            "~0; the binding residual is the engine's VALUE prediction, and the .449 metric was degenerate."
        )
    },
    "exact_match_metric_was_degenerate": {
        "principle": (
            "true -- exact-full-grid-match held-out accuracy was 0.0 on every game incl tu93 "
            "(FALSE_NEGATIVE_RISK); .450 A1 uses the corrigendum-corrected GRADED metric instead."
        )
    },
    "energy_program_concluded": {
        "principle": (
            "true -- the energy-as-ARC-lever program is concluded (negative); the planner must NOT "
            "re-propose energy stages."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric carried from the registry (67 after s5i5 L2), not "
            "re-counted."
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
    "a449_fork_untrusted_non_test",
    "corrigendum_change_location_learnable",
    "exact_match_metric_was_degenerate",
    "energy_program_concluded",
    "reproducible_total_levels",
    "a449_close_state",
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
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


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


def _cell_recall_by_game(corrigendum: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row.get("game")): round(_float(row.get("cell_recall_prior_engine")), 4)
        for row in _list(corrigendum.get("per_game"))
        if isinstance(row, Mapping) and row.get("game")
    }


def _max_change_value_accuracy(corrigendum: Mapping[str, Any]) -> float:
    values = [
        _float(row.get("changing_acc_prior_engine"))
        for row in _list(corrigendum.get("per_game"))
        if isinstance(row, Mapping)
    ]
    values.append(_float(corrigendum.get("mean_changing_acc_prior")))
    return round(max(values), 4) if values else 0.0


def _energy_program_concluded(root: Path) -> bool:
    texts = []
    for rel_path in (ROADMAP_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        texts.append(path.read_text(encoding="utf-8") if path.exists() else "")
    joined = "\n".join(texts)
    return "Energy CONCLUDED" in joined or "Energy-as-ARC-lever program CONCLUDED" in joined


def build_close_state(root: Path) -> JsonDict:
    """Aggregate the .449 close-state from the registry, capstone, and corrigendum."""

    registry = read_yaml_object(root / REGISTRY_REL_PATH)
    capstone = read_json_object(root / CAPSTONE_REL_PATH)
    corrigendum = read_json_object(root / CORRIGENDUM_REL_PATH)
    a1 = _mapping(capstone.get("a1_generation_wall_fork_verdict"))
    a1_checks = _mapping(a1.get("checks"))
    a1_live = _mapping(a1_checks.get("a1_live_gpu"))
    a1_positive_control = _mapping(a1_checks.get("a1_positive_control"))
    a1b = _mapping(capstone.get("a1b_inducer_swing"))
    levelup = _mapping(capstone.get("levelup_bank"))
    self_play = _mapping(capstone.get("self_play_checkpoint"))
    heldout = _mapping(capstone.get("heldout_readiness"))
    package = _mapping(capstone.get("submission_package_state"))
    hardware = _mapping(capstone.get("hardware_continuity"))
    sota = _mapping(capstone.get("sota_handoff"))
    board = _mapping(hardware.get("board_state"))
    cell_recall = _cell_recall_by_game(corrigendum)
    max_cell_recall = round(max(cell_recall.values()), 4) if cell_recall else 0.0
    max_change_value_accuracy = _max_change_value_accuracy(corrigendum)
    registry_total = _int(registry.get("reproducible_total_levels"))
    fork_untrusted = a1.get("verdict") == "non_test_b1_untrusted" and a1.get("b1_trusted") is False
    exact_degenerate = (
        fork_untrusted
        and a1_positive_control.get("passed") is False
        and _float(a1_positive_control.get("engine_heldout_accuracy")) == 0.0
        and max_change_value_accuracy == 0.0
    )
    location_learnable = max_cell_recall >= 0.75 and max_change_value_accuracy == 0.0

    return {
        "summary": (
            "a1_untrusted_non_test_metric_degenerate_corrigendum_location_signal_"
            "value_gap_energy_concluded"
        ),
        "capstone_honest_verdict": str(capstone.get("honest_verdict", "")),
        "a1": {
            "honest_verdict": str(a1.get("upstream_honest_verdict", "")),
            "verdict": str(a1.get("verdict", "")),
            "b1_trusted": a1.get("b1_trusted") is True,
            "a1_genuinely_diagnostic": a1.get("a1_genuinely_diagnostic") is True,
            "generator_backend": str(a1_live.get("generator_backend", "")),
            "gpu0_duration_s": round(_float(a1_live.get("duration_s")), 4),
            "live_path_reachable": _mapping(a1_checks.get("a1_live_path")).get("passed") is True,
            "planner_blind": _mapping(a1_checks.get("a1_planner_blind_to_banked_answer")).get(
                "passed"
            )
            is True,
            "positive_control_game": str(a1_positive_control.get("positive_control_game", "tu93")),
            "positive_control_exact_accuracy": _float(
                a1_positive_control.get("engine_heldout_accuracy")
            ),
            "positive_control_passed": a1_positive_control.get("passed") is True,
            "next_450_pivot": str(a1.get("next_450_pivot", "")),
            "computed_fork_verdict": str(a1.get("computed_fork_verdict", "")),
            "failure_reasons": list(_list(a1.get("a1_failure_reasons"))),
        },
        "corrigendum": {
            "honest_verdict": str(corrigendum.get("honest_verdict", "")),
            "cell_recall_by_game": cell_recall,
            "max_cell_recall": max_cell_recall,
            "max_change_value_accuracy": max_change_value_accuracy,
            "fork_probe_number_is_artifact": corrigendum.get("fork_probe_number_is_artifact") is True,
        },
        "a1b": {
            "cegis_heldout_accuracy_delta_median": _float(
                a1b.get("cegis_heldout_accuracy_delta_median")
            ),
            "delta_ci95": list(_list(a1b.get("delta_ci95"))),
            "positive_control_passed": a1b.get("positive_control_passed") is True,
            "status": str(a1b.get("status", "")),
        },
        "a2": {
            "target_game": str(levelup.get("target_game", "")),
            "new_levels_banked": _int(levelup.get("new_levels_banked")),
            "solve_provenance": str(levelup.get("solve_provenance", "")),
        },
        "a3": {
            "target_game": str(self_play.get("target_game", "")),
            "checkpoint_path": str(self_play.get("checkpoint_path", "")),
            "verifier_checkpoint_refreshed": self_play.get("verifier_checkpoint_refreshed") is True,
        },
        "a4": {
            "heldout_first_win_rate": _float(heldout.get("heldout_first_win_rate")),
            "live_agent_ran": heldout.get("live_agent_ran") is True,
            "generator_backend": str(heldout.get("generator_backend", "")),
            "positive_control_passed": heldout.get("positive_control_passed") is True,
        },
        "b2": {
            "submission_package_ready": package.get("submission_package_ready") is True,
            "operator_only": package.get("operator_only") is True,
            "vram_estimate_gb": _float(package.get("vram_estimate_gb")),
        },
        "c": {
            "kv260_ssh_reachable": hardware.get("kv260_ssh_reachable") is True,
            "uio_device_count": _int(board.get("uio_device_count")),
        },
        "d": {
            "aimed_at_fork_verdict": str(sota.get("aimed_at_fork_verdict", "")),
            "priority_1": "test_time_dynamics_adaptation",
            "priority_2": "family_b_vs_local_open_code_inducer_ab",
        },
        "energy_program_concluded": _energy_program_concluded(root),
        "reproducible_total_levels": registry_total,
        "a449_fork_untrusted_non_test": fork_untrusted,
        "corrigendum_change_location_learnable": location_learnable,
        "exact_match_metric_was_degenerate": exact_degenerate,
    }


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance records for the source files this task aggregates."""

    return [
        {"path": str(REGISTRY_REL_PATH), "sha256": file_sha256(root / REGISTRY_REL_PATH)},
        {"path": str(CAPSTONE_REL_PATH), "sha256": file_sha256(root / CAPSTONE_REL_PATH)},
        {"path": str(CORRIGENDUM_REL_PATH), "sha256": file_sha256(root / CORRIGENDUM_REL_PATH)},
    ]


def build_artifact(
    *,
    root: Path,
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    pretest_gate: Mapping[str, Any],
    transition_performed: bool,
    duration_s: float,
) -> JsonDict:
    """Build the Exp 4881 artifact from upstream source files."""

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
        "a449_fork_untrusted_non_test": close_state["a449_fork_untrusted_non_test"],
        "corrigendum_change_location_learnable": close_state[
            "corrigendum_change_location_learnable"
        ],
        "exact_match_metric_was_degenerate": close_state["exact_match_metric_was_degenerate"],
        "energy_program_concluded": close_state["energy_program_concluded"],
        "reproducible_total_levels": close_state["reproducible_total_levels"],
        "a449_close_state": close_state,
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
    """Run the record-only .449/.450 transition workflow."""

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
            honest_verdict="blocked_450_not_active",
            preconditions_checked=preconditions,
            pretest_gate={"ran": False, "green": False, "reason": "skipped_until_450_active"},
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
        honest_verdict="complete_449_archived_450_activated_a1_untrusted_non_test_value_gap_focus",
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
    """Return schema-contract errors for the Exp 4881 artifact."""

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
        "a449_fork_untrusted_non_test",
        "corrigendum_change_location_learnable",
        "exact_match_metric_was_degenerate",
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
    """Run the Exp 4881 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 4881 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))
