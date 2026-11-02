"""Experiment 4845: hostile audit of the A1 perception probe.

Spec refs: REQ-ARC-WMTE-4845,
SCENARIO-ARC-WMTE-4845-A1-HOSTILE-AUDIT,
SCENARIO-ARC-WMTE-4845-NON-TEST-CLASSIFICATION.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4845_perception_probe_audit"
EXPERIMENT_ID = 4845
SCHEMA = "carnot.arc.a1_perception_probe_audit_4845.v1"
SOURCE_ARTIFACT_RELATIVE_PATH = "results/experiment_4841_object_identity_perception_probe.json"
RESULT_RELATIVE_PATH = "results/experiment_4845_perception_probe_audit.json"
AUDIT_REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 4845
DURATION_FLOOR_S = 0.0001
TARGET_GAMES = ("lp85", "r11l", "tu93")
REAL_SOURCE_KINDS = {"banked_replay", "transition_corpus"}
TERMINAL_PREFIXES = ("success_", "complete_", "blocked_")

SPEC_REFS = [
    "REQ-ARC-WMTE-4845",
    "SCENARIO-ARC-WMTE-4845-A1-HOSTILE-AUDIT",
    "SCENARIO-ARC-WMTE-4845-NON-TEST-CLASSIFICATION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; audit complete is complete_/success_."},
    "a1_genuinely_exercised": {
        "principle": (
            "the load-bearing check -- measured_on_real_frames AND tracker!=baseline no-op "
            "AND positive control real AND verdict matches numbers; else A1 is a non-test "
            "(synthetic-only or a silent degenerate-to-baseline)."
        )
    },
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts (0.0001s floor)."},
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "source_artifact_path",
    "source_artifact_checksum",
    "field_principles",
    "checks",
    "non_test_reasons",
    "per_game_correspondence_deltas",
    "recovered_games_from_rows",
    "claimed_recovery_matches_rows",
    "summarizer_result",
    "adversarial_result",
    "live_lint_result",
    "preconditions_checked",
    "audit_report_path",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

JsonDict = dict[str, Any]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _safe_suffix(reasons: list[str]) -> str:
    if not reasons:
        return "genuinely_exercised"
    joined = "_".join(reasons[:3])
    return re.sub(r"[^a-z0-9_]+", "_", joined.lower()).strip("_") or "failed_checks"


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean["reproducibility_checksum"] = ""
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def run_summarizer(path: Path) -> JsonDict:
    from scripts import summarize_artifact

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        returncode = summarize_artifact.summarize(path)
    return {"returncode": int(returncode), "stdout": buffer.getvalue(), "stderr": ""}


def run_adversarial_verify(path: Path) -> JsonDict:
    from scripts import adversarial_verify

    return dict(adversarial_verify.verify_artifact(path))


def run_arc_orphan_solver_lint(root: Path) -> JsonDict:
    command = [sys.executable, str(root / "scripts" / "arc_orphan_solver_lint.py")]
    proc = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _real_frame_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    per_game = _mapping(artifact.get("per_game_correspondence"))
    rows: dict[str, JsonDict] = {}
    bad_games: list[str] = []
    for game in TARGET_GAMES:
        row = _mapping(per_game.get(game))
        source_kind = row.get("source_kind")
        n_frames = row.get("n_frames")
        present = bool(row)
        real = bool(source_kind in REAL_SOURCE_KINDS)
        enough_frames = isinstance(n_frames, int) and n_frames >= 2
        rows[game] = {
            "present": present,
            "source_kind": source_kind,
            "n_frames": n_frames,
            "real_frame_backed": real,
            "enough_frames": enough_frames,
        }
        if not (present and real and enough_frames):
            bad_games.append(game)

    passed = bool(artifact.get("measured_on_real_frames") is True and per_game and not bad_games)
    reasons = [] if passed else ["synthetic_or_missing_real_frames"]
    return (
        {
            "passed": passed,
            "artifact_measured_on_real_frames": artifact.get("measured_on_real_frames"),
            "target_rows": rows,
            "bad_games": bad_games,
        },
        reasons,
    )


def _tracker_changed_check(
    artifact: Mapping[str, Any],
) -> tuple[JsonDict, list[str], dict[str, float]]:
    per_game = _mapping(artifact.get("per_game_correspondence"))
    deltas: dict[str, float] = {}
    missing_numeric: list[str] = []
    for game, row_value in per_game.items():
        row = _mapping(row_value)
        shape = _finite_float(row.get("shape_motion_score"))
        baseline = _finite_float(row.get("color_centroid_baseline_score"))
        if shape is None or baseline is None:
            missing_numeric.append(str(game))
            continue
        deltas[str(game)] = round(shape - baseline, 6)

    nonzero_games = [game for game, delta in deltas.items() if abs(delta) > 1e-9]
    distinct_delta_count = len(set(deltas.values()))
    passed = bool(deltas and not missing_numeric and nonzero_games and distinct_delta_count > 1)
    reasons = [] if passed else ["tracker_degenerate_to_baseline"]
    return (
        {
            "passed": passed,
            "deltas": deltas,
            "nonzero_delta_games": nonzero_games,
            "distinct_delta_count": distinct_delta_count,
            "missing_numeric_games": missing_numeric,
        },
        reasons,
        deltas,
    )


def _positive_control_and_recovery_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    per_game = _mapping(artifact.get("per_game_correspondence"))
    positive = _mapping(artifact.get("positive_control_tu93"))
    player_id = positive.get("player_track_id")
    goal_id = positive.get("goal_track_id")
    player_motion = _finite_float(positive.get("player_motion"))
    goal_persistence = _finite_float(positive.get("goal_persistence"))
    positive_passed = bool(
        artifact.get("positive_control_tu93_passed") is True
        and positive.get("passed") is True
        and player_id is not None
        and goal_id is not None
        and player_id != goal_id
        and player_motion is not None
        and player_motion > 0.0
        and goal_persistence is not None
        and goal_persistence > 0.0
    )

    recovered_games = [
        str(game) for game, row in per_game.items() if _bool(_mapping(row).get("recovered"))
    ]
    recovered_count = len(recovered_games)
    claimed_recovery = artifact.get("games_with_recovery")
    claim_matches_rows = claimed_recovery == recovered_count
    success_claimed = str(artifact.get("honest_verdict") or "").startswith("success_")
    complete_claimed = str(artifact.get("honest_verdict") or "").startswith("complete_")
    should_be_success = bool(positive_passed and recovered_count >= 2)
    verdict_matches_numbers = bool(
        claim_matches_rows
        and (
            (success_claimed and should_be_success) or (complete_claimed and not should_be_success)
        )
    )

    reasons: list[str] = []
    if not positive_passed:
        reasons.append("tu93_positive_control_failed")
    if not claim_matches_rows:
        reasons.append("games_with_recovery_mismatch")
    if success_claimed and recovered_count < 2:
        reasons.append("success_claim_without_two_recovered_games")
    elif not verdict_matches_numbers:
        reasons.append("verdict_does_not_match_recovery_numbers")

    return (
        {
            "passed": not reasons,
            "positive_control_passed": positive_passed,
            "player_track_id": player_id,
            "goal_track_id": goal_id,
            "player_motion": player_motion,
            "goal_persistence": goal_persistence,
            "recovered_games": recovered_games,
            "recovered_count": recovered_count,
            "artifact_games_with_recovery": claimed_recovery,
            "claimed_recovery_matches_rows": claim_matches_rows,
            "success_claimed": success_claimed,
            "complete_claimed": complete_claimed,
            "should_be_success": should_be_success,
            "verdict_matches_numbers": verdict_matches_numbers,
        },
        reasons,
    )


def _live_path_and_provenance_check(
    artifact: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
) -> tuple[JsonDict, list[str]]:
    lint_passed = bool(live_lint_result.get("passed"))
    artifact_live = artifact.get("live_path_reachable") is True
    provenance = artifact.get("solve_provenance")
    reasons: list[str] = []
    if not lint_passed or not artifact_live:
        reasons.append("live_path_unreachable")
    if provenance != "development_proxy":
        reasons.append("solve_provenance_not_development_proxy")
    return (
        {
            "passed": not reasons,
            "arc_orphan_solver_lint_passed": lint_passed,
            "artifact_live_path_reachable": artifact_live,
            "solve_provenance": provenance,
            "not_live_agent_self_discovery": provenance != "live_agent_self_discovery",
        },
        reasons,
    )


def _tool_cleanliness_check(
    summarizer_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
) -> tuple[JsonDict, list[str]]:
    summarizer_clean = summarizer_result.get("returncode") == 0
    flag_count = adversarial_result.get("flag_count")
    adversarial_clean = adversarial_result.get("loaded") is not False and flag_count == 0
    reasons: list[str] = []
    if not summarizer_clean:
        reasons.append("summarizer_reported_live_flags")
    if not adversarial_clean:
        reasons.append("adversarial_verify_flagged")
    return (
        {
            "passed": not reasons,
            "summarizer_returncode": summarizer_result.get("returncode"),
            "adversarial_flag_count": flag_count,
            "adversarial_loaded": adversarial_result.get("loaded"),
        },
        reasons,
    )


def audit_a1_artifact(
    artifact: Mapping[str, Any],
    *,
    summarizer_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
) -> JsonDict:
    real_check, real_reasons = _real_frame_check(artifact)
    tracker_check, tracker_reasons, deltas = _tracker_changed_check(artifact)
    positive_check, positive_reasons = _positive_control_and_recovery_check(artifact)
    live_check, live_reasons = _live_path_and_provenance_check(artifact, live_lint_result)
    tool_check, tool_reasons = _tool_cleanliness_check(summarizer_result, adversarial_result)

    reasons = real_reasons + tracker_reasons + positive_reasons + live_reasons + tool_reasons
    exercised = not reasons
    return {
        "honest_verdict": (
            "complete_a1_perception_probe_audit_genuinely_exercised"
            if exercised
            else f"complete_a1_perception_probe_non_test_{_safe_suffix(reasons)}"
        ),
        "a1_genuinely_exercised": exercised,
        "non_test_reasons": reasons,
        "checks": {
            "measured_on_real_frames": real_check,
            "tracker_changed_vs_baseline": tracker_check,
            "positive_control_and_recovery_claim": positive_check,
            "live_path_and_provenance": live_check,
            "summarizer_and_adversarial_verify": tool_check,
        },
        "per_game_correspondence_deltas": deltas,
        "recovered_games_from_rows": positive_check["recovered_count"],
        "claimed_recovery_matches_rows": positive_check["claimed_recovery_matches_rows"],
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    repo = Path(root)
    source = repo / SOURCE_ARTIFACT_RELATIVE_PATH
    spec = repo / SPEC_RELATIVE_PATH
    spec_text = spec.read_text(encoding="utf-8") if spec.exists() else ""
    return {
        "ok": source.exists() and "REQ-ARC-WMTE-4845" in spec_text,
        "source_artifact_present": source.exists(),
        "spec_has_req_4845": "REQ-ARC-WMTE-4845" in spec_text,
        "summarizer_script_present": (repo / "scripts/summarize_artifact.py").exists(),
        "adversarial_verify_script_present": (repo / "scripts/adversarial_verify.py").exists(),
        "arc_orphan_solver_lint_present": (repo / "scripts/arc_orphan_solver_lint.py").exists(),
    }


def _blocked_artifact(checks: Mapping[str, Any]) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "source_artifact_path": SOURCE_ARTIFACT_RELATIVE_PATH,
        "source_artifact_checksum": None,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": "blocked_missing_exp4841_perception_artifact",
        "a1_genuinely_exercised": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "checks": {},
        "non_test_reasons": ["missing_exp4841_perception_artifact"],
        "per_game_correspondence_deltas": {},
        "recovered_games_from_rows": 0,
        "claimed_recovery_matches_rows": False,
        "summarizer_result": {},
        "adversarial_result": {},
        "live_lint_result": {},
        "preconditions_checked": dict(checks),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_FLOOR_S,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    source_path: Path,
    source_artifact: Mapping[str, Any],
    audit: Mapping[str, Any],
    summarizer_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "source_artifact_path": SOURCE_ARTIFACT_RELATIVE_PATH,
        "source_artifact_checksum": file_checksum(source_path),
        "source_honest_verdict": source_artifact.get("honest_verdict"),
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": audit.get("honest_verdict"),
        "a1_genuinely_exercised": audit.get("a1_genuinely_exercised"),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "checks": dict(_mapping(audit.get("checks"))),
        "non_test_reasons": list(audit.get("non_test_reasons") or []),
        "per_game_correspondence_deltas": dict(
            _mapping(audit.get("per_game_correspondence_deltas"))
        ),
        "recovered_games_from_rows": audit.get("recovered_games_from_rows"),
        "claimed_recovery_matches_rows": audit.get("claimed_recovery_matches_rows"),
        "summarizer_result": dict(summarizer_result),
        "adversarial_result": dict(adversarial_result),
        "live_lint_result": dict(live_lint_result),
        "preconditions_checked": dict(preconditions_checked),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(DURATION_FLOOR_S, duration_s), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if not isinstance(artifact.get("a1_genuinely_exercised"), bool):
        errors.append("a1_genuinely_exercised_must_be_bool")
    if not isinstance(artifact.get("checks"), dict):
        errors.append("checks_must_be_dict")
    if not isinstance(artifact.get("non_test_reasons"), list):
        errors.append("non_test_reasons_must_be_list")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    duration = _finite_float(artifact.get("duration_s"))
    if duration is None or duration < DURATION_FLOOR_S:
        errors.append("duration_below_aggregation_floor")
    expected = "sha256:" + payload_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected:
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_markdown_section(artifact: Mapping[str, Any]) -> str:
    checks = _mapping(artifact.get("checks"))
    rows = [
        "",
        "## Experiment 4845 .446 A1 Perception Probe Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- a1_genuinely_exercised: `{artifact.get('a1_genuinely_exercised')}`",
        f"- Non-test reasons: `{', '.join(artifact.get('non_test_reasons') or []) or '-'}`",
        f"- Inference substrate: `{artifact.get('inference_substrate')}`",
        "",
        "| Check | Passed | Detail |",
        "|---|---:|---|",
    ]
    for name, check in checks.items():
        if not isinstance(check, Mapping):
            continue
        detail = {
            key: value
            for key, value in check.items()
            if key != "passed" and key not in {"target_rows"}
        }
        rows.append(
            f"| `{name}` | `{check.get('passed')}` | `{json.dumps(detail, sort_keys=True)}` |"
        )
    rows.extend(
        [
            "",
            f"- Per-game deltas: `{artifact.get('per_game_correspondence_deltas')}`",
            f"- Source checksum: `{artifact.get('source_artifact_checksum')}`",
            "",
        ]
    )
    return "\n".join(rows)


def append_markdown_report(
    artifact: Mapping[str, Any],
    *,
    root: Path | str = REPO_ROOT,
) -> Path:
    report_path = Path(root) / AUDIT_REPORT_RELATIVE_PATH
    marker = "## Experiment 4845 .446 A1 Perception Probe Audit"
    if report_path.exists():
        current = report_path.read_text(encoding="utf-8")
        if marker in current:
            return report_path
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        current = "# ARC Null Silent-Bug Audit\n"
    report_path.write_text(current.rstrip() + render_markdown_section(artifact), encoding="utf-8")
    return report_path


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    now: Callable[[], float] | None = None,
) -> JsonDict:
    repo = Path(root)
    clock = now or time.monotonic
    start = clock()
    checks = check_preconditions(repo)
    if not checks["ok"]:
        artifact = _blocked_artifact(checks)
        if write:
            write_artifact(artifact, root=repo)
            append_markdown_report(artifact, root=repo)
        return artifact

    source_path = repo / SOURCE_ARTIFACT_RELATIVE_PATH
    source_artifact = _read_json(source_path)
    summarizer_result = run_summarizer(source_path)
    adversarial_result = run_adversarial_verify(source_path)
    live_lint_result = run_arc_orphan_solver_lint(repo)
    audit = audit_a1_artifact(
        source_artifact,
        summarizer_result=summarizer_result,
        adversarial_result=adversarial_result,
        live_lint_result=live_lint_result,
    )
    artifact = build_artifact(
        source_path=source_path,
        source_artifact=source_artifact,
        audit=audit,
        summarizer_result=summarizer_result,
        adversarial_result=adversarial_result,
        live_lint_result=live_lint_result,
        preconditions_checked=checks,
        duration_s=clock() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=repo)
        append_markdown_report(artifact, root=repo)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "a1_genuinely_exercised": artifact["a1_genuinely_exercised"],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
