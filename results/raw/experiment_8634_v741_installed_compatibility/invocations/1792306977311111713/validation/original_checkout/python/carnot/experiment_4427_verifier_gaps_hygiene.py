"""Exp 4427: reconcile verifier gaps emitted by .409 ARC solve tasks.

Spec refs: REQ-VERIFY-4427, SCENARIO-VERIFY-4427.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4427_verifier_gaps_hygiene.json"
GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"

EXP4421_PATH = "results/experiment_4421_config_rule_solve_unseen.json"
EXP4422_PATH = "results/experiment_4422_glyph_rewrite_perception.json"
EXP4423_PATH = "results/experiment_4423_generic_first_contact_breadth.json"
EXP4424_PATH = "results/experiment_4424_deeper_solved_game.json"
EXP4425_PATH = "results/experiment_4425_config_rule_vocabulary_transfer.json"
EXP4426_PATH = "results/experiment_4426_arc_registry_repro_audit.json"

SOURCE_ARTIFACTS = {
    "exp4421": EXP4421_PATH,
    "exp4422": EXP4422_PATH,
    "exp4423": EXP4423_PATH,
    "exp4424": EXP4424_PATH,
    "exp4425": EXP4425_PATH,
    "exp4426": EXP4426_PATH,
}

RANDOM_SEED = 4427
INFERENCE_SUBSTRATE = "cpu_artifact_reconciliation_no_llm"
SPEC_REFS = ("REQ-VERIFY-4427", "SCENARIO-VERIFY-4427")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_gaps_reconciled",
    "appended_gaps",
    "filled_gaps",
    "build_target_for_410",
    "source_artifacts",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed final state for conductor/reconciler classification.",
    "inference_substrate": "CPU-only artifact and markdown reconciliation; no LLM, GPU, or live solve.",
    "verifier_gaps_reconciled": (
        "Bare bool: true only when emitted gaps are present, filled gaps are recorded, "
        "and one .410 build target is selected."
    ),
    "appended_gaps": "Structured gap entries appended or refreshed in ops/verifier_gaps.md.",
    "filled_gaps": "Structured gap entries moved to status: filled by .409 reproduction evidence.",
    "build_target_for_410": "The single highest-priority open verifier gap for the .410 planner.",
    "reproducibility_checksum": "Hash of source artifact checksums and reconciliation decisions.",
}

TERMINAL_PREFIXES = ("complete:", "blocked:", "failed:", "success:", "partial:")
PRIORITY_RANK = {"high": 3, "medium": 2, "low": 1}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(value: Any) -> str:
    return _sha256_text(_stable_json(value))


def _load_json(root: Path, rel_path: str) -> tuple[dict[str, Any], str]:
    path = root / rel_path
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(loaded, dict):
        return {}, "JSON top-level is not an object"
    return loaded, ""


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_str(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def check_preconditions(root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """REQ-VERIFY-4427: load the ledger and every required .409 source artifact first."""

    ledger_path = root / GAPS_RELATIVE_PATH
    try:
        ledger_text = ledger_path.read_text(encoding="utf-8")
        ledger_error = ""
        ledger_readable = True
    except OSError as exc:
        ledger_text = ""
        ledger_error = f"{type(exc).__name__}: {exc}"
        ledger_readable = False

    payloads: dict[str, dict[str, Any]] = {}
    source_report: dict[str, dict[str, Any]] = {}
    for key, rel_path in SOURCE_ARTIFACTS.items():
        payload, error = _load_json(root, rel_path)
        payloads[key] = payload
        source_report[key] = {
            "path": rel_path,
            "readable": error == "",
            "error": error,
            "sha256": _file_sha256(root / rel_path),
            "honest_verdict": _as_str(payload.get("honest_verdict")),
            "missing_verifier_gaps_count": len(payload.get("missing_verifier_gaps", []))
            if isinstance(payload.get("missing_verifier_gaps"), list)
            else 0,
        }

    blocked_sources = [key for key, row in source_report.items() if not row["readable"]]
    preconditions = {
        "ok": ledger_readable and not blocked_sources,
        "ledger": {
            "path": GAPS_RELATIVE_PATH,
            "readable": ledger_readable,
            "error": ledger_error,
            "sha256": _sha256_text(ledger_text) if ledger_readable else "",
        },
        "source_artifacts": source_report,
        "blocked_sources": blocked_sources,
    }
    return preconditions, payloads


def _priority_from_headroom(gap: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    explicit = str(gap.get("priority", "")).lower()
    if explicit in PRIORITY_RANK:
        return explicit
    summary = gap.get("loop_result_summary")
    if isinstance(summary, Mapping):
        if summary.get("offline_reproduced") is False and _as_int(summary.get("reproduced_levels")) == 0:
            return "high"
    if payload.get("offline_reproduced") is False and _as_int(payload.get("reproduced_levels")) == 0:
        return "high"
    return "medium"


def _headroom_score(gap: Mapping[str, Any], payload: Mapping[str, Any]) -> int:
    summary = gap.get("loop_result_summary")
    if isinstance(summary, Mapping) and _as_int(summary.get("reproduced_levels")) == 0:
        return 1
    target = _as_int(payload.get("target_level"))
    reproduced = _as_int(payload.get("reproduced_levels"))
    if target:
        return max(0, target - reproduced)
    return 1 if payload.get("offline_reproduced") is False else 0


def _normalize_emitted_gap(
    gap: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    source_artifact: str,
) -> dict[str, Any]:
    gap_id = _as_str(gap.get("gap_id")).strip()
    game = _as_str(gap.get("game") or payload.get("target_game") or payload.get("game"))
    priority = _priority_from_headroom(gap, payload)
    return {
        "gap_id": gap_id,
        "status": _as_str(gap.get("status"), "open") or "open",
        "game": game,
        "evidence": (
            f"{source_artifact}; game={game}; honest_verdict={payload.get('honest_verdict')}; "
            f"offline_reproduced={payload.get('offline_reproduced')}; "
            f"reproduced_levels={payload.get('reproduced_levels')}"
        ),
        "failure_mode": _as_str(gap.get("failure_mode"), "present-but-unselectable residual"),
        "missing_discriminator": _as_str(gap.get("missing_discriminator")),
        "candidate_design": _as_str(gap.get("candidate_design")),
        "priority": priority,
        "headroom": _headroom_score(gap, payload),
        "source_artifact": source_artifact,
        "movement": "newly_logged",
    }


def collect_emitted_gaps(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """SCENARIO-VERIFY-4427: collect structured missing gaps emitted by .409 solve tasks."""

    gaps: dict[str, dict[str, Any]] = {}
    for key in ("exp4421", "exp4422", "exp4423", "exp4424"):
        payload = payloads.get(key, {})
        source = SOURCE_ARTIFACTS[key]
        raw_gaps = payload.get("missing_verifier_gaps")
        if not isinstance(raw_gaps, list):
            continue
        for raw_gap in raw_gaps:
            if not isinstance(raw_gap, Mapping):
                continue
            gap_id = _as_str(raw_gap.get("gap_id")).strip()
            if not gap_id:
                continue
            gaps[gap_id] = _normalize_emitted_gap(raw_gap, payload=payload, source_artifact=source)
    return list(gaps.values())


def collect_residual_gaps(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4427: keep unresolved .409 deeper-level residuals visible."""

    exp4424 = payloads.get("exp4424", {})
    residual = _as_str(exp4424.get("residual_failing_mechanic"))
    if exp4424.get("offline_reproduced") is True or not residual:
        return []
    game = _as_str(exp4424.get("game"), "unknown")
    target_level = _as_int(exp4424.get("target_level"))
    gap_id = f"GAP-4424-{game.upper()}-L{target_level}-ROUTE-SEARCH"
    return [
        {
            "gap_id": gap_id,
            "status": "open",
            "game": game,
            "evidence": (
                f"{EXP4424_PATH}; game={game}; target_level={target_level}; "
                f"offline_reproduced=False; reproduced_levels={exp4424.get('reproduced_levels')}; "
                f"residual={residual}"
            ),
            "failure_mode": f"{game} L{target_level} remains unreproduced after .409 HUD/mechanic cleanup",
            "missing_discriminator": (
                f"{game} route-search verifier that proves the complete L{target_level} path "
                "after the recorded mechanic cleanup"
            ),
            "candidate_design": (
                f"build an executable route-search verifier over the {game} world model and count "
                "only arc_solver_kit.reproduce success"
            ),
            "priority": "medium",
            "headroom": max(0, target_level - _as_int(exp4424.get("reproduced_levels"))),
            "source_artifact": EXP4424_PATH,
            "movement": "newly_logged",
        }
    ]


def collect_filled_gaps(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4427: turn offline reproduction successes into filled gap updates."""

    filled: list[dict[str, Any]] = []
    exp4421 = payloads.get("exp4421", {})
    if exp4421.get("offline_reproduced") is True and _as_int(exp4421.get("reproduced_levels")) >= 1:
        filled.append(
            {
                "gap_id": "GAP-4421-S5I5-MARKER-COVERAGE",
                "status": "filled (exp4421_s5i5_marker_coverage)",
                "game": "s5i5",
                "evidence": (
                    f"{EXP4421_PATH}; offline_reproduced=True; "
                    f"reproduced_levels={exp4421.get('reproduced_levels')}; "
                    f"new_levels_reproduced={exp4421.get('new_levels_reproduced')}"
                ),
                "failure_mode": "s5i5 marker-coverage config rule is now grounded and offline reproduced",
                "missing_discriminator": "grounded marker-coverage predicate",
                "candidate_design": "reuse Exp 4421 marker-coverage verifier for related marker-toggle games",
                "priority": "medium",
                "headroom": 0,
                "source_artifact": EXP4421_PATH,
                "movement": "filled",
            }
        )

    exp4422 = payloads.get("exp4422", {})
    if exp4422.get("offline_reproduced") is True and _as_int(exp4422.get("reproduced_levels")) >= 6:
        filled.append(
            {
                "gap_id": "GAP-4422-TR87-GLYPH-REWRITE-PERCEPTION",
                "status": "filled (exp4422_tr87_glyph_rewrite_perception)",
                "game": "tr87",
                "evidence": (
                    f"{EXP4422_PATH}; offline_reproduced=True; "
                    f"reproduced_levels={exp4422.get('reproduced_levels')}; "
                    f"fires_on_win={exp4422.get('fires_on_win')}; "
                    f"false_positive_rate={exp4422.get('false_positive_rate')}"
                ),
                "failure_mode": "tr87 glyph rewrite perception is now grounded through L6 replay",
                "missing_discriminator": "glyph rewrite perception predicate",
                "candidate_design": "reuse segmented glyph rewrite predicates for future rewrite games",
                "priority": "medium",
                "headroom": 0,
                "source_artifact": EXP4422_PATH,
                "movement": "filled",
            }
        )
    return filled


def choose_build_target(open_gaps: list[dict[str, Any]]) -> dict[str, Any]:
    """REQ-VERIFY-4427: pick one highest-priority open gap for the .410 planner."""

    candidates = [gap for gap in open_gaps if str(gap.get("status", "open")).startswith("open")]
    if not candidates:
        return {}
    return dict(
        sorted(
            candidates,
            key=lambda gap: (
                -PRIORITY_RANK.get(str(gap.get("priority", "low")), 0),
                -_as_int(gap.get("headroom")),
                str(gap.get("gap_id")),
            ),
        )[0]
    )


def _marker(gap_id: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", gap_id.lower()).strip("-")
    return f"exp4427-{safe}"


def _gap_present(ledger_text: str, gap_id: str) -> bool:
    return f"### {gap_id}" in ledger_text


def _gap_block(gap: Mapping[str, Any], *, is_build_target: bool) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4427 .409 verifier gap hygiene\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'medium')}\n"
        f"- headroom: {gap.get('headroom', 0)}\n"
        f"- build target for .410 planner: {str(is_build_target).lower()}\n"
        f"- movement: {gap.get('movement', 'newly_logged')}\n"
    )


def _replace_marked_block(ledger_text: str, gap: Mapping[str, Any], *, is_build_target: bool) -> str:
    marker = _marker(str(gap["gap_id"]))
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    replacement = f"{start}\n{_gap_block(gap, is_build_target=is_build_target).rstrip()}\n{end}"
    if start in ledger_text and end in ledger_text:
        prefix, rest = ledger_text.split(start, 1)
        _, suffix = rest.split(end, 1)
        return f"{prefix}{replacement}{suffix}"
    return ledger_text.rstrip() + "\n\n" + replacement + "\n"


def _replace_existing_gap_status(ledger_text: str, gap: Mapping[str, Any]) -> tuple[str, bool]:
    heading = f"### {gap['gap_id']}"
    start = ledger_text.find(heading)
    if start < 0:
        return ledger_text, False
    next_heading = ledger_text.find("\n### ", start + len(heading))
    next_marker = ledger_text.find("\n<!-- ", start + len(heading))
    candidates = [pos for pos in (next_heading, next_marker) if pos >= 0]
    end = min(candidates) if candidates else len(ledger_text)
    section = ledger_text[start:end]
    status_line = f"- status: {gap.get('status', 'filled')}"
    if re.search(r"(?m)^- status: .*$", section):
        section = re.sub(r"(?m)^- status: .*$", status_line, section, count=1)
    else:
        section = section.replace("\n", f"\n{status_line}\n", 1)
    if f"- filled evidence: {gap.get('evidence', '')}" not in section:
        section = section.rstrip() + f"\n- filled evidence: {gap.get('evidence', '')}\n"
    return ledger_text[:start] + section + ledger_text[end:], True


def reconcile_ledger(
    ledger_text: str,
    *,
    open_gaps: list[dict[str, Any]],
    filled_gaps: list[dict[str, Any]],
    build_target: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """SCENARIO-VERIFY-4427: idempotently update markdown without pruning history."""

    updated = ledger_text
    target_id = build_target.get("gap_id")
    appended: list[dict[str, Any]] = []

    for gap in filled_gaps:
        updated, replaced = _replace_existing_gap_status(updated, gap)
        if not replaced:
            updated = _replace_marked_block(updated, gap, is_build_target=False)
            appended.append(gap)

    for gap in open_gaps:
        is_target = gap.get("gap_id") == target_id
        if _gap_present(updated, str(gap["gap_id"])):
            if f"<!-- {_marker(str(gap['gap_id']))}:start -->" in updated:
                updated = _replace_marked_block(updated, gap, is_build_target=is_target)
            continue
        updated = _replace_marked_block(updated, gap, is_build_target=is_target)
        appended.append(gap)

    return updated, appended


def _checksum_for_artifact(artifact: Mapping[str, Any]) -> str:
    payload = {
        "appended_gaps": artifact.get("appended_gaps"),
        "filled_gaps": artifact.get("filled_gaps"),
        "build_target_for_410": artifact.get("build_target_for_410"),
        "source_artifacts": artifact.get("source_artifacts"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "random_seed": artifact.get("random_seed"),
        "spec_refs": artifact.get("spec_refs"),
        "inference_substrate": artifact.get("inference_substrate"),
    }
    return _sha256_payload(payload)


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked_artifact(preconditions: Mapping[str, Any], started_at: float, ended_at: float) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": "experiment_4427_verifier_gaps_hygiene",
        "schema": "carnot.exp4427.verifier_gaps_hygiene.v1",
        "honest_verdict": "blocked: verifier_gap_hygiene_precondition_failed",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_gaps_reconciled": False,
        "appended_gaps": [],
        "filled_gaps": [],
        "build_target_for_410": {},
        "source_artifacts": preconditions.get("source_artifacts", {}),
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": max(0.0, ended_at - started_at),
    }
    artifact["reproducibility_checksum"] = _checksum_for_artifact(artifact)
    return artifact


def run(root: Path = REPO_ROOT, *, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """REQ-VERIFY-4427: reconcile ops/verifier_gaps.md and write the terminal artifact."""

    started_at = now()
    preconditions, payloads = check_preconditions(root)
    if preconditions.get("ok") is not True:
        artifact = _blocked_artifact(preconditions, started_at, now())
        _write_artifact(root, artifact)
        return artifact

    ledger_path = root / GAPS_RELATIVE_PATH
    ledger_text = ledger_path.read_text(encoding="utf-8")
    open_gaps = collect_emitted_gaps(payloads) + collect_residual_gaps(payloads)
    filled_gaps = collect_filled_gaps(payloads)
    build_target = choose_build_target(open_gaps)
    updated_ledger, appended_gaps = reconcile_ledger(
        ledger_text,
        open_gaps=open_gaps,
        filled_gaps=filled_gaps,
        build_target=build_target,
    )
    ledger_path.write_text(updated_ledger, encoding="utf-8")

    verifier_gaps_reconciled = bool(build_target) and all(
        _gap_present(updated_ledger, str(gap["gap_id"])) for gap in open_gaps + filled_gaps
    )
    artifact: dict[str, Any] = {
        "experiment": "experiment_4427_verifier_gaps_hygiene",
        "schema": "carnot.exp4427.verifier_gaps_hygiene.v1",
        "honest_verdict": (
            "complete: verifier_gaps_reconciled_for_409_build_target_"
            f"{str(build_target.get('gap_id', 'none')).lower()}"
        )
        if verifier_gaps_reconciled
        else "blocked: verifier_gap_hygiene_no_open_build_target",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_gaps_reconciled": verifier_gaps_reconciled,
        "appended_gaps": appended_gaps,
        "filled_gaps": filled_gaps,
        "build_target_for_410": build_target,
        "source_artifacts": preconditions["source_artifacts"],
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": max(0.0, now() - started_at),
    }
    artifact["reproducibility_checksum"] = _checksum_for_artifact(artifact)
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-VERIFY-4427: reject malformed or fabricated terminal artifacts."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be cpu_artifact_reconciliation_no_llm")
    if not isinstance(artifact.get("verifier_gaps_reconciled"), bool):
        raise ValueError("verifier_gaps_reconciled must be bare bool")
    for field in ("appended_gaps", "filled_gaps", "spec_refs"):
        if not isinstance(artifact.get(field), list):
            raise ValueError(f"{field} must be list")
    for field in ("build_target_for_410", "source_artifacts", "preconditions_checked", "field_principles"):
        if not isinstance(artifact.get(field), Mapping):
            raise ValueError(f"{field} must be dict")
    if artifact.get("verifier_gaps_reconciled") and not artifact.get("build_target_for_410"):
        raise ValueError("build_target_for_410 must be non-empty when reconciled")
    if not isinstance(artifact.get("random_seed"), int) or isinstance(artifact.get("random_seed"), bool):
        raise ValueError("random_seed must be bare int")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError("reproducibility_checksum must be 64-char sha256 hex")
    if set(SPEC_REFS) - set(artifact.get("spec_refs", [])):
        raise ValueError("spec_refs must include REQ-VERIFY-4427 and SCENARIO-VERIFY-4427")


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
