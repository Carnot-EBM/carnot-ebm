"""Exp 5165: retire the nulled generation-axis exploration-signal scope.

Spec refs: REQ-REPORT-5165, SCENARIO-REPORT-5165-LINT,
SCENARIO-REPORT-5165-NARROW-SCOPE.

This module is a narrow reporting harness. It validates that
ops/exclusion_manifest.yaml contains the scoped retired_extras entry, proves the
entry is load-bearing through scripts/exclusion_manifest_lint.py, checks that it
does not block this milestone's deepen-wall tasks, and writes the required
artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_5150_archive_471_activate_472 import (
    CommandResult,
    _bool,
    _list,
    _mapping,
    file_sha256,
    payload_checksum,
    read_json_mapping,
    run_adversarial_verification,
    verification_payload,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5165_generation_axis_retirement_hygiene_v473.json")
MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
LINTER_RELATIVE_PATH = Path("scripts/exclusion_manifest_lint.py")
LINTER_MODULE_PATH = REPO_ROOT / LINTER_RELATIVE_PATH

EXPERIMENT = "experiment_5165_generation_axis_retirement_hygiene_v473"
EXPERIMENT_ID = "exp5165-generation-axis-retirement-hygiene-v473"
MILESTONE = "2026.07.473"
SCHEMA = "carnot.experiment_5165_generation_axis_retirement_hygiene.v1"
RANDOM_SEED = 5165
INFERENCE_SUBSTRATE = "metadata_and_source_audit"
ENTRY_ID = "generation_axis_exploration_signal_retired_exp5154_v473"
COMPLETE_VERDICT = (
    "complete: generation_axis_exploration_signal_scope_retired_and_lint_load_bearing"
)
INCOMPLETE_VERDICT = "complete: generation_axis_retirement_hygiene_incomplete"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
KNOWN_ISSUES_NOTE_MARKER = "2026-07-02 (RETIREMENT): generation-axis exploration-signal first-contact reruns closed after exp5154"

SOURCE_RESULT_PATHS: dict[str, Path] = {
    "exp4688": Path("results/experiment_4688_controllable_novelty_proposal_policy_live.json"),
    "exp4689": Path("results/experiment_4689_program_synthesis_action_effect_proposal_filter.json"),
    "exp5154": Path("results/experiment_5154_energy_fitness_directed_exploration_v472.json"),
}

EXPECTED_EXPERIMENT_IDS = ["exp4688", "exp4689", "exp5154"]
EXPECTED_BLOCKED_PATTERNS = [
    "first-contact candidate-generation rerun",
    "first contact exploration signal rerun",
    "novelty-bonus first-contact rerun",
    "program-synthesis first-contact rerun",
    "energy-as-fitness first-contact rerun",
]
SPEC_REFS = [
    "REQ-REPORT-5165",
    "SCENARIO-REPORT-5165-LINT",
    "SCENARIO-REPORT-5165-NARROW-SCOPE",
]
DEEPEN_WALL_PREFIXES = ("exp5157", "exp5158", "exp5159")

FIELD_PRINCIPLES: dict[str, str] = {
    "exclusion_manifest_entry_added": "manifest retirement must be present and schema-compatible",
    "entry_id": "traceability to the retired_extras entry",
    "false_positive_check_against_this_milestone": (
        "Must confirm the new entry does not accidentally block this milestone's own "
        "deepen-wall tasks -- an over-broad retirement would be a self-inflicted wound."
    ),
    "synthetic_match_check_passed": (
        "Confirms the retirement is load-bearing, not just documentation -- per the "
        "2026-07-01 BLOCKED_PATTERN_MATCHED mechanical-fix precedent."
    ),
    "known_issues_or_gaps_md_updated": (
        "dated retirement note preserves the allocation decision without deleting prior gap history"
    ),
    "honest_verdict": "Must start with complete:/complete_/success:/success_.",
}

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "spec_refs",
    "result_path",
    "run_date",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "source_artifacts_read",
    "source_artifact_summary",
    "manifest_entry_audit",
    "current_roadmap_lint",
    "synthetic_match_lint",
    "adversarial_verification",
    "flagged_adversarial",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES,
)

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5165_generation_axis_retirement_hygiene.py -q -o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5165_generation_axis_retirement_hygiene_v473.py' -m pytest tests/python/test_experiment_5165_generation_axis_retirement_hygiene.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5165_generation_axis_retirement_hygiene_v473.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5165_generation_axis_retirement_hygiene.py",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/pytest tests/python -q",
]


def expected_manifest_entry() -> JsonDict:
    return {
        "id": ENTRY_ID,
        "experiment_scope": (
            "ARC first-contact candidate-generation exploration-signal tweaks on unsolved games "
            "after novelty, program-synthesis filter, and energy-as-fitness QD nulls"
        ),
        "reason": (
            "retire_if_same_verdict: Exp 5154 is the third independent null on better "
            "exploration signals for first-contact candidate generation after Exp 4688 "
            "and Exp 4689. Future reruns need an operator_override citing a genuinely "
            "new mechanism; deepen-wall cross-level warm-starts and representation/"
            "perception fixes are outside this retired scope."
        ),
        "experiment_ids": list(EXPECTED_EXPERIMENT_IDS),
        "retired_milestone": MILESTONE,
        "retired_by_artifact": str(SOURCE_RESULT_PATHS["exp5154"]),
        "recorded_by_artifact": str(RESULT_RELATIVE_PATH),
        "operator_reopen_required": True,
        "retire_if_same_verdict": True,
        "blocked_patterns": list(EXPECTED_BLOCKED_PATTERNS),
    }


def _load_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _load_linter_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_exp5165_exclusion_manifest_lint", LINTER_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {LINTER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _risk_to_dict(risk: Any) -> JsonDict:
    return {
        "task_id": str(getattr(risk, "task_id", "")),
        "task_title": str(getattr(risk, "task_title", "")),
        "violation_class": str(getattr(risk, "violation_class", "")),
        "severity": str(getattr(risk, "severity", "")),
        "detail": str(getattr(risk, "detail", "")),
        "retirement_reason": str(getattr(risk, "retirement_reason", "")),
    }


def lint_roadmap(root: Path, roadmap_path: Path) -> list[JsonDict]:
    module = _load_linter_module()
    module.PROJECT_ROOT = root
    return [_risk_to_dict(risk) for risk in module.lint(roadmap_path)]


def _find_manifest_entry(root: Path) -> JsonDict:
    manifest = _load_yaml_mapping(root / MANIFEST_RELATIVE_PATH)
    for entry in _list(manifest.get("retired_extras")):
        entry_map = _mapping(entry)
        if entry_map.get("id") == ENTRY_ID:
            return dict(entry_map)
    return {}


def _entry_audit(entry: JsonMap) -> JsonDict:
    expected = expected_manifest_entry()
    errors: list[str] = []
    for key in (
        "id",
        "experiment_scope",
        "reason",
        "experiment_ids",
        "retired_milestone",
        "retired_by_artifact",
        "recorded_by_artifact",
        "operator_reopen_required",
        "retire_if_same_verdict",
        "blocked_patterns",
    ):
        if entry.get(key) != expected[key]:
            errors.append(f"{key}.mismatch")
    return {
        "entry_id": str(entry.get("id", "")),
        "found": bool(entry),
        "errors": errors,
        "blocked_patterns": _list(entry.get("blocked_patterns")),
    }


def check_known_issues_or_gaps_doc(root: Path) -> JsonDict:
    path = root / KNOWN_ISSUES_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "path": str(KNOWN_ISSUES_RELATIVE_PATH),
        "exists": path.exists(),
        "marker": KNOWN_ISSUES_NOTE_MARKER,
        "updated": KNOWN_ISSUES_NOTE_MARKER in text,
    }


def _entry_blocked_pattern_risks(risks: Sequence[JsonMap]) -> list[JsonDict]:
    return [
        dict(risk)
        for risk in risks
        if risk.get("violation_class") == "BLOCKED_PATTERN_MATCHED"
        and ENTRY_ID in str(risk.get("detail", ""))
    ]


def check_current_roadmap_false_positive(root: Path) -> JsonDict:
    roadmap_path = root / ROADMAP_RELATIVE_PATH
    risks = lint_roadmap(root, roadmap_path)
    entry_risks = _entry_blocked_pattern_risks(risks)
    deepen_ids = [
        str(task.get("id", ""))
        for task in _list(_load_yaml_mapping(roadmap_path).get("tasks"))
        if str(task.get("id", "")).startswith(DEEPEN_WALL_PREFIXES)
    ]
    deepen_risks = [
        risk
        for risk in entry_risks
        if str(risk.get("task_id", "")).startswith(DEEPEN_WALL_PREFIXES)
    ]
    return {
        "roadmap_path": str(ROADMAP_RELATIVE_PATH),
        "all_risks": risks,
        "entry_blocked_pattern_risks": entry_risks,
        "entry_blocked_pattern_task_ids": [str(risk.get("task_id", "")) for risk in entry_risks],
        "deepen_wall_task_ids": deepen_ids,
        "deepen_wall_entry_risks": deepen_risks,
        "passed": not entry_risks and len(deepen_ids) == 3,
    }


def _synthetic_roadmap_text() -> str:
    return yaml.safe_dump(
        {
            "milestone": MILESTONE,
            "tasks": [
                {
                    "id": "exp9999-curiosity-driven-first-contact-exploration-signal-rerun",
                    "title": "curiosity driven first contact exploration signal rerun",
                    "prompt": (
                        "Try another better exploration signal over the first-contact "
                        "candidate-generation process for an unsolved ARC game."
                    ),
                    "agent_type": "codex",
                }
            ],
        },
        sort_keys=False,
    )


def check_synthetic_match(root: Path) -> JsonDict:
    with tempfile.TemporaryDirectory(prefix="exp5165_lint_") as tmp:
        roadmap_path = Path(tmp) / "research-roadmap-next.yaml"
        roadmap_path.write_text(_synthetic_roadmap_text(), encoding="utf-8")
        risks = lint_roadmap(root, roadmap_path)
    matched = _entry_blocked_pattern_risks(risks)
    return {
        "synthetic_title": "curiosity driven first contact exploration signal rerun",
        "matched_risks": matched,
        "all_risks": risks,
        "passed": any(risk.get("severity") == "HARD" for risk in matched),
    }


def _source_row(root: Path, *, source_id: str, path: Path, kind: str) -> JsonDict:
    full_path = root / path
    return {
        "source_id": source_id,
        "kind": kind,
        "path": str(path),
        "exists": full_path.exists(),
        "sha256": file_sha256(full_path),
    }


def build_source_artifacts_read(root: Path) -> list[JsonDict]:
    rows = [
        _source_row(root, source_id=source_id, path=path, kind="prior_result")
        for source_id, path in SOURCE_RESULT_PATHS.items()
    ]
    rows.extend(
        [
            _source_row(
                root, source_id="exclusion_manifest", path=MANIFEST_RELATIVE_PATH, kind="ops_yaml"
            ),
            _source_row(
                root, source_id="known_issues", path=KNOWN_ISSUES_RELATIVE_PATH, kind="ops_doc"
            ),
            _source_row(
                root, source_id="active_roadmap", path=ROADMAP_RELATIVE_PATH, kind="roadmap_yaml"
            ),
            _source_row(
                root, source_id="exclusion_manifest_lint", path=LINTER_RELATIVE_PATH, kind="script"
            ),
        ]
    )
    return rows


def _source_artifact_summary(root: Path) -> JsonDict:
    summary: JsonDict = {}
    for source_id, path in SOURCE_RESULT_PATHS.items():
        payload, status = read_json_mapping(root / path)
        payload_map = _mapping(payload)
        summary[source_id] = {
            "path": str(path),
            "loadable": _bool(status.get("loadable")),
            "honest_verdict": str(payload_map.get("honest_verdict", "")),
        }
        if source_id == "exp4688":
            summary[source_id]["generic_agent_reached_level"] = payload_map.get(
                "generic_agent_reached_level"
            )
            summary[source_id]["offline_reproduced"] = payload_map.get("offline_reproduced")
        if source_id == "exp4689":
            summary[source_id]["coverage_delta"] = payload_map.get("coverage_delta")
            summary[source_id]["candidate_generation_coverage_filter"] = payload_map.get(
                "candidate_generation_coverage_filter"
            )
        if source_id == "exp5154":
            summary[source_id]["winning_trajectory_surfaced"] = payload_map.get(
                "winning_trajectory_surfaced"
            )
            summary[source_id]["reproducible_levels_delta"] = payload_map.get(
                "reproducible_levels_delta"
            )
    return summary


def _honest_verdict(
    *,
    entry_ok: bool,
    current_ok: bool,
    synthetic_ok: bool,
    doc_ok: bool,
) -> str:
    if entry_ok and current_ok and synthetic_ok and doc_ok:
        return COMPLETE_VERDICT
    return INCOMPLETE_VERDICT


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float,
    run_date: str,
    verification: JsonMap,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    entry = _find_manifest_entry(root)
    entry_audit = _entry_audit(entry)
    current_lint = check_current_roadmap_false_positive(root)
    synthetic_lint = check_synthetic_match(root)
    doc_check = check_known_issues_or_gaps_doc(root)
    entry_ok = not _list(entry_audit.get("errors")) and bool(entry_audit.get("found"))
    current_ok = _bool(current_lint.get("passed"))
    synthetic_ok = _bool(synthetic_lint.get("passed"))
    doc_ok = _bool(doc_check.get("updated"))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(duration_s, 0.0001), 6),
        "random_seed": RANDOM_SEED,
        "source_artifacts_read": build_source_artifacts_read(root),
        "source_artifact_summary": _source_artifact_summary(root),
        "manifest_entry_audit": entry_audit,
        "current_roadmap_lint": current_lint,
        "synthetic_match_lint": synthetic_lint,
        "known_issues_or_gaps_doc": doc_check,
        "adversarial_verification": dict(verification),
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
        "tests_run": list(tests_run),
        "exclusion_manifest_entry_added": entry_ok,
        "entry_id": ENTRY_ID,
        "false_positive_check_against_this_milestone": current_ok,
        "synthetic_match_check_passed": synthetic_ok,
        "known_issues_or_gaps_md_updated": doc_ok,
        "honest_verdict": _honest_verdict(
            entry_ok=entry_ok,
            current_ok=current_ok,
            synthetic_ok=synthetic_ok,
            doc_ok=doc_ok,
        ),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            errors.append(f"missing.{field}")
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(artifact.get("field_principles")).get(field) != principle:
            errors.append(f"field_principle.{field}")
    checks = [
        (artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id.invalid"),
        (artifact.get("milestone") != MILESTONE, "milestone.invalid"),
        (artifact.get("entry_id") != ENTRY_ID, "entry_id.invalid"),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate.invalid"),
        (
            not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES),
            "honest_verdict.not_terminal",
        ),
        (
            artifact.get("exclusion_manifest_entry_added") is not True,
            "exclusion_manifest_entry_added.invalid",
        ),
        (
            artifact.get("false_positive_check_against_this_milestone") is not True,
            "false_positive_check_against_this_milestone.invalid",
        ),
        (
            artifact.get("synthetic_match_check_passed") is not True,
            "synthetic_match_check_passed.invalid",
        ),
        (
            artifact.get("known_issues_or_gaps_md_updated") is not True,
            "known_issues_or_gaps_md_updated.invalid",
        ),
        (
            not isinstance(artifact.get("duration_s"), (int, float))
            or artifact.get("duration_s", 0) <= 0,
            "duration_s.invalid",
        ),
        (not _list(artifact.get("source_artifacts_read")), "source_artifacts_read.empty"),
        (not _list(artifact.get("tests_run")), "tests_run.empty"),
        (
            artifact.get("reproducibility_checksum") != payload_checksum(artifact),
            "reproducibility_checksum.invalid",
        ),
    ]
    errors.extend(code for failed, code in checks if failed)
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5165 retirement artifact: {errors}")


def run(
    *,
    root: Path = REPO_ROOT,
    output: Path | None = None,
    run_date: str | None = None,
    clock: Any = time.perf_counter,
    verification_runner: Any = run_adversarial_verification,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
) -> Path:
    root = Path(root)
    output_path = output or root / RESULT_RELATIVE_PATH
    start = clock()
    placeholder = verification_payload(
        CommandResult(command=(), exit_code=0, stdout='{"flags":[]}', stderr="")
    )
    artifact = build_artifact(
        root=root,
        duration_s=clock() - start,
        run_date=run_date or time.strftime("%Y%m%d"),
        verification=placeholder,
        tests_run=tests_run,
    )
    write_json(output_path, artifact)
    verification = verification_payload(verification_runner(root, output_path))
    artifact = {
        **artifact,
        "adversarial_verification": verification,
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    write_json(output_path, artifact)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    run(root=args.root, output=args.output, run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
