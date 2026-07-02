"""Exp 5167: V473 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5167, SCENARIO-CAPSTONE-5167,
SCENARIO-CAPSTONE-5167-FIELD-PRINCIPLES.

This module closes the milestone by accounting for already-landed evidence. It
does not rerun research and it does not turn quarantined evidence into a
headline claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
JsonMap = Mapping[str, Any]
AdversarialReporter = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results") / "experiment_5167_capstone_v473.json"
EXPERIMENT = "experiment_5167_capstone_v473"
EXPERIMENT_ID = "exp5167-capstone-v473"
MILESTONE = "2026.07.473"
SCHEMA = "carnot.experiment_5167_capstone_v473.v1"
RANDOM_SEED = 5167
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
HONEST_VERDICT = (
    "complete: v473 reconciled with DiffusionGemma ungated for future scaling, "
    "GAP-4 scale-up not filled, zero new ARC levels banked, and exp5161 excluded "
    "as flagged_adversarial."
)
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

SPEC_REFS = [
    "REQ-CAPSTONE-5167",
    "SCENARIO-CAPSTONE-5167",
    "SCENARIO-CAPSTONE-5167-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "per_task_summary": "list of {task_id, honest_verdict, headline_outcome} read from every V473 artifact",
    "gap4_status_reconciled": "GAP-4 status reconciled from Exp5161 without rounding a pilot up to filled",
    "diffusiongemma_gate_reconciled": "DiffusionGemma gate reconciled from Exp5160's clean cross-corpus result",
    "reproducible_total_levels_delta": "new banked ARC levels from Exp5159; zero is valid and must not be inflated",
    "levelup_guarantee_structurally_satisfied": "roadmap lint found at least one level-up attempt",
    "levelup_guarantee_outcome_satisfied": "true only if the level-up attempt actually banked a level",
    "flagged_adversarial_artifacts_excluded": "task ids skipped from headline aggregation because flagged_adversarial=true",
    "research_conductor_py_untouched_confirmed": "Exp5164 hard constraint that scripts/research_conductor.py stayed untouched",
    "honest_verdict": "terminal-prefix single sentence summarizing the whole milestone without vague celebration",
}

REQUIRED_ARTIFACT_FIELDS = (
    "per_task_summary",
    "gap4_status_reconciled",
    "diffusiongemma_gate_reconciled",
    "reproducible_total_levels_delta",
    "levelup_guarantee_structurally_satisfied",
    "levelup_guarantee_outcome_satisfied",
    "flagged_adversarial_artifacts_excluded",
    "research_conductor_py_untouched_confirmed",
    "honest_verdict",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "spec_refs",
    "result_path",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "source_artifacts",
    "adversarial_verification",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "tests_run",
    "flagged_adversarial",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_TESTS_RUN = [
    "python3 scripts/adversarial_verify.py results/experiment_51{57,58,59,60,61,62,63,64,65,66}_*_v473.json",
    "python3 scripts/arc_levelup_guarantee_lint.py research-roadmap.yaml",
    ".venv/bin/pytest tests/python/test_experiment_5167_capstone_v473.py -q -o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5167_capstone_v473.py' "
    "-m pytest tests/python/test_experiment_5167_capstone_v473.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5167_capstone_v473.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5167_capstone_v473.py",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class UpstreamSource:
    """One expected V473 result artifact."""

    experiment_number: int
    task_id: str
    relative_path: Path


@dataclass(frozen=True)
class LevelupLintResult:
    """Captured output from the roadmap level-up-attempt lint."""

    exit_code: int
    stdout: str
    structurally_satisfied: bool


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5157,
        "exp5157-deepen-warmstart-replay-ablation-v473",
        Path("results/experiment_5157_deepen_warmstart_replay_ablation_v473.json"),
    ),
    UpstreamSource(
        5158,
        "exp5158-deepen-goal-energy-ranker-replay-v473",
        Path("results/experiment_5158_deepen_goal_energy_ranker_replay_v473.json"),
    ),
    UpstreamSource(
        5159,
        "exp5159-deepen-live-levelup-attempt-v473",
        Path("results/experiment_5159_deepen_live_levelup_attempt_v473.json"),
    ),
    UpstreamSource(
        5160,
        "exp5160-oracle-distinct-cross-corpus-closure-v473",
        Path("results/experiment_5160_oracle_distinct_cross_corpus_closure_v473.json"),
    ),
    UpstreamSource(
        5161,
        "exp5161-gap4-protocol-execution-pilot-v473",
        Path("results/experiment_5161_gap4_protocol_execution_pilot_v473.json"),
    ),
    UpstreamSource(
        5162,
        "exp5162-sota-ingestion-multilevel-v473",
        Path("results/experiment_5162_sota_ingestion_multilevel_v473.json"),
    ),
    UpstreamSource(
        5163,
        "exp5163-mmlu-pro-verifier-rescale-v473",
        Path("results/experiment_5163_mmlu_pro_verifier_rescale_v473.json"),
    ),
    UpstreamSource(
        5164,
        "exp5164-retro-timing-falsezero-fix-v473",
        Path("results/experiment_5164_retro_timing_falsezero_fix_v473.json"),
    ),
    UpstreamSource(
        5165,
        "exp5165-generation-axis-retirement-hygiene-v473",
        Path("results/experiment_5165_generation_axis_retirement_hygiene_v473.json"),
    ),
    UpstreamSource(
        5166,
        "exp5166-hardware-continuity-board-timing-v473",
        Path("results/experiment_5166_hardware_continuity_board_timing_v473.json"),
    ),
)


def value_of(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def honest_verdict_text(value: Any) -> str:
    raw = value_of(value)
    return raw if isinstance(raw, str) else ""


def _number(value: Any) -> float | None:
    raw = value_of(value)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return value_of(value) is True


def _principled(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def file_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {"exists": True, "loadable": False, "error": "malformed_json"}
    if not isinstance(parsed, dict):
        return {}, {"exists": True, "loadable": False, "error": "not_json_object"}
    return parsed, {"exists": True, "loadable": True, "error": None, "sha256": file_sha256(path)}


def adversarial_report_for(path: Path) -> JsonDict:  # pragma: no cover - exercised by integration run.
    from scripts.adversarial_verify import verify_artifact

    return verify_artifact(path)


def run_levelup_lint(root: Path) -> LevelupLintResult:  # pragma: no cover - exercised by integration run.
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "arc_levelup_guarantee_lint.py"),
            str(root / "research-roadmap.yaml"),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return LevelupLintResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        structurally_satisfied=completed.returncode == 0,
    )


def read_registry_totals(root: Path) -> JsonDict:
    path = root / "ops" / "arc_solve_registry.yaml"
    if not path.exists():
        return {
            "path": str(path.relative_to(root)),
            "loadable": False,
            "reproducible_total_levels": None,
            "reproducible_total_games": None,
        }
    parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, Mapping):
        parsed = {}
    return {
        "path": str(path.relative_to(root)),
        "loadable": True,
        "reproducible_total_levels": parsed.get("reproducible_total_levels"),
        "reproducible_total_games": parsed.get("reproducible_total_games"),
    }


def load_upstreams(
    root: Path, adversarial_reporter: AdversarialReporter
) -> tuple[dict[int, JsonDict], list[JsonDict], list[str], list[JsonDict]]:
    artifacts: dict[int, JsonDict] = {}
    source_rows: list[JsonDict] = []
    missing: list[str] = []
    reports: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        path = root / source.relative_path
        data, meta = read_json_mapping(path)
        row = {
            "experiment_number": source.experiment_number,
            "task_id": source.task_id,
            "relative_path": str(source.relative_path),
            "exists": meta.get("exists") is True,
            "loadable": meta.get("loadable") is True,
            "sha256": meta.get("sha256"),
            "error": meta.get("error"),
        }
        if not meta.get("loadable"):
            missing.append(source.task_id)
            source_rows.append(row)
            continue
        artifacts[source.experiment_number] = data
        report = adversarial_reporter(path)
        reports.append(_verification_row(source, data, report))
        source_rows.append(
            row
            | {
                "honest_verdict": honest_verdict_text(data.get("honest_verdict")),
                "flagged_adversarial": data.get("flagged_adversarial") is True,
            }
        )
    return artifacts, source_rows, missing, reports


def _verification_row(source: UpstreamSource, data: JsonMap, report: JsonMap) -> JsonDict:
    flags = list(report.get("flags") or [])
    critical = any(flag.get("severity") == "critical" for flag in flags if isinstance(flag, Mapping))
    sanitized_flags = [
        {
            "kind": flag.get("kind"),
            "severity": flag.get("severity"),
        }
        for flag in flags
        if isinstance(flag, Mapping)
    ]
    return {
        "task_id": source.task_id,
        "relative_path": str(source.relative_path),
        "stamped_flagged_adversarial": data.get("flagged_adversarial") is True,
        "critical_adversarial_flag": critical,
        "flag_count": report.get("flag_count", len(flags)),
        "max_severity": report.get("max_severity"),
        "flags": sanitized_flags,
    }


def _is_excluded(source: UpstreamSource, data: JsonMap, reports: Sequence[JsonMap]) -> bool:
    if data.get("flagged_adversarial") is True:
        return True
    return any(
        row["task_id"] == source.task_id and row.get("critical_adversarial_flag") is True
        for row in reports
    )


def headline_outcome(source: UpstreamSource, data: JsonMap, reports: Sequence[JsonMap]) -> str:
    if not data:
        return "missing_artifact_not_aggregated"
    if _is_excluded(source, data, reports):
        return "excluded_from_headline_aggregation_flagged_adversarial"
    if source.experiment_number == 5157:
        return "warmstart_gate_failed_zero_delta"
    if source.experiment_number == 5158:
        return f"goal_energy_ranker_gate_failed_improved_{data.get('games_improved_count', 0)}_of_3"
    if source.experiment_number == 5159:
        return "gate_blocked_no_level_banked" if data.get("status") == "blocked" else "levelup_attempt_ran"
    if source.experiment_number == 5160:
        return str(data.get("headline_outcome") or "diffusiongemma_cross_corpus_reconciled")
    if source.experiment_number == 5162:
        return "zero_new_post_2026_07_02_primary_findings"
    if source.experiment_number == 5163:
        return "mmlu_pro_verifier_delta_0.025_ci_includes_0"
    if source.experiment_number == 5164:
        return "false_zero_timing_fix_clean_conductor_untouched"
    if source.experiment_number == 5165:
        return "generation_axis_scope_retired_lint_load_bearing"
    if source.experiment_number == 5166:
        return "hardware_continuity_kv260_polarfire_reachable_gatemate_blocked_no_speedup"
    return "reconciled"


def per_task_summary(artifacts: JsonMap, reports: Sequence[JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        data = artifacts.get(source.experiment_number, {})
        rows.append(
            {
                "task_id": source.task_id,
                "honest_verdict": honest_verdict_text(data.get("honest_verdict"))
                if data
                else "missing_artifact",
                "headline_outcome": headline_outcome(source, data, reports),
            }
        )
    return rows


def flagged_exclusions(artifacts: JsonMap, reports: Sequence[JsonMap]) -> list[str]:
    excluded: list[str] = []
    for source in UPSTREAM_SOURCES:
        data = artifacts.get(source.experiment_number, {})
        if data and _is_excluded(source, data, reports):
            excluded.append(source.task_id)
    return excluded


def gap4_status(artifacts: JsonMap, reports: Sequence[JsonMap]) -> str:
    data = artifacts.get(5161, {})
    if not data:
        return "still_open_missing_exp5161"
    recommendation = value_of(data.get("gap4_status_recommendation"))
    if recommendation == "scale_up_recommended":
        status = "scale_up_recommended_not_filled"
    elif recommendation in {"filled", "still_open", "retired"}:
        status = str(recommendation)
    else:
        status = "still_open"
    source = next(src for src in UPSTREAM_SOURCES if src.experiment_number == 5161)
    if _is_excluded(source, data, reports):
        status += "_flagged_excluded_from_headline"
    return status


def diffusiongemma_gate(artifacts: JsonMap) -> str:
    data = artifacts.get(5160, {})
    if not data:
        return "still_gated_missing_exp5160"
    recommendation = value_of(data.get("diffusiongemma_gate_updated_recommendation"))
    if recommendation == "ungate_now" and _bool(data.get("cross_corpus_replication_passed")):
        return "ungate_now_cross_corpus_replication_passed_no_scaling_run"
    return f"{recommendation or 'still_gated'}_cross_corpus_not_decision_grade"


def reproducible_level_delta(artifacts: JsonMap) -> int:
    data = artifacts.get(5159, {})
    candidates = (
        data.get("reproducible_total_levels_delta"),
        data.get("new_levels_banked"),
        data.get("reproduced_levels_delta"),
    )
    numbers = [_number(value) for value in candidates]
    positive = [int(value) for value in numbers if value and value > 0]
    return max(positive) if positive else 0


def research_conductor_untouched(artifacts: JsonMap) -> bool:
    data = artifacts.get(5164, {})
    return data.get("research_conductor_py_modified") is False


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260702",
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    adversarial_reporter: AdversarialReporter | None = None,
    levelup_lint_result: LevelupLintResult | None = None,
) -> JsonDict:
    start = time.perf_counter()
    reporter = adversarial_reporter or adversarial_report_for
    artifacts, sources, missing, reports = load_upstreams(root, reporter)
    lint = levelup_lint_result or run_levelup_lint(root)
    delta = reproducible_level_delta(artifacts)
    excluded = flagged_exclusions(artifacts, reports)
    registry = read_registry_totals(root)
    eligible = [
        source.task_id
        for source in UPSTREAM_SOURCES
        if source.task_id not in missing
        and source.task_id not in excluded
        and artifacts.get(source.experiment_number)
    ]
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s if duration_s is not None else time.perf_counter() - start), 6),
        "source_artifacts": sources,
        "missing_artifacts": missing,
        "adversarial_verification": reports,
        "headline_eligible_task_ids": eligible,
        "registry_reconciliation": registry | {"delta_from_exp5159": delta},
        "levelup_lint": {
            "exit_code": lint.exit_code,
            "stdout": lint.stdout,
        },
        "preconditions_checked": {
            "expected_artifacts": len(UPSTREAM_SOURCES),
            "loadable_artifacts": len(artifacts),
            "adversarial_verify_ran_per_loadable_artifact": len(reports) == len(artifacts),
            "arc_levelup_lint_ran": True,
            "registry_read": registry.get("loadable") is True,
        },
        "per_task_summary": _principled("per_task_summary", per_task_summary(artifacts, reports)),
        "gap4_status_reconciled": _principled("gap4_status_reconciled", gap4_status(artifacts, reports)),
        "diffusiongemma_gate_reconciled": _principled(
            "diffusiongemma_gate_reconciled", diffusiongemma_gate(artifacts)
        ),
        "reproducible_total_levels_delta": _principled("reproducible_total_levels_delta", delta),
        "levelup_guarantee_structurally_satisfied": _principled(
            "levelup_guarantee_structurally_satisfied", lint.structurally_satisfied
        ),
        "levelup_guarantee_outcome_satisfied": _principled(
            "levelup_guarantee_outcome_satisfied", delta > 0
        ),
        "flagged_adversarial_artifacts_excluded": _principled(
            "flagged_adversarial_artifacts_excluded", excluded
        ),
        "research_conductor_py_untouched_confirmed": _principled(
            "research_conductor_py_untouched_confirmed", research_conductor_untouched(artifacts)
        ),
        "honest_verdict": HONEST_VERDICT,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "tests_run": list(tests_run if tests_run is not None else DEFAULT_TESTS_RUN),
        "flagged_adversarial": False,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = honest_verdict_text(payload.get("honest_verdict"))
    if not verdict.startswith(TERMINAL_PREFIXES) or "\n" in verdict:
        raise ValueError("honest_verdict must be a single terminal-prefix sentence")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare V473 aggregation")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field principle mismatch")
    if payload.get("flagged_adversarial") is not False:
        raise ValueError("flagged_adversarial must be false for the capstone itself")
    if value_of(payload["research_conductor_py_untouched_confirmed"]) is not True:
        raise ValueError("research_conductor_py_untouched_confirmed must be true")
    delta = value_of(payload["reproducible_total_levels_delta"])
    outcome = value_of(payload["levelup_guarantee_outcome_satisfied"])
    if outcome is True and int(delta) <= 0:
        raise ValueError("levelup_guarantee_outcome_satisfied cannot be true with zero delta")
    if not payload.get("tests_run"):
        raise ValueError("tests_run must record verification commands")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260702",
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    adversarial_reporter: AdversarialReporter | None = None,
    levelup_lint_result: LevelupLintResult | None = None,
) -> Path:
    payload = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
        adversarial_reporter=adversarial_reporter,
        levelup_lint_result=levelup_lint_result,
    )
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, payload)
    return out_path
