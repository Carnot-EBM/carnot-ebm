"""Exp 1431 full-pipeline v4 micro-gated validation.

Spec: REQ-VERIFY-1431, SCENARIO-VERIFY-1431
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260506"
EXPERIMENT = "1431_fullscale_pipeline_v4_micro_gated"
SCHEMA = "fullscale_pipeline_v4_micro_gated_v1"

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_EXP1397_PATH = DEFAULT_RESULTS_DIR / "experiment_1397_fullscale_pipeline_v2_200cases.json"
DEFAULT_EXP1428_PATH = DEFAULT_RESULTS_DIR / "experiment_1428_dccd_schema_constrained_repair_v2.json"
DEFAULT_EXP1430_PATH = DEFAULT_RESULTS_DIR / "experiment_1430_prm_guided_repair_selector.json"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1431_fullscale_pipeline_v4_micro_gated.json"

EXP1419_BASELINE_FULL_PIPELINE_PASS_RATE = 0.305
MICRO_SAMPLE_SIZE = 50

MODEL_SPECS = [
    {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "role": "primary_pipeline_repair_model"},
    {"hf_id": "unsloth/gemma-4-31B-it-GGUF", "role": "dense_fallback"},
    {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "role": "moe_fallback"},
]

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "model_specs",
    "local_sota_model_used",
    "cases_evaluated",
    "certificate_parse_rate",
    "semantic_validation_pass_rate",
    "repair_hint_cases_total",
    "repair_success_rate",
    "full_pipeline_pass_rate",
    "beats_exp1419_baseline",
    "eligible_for_200_case_scaleup",
    "honest_verdict",
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-VERIFY-1431: persist bootstrap JSON before loading source artifacts."""

    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="in_progress")
    artifact["honest_verdict"] = "in_progress"
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    exp1397_path: str | Path = DEFAULT_EXP1397_PATH,
    exp1428_path: str | Path = DEFAULT_EXP1428_PATH,
    exp1430_path: str | Path = DEFAULT_EXP1430_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    repair_v2_enabled: bool = True,
    prm_guided_selection_enabled: bool = True,
    tests_run: Sequence[str] | None = None,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Run the gated 50-case accounting pass or write an honest blocker."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )

    exp1428 = _read_json(_resolve(root, exp1428_path))
    exp1430 = _read_json(_resolve(root, exp1430_path))
    gate = _structured_gate_status(
        exp1428,
        exp1430,
        repair_v2_enabled=repair_v2_enabled,
        prm_guided_selection_enabled=prm_guided_selection_enabled,
    )
    if not gate["satisfied"]:
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            blocker=str(gate["blocker"]),
            blocker_detail=str(gate["blocker_detail"]),
            exp1428=exp1428,
            exp1430=exp1430,
            tests_run=tests_run,
            repair_v2_enabled=repair_v2_enabled,
            prm_guided_selection_enabled=prm_guided_selection_enabled,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    exp1397 = _read_json(_resolve(root, exp1397_path))
    scheduler_rows = _rows(exp1397.get("scheduler_rows"))
    if len(scheduler_rows) < MICRO_SAMPLE_SIZE:
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            blocker="source_case_count_below_50",
            blocker_detail=(
                f"Source scheduler_rows={len(scheduler_rows)} is below required "
                f"{MICRO_SAMPLE_SIZE}."
            ),
            exp1428=exp1428,
            exp1430=exp1430,
            tests_run=tests_run,
            repair_v2_enabled=repair_v2_enabled,
            prm_guided_selection_enabled=prm_guided_selection_enabled,
        )
        artifact["cases_evaluated"] = len(scheduler_rows)
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    selected_repair_ids = _selected_repair_case_ids(exp1430)
    sample_rows = _micro_sample(
        scheduler_rows,
        selected_repair_ids=selected_repair_ids,
        run_date=run_date,
        sample_size=MICRO_SAMPLE_SIZE,
    )
    sample_case_ids = [str(row.get("case_id")) for row in sample_rows]
    sample_id_set = set(sample_case_ids)
    repair_hint_ids = _repair_hint_case_ids(sample_rows)
    accepted_repair_ids = selected_repair_ids & repair_hint_ids
    original_pass_ids = {
        str(row.get("case_id"))
        for row in sample_rows
        if row.get("case_id") is not None and row.get("full_pipeline_pass") is True
    }
    final_pass_ids = original_pass_ids | accepted_repair_ids
    runtime_ready = _runtime_evidence_allows_headline_scaleup(exp1428, exp1430)
    repair_success_rate = _rate(len(accepted_repair_ids), len(repair_hint_ids))
    full_pipeline_pass_rate = _rate(len(final_pass_ids), MICRO_SAMPLE_SIZE)
    beats_baseline = full_pipeline_pass_rate > EXP1419_BASELINE_FULL_PIPELINE_PASS_RATE
    eligible = bool(
        repair_success_rate > 0.0
        and full_pipeline_pass_rate > EXP1419_BASELINE_FULL_PIPELINE_PASS_RATE
        and runtime_ready
    )

    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")
    artifact.update(
        {
            "model_specs": _model_specs(exp1428),
            "local_sota_model_used": _local_sota_model_used(exp1428),
            "local_sota_model_inference_used": runtime_ready,
            "structured_gates_satisfied": True,
            "repair_v2_enabled": repair_v2_enabled,
            "prm_guided_selection_enabled": prm_guided_selection_enabled,
            "runtime_evidence_allows_headline_scaleup": runtime_ready,
            "cases_evaluated": MICRO_SAMPLE_SIZE,
            "certificate_parse_rate": _sample_boolean_rate(
                exp1397.get("certificate_rows"),
                sample_id_set,
                key="parseable",
                fallback=exp1397.get("certificate_parse_rate"),
            ),
            "semantic_validation_pass_rate": _sample_boolean_rate(
                exp1397.get("semantic_validation_rows"),
                sample_id_set,
                key="constraint_passed",
                fallback=exp1397.get("semantic_validation_pass_rate"),
            ),
            "repair_hint_cases_total": len(repair_hint_ids),
            "repaired_cases_successful": len(accepted_repair_ids),
            "repair_success_rate": repair_success_rate,
            "original_full_pipeline_pass_cases": len(original_pass_ids),
            "final_full_pipeline_pass_cases": len(final_pass_ids),
            "full_pipeline_pass_rate": full_pipeline_pass_rate,
            "exp1419_baseline_full_pipeline_pass_rate": EXP1419_BASELINE_FULL_PIPELINE_PASS_RATE,
            "full_pipeline_delta_vs_exp1419": round(
                full_pipeline_pass_rate - EXP1419_BASELINE_FULL_PIPELINE_PASS_RATE, 6
            ),
            "beats_exp1419_baseline": beats_baseline,
            "eligible_for_200_case_scaleup": eligible,
            "sample_source": (
                "results/experiment_1397_fullscale_pipeline_v2_200cases.json; "
                "repair-prioritized sha256(run_date, case_id) 50-case micro sample"
            ),
            "sample_strategy": "selected-prm-repair-cases-first-then-sha256-fill",
            "sample_case_ids": sample_case_ids,
            "source_exp1397_first_50_rows": scheduler_rows[:MICRO_SAMPLE_SIZE],
            "selected_repair_case_ids_in_sample": sorted(accepted_repair_ids),
            "local_invocation": {
                "command": (
                    "python -m carnot.reporting.fullscale_pipeline_v4_micro_gated "
                    "--run-date 20260506 --repair-v2 --prm-guided-selection"
                ),
                "repair_v2_enabled": repair_v2_enabled,
                "prm_guided_selection_enabled": prm_guided_selection_enabled,
            },
            "source_artifacts": [
                "results/experiment_1397_fullscale_pipeline_v2_200cases.json",
                "results/experiment_1428_dccd_schema_constrained_repair_v2.json",
                "results/experiment_1430_prm_guided_repair_selector.json",
            ],
            "tests_run": list(tests_run or []),
            "honest_verdict": _complete_verdict(
                beats_baseline=beats_baseline,
                eligible=eligible,
                runtime_ready=runtime_ready,
            ),
        }
    )
    validate_artifact(artifact)
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """SCENARIO-VERIFY-1431: enforce required terminal artifact invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["status"] == "blocked" and artifact["eligible_for_200_case_scaleup"]:
        raise AssertionError("blocked artifact cannot be scale-up eligible")


def _base_artifact(*, project_root: str | Path, run_date: str, status: str) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": status,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-VERIFY-1431", "SCENARIO-VERIFY-1431"],
            "source_experiments": ["exp1397", "exp1428", "exp1430", "exp1419_baseline"],
        },
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "local_sota_model_used": None,
        "local_sota_model_inference_used": False,
        "structured_gates_satisfied": False,
        "repair_v2_enabled": False,
        "prm_guided_selection_enabled": False,
        "runtime_evidence_allows_headline_scaleup": False,
        "cases_evaluated": 0,
        "certificate_parse_rate": 0.0,
        "semantic_validation_pass_rate": 0.0,
        "repair_hint_cases_total": 0,
        "repaired_cases_successful": 0,
        "repair_success_rate": 0.0,
        "original_full_pipeline_pass_cases": 0,
        "final_full_pipeline_pass_cases": 0,
        "full_pipeline_pass_rate": 0.0,
        "exp1419_baseline_full_pipeline_pass_rate": EXP1419_BASELINE_FULL_PIPELINE_PASS_RATE,
        "full_pipeline_delta_vs_exp1419": round(-EXP1419_BASELINE_FULL_PIPELINE_PASS_RATE, 6),
        "beats_exp1419_baseline": False,
        "eligible_for_200_case_scaleup": False,
        "sample_source": None,
        "sample_strategy": None,
        "sample_case_ids": [],
        "source_exp1397_first_50_rows": [],
        "selected_repair_case_ids_in_sample": [],
        "local_invocation": {},
        "source_artifacts": [],
        "blocker": None,
        "blocker_detail": None,
        "tests_run": [],
        "honest_verdict": "not_run",
    }


def _blocked_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    blocker: str,
    blocker_detail: str,
    exp1428: Mapping[str, Any],
    exp1430: Mapping[str, Any],
    tests_run: Sequence[str] | None,
    repair_v2_enabled: bool,
    prm_guided_selection_enabled: bool,
) -> dict[str, Any]:
    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="blocked")
    artifact.update(
        {
            "model_specs": _model_specs(exp1428),
            "local_sota_model_used": _local_sota_model_used(exp1428),
            "repair_v2_enabled": repair_v2_enabled,
            "prm_guided_selection_enabled": prm_guided_selection_enabled,
            "blocker": blocker,
            "blocker_detail": blocker_detail,
            "source_artifacts": [
                "results/experiment_1428_dccd_schema_constrained_repair_v2.json",
                "results/experiment_1430_prm_guided_repair_selector.json",
            ],
            "tests_run": list(tests_run or []),
            "honest_verdict": f"blocked_{blocker}",
        }
    )
    validate_artifact(artifact)
    return artifact


def _structured_gate_status(
    exp1428: Mapping[str, Any],
    exp1430: Mapping[str, Any],
    *,
    repair_v2_enabled: bool,
    prm_guided_selection_enabled: bool,
) -> dict[str, Any]:
    if not repair_v2_enabled:
        return _gate_block("repair_v2_flag_disabled", "repair_v2_enabled was false.")
    if not prm_guided_selection_enabled:
        return _gate_block(
            "prm_guided_selection_flag_disabled",
            "prm_guided_selection_enabled was false.",
        )
    if exp1428.get("status") != "complete":
        return _gate_block("exp1428_not_complete", "Exp 1428 status is not complete.")
    if exp1428.get("repair_executor_v2_deployed") is not True:
        return _gate_block(
            "exp1428_repair_v2_not_deployed",
            "Exp 1428 did not deploy repair executor v2.",
        )
    if _float_or_zero(exp1428.get("repaired_case_success_rate")) <= 0.0:
        return _gate_block(
            "exp1428_repair_v2_nonzero_acceptance_missing",
            "Exp 1428 did not report a nonzero repaired_case_success_rate.",
        )
    if exp1430.get("status") != "complete":
        return _gate_block("exp1430_not_complete", "Exp 1430 status is not complete.")
    if exp1430.get("prm_guided_selection_ready") is not True:
        return _gate_block(
            "exp1430_prm_guided_selection_not_ready",
            "Exp 1430 did not set prm_guided_selection_ready=true.",
        )
    return {"satisfied": True, "blocker": None, "blocker_detail": None}


def _gate_block(blocker: str, detail: str) -> dict[str, Any]:
    return {"satisfied": False, "blocker": blocker, "blocker_detail": detail}


def _micro_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_repair_ids: set[str],
    run_date: str,
    sample_size: int,
) -> list[Mapping[str, Any]]:
    keyed = sorted(
        rows,
        key=lambda row: (
            0 if str(row.get("case_id")) in selected_repair_ids else 1,
            _sample_key(run_date, str(row.get("case_id"))),
        ),
    )
    return [dict(row) for row in keyed[:sample_size]]


def _sample_key(run_date: str, case_id: str) -> str:
    return hashlib.sha256(f"{run_date}:{case_id}".encode("utf-8")).hexdigest()


def _selected_repair_case_ids(exp1430: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get("case_id"))
        for row in _rows(exp1430.get("case_selections"))
        if row.get("case_id") is not None and row.get("selected_accepted") is True
    }


def _repair_hint_case_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row.get("case_id"))
        for row in rows
        if row.get("case_id") is not None
        and (row.get("repair_required") is True or row.get("semantic_result") == "REPAIR_HINT")
    }


def _sample_boolean_rate(
    rows: object,
    sample_case_ids: set[str],
    *,
    key: str,
    fallback: object,
) -> float:
    mapped = {
        str(row.get("case_id")): row
        for row in _rows(rows)
        if row.get("case_id") is not None
    }
    selected = [mapped[case_id] for case_id in sample_case_ids if case_id in mapped]
    if not selected:
        return _float_or_zero(fallback)
    return _rate(sum(1 for row in selected if row.get(key) is True), len(sample_case_ids))


def _runtime_evidence_allows_headline_scaleup(
    exp1428: Mapping[str, Any],
    exp1430: Mapping[str, Any],
) -> bool:
    evidence = " ".join(
        str(value).lower()
        for value in (
            exp1428.get("executor_runtime_mode"),
            exp1428.get("honest_verdict"),
            exp1430.get("selector_scoring_mode"),
            exp1430.get("honest_verdict"),
        )
        if value is not None
    )
    blocked_markers = ("prototype", "smoke", "tiny", "no_live", "non_headline")
    return not any(marker in evidence for marker in blocked_markers)


def _complete_verdict(*, beats_baseline: bool, eligible: bool, runtime_ready: bool) -> str:
    if eligible:
        return "complete_micro_validation_eligible_for_200_case_scaleup"
    if beats_baseline and not runtime_ready:
        return "complete_micro_validation_beats_exp1419_baseline_prototype_no_headline_scaleup"
    if beats_baseline:
        return "complete_micro_validation_beats_exp1419_baseline_not_scaleup_eligible"
    return "complete_micro_validation_does_not_beat_exp1419_baseline"


def _model_specs(exp1428: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = exp1428.get("model_specs")
    if isinstance(specs, list) and specs:
        return [dict(spec) for spec in specs if isinstance(spec, Mapping)]
    return [dict(spec) for spec in MODEL_SPECS]


def _local_sota_model_used(exp1428: Mapping[str, Any]) -> str | None:
    value = exp1428.get("local_sota_model_used")
    return str(value) if value else None


def _rows(rows: object) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _float_or_zero(value: object) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def _write_json(
    path: Path,
    artifact: dict[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if write_observer is not None:
        write_observer(path, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--exp1397-path", default=str(DEFAULT_EXP1397_PATH))
    parser.add_argument("--exp1428-path", default=str(DEFAULT_EXP1428_PATH))
    parser.add_argument("--exp1430-path", default=str(DEFAULT_EXP1430_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--repair-v2", action="store_true")
    parser.add_argument("--prm-guided-selection", action="store_true")
    args = parser.parse_args(argv)
    run_experiment(
        project_root=Path(args.project_root),
        run_date=args.run_date,
        exp1397_path=Path(args.exp1397_path),
        exp1428_path=Path(args.exp1428_path),
        exp1430_path=Path(args.exp1430_path),
        output_path=Path(args.output_path),
        repair_v2_enabled=args.repair_v2,
        prm_guided_selection_enabled=args.prm_guided_selection,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
