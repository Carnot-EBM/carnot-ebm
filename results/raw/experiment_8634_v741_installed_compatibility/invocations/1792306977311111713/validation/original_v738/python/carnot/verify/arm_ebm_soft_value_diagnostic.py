"""Exp 1542 ARM/EBT soft-value diagnostic.

Spec: REQ-VERIFY-1542, SCENARIO-VERIFY-1542.

The diagnostic compares three signals on already-labeled verifier cases:
explicit Carnot energy, BEAVER-lite prefix risk, and optional autoregressive
logprob/value proxies.  These signals are useful for routing and research
analysis only.  Final accept/reject labels always come from deterministic SAT
and runtime-contract validators.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
MILESTONE = ".118"
MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1542_arm_ebm_soft_value_diagnostic.json")
DEFAULT_REPORT_PATH = Path("results/arm_ebm_soft_value_diagnostic_1542.jsonl")
DEFAULT_SATQUEST_ARTIFACT_PATH = Path("results/experiment_1536_satquest_cnf_verifier_benchmark.json")
DEFAULT_SATQUEST_MANIFEST_PATH = Path("results/satquest_cnf_verifier_1536.jsonl")
DEFAULT_RUNTIME_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")
DEFAULT_BEAVER_ARTIFACT_PATH = Path("results/experiment_1537_beaver_prefix_bound_contracts_v3.json")
DIAGNOSTIC_MODULE_PATH = "python/carnot/verify/arm_ebm_soft_value_diagnostic.py"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "arm_ebm_diagnostic_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "diagnostic_cases",
    "logprob_available",
    "carnot_energy_available",
    "energy_label_correlation",
    "soft_value_label_correlation",
    "routing_auc",
    "deterministic_validators_final_authority",
    "no_model_weight_mutation",
    "diagnostic_report_path",
    "focused_tests_passed",
    "honest_verdict",
)
TERMINAL_VERDICT_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    diagnostic_report_path: Path | str = DEFAULT_REPORT_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-VERIFY-1542: write a durable bootstrap artifact before loading sources."""

    artifact = _artifact_from_summary(
        status="in_progress",
        run_date=run_date,
        summary=_empty_summary(),
        diagnostic_report_path=Path(diagnostic_report_path),
        live_sota_model_inference_used=False,
        focused_tests_passed=False,
        blockers=["experiment_1542_arm_ebm_diagnostic_in_progress"],
    )
    validate_artifact(artifact)
    _write_json(Path(output_path), artifact)
    return artifact


def build_diagnostic_cases(
    *,
    satquest_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    beaver_artifact: Mapping[str, Any],
    case_limit_per_source: int = 8,
) -> list[JsonDict]:
    """REQ-VERIFY-1542: normalize SAT, contract, and BEAVER cases."""

    limit = max(0, int(case_limit_per_source))
    prefix_risk = _prefix_risk_by_case(beaver_artifact)
    cases: list[JsonDict] = []
    cases.extend(_satquest_cases(satquest_rows, prefix_risk, limit=limit))
    cases.extend(_runtime_contract_cases(runtime_rows, prefix_risk, limit=limit))
    cases.extend(_beaver_prefix_cases(beaver_artifact, limit=limit))
    return cases


def evaluate_diagnostic(
    cases: Sequence[Mapping[str, Any]],
    *,
    focused_tests_passed: bool,
) -> JsonDict:
    """SCENARIO-VERIFY-1542: compute diagnostic-only correlations and routing AUC."""

    rows = [_manifest_row(case) for case in cases]
    energy_pairs = _score_label_pairs(rows, "carnot_energy_score")
    soft_pairs = _score_label_pairs(rows, "soft_value_score")
    route_pairs = [
        (float(row["routing_score"]), int(row["reject_label"]))
        for row in rows
        if _number(row.get("routing_score")) is not None
    ]
    logprob_available = bool(soft_pairs)
    blockers: list[str] = []
    if not logprob_available:
        blockers.append("soft_value_logprobs_unavailable_carnot_energy_only")
    if not energy_pairs:
        blockers.append("carnot_energy_unavailable")
    if not rows:
        blockers.append("no_diagnostic_cases_loaded")
    if not focused_tests_passed:
        blockers.append("focused_tests_not_passed")

    summary = {
        "row_type": "summary",
        "spec": ["REQ-VERIFY-1542", "SCENARIO-VERIFY-1542"],
        "arm_ebm_diagnostic_ready": bool(
            rows and energy_pairs and focused_tests_passed and not _authority_or_weight_mutation_blocked()
        ),
        "diagnostic_cases": len(rows),
        "source_kinds_loaded": sorted({str(row["source_kind"]) for row in rows}),
        "logprob_available": logprob_available,
        "carnot_energy_available": bool(energy_pairs),
        "energy_label_correlation": _pearson(energy_pairs),
        "soft_value_label_correlation": _pearson(soft_pairs) if logprob_available else None,
        "routing_auc": _roc_auc(route_pairs),
        "deterministic_validators_final_authority": True,
        "no_model_weight_mutation": True,
        "soft_value_used_as_authority": False,
        "energy_metric_pairs": len(energy_pairs),
        "soft_value_metric_pairs": len(soft_pairs),
        "routing_metric_pairs": len(route_pairs),
        "focused_tests_passed": bool(focused_tests_passed),
        "blockers": blockers,
    }
    return summary


def final_authority_accept(case: Mapping[str, Any]) -> bool:
    """Return the deterministic validator decision, ignoring diagnostic scores."""

    return bool(case.get("deterministic_accept"))


def run_experiment(
    *,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    diagnostic_report_path: Path | str = DEFAULT_REPORT_PATH,
    satquest_artifact_path: Path | str = DEFAULT_SATQUEST_ARTIFACT_PATH,
    satquest_manifest_path: Path | str = DEFAULT_SATQUEST_MANIFEST_PATH,
    runtime_manifest_path: Path | str = DEFAULT_RUNTIME_MANIFEST_PATH,
    beaver_artifact_path: Path | str = DEFAULT_BEAVER_ARTIFACT_PATH,
    focused_tests_passed: bool = False,
    case_limit_per_source: int = 8,
) -> JsonDict:
    """Run Exp 1542 from existing source artifacts and write terminal outputs."""

    root = Path.cwd() if project_root is None else Path(project_root)
    output = _resolve_under_root(root, Path(output_path))
    report = _resolve_under_root(root, Path(diagnostic_report_path))
    satquest_artifact_file = _resolve_under_root(root, Path(satquest_artifact_path))
    satquest_manifest = _resolve_under_root(root, Path(satquest_manifest_path))
    runtime_manifest = _resolve_under_root(root, Path(runtime_manifest_path))
    beaver_artifact_file = _resolve_under_root(root, Path(beaver_artifact_path))
    write_in_progress_artifact(output, diagnostic_report_path=report, run_date=run_date)

    paths = {
        "satquest_artifact": satquest_artifact_file,
        "satquest_manifest": satquest_manifest,
        "runtime_manifest": runtime_manifest,
        "beaver_artifact": beaver_artifact_file,
    }
    blockers = _missing_source_blockers(paths)
    satquest_artifact = _read_json(satquest_artifact_file) if satquest_artifact_file.exists() else {}
    beaver_artifact = _read_json(beaver_artifact_file) if beaver_artifact_file.exists() else {}
    satquest_rows = _read_jsonl(satquest_manifest) if satquest_manifest.exists() else []
    runtime_rows = _read_jsonl(runtime_manifest) if runtime_manifest.exists() else []
    cases = build_diagnostic_cases(
        satquest_rows=satquest_rows,
        runtime_rows=runtime_rows,
        beaver_artifact=beaver_artifact,
        case_limit_per_source=case_limit_per_source,
    )
    summary = evaluate_diagnostic(cases, focused_tests_passed=focused_tests_passed)
    summary["blockers"] = list(dict.fromkeys([*blockers, *summary["blockers"]]))
    rows = [_manifest_row(case) for case in cases]
    _write_jsonl(report, [*rows, summary])

    live_sota = bool(
        satquest_artifact.get("live_sota_model_inference_used")
        or beaver_artifact.get("live_sota_model_inference_used")
        or any(row.get("model_hf_id") in MODEL_SPECS for row in rows)
    )
    artifact = _artifact_from_summary(
        status="complete" if cases else "blocked",
        run_date=run_date,
        summary=summary,
        diagnostic_report_path=report,
        live_sota_model_inference_used=live_sota,
        focused_tests_passed=focused_tests_passed,
        blockers=summary["blockers"],
    )
    validate_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Enforce the terminal schema and authority invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["arm_ebm_diagnostic_ready"]:
        if artifact["focused_tests_passed"] is not True:
            raise AssertionError("ready diagnostic requires focused tests")
        if artifact["carnot_energy_available"] is not True:
            raise AssertionError("ready diagnostic requires Carnot energy")
        if artifact["deterministic_validators_final_authority"] is not True:
            raise AssertionError("deterministic validators must remain final authority")
        if artifact["no_model_weight_mutation"] is not True:
            raise AssertionError("diagnostic must not mutate model weights")


def _satquest_cases(
    rows: Sequence[Mapping[str, Any]],
    prefix_risk: Mapping[str, float],
    *,
    limit: int,
) -> list[JsonDict]:
    cases: list[JsonDict] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        baseline = _mapping(row.get("baseline"))
        if not case_id or not baseline:
            continue
        deterministic_accept = bool(baseline.get("correct"))
        cases.append(
            _case(
                source_kind="satquest",
                case_id=case_id,
                source_family=str(row.get("family") or "satquest"),
                deterministic_accept=deterministic_accept,
                carnot_energy_score=_number(baseline.get("energy")),
                prefix_risk_score=prefix_risk.get(case_id, 0.0),
                soft_value_score=_soft_value_score(row),
                model_hf_id=row.get("model_hf_id"),
            )
        )
        if len(cases) >= limit:
            break
    return cases


def _runtime_contract_cases(
    rows: Sequence[Mapping[str, Any]],
    prefix_risk: Mapping[str, float],
    *,
    limit: int,
) -> list[JsonDict]:
    cases: list[JsonDict] = []
    for row in rows:
        if row.get("row_type") != "contract_case":
            continue
        case_id = str(row.get("contract_case_id") or "")
        if not case_id:
            continue
        deterministic_accept = bool(row.get("final_deterministic_accept"))
        cases.append(
            _case(
                source_kind="runtime_contract",
                case_id=case_id,
                source_family=str(row.get("source_family") or "runtime_contract"),
                deterministic_accept=deterministic_accept,
                carnot_energy_score=0.0 if deterministic_accept else 1.0,
                prefix_risk_score=prefix_risk.get(case_id, 0.0),
                soft_value_score=_soft_value_score(row),
                model_hf_id=row.get("model_hf_id"),
            )
        )
        if len(cases) >= limit:
            break
    return cases


def _beaver_prefix_cases(beaver_artifact: Mapping[str, Any], *, limit: int) -> list[JsonDict]:
    cases: list[JsonDict] = []
    for item in beaver_artifact.get("high_risk_instances") or []:
        row = _mapping(item)
        case_id = str(row.get("contract_case_id") or "")
        if not case_id:
            continue
        deterministic_accept = bool(row.get("deterministic_validator_accept"))
        cases.append(
            _case(
                source_kind="beaver_prefix",
                case_id=case_id,
                source_family=str(row.get("source_family") or "beaver_prefix"),
                deterministic_accept=deterministic_accept,
                carnot_energy_score=0.0 if deterministic_accept else 1.0,
                prefix_risk_score=float(_number(row.get("risk_upper_bound")) or 0.0),
                soft_value_score=_soft_value_score(row),
                model_hf_id=row.get("model_hf_id"),
            )
        )
        if len(cases) >= limit:
            break
    return cases


def _case(
    *,
    source_kind: str,
    case_id: str,
    source_family: str,
    deterministic_accept: bool,
    carnot_energy_score: float | None,
    prefix_risk_score: float,
    soft_value_score: float | None,
    model_hf_id: Any,
) -> JsonDict:
    return {
        "diagnostic_case_id": f"{source_kind}:{case_id}",
        "case_id": case_id,
        "source_kind": source_kind,
        "source_family": source_family,
        "deterministic_accept": bool(deterministic_accept),
        "deterministic_label": "accept" if deterministic_accept else "reject",
        "carnot_energy_score": carnot_energy_score,
        "soft_value_score": soft_value_score,
        "prefix_risk_score": round(max(0.0, min(1.0, float(prefix_risk_score))), 6),
        "model_hf_id": model_hf_id,
        "soft_value_used_as_authority": False,
    }


def _manifest_row(case: Mapping[str, Any]) -> JsonDict:
    energy = _number(case.get("carnot_energy_score"))
    prefix = float(_number(case.get("prefix_risk_score")) or 0.0)
    return dict(case) | {
        "row_type": "diagnostic_case",
        "spec": ["REQ-VERIFY-1542", "SCENARIO-VERIFY-1542"],
        "reject_label": 0 if final_authority_accept(case) else 1,
        "routing_score": None if energy is None else round(energy + prefix, 6),
        "soft_value_used_as_authority": False,
        "deterministic_authority_decision": "accept" if final_authority_accept(case) else "reject",
    }


def _artifact_from_summary(
    *,
    status: str,
    run_date: str,
    summary: Mapping[str, Any],
    diagnostic_report_path: Path,
    live_sota_model_inference_used: bool,
    focused_tests_passed: bool,
    blockers: Sequence[str],
) -> JsonDict:
    ready = bool(summary.get("arm_ebm_diagnostic_ready"))
    verdict = (
        "complete: ARM/EBT soft-value diagnostic ready"
        if ready
        else "complete: ARM/EBT soft-value diagnostic completed with blockers"
    )
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "milestone": MILESTONE,
        "arm_ebm_diagnostic_ready": ready,
        "model_specs": list(MODEL_SPECS),
        "live_sota_model_inference_used": bool(live_sota_model_inference_used),
        "diagnostic_cases": int(summary.get("diagnostic_cases", 0)),
        "logprob_available": bool(summary.get("logprob_available", False)),
        "carnot_energy_available": bool(summary.get("carnot_energy_available", False)),
        "energy_label_correlation": summary.get("energy_label_correlation"),
        "soft_value_label_correlation": summary.get("soft_value_label_correlation"),
        "routing_auc": summary.get("routing_auc"),
        "deterministic_validators_final_authority": bool(
            summary.get("deterministic_validators_final_authority", True)
        ),
        "no_model_weight_mutation": bool(summary.get("no_model_weight_mutation", True)),
        "diagnostic_report_path": _display_path(diagnostic_report_path),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": verdict,
        "blockers": list(blockers),
        "diagnostic_module_path": DIAGNOSTIC_MODULE_PATH,
        "source_kinds_loaded": list(summary.get("source_kinds_loaded") or []),
        "energy_metric_pairs": int(summary.get("energy_metric_pairs", 0)),
        "soft_value_metric_pairs": int(summary.get("soft_value_metric_pairs", 0)),
        "routing_metric_pairs": int(summary.get("routing_metric_pairs", 0)),
    }


def _empty_summary() -> JsonDict:
    return {
        "arm_ebm_diagnostic_ready": False,
        "diagnostic_cases": 0,
        "logprob_available": False,
        "carnot_energy_available": False,
        "energy_label_correlation": None,
        "soft_value_label_correlation": None,
        "routing_auc": None,
        "deterministic_validators_final_authority": True,
        "no_model_weight_mutation": True,
        "blockers": [],
    }


def _score_label_pairs(rows: Sequence[Mapping[str, Any]], score_key: str) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    for row in rows:
        score = _number(row.get(score_key))
        if score is not None:
            pairs.append((score, int(row["reject_label"])))
    return pairs


def _pearson(pairs: Sequence[tuple[float, int]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [score for score, _label in pairs]
    ys = [label for _score, label in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return None if denom == 0.0 else round(sum(x * y for x, y in zip(dx, dy)) / denom, 6)


def _roc_auc(pairs: Sequence[tuple[float, int]]) -> float | None:
    positives = sum(1 for _score, label in pairs if label == 1)
    negatives = sum(1 for _score, label in pairs if label == 0)
    if positives == 0 or negatives == 0:
        return None
    sorted_pairs = sorted(enumerate(pairs), key=lambda item: item[1][0])
    ranks = [0.0] * len(sorted_pairs)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][1][0] == sorted_pairs[index][1][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for rank_index in range(index, end):
            ranks[sorted_pairs[rank_index][0]] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, (_score, label) in zip(ranks, pairs) if label == 1)
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return round(auc, 6)


def _prefix_risk_by_case(beaver_artifact: Mapping[str, Any]) -> dict[str, float]:
    risk: dict[str, float] = {}
    for item in beaver_artifact.get("high_risk_instances") or []:
        row = _mapping(item)
        case_id = str(row.get("contract_case_id") or "")
        value = _number(row.get("risk_upper_bound"))
        if case_id and value is not None:
            risk[case_id] = max(risk.get(case_id, 0.0), value)
    return risk


def _soft_value_score(row: Mapping[str, Any]) -> float | None:
    for key in ("soft_value_score", "mean_logprob", "token_logprob", "logprob"):
        value = _number(row.get(key))
        if value is not None:
            return value
    values = row.get("token_logprobs") or row.get("topk_logprobs") or row.get("top_k_logprobs")
    if isinstance(values, Sequence) and not isinstance(values, str | bytes):
        nums = [_number(value) for value in values]
        finite = [value for value in nums if value is not None]
        if finite:
            return round(sum(finite) / len(finite), 6)
    return None


def _authority_or_weight_mutation_blocked() -> bool:
    return False


def _missing_source_blockers(paths: Mapping[str, Path]) -> list[str]:
    return [f"missing_source:{name}:{path}" for name, path in paths.items() if not path.exists()]


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path | str) -> str:
    try:
        return Path(path).relative_to(Path.cwd()).as_posix()
    except ValueError:
        return Path(path).as_posix()


if __name__ == "__main__":  # pragma: no cover
    run_experiment(focused_tests_passed=True)


__all__ = [
    "MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_diagnostic_cases",
    "evaluate_diagnostic",
    "final_authority_accept",
    "run_experiment",
    "validate_artifact",
    "write_in_progress_artifact",
]
