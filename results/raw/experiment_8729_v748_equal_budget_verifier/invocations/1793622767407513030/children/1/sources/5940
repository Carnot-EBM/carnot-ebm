"""Exp 1556 ARM/EBT logprob telemetry repair diagnostic.

Spec: REQ-VERIFY-1556, SCENARIO-VERIFY-1556.

The adapter keeps the Exp 1542 authority boundary but repairs the missing
telemetry lane by reusing the local SOTA GGUF llama.cpp telemetry path.  Token
logprobs and top-k alternatives are treated as research diagnostics only: they
can help explain routing scores, but deterministic validators still make every
final accept/reject decision.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.reporting.live_sota_logprob_telemetry_preflight import (
    TelemetryCase,
    build_telemetry_artifact,
)

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
MILESTONE = "20260508"
MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1556_arm_ebm_logprob_telemetry_repair.json")
DEFAULT_TELEMETRY_MANIFEST_PATH = Path("results/arm_ebm_logprob_telemetry_manifest_1556.jsonl")
DEFAULT_DIAGNOSTIC_REPORT_PATH = Path("results/arm_ebm_logprob_telemetry_diagnostic_1556.jsonl")
DEFAULT_SATQUEST_MANIFEST_PATH = Path("results/satquest_sota_reeval_zero_false_accepts_1550.jsonl")
TELEMETRY_ADAPTER_PATH = "python/carnot/verify/arm_ebm_logprob_telemetry_repair.py"
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "arm_ebm_logprob_telemetry_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "logprob_available",
    "topk_available",
    "telemetry_adapter_path",
    "diagnostic_cases",
    "energy_label_correlation",
    "routing_auc",
    "deterministic_validators_final_authority",
    "telemetry_blockers",
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

TelemetryBuilder = Callable[..., Mapping[str, Any]]


def write_in_progress_artifact(output_path: Path | str = DEFAULT_ARTIFACT_PATH) -> JsonDict:
    """REQ-VERIFY-1556: write a durable bootstrap artifact before runtime work."""

    artifact = _artifact_from_summary(
        summary=_empty_summary(),
        status="in_progress",
        focused_tests_passed=False,
        telemetry_artifact={},
    )
    _write_json(Path(output_path), artifact)
    return artifact


def select_satquest_rows(rows: Sequence[Mapping[str, Any]], *, limit: int) -> list[JsonDict]:
    """Select a bounded SATQuest mix with both deterministic labels when present."""

    usable = [
        dict(row)
        for row in rows
        if row.get("case_id")
        and row.get("prompt")
        and _number(_mapping(row.get("baseline")).get("energy")) is not None
    ]
    accepts = [row for row in usable if bool(_mapping(row.get("baseline")).get("correct"))]
    rejects = [row for row in usable if not bool(_mapping(row.get("baseline")).get("correct"))]
    if accepts and rejects:
        half = max(1, int(limit) // 2)
        selected = [*accepts[:half], *rejects[: max(0, int(limit) - half)]]
    else:
        selected = usable[: max(0, int(limit))]
    return selected[: max(0, int(limit))]


def build_telemetry_cases(rows: Sequence[Mapping[str, Any]]) -> list[TelemetryCase]:
    """Turn deterministic source rows into prompts for the SOTA telemetry runner."""

    cases: list[TelemetryCase] = []
    for row in rows:
        oracle = _mapping(row.get("solver_oracle"))
        baseline = _mapping(row.get("baseline"))
        expected = str(oracle.get("label") or baseline.get("answer") or "")
        cases.append(
            TelemetryCase(
                case_id=str(row["case_id"]),
                family=str(row.get("family") or "satquest"),
                prompt=str(row["prompt"]),
                expected_answer=expected,
            )
        )
    return cases


def parse_telemetry_row(row: Mapping[str, Any]) -> JsonDict:
    """REQ-VERIFY-1556: normalize token logprob and top-k telemetry fields."""

    token_logprobs = [_value for value in row.get("token_logprobs") or [] if (_value := _number(value)) is not None]
    top_logprobs = []
    for top in row.get("top_logprobs") or []:
        if isinstance(top, Mapping):
            converted = {
                str(key): number
                for key, value in top.items()
                if (number := _number(value)) is not None
            }
            if converted:
                top_logprobs.append(converted)
    mean_logprob = round(sum(token_logprobs) / len(token_logprobs), 6) if token_logprobs else None
    topk_available = bool(row.get("topk_alternatives_available")) or bool(top_logprobs)
    blockers = _unique(
        [
            str(row.get("blocker")) if row.get("blocker") else None,
            None if token_logprobs else "token_logprobs_missing",
            None if topk_available else "topk_logprobs_missing",
        ]
    )
    return {
        "response_text": str(row.get("response_text") or ""),
        "token_logprobs": token_logprobs,
        "mean_logprob": mean_logprob,
        "semantic_energy": round(-mean_logprob, 6) if mean_logprob is not None else None,
        "logprob_available": bool(token_logprobs),
        "top_logprobs": top_logprobs,
        "topk_available": topk_available,
        "topk_position_count": len(top_logprobs),
        "telemetry_blockers": blockers,
    }


def build_diagnostic_rows(
    source_rows: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join SATQuest deterministic labels, energy scores, and runtime telemetry."""

    telemetry_by_case = {str(row.get("case_id")): row for row in telemetry_rows}
    diagnostics: list[JsonDict] = []
    for source in source_rows:
        baseline = _mapping(source.get("baseline"))
        case_id = str(source.get("case_id") or "")
        signals = parse_telemetry_row(telemetry_by_case.get(case_id, {}))
        deterministic_accept = bool(baseline.get("correct"))
        energy = _number(baseline.get("energy"))
        row = {
            "row_type": "diagnostic_case",
            "spec": ["REQ-VERIFY-1556", "SCENARIO-VERIFY-1556"],
            "case_id": case_id,
            "source_family": str(source.get("family") or "satquest"),
            "deterministic_accept": deterministic_accept,
            "deterministic_final_accept": final_authority_accept(
                {"deterministic_accept": deterministic_accept}
            ),
            "reject_label": 0 if deterministic_accept else 1,
            "carnot_energy_score": energy,
            "model_declared_accept": _mapping(source.get("parse_result")).get("model_declared_accept"),
            "soft_signal_overrode_validator": False,
        } | signals
        semantic_energy = _number(row.get("semantic_energy"))
        row["routing_score"] = (
            round(float(energy) + float(semantic_energy), 6)
            if energy is not None and semantic_energy is not None
            else energy
        )
        diagnostics.append(row)
    return diagnostics


def evaluate_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    focused_tests_passed: bool,
    telemetry_artifact: Mapping[str, Any],
) -> JsonDict:
    """SCENARIO-VERIFY-1556: compute diagnostic-only metrics and blockers."""

    energy_pairs = _score_label_pairs(rows, "carnot_energy_score")
    route_pairs = _score_label_pairs(rows, "routing_score")
    logprob_available = any(bool(row.get("logprob_available")) for row in rows)
    topk_available = any(bool(row.get("topk_available")) for row in rows)
    live_used = bool(telemetry_artifact.get("live_sota_model_inference_used"))
    blockers = _unique(
        [
            *[str(blocker) for blocker in telemetry_artifact.get("blockers") or []],
            *[
                str(blocker)
                for row in rows
                for blocker in row.get("telemetry_blockers", [])
            ],
            None if rows else "no_diagnostic_cases_loaded",
            None if energy_pairs else "carnot_energy_unavailable",
            None if live_used else "live_sota_model_inference_not_used",
            None if logprob_available else "token_logprobs_missing",
            None if topk_available else "topk_logprobs_missing",
            None if focused_tests_passed else "focused_tests_not_passed",
        ]
    )
    ready = bool(
        rows
        and energy_pairs
        and live_used
        and logprob_available
        and topk_available
        and focused_tests_passed
    )
    return {
        "arm_ebm_logprob_telemetry_ready": ready,
        "diagnostic_cases": len(rows),
        "live_sota_model_inference_used": live_used,
        "logprob_available": logprob_available,
        "topk_available": topk_available,
        "energy_label_correlation": _pearson(energy_pairs),
        "energy_logprob_correlation": _energy_logprob_correlation(rows),
        "routing_auc": _roc_auc(route_pairs),
        "deterministic_validators_final_authority": True,
        "telemetry_blockers": blockers,
    }


def final_authority_accept(case: Mapping[str, Any]) -> bool:
    """Return only the deterministic validator decision, ignoring soft signals."""

    return bool(case.get("deterministic_accept"))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Enforce the Exp1556 terminal schema and authority invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["deterministic_validators_final_authority"] is not True:
        raise AssertionError("deterministic validators must remain final authority")
    if artifact["arm_ebm_logprob_telemetry_ready"]:
        if artifact["focused_tests_passed"] is not True:
            raise AssertionError("ready telemetry requires focused tests")
        if not artifact["live_sota_model_inference_used"]:
            raise AssertionError("ready telemetry requires live SOTA inference")
        if not (artifact["logprob_available"] and artifact["topk_available"]):
            raise AssertionError("ready telemetry requires live SOTA logprob and top-k telemetry")


def run_experiment(
    *,
    project_root: Path | str = ".",
    run_date: str = RUN_DATE,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    telemetry_manifest_path: Path | str = DEFAULT_TELEMETRY_MANIFEST_PATH,
    diagnostic_report_path: Path | str = DEFAULT_DIAGNOSTIC_REPORT_PATH,
    satquest_manifest_path: Path | str = DEFAULT_SATQUEST_MANIFEST_PATH,
    focused_tests_passed: bool = False,
    case_limit: int = 4,
    telemetry_builder: TelemetryBuilder | None = None,
) -> JsonDict:
    """Run the bounded Exp1556 diagnostic and write terminal artifacts."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    telemetry_manifest = _resolve(root, telemetry_manifest_path)
    diagnostic_report = _resolve(root, diagnostic_report_path)
    satquest_manifest = _resolve(root, satquest_manifest_path)
    write_in_progress_artifact(output)
    source_rows = select_satquest_rows(_read_jsonl(satquest_manifest), limit=case_limit)
    cases = build_telemetry_cases(source_rows)
    builder = telemetry_builder or _default_telemetry_builder
    telemetry_artifact = dict(
        builder(
            project_root=root,
            run_date=run_date,
            cases=cases,
            manifest_path=telemetry_manifest,
        )
    )
    telemetry_rows = _read_jsonl(telemetry_manifest) if telemetry_manifest.exists() else []
    diagnostic_rows = build_diagnostic_rows(source_rows, telemetry_rows)
    summary = evaluate_diagnostic_rows(
        diagnostic_rows,
        focused_tests_passed=focused_tests_passed,
        telemetry_artifact=telemetry_artifact,
    )
    _write_jsonl(diagnostic_report, diagnostic_rows)
    artifact = _artifact_from_summary(
        summary=summary,
        status="complete",
        focused_tests_passed=focused_tests_passed,
        telemetry_artifact=telemetry_artifact,
    )
    artifact["run_date"] = run_date
    artifact["diagnostic_report_path"] = _display(root, diagnostic_report)
    artifact["telemetry_manifest_path"] = _display(root, telemetry_manifest)
    artifact["models_used"] = [str(model) for model in telemetry_artifact.get("models_used") or []]
    artifact["energy_logprob_correlation"] = summary.get("energy_logprob_correlation")
    validate_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def _default_telemetry_builder(
    *,
    project_root: Path,
    run_date: str,
    cases: Sequence[TelemetryCase],
    manifest_path: Path,
) -> JsonDict:  # pragma: no cover - exercised by the live diagnostic run, not unit tests.
    return build_telemetry_artifact(
        project_root=project_root,
        run_date=run_date,
        manifest_path=manifest_path,
        cases=cases,
        generation_source="live_sota_llamacpp",
    )


def _artifact_from_summary(
    *,
    summary: Mapping[str, Any],
    status: str,
    focused_tests_passed: bool,
    telemetry_artifact: Mapping[str, Any],
) -> JsonDict:
    ready = bool(summary.get("arm_ebm_logprob_telemetry_ready"))
    verdict = (
        "complete: ARM/EBT logprob telemetry ready diagnostic-only"
        if ready
        else "complete: ARM/EBT logprob telemetry diagnostic completed with blockers"
    )
    return {
        "status": status,
        "milestone": MILESTONE,
        "arm_ebm_logprob_telemetry_ready": ready,
        "model_specs": list(MODEL_SPECS),
        "live_sota_model_inference_used": bool(
            summary.get(
                "live_sota_model_inference_used",
                telemetry_artifact.get("live_sota_model_inference_used", False),
            )
        ),
        "logprob_available": bool(summary.get("logprob_available", False)),
        "topk_available": bool(summary.get("topk_available", False)),
        "telemetry_adapter_path": TELEMETRY_ADAPTER_PATH,
        "diagnostic_cases": int(summary.get("diagnostic_cases", 0)),
        "energy_label_correlation": summary.get("energy_label_correlation"),
        "routing_auc": summary.get("routing_auc"),
        "deterministic_validators_final_authority": bool(
            summary.get("deterministic_validators_final_authority", True)
        ),
        "telemetry_blockers": list(summary.get("telemetry_blockers") or []),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": verdict,
    }


def _empty_summary() -> JsonDict:
    return {
        "arm_ebm_logprob_telemetry_ready": False,
        "diagnostic_cases": 0,
        "live_sota_model_inference_used": False,
        "logprob_available": False,
        "topk_available": False,
        "energy_label_correlation": None,
        "routing_auc": None,
        "deterministic_validators_final_authority": True,
        "telemetry_blockers": ["experiment_1556_arm_ebm_logprob_telemetry_in_progress"],
    }


def _score_label_pairs(rows: Sequence[Mapping[str, Any]], score_key: str) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    for row in rows:
        score = _number(row.get(score_key))
        if score is not None:
            pairs.append((score, int(row["reject_label"])))
    return pairs


def _energy_logprob_correlation(rows: Sequence[Mapping[str, Any]]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        energy = _number(row.get("carnot_energy_score"))
        semantic_energy = _number(row.get("semantic_energy"))
        if energy is not None and semantic_energy is not None:
            pairs.append((energy, semantic_energy))
    return _pearson_float(pairs)


def _pearson(pairs: Sequence[tuple[float, int]]) -> float | None:
    return _pearson_float([(score, float(label)) for score, label in pairs])


def _pearson_float(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [x for x, _y in pairs]
    ys = [y for _x, y in pairs]
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


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unique(values: Sequence[str | None]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
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


def _resolve(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _display(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "TELEMETRY_ADAPTER_PATH",
    "build_diagnostic_rows",
    "build_telemetry_cases",
    "evaluate_diagnostic_rows",
    "final_authority_accept",
    "parse_telemetry_row",
    "run_experiment",
    "select_satquest_rows",
    "validate_artifact",
    "write_in_progress_artifact",
]
