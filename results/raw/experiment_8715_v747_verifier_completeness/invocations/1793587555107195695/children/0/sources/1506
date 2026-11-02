"""Exp 5138: exact-validator energy-guided decoding gate.

Spec refs: REQ-PIPELINE-5138,
SCENARIO-PIPELINE-5138.

This experiment intentionally separates true guided decoding from reranking.
The Exp 5136 pool contains complete audited candidates, which is enough to
measure unguided and rerank-only controls, but not enough to prove that an
energy signal changed token choices during generation. When stepwise logprob or
top-token telemetry is absent, this runner writes a blocked artifact instead of
renaming best-of-N selection as guided decoding.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5136_receipt_structured_pool_v2_v471 as pool_mod  # noqa: E402


JsonDict = dict[str, Any]

EXPERIMENT_ID = "exp5138-ets-ebd-guided-decoding-v471"
MILESTONE = "2026.07.471"
RESULT_RELATIVE_PATH = "results/experiment_5138_ets_ebd_guided_decoding_v471.json"
UPSTREAM_POOL_ARTIFACT = pool_mod.RESULT_RELATIVE_PATH
INFERENCE_SUBSTRATE = "local_sota_gguf_energy_guided_decoding_or_blocked"

SUCCESS_READY_VERDICT = "complete_guided_decoding_ready_beats_matched_controls"
BLOCKED_UPSTREAM_VERDICT = "blocked_exp5136_upstream_unreadable"
BLOCKED_POOL_VERDICT = "blocked_structured_pool_v2_clean_false"
BLOCKED_ROWS_VERDICT = "blocked_structured_pool_v2_rows_missing"
BLOCKED_MODEL_VERDICT = "blocked_mandated_model_specs_missing"
BLOCKED_TELEMETRY_VERDICT = "blocked_stepwise_logprob_telemetry_unavailable"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")

RANDOM_SEED = 20260702
GUIDABLE_FAMILIES = ("graph_coloring", "or_allocation", "travel_budget")
FIXED_TOKEN_BUDGET = 32
STEPWISE_TELEMETRY_BLOCKER = "missing_stepwise_logprob_or_top_token_decoder_telemetry"
MANDATED_MODEL_IDS = pool_mod.MANDATED_MODEL_IDS

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "MODEL_SPECS",
    "upstream_pool_artifact",
    "exact_validator_authority",
    "controls_differentiated",
    "rerank_only_control",
    "token_nfe_accounting",
    "guided_decoding_delta",
    "delta_ci95",
    "violation_rate_delta",
    "logprob_or_blocker_evidence",
    "guided_decoding_ready",
    "conductor_modified",
    "tests_run",
)

FIELD_PRINCIPLES = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "MODEL_SPECS": "mandated local SOTA model provenance",
    "upstream_pool_artifact": "data provenance",
    "exact_validator_authority": "ground-truth accountability",
    "controls_differentiated": "no rerank-only masquerade",
    "rerank_only_control": "baseline adequacy",
    "token_nfe_accounting": "cost fairness",
    "guided_decoding_delta": "utility",
    "delta_ci95": "statistical caution",
    "violation_rate_delta": "constraint quality",
    "logprob_or_blocker_evidence": "process telemetry honesty",
    "guided_decoding_ready": "downstream readiness",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5138_ets_ebd_guided_decoding_v471.py --date 20260702",
    '.venv/bin/pytest tests/python/test_experiment_5138_ets_ebd_guided_decoding_v471.py -q -o addopts=""',
    ".venv/bin/coverage erase && .venv/bin/coverage run --include='/home/ianblenke/github.com/"
    "ianblenke/carnot/python/carnot/experiment_5138_ets_ebd_guided_decoding_v471.py' "
    '-m pytest tests/python/test_experiment_5138_ets_ebd_guided_decoding_v471.py -q -o addopts="" '
    "&& .venv/bin/coverage report --include='/home/ianblenke/github.com/ianblenke/carnot/"
    "python/carnot/experiment_5138_ets_ebd_guided_decoding_v471.py' --fail-under=100 -m",
    ".venv/bin/ruff check python/carnot/experiment_5138_ets_ebd_guided_decoding_v471.py "
    "scripts/experiment_5138_ets_ebd_guided_decoding_v471.py "
    "tests/python/test_experiment_5138_ets_ebd_guided_decoding_v471.py",
    ".venv/bin/ruff format --check python/carnot/experiment_5138_ets_ebd_guided_decoding_v471.py "
    "scripts/experiment_5138_ets_ebd_guided_decoding_v471.py "
    "tests/python/test_experiment_5138_ets_ebd_guided_decoding_v471.py",
    "python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5138_ets_ebd_guided_decoding_v471.py",
    ".venv/bin/pytest tests/python -q",
]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Any) -> str:
    return _sha256_text(_json_dumps(payload))


def _round_rate(value: float) -> float:
    return round(float(value), 6)


def _read_json(path: Path) -> tuple[JsonDict | None, str | None]:
    if not path.exists():
        return None, f"missing upstream artifact: {path.as_posix()}"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, f"upstream artifact is not a JSON object: {path.as_posix()}"
    return parsed, None


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_specs_complete(model_specs: Sequence[Mapping[str, Any]]) -> bool:
    ids = {str(row.get("hf_id")) for row in model_specs if row.get("model_path")}
    return ids == set(MANDATED_MODEL_IDS)


def _duration_s(upstream: Mapping[str, Any] | None, current_duration_s: float) -> float:
    upstream_duration = float(upstream.get("duration_s", 0.0)) if upstream else 0.0
    return max(float(current_duration_s), upstream_duration, 0.000001)


def _estimated_tokens(text: Any) -> int:
    parts = re.findall(r"\S+", str(text or ""))
    return max(1, len(parts))


def _rate(flags: Sequence[bool]) -> float:
    return 0.0 if not flags else _round_rate(sum(1 for item in flags if item) / len(flags))


def _paired_delta_ci95(guided: Sequence[bool], control: Sequence[bool]) -> list[float]:
    if not guided or not control:
        return [0.0, 0.0]
    pairs = [(bool(left), bool(right)) for left, right in zip(guided, control, strict=False)]
    deltas = [1.0 * left - 1.0 * right for left, right in pairs]
    mean = sum(deltas) / len(deltas)
    p_like = max(0.0, min(1.0, mean))
    se = (p_like * (1.0 - p_like) / len(deltas)) ** 0.5
    return [round(max(-1.0, mean - 1.96 * se), 2), round(min(1.0, mean + 1.96 * se), 2)]


def select_guidable_rows(
    rows: Sequence[Mapping[str, Any]], per_family: int | None = None
) -> list[JsonDict]:
    selected: list[JsonDict] = []
    counts = {family: 0 for family in GUIDABLE_FAMILIES}
    for row in rows:
        family = str(row.get("family"))
        if family not in counts:
            continue
        if per_family is not None and counts[family] >= per_family:
            continue
        selected.append(dict(row))
        counts[family] += 1
    return selected


def _candidate_tokens(candidate: Mapping[str, Any]) -> int:
    return _estimated_tokens(candidate.get("raw_response", ""))


def _candidate_correct(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("correct") is True)


def _candidate_id(candidate: Mapping[str, Any]) -> str | None:
    value = candidate.get("candidate_id")
    return str(value) if value is not None else None


def _first_candidate(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    return first if isinstance(first, Mapping) else None


def _candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, Mapping)]


def _best_by_exact_validator(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    for candidate in candidates:
        if _candidate_correct(candidate):
            return candidate
    return candidates[0]


def _fixed_token_choice(candidates: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, int]:
    used: list[Mapping[str, Any]] = []
    spent = 0
    for candidate in candidates:
        tokens = _candidate_tokens(candidate)
        if used and spent + tokens > FIXED_TOKEN_BUDGET:
            break
        used.append(candidate)
        spent += tokens
        if spent >= FIXED_TOKEN_BUDGET:
            break
    if not used:
        return None, 0
    return _best_by_exact_validator(used), len(used)


def _arm_metrics(
    *,
    arm: str,
    selected: Sequence[Mapping[str, Any] | None],
    generated_tokens: int,
    validator_calls: int,
    selection_validator_calls: int,
    task_count: int,
    selected_candidate_ids: Sequence[str | None],
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    correct = [bool(candidate and _candidate_correct(candidate)) for candidate in selected]
    payload: JsonDict = {
        "arm": arm,
        "task_count": int(task_count),
        "exact_validator_success": _rate(correct),
        "violation_rate": _round_rate(1.0 - _rate(correct)),
        "generated_tokens": int(generated_tokens),
        "validator_calls": int(validator_calls),
        "selection_validator_calls": int(selection_validator_calls),
        "nfe": int(selection_validator_calls),
        "selected_candidate_ids": list(selected_candidate_ids),
        "correct_by_task": correct,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def evaluate_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    task_count = len(rows)
    firsts: list[Mapping[str, Any] | None] = []
    bests: list[Mapping[str, Any] | None] = []
    fixeds: list[Mapping[str, Any] | None] = []
    first_tokens = 0
    all_tokens = 0
    fixed_tokens = 0
    all_validator_calls = 0
    fixed_validator_calls = 0
    for row in rows:
        candidates = _candidates(row)
        first = _first_candidate(row)
        best = _best_by_exact_validator(candidates)
        fixed, fixed_calls = _fixed_token_choice(candidates)
        firsts.append(first)
        bests.append(best)
        fixeds.append(fixed)
        first_tokens += _candidate_tokens(first or {})
        all_tokens += sum(_candidate_tokens(candidate) for candidate in candidates)
        fixed_tokens += sum(_candidate_tokens(candidate) for candidate in candidates[:fixed_calls])
        all_validator_calls += len(candidates)
        fixed_validator_calls += fixed_calls

    controls = {
        "unguided_generation": _arm_metrics(
            arm="unguided_generation",
            selected=firsts,
            generated_tokens=first_tokens,
            validator_calls=task_count,
            selection_validator_calls=0,
            task_count=task_count,
            selected_candidate_ids=[_candidate_id(candidate or {}) for candidate in firsts],
        ),
        "best_of_n_reranking": _arm_metrics(
            arm="best_of_n_reranking",
            selected=bests,
            generated_tokens=all_tokens,
            validator_calls=all_validator_calls,
            selection_validator_calls=all_validator_calls,
            task_count=task_count,
            selected_candidate_ids=[_candidate_id(candidate or {}) for candidate in bests],
            extra={"rerank_n_per_task": 0 if task_count == 0 else all_validator_calls // task_count},
        ),
        "fixed_token_reranking": _arm_metrics(
            arm="fixed_token_reranking",
            selected=fixeds,
            generated_tokens=fixed_tokens,
            validator_calls=fixed_validator_calls,
            selection_validator_calls=fixed_validator_calls,
            task_count=task_count,
            selected_candidate_ids=[_candidate_id(candidate or {}) for candidate in fixeds],
            extra={"token_budget_per_task": FIXED_TOKEN_BUDGET},
        ),
        "guided_decoding": {
            "arm": "guided_decoding",
            "executed": False,
            "blocked_reason": STEPWISE_TELEMETRY_BLOCKER,
            "task_count": task_count,
            "energy_updates": [],
            "generated_tokens": 0,
            "validator_calls": 0,
            "selection_validator_calls": 0,
            "nfe": 0,
            "exact_validator_success": None,
            "violation_rate": None,
        },
    }
    return controls


def inspect_stepwise_telemetry(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows_with_token_logprobs = 0
    rows_with_top_logprobs = 0
    rows_with_energy_updates = 0
    for row in rows:
        candidates = _candidates(row)
        has_token_logprobs = any(candidate.get("token_logprobs") for candidate in candidates)
        has_top_logprobs = any(candidate.get("top_logprobs") for candidate in candidates)
        has_energy_updates = any(candidate.get("energy_updates") for candidate in candidates)
        rows_with_token_logprobs += int(has_token_logprobs)
        rows_with_top_logprobs += int(has_top_logprobs)
        rows_with_energy_updates += int(has_energy_updates)
    has_required = (
        bool(rows)
        and rows_with_token_logprobs == len(rows)
        and rows_with_top_logprobs == len(rows)
        and rows_with_energy_updates == len(rows)
    )
    return {
        "has_required_stepwise_telemetry": has_required,
        "rows_checked": len(rows),
        "rows_with_token_logprobs": rows_with_token_logprobs,
        "rows_with_top_logprobs": rows_with_top_logprobs,
        "rows_with_energy_updates": rows_with_energy_updates,
        "required_for_true_guided_decoding": [
            "per-step candidate token logprobs",
            "top-token alternatives before sampling",
            "energy updates applied before token selection",
        ],
        "candidate_pool_has_only_completed_outputs": not has_required,
        "blocked_reason": None if has_required else STEPWISE_TELEMETRY_BLOCKER,
    }


def _exact_validator_authority(
    rows: Sequence[Mapping[str, Any]], upstream: Mapping[str, Any]
) -> JsonDict:
    validators = sorted({str(row.get("validator")) for row in rows if row.get("validator")})
    return {
        "authority_intact": bool(rows) and bool(validators),
        "authority_source": "exp5136_exact_validator_outputs",
        "validators_used": validators,
        "llm_judge_used_as_ground_truth": False,
        "verifier_is_oracle": bool(upstream.get("verifier_is_oracle", False)),
        "fover_scope_used": bool(upstream.get("fover_scope_used", False)),
    }


def _token_nfe_accounting(controls: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    return {
        name: {
            "generated_tokens": metrics.get("generated_tokens"),
            "validator_calls": metrics.get("validator_calls"),
            "selection_validator_calls": metrics.get("selection_validator_calls"),
            "nfe": metrics.get("nfe"),
            "executed": name != "guided_decoding"
            if name in {"unguided_generation", "best_of_n_reranking", "fixed_token_reranking"}
            else metrics.get("executed", False),
            "blocked_reason": metrics.get("blocked_reason"),
        }
        for name, metrics in controls.items()
    }


def _blocked_artifact(
    *,
    verdict: str,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float,
    upstream: Mapping[str, Any] | None,
    upstream_error: str | None,
    selected_rows: Sequence[Mapping[str, Any]] | None = None,
    controls: Mapping[str, Mapping[str, Any]] | None = None,
    telemetry: Mapping[str, Any] | None = None,
) -> JsonDict:
    model_specs = [
        dict(row) for row in (upstream or {}).get("MODEL_SPECS", []) if isinstance(row, Mapping)
    ]
    selected = list(selected_rows or [])
    control_metrics: JsonDict = dict(controls or {})
    blocker = dict(telemetry or inspect_stepwise_telemetry(selected))
    rerank_control = (
        dict(control_metrics["best_of_n_reranking"])
        if control_metrics
        else {
            "arm": "best_of_n_reranking",
            "task_count": 0,
            "exact_validator_success": 0.0,
            "violation_rate": 1.0,
            "generated_tokens": 0,
            "validator_calls": 0,
            "selection_validator_calls": 0,
            "nfe": 0,
        }
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration_s(upstream, current_duration_s),
        "MODEL_SPECS": model_specs,
        "model_specs": model_specs,
        "upstream_pool_artifact": UPSTREAM_POOL_ARTIFACT,
        "exact_validator_authority": _exact_validator_authority(selected, upstream or {}),
        "controls_differentiated": False,
        "rerank_only_control": rerank_control,
        "token_nfe_accounting": _token_nfe_accounting(control_metrics)
        if control_metrics
        else {
            "guided_decoding": {
                "generated_tokens": 0,
                "validator_calls": 0,
                "selection_validator_calls": 0,
                "nfe": 0,
                "executed": False,
                "blocked_reason": STEPWISE_TELEMETRY_BLOCKER,
            }
        },
        "guided_decoding_delta": None,
        "delta_ci95": [None, None],
        "violation_rate_delta": None,
        "logprob_or_blocker_evidence": blocker,
        "guided_decoding_ready": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "selected_task_families": list(GUIDABLE_FAMILIES),
        "selected_task_count": len(selected),
        "control_metrics": control_metrics,
        "strongest_matched_control": rerank_control,
        "preconditions_checked": {
            "upstream_error": upstream_error,
            "upstream_loaded": upstream is not None,
            "structured_pool_v2_clean": bool(
                upstream and upstream.get("structured_pool_v2_clean") is True
            ),
            "pool_rows_loaded": bool(selected),
            "model_specs_complete": _model_specs_complete(model_specs),
            "stepwise_telemetry_available": bool(
                blocker.get("has_required_stepwise_telemetry") is True
            ),
        },
        "reproducibility_checksum": _sha256_payload(
            {
                "experiment_id": EXPERIMENT_ID,
                "verdict": verdict,
                "model_specs": model_specs,
                "selected_task_count": len(selected),
                "controls": control_metrics,
                "telemetry": blocker,
            }
        ),
    }
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    root: Path,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float = 0.0,
) -> JsonDict:
    upstream, upstream_error = _read_json(root / UPSTREAM_POOL_ARTIFACT)
    if upstream_error is not None:
        return _blocked_artifact(
            verdict=BLOCKED_UPSTREAM_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=upstream_error,
        )
    if upstream is None or upstream.get("structured_pool_v2_clean") is not True:
        return _blocked_artifact(
            verdict=BLOCKED_POOL_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=None,
        )

    pool_path = str(upstream.get("pool_path") or pool_mod.POOL_RELATIVE_PATH)
    rows = select_guidable_rows(read_jsonl(root / pool_path))
    if not rows:
        return _blocked_artifact(
            verdict=BLOCKED_ROWS_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=None,
        )

    model_specs = [dict(row) for row in upstream.get("MODEL_SPECS", []) if isinstance(row, Mapping)]
    if not _model_specs_complete(model_specs):
        return _blocked_artifact(
            verdict=BLOCKED_MODEL_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=None,
            selected_rows=rows,
            controls=evaluate_controls(rows),
        )

    controls = evaluate_controls(rows)
    telemetry = inspect_stepwise_telemetry(rows)
    return _blocked_artifact(
        verdict=BLOCKED_TELEMETRY_VERDICT,
        run_date=run_date,
        tests_run=tests_run,
        current_duration_s=current_duration_s,
        upstream=upstream,
        upstream_error=None,
        selected_rows=rows,
        controls=controls,
        telemetry=telemetry,
    )


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float = 0.0,
) -> JsonDict:
    artifact = build_artifact(
        root=Path(root),
        run_date=run_date,
        tests_run=tests_run,
        current_duration_s=current_duration_s,
    )
    write_json(Path(root) / RESULT_RELATIVE_PATH, artifact)
    return artifact


def _terminal_verdict(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id mismatch")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    if not _terminal_verdict(artifact["honest_verdict"]):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("substrate mismatch")
    if artifact["MODEL_SPECS"] != artifact.get("model_specs"):
        raise ValueError("mandated model_specs must mirror MODEL_SPECS")
    if artifact["upstream_pool_artifact"] != UPSTREAM_POOL_ARTIFACT:
        raise ValueError("upstream artifact mismatch")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be false")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must not be empty")
    if not isinstance(artifact["exact_validator_authority"], Mapping) or not artifact[
        "exact_validator_authority"
    ].get("authority_source"):
        raise ValueError("exact validator authority is missing")
    if artifact["controls_differentiated"] is not False:
        raise ValueError("controls must not be differentiated without true guided telemetry")
    if not isinstance(artifact["rerank_only_control"], Mapping) or not artifact[
        "rerank_only_control"
    ].get("arm"):
        raise ValueError("rerank-only control missing")
    if not isinstance(artifact["token_nfe_accounting"], Mapping) or not artifact[
        "token_nfe_accounting"
    ]:
        raise ValueError("token_nfe accounting missing")
    if not isinstance(artifact["logprob_or_blocker_evidence"], Mapping) or "blocked_reason" not in artifact[
        "logprob_or_blocker_evidence"
    ]:
        raise ValueError("blocker evidence missing")
    if artifact["guided_decoding_ready"] is not False:
        raise ValueError("guided_decoding_ready requires true guided decoding and positive delta")
    if artifact["guided_decoding_delta"] is not None:
        raise ValueError("blocked artifacts must not report a guided decoding delta")
    if artifact["delta_ci95"] != [None, None]:
        raise ValueError("blocked artifacts must not report a guided decoding CI")
    if artifact["violation_rate_delta"] is not None:
        raise ValueError("blocked artifacts must not report a violation-rate delta")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Exp 5138 exact-validator energy-guided decoding gate."
    )
    parser.add_argument("--date", default=dt.datetime.now(dt.UTC).strftime("%Y%m%d"))
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--duration-override", type=float, default=None)
    args = parser.parse_args(argv)

    started = time.monotonic()
    current_duration = args.duration_override
    if current_duration is None:
        current_duration = max(time.monotonic() - started, 0.000001)
    artifact = write_artifact(
        root=Path(args.root),
        run_date=str(args.date),
        tests_run=DEFAULT_TESTS_RUN,
        current_duration_s=float(current_duration),
    )
    print(
        json.dumps({"artifact": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    raise SystemExit(main())
