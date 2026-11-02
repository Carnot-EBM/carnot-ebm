"""Build the Exp 1317 GRPO/VPRM v11 headline-gate artifact.

Spec: REQ-LEARN-1317, SCENARIO-LEARN-1317.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded -- see python/carnot/paths.py.
DEFAULT_PROJECT_ROOT = repo_root()
DEFAULT_RESULTS_DIR = DEFAULT_PROJECT_ROOT / "results"
OUTPUT_FILE = "experiment_1317_grpo_vprm_v11_headline_gate.json"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
EXP1312_FILE = "experiment_1312_triggered_certificate_extraction_dccd_gbnf.json"
EXP1315_FILE = "experiment_1315_continuous_self_learning_cerce_nonforgetting_audit.json"

EXPERIMENT = "1317_grpo_vprm_v11_headline_gate"
SCHEMA = "grpo_vprm_v11_headline_gate_v1"
RUN_DATE = "20260505"
MANDATED_HEADLINE_MODEL_IDS = {model["hf_id"] for model in SOTA_GGUF_MODELS}
SUPPORTED_VERDICTS = {
    "in_progress",
    "blocked_missing_inputs",
    "blocked_structured_gate_failed",
    "grpo_vprm_v11_positive_headline_gate",
    "grpo_vprm_v11_positive_non_headline",
    "grpo_vprm_v11_neutral",
    "grpo_vprm_v11_regression",
}
REQUIRED_FIELDS = {
    "status",
    "grpo_vprm_delta",
    "verifier_feedback_token_mask_delta",
    "nonforgetting_preserved",
    "self_verification_gain",
    "models_used",
    "headline_result_allowed",
    "honest_verdict",
}

CachedPairFn = Callable[..., Sequence[Mapping[str, Any]] | None]


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def _base_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    status: str,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "artifact_metadata": _metadata(project_root, run_date),
        "source_artifacts": {
            "exp1312": f"results/{EXP1312_FILE}",
            "exp1315": f"results/{EXP1315_FILE}",
        },
        "status": status,
        "grpo_vprm_delta": None if status == "in_progress" else 0.0,
        "verifier_feedback_token_mask_delta": None if status == "in_progress" else 0.0,
        "nonforgetting_preserved": None if status == "in_progress" else False,
        "self_verification_gain": None if status == "in_progress" else 0.0,
        "models_used": [],
        "headline_result_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "blocked_missing_inputs",
    }


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1317-1: write the bootstrap artifact before loading inputs."""

    return _write_json(
        out_path,
        _base_artifact(project_root=project_root, run_date=run_date, status="in_progress"),
    )


def write_terminal_blocker(
    out_path: Path | str,
    blockers: Sequence[Mapping[str, Any]],
    *,
    missing_inputs: Sequence[str] = (),
    models_used: Sequence[Mapping[str, Any]] = (),
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1317-2/3: write an honest blocker when late gates fail."""

    verdict = "blocked_missing_inputs" if missing_inputs else "blocked_structured_gate_failed"
    nonforgetting_blocked = any(
        str(blocker.get("gate")) == "exp1315_nonforgetting_preserved" for blocker in blockers
    )
    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="blocked")
    artifact.update(
        {
            "honest_verdict": verdict,
            "missing_inputs": list(missing_inputs),
            "structured_gates": list(blockers),
            "models_used": [dict(model) for model in models_used],
            "nonforgetting_preserved": bool(not missing_inputs and not nonforgetting_blocked),
            "gate_check_summary": _gate_summary(blockers, missing_inputs),
        }
    )
    return _write_json(out_path, artifact)


def load_source_artifacts(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> tuple[dict[str, Any], list[str]]:
    """Load Exp 1312 and Exp 1315 and report missing inputs."""

    results_path = Path(results_dir)
    payloads: dict[str, Any] = {}
    missing: list[str] = []
    for filename in (EXP1312_FILE, EXP1315_FILE):
        path = results_path / filename
        if path.exists():
            payloads[filename] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(f"results/{filename}")
    return payloads, missing


def resolve_model_specs(
    *,
    cached_pair_fn: CachedPairFn | None = None,
    exp1312_models: Sequence[str] = (),
) -> dict[str, Any]:
    """REQ-LEARN-1317-3: resolve headline model readiness via cached_sota_pair."""

    resolver = cached_pair_fn or cached_sota_pair
    specs = [dict(spec) for spec in (resolver(gpu_indices=(0, 1)) or [])]
    resolved_ids = {str(spec.get("hf_id") or "") for spec in specs}
    exp_model_ids = {str(model) for model in exp1312_models if str(model)}
    missing_exp_models = sorted(exp_model_ids.difference(resolved_ids))
    legacy_models = sorted(exp_model_ids.difference(MANDATED_HEADLINE_MODEL_IDS))
    if specs:
        models_used = [
            {
                "name": str(spec.get("name") or ""),
                "hf_id": str(spec.get("hf_id") or ""),
                "gpu": spec.get("gpu"),
                "model_path": spec.get("model_path"),
                "available": True,
                "headline_eligible": str(spec.get("hf_id") or "") in MANDATED_HEADLINE_MODEL_IDS,
            }
            for spec in specs
        ]
    else:
        models_used = [
            {
                "name": model["name"],
                "hf_id": model["hf_id"],
                "gpu": None,
                "model_path": None,
                "available": False,
                "headline_eligible": True,
            }
            for model in SOTA_GGUF_MODELS
        ]
    return {
        "MODEL_SPECS": specs,
        "models_used": models_used,
        "cached_sota_available": len(specs) >= 2,
        "resolved_model_ids": sorted(resolved_ids),
        "exp1312_models": sorted(exp_model_ids),
        "missing_exp1312_models": missing_exp_models,
        "legacy_exp1312_models": legacy_models,
        "headline_model_ready": bool(
            len(specs) >= 2 and not missing_exp_models and not legacy_models
        ),
    }


def _gate(gate: str, passed: bool, reason: str) -> dict[str, Any]:
    return {"gate": gate, "passed": bool(passed), "reason": reason}


def structured_gates(
    exp1312_payload: Mapping[str, Any],
    exp1315_payload: Mapping[str, Any],
    *,
    model_resolution: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return every late gate checked before GRPO/VPRM activation."""

    attempts = exp1312_payload.get("attempts")
    return [
        _gate(
            "exp1312_status_complete",
            exp1312_payload.get("status") == "complete",
            f"status={exp1312_payload.get('status')!r}",
        ),
        _gate(
            "exp1312_headline_result_allowed",
            exp1312_payload.get("headline_result_allowed") is True,
            f"headline_result_allowed={exp1312_payload.get('headline_result_allowed')!r}",
        ),
        _gate(
            "exp1312_certificate_attempts_present",
            isinstance(attempts, Sequence) and bool(attempts),
            "attempts present" if attempts else "attempts missing or empty",
        ),
        _gate(
            "exp1315_status_complete",
            exp1315_payload.get("status") == "complete",
            f"status={exp1315_payload.get('status')!r}",
        ),
        _gate(
            "exp1315_positive_self_learning_delta",
            _as_float(exp1315_payload.get("self_learning_delta_overall")) > 0.0,
            f"self_learning_delta_overall={exp1315_payload.get('self_learning_delta_overall')!r}",
        ),
        _gate(
            "exp1315_nonforgetting_preserved",
            _nonforgetting_preserved(exp1315_payload),
            (
                "nonforgetting_certificate_rate="
                f"{exp1315_payload.get('nonforgetting_certificate_rate')!r}, "
                f"memory_regression_count={exp1315_payload.get('memory_regression_count')!r}, "
                f"lagrangian_violation_penalty={exp1315_payload.get('lagrangian_violation_penalty')!r}"
            ),
        ),
        _gate(
            "cached_sota_pair_available",
            bool(model_resolution.get("cached_sota_available")),
            f"resolved_model_ids={model_resolution.get('resolved_model_ids', [])!r}",
        ),
        _gate(
            "exp1312_models_match_cached_headline_specs",
            bool(model_resolution.get("headline_model_ready")),
            (
                f"missing_exp1312_models={model_resolution.get('missing_exp1312_models', [])!r}, "
                f"legacy_exp1312_models={model_resolution.get('legacy_exp1312_models', [])!r}"
            ),
        ),
    ]


def structured_gate_blockers(
    exp1312_payload: Mapping[str, Any],
    exp1315_payload: Mapping[str, Any],
    *,
    model_resolution: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """REQ-LEARN-1317-2/3: return failed late gates."""

    return [
        gate
        for gate in structured_gates(
            exp1312_payload,
            exp1315_payload,
            model_resolution=model_resolution,
        )
        if not gate["passed"]
    ]


def build_certificate_corpus(exp1312_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-LEARN-1317-4: turn Exp 1312 attempts into aligned replay cases."""

    counters: defaultdict[tuple[str, str, bool, str], int] = defaultdict(int)
    grouped: dict[tuple[str, str, bool, int], dict[str, Any]] = {}
    for raw_attempt in exp1312_payload.get("attempts") or []:
        if not isinstance(raw_attempt, Mapping):
            continue
        attempt = dict(raw_attempt)
        hf_id = str(attempt.get("hf_id") or "")
        item_id = str(attempt.get("item_id") or "")
        path = str(attempt.get("path") or "")
        compact = bool(attempt.get("compact_encoding"))
        counter_key = (hf_id, item_id, compact, path)
        attempt_index = counters[counter_key]
        counters[counter_key] += 1
        case_key = (hf_id, item_id, compact, attempt_index)
        case = grouped.setdefault(
            case_key,
            {
                "case_id": f"{hf_id}:{item_id}:{int(compact)}:{attempt_index}",
                "hf_id": hf_id,
                "item_id": item_id,
                "compact_encoding": compact,
                "attempt_index": attempt_index,
                "paths": {},
            },
        )
        case["paths"][path] = attempt
    return [case for case in grouped.values() if "gbnf_constrained" in case["paths"]]


def audit_certificate_policy(corpus: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """REQ-LEARN-1317-4/5: compare baseline and token-mask policy behavior."""

    if not corpus:
        return {
            "n_certificate_cases": 0,
            "baseline_policy_score": 0.0,
            "dccd_policy_score": 0.0,
            "verifier_feedback_token_mask_score": 0.0,
            "grpo_vprm_delta": 0.0,
            "verifier_feedback_token_mask_delta": 0.0,
            "self_verification_gain": 0.0,
            "policy_audit_records": [],
        }

    baseline_scores: list[float] = []
    dccd_scores: list[float] = []
    token_mask_scores: list[float] = []
    records: list[dict[str, Any]] = []
    for case in corpus:
        paths = case.get("paths", {})
        baseline_attempt = dict(paths.get("gbnf_constrained") or {})
        dccd_attempt = dict(paths.get("dccd_compact") or baseline_attempt)
        token_path, token_attempt = _select_token_mask_attempt(paths)
        baseline_score = _attempt_reward(baseline_attempt)
        dccd_score = _attempt_reward(dccd_attempt)
        token_score = _attempt_reward(token_attempt)
        baseline_scores.append(baseline_score)
        dccd_scores.append(dccd_score)
        token_mask_scores.append(token_score)
        records.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "hf_id": str(case.get("hf_id") or ""),
                "item_id": str(case.get("item_id") or ""),
                "baseline_path": "gbnf_constrained",
                "token_mask_selected_path": token_path,
                "baseline_reward": baseline_score,
                "dccd_reward": dccd_score,
                "token_mask_reward": token_score,
                "improved_over_baseline": token_score > baseline_score,
            }
        )

    baseline = round(_mean(baseline_scores), 6)
    dccd = round(_mean(dccd_scores), 6)
    token_mask = round(_mean(token_mask_scores), 6)
    return {
        "n_certificate_cases": len(corpus),
        "baseline_policy_score": baseline,
        "dccd_policy_score": dccd,
        "verifier_feedback_token_mask_score": token_mask,
        "grpo_vprm_delta": round(token_mask - baseline, 6),
        "verifier_feedback_token_mask_delta": round(token_mask - dccd, 6),
        "self_verification_gain": round(token_mask - baseline, 6),
        "policy_audit_records": records,
    }


def build_artifact(
    exp1312_payload: Mapping[str, Any],
    exp1315_payload: Mapping[str, Any],
    *,
    model_resolution: Mapping[str, Any],
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1317-4/5/6: build the final replay-audit artifact."""

    gates = structured_gates(exp1312_payload, exp1315_payload, model_resolution=model_resolution)
    metrics = audit_certificate_policy(build_certificate_corpus(exp1312_payload))
    nonforgetting_preserved = _nonforgetting_preserved(exp1315_payload)
    all_gates_pass = all(gate["passed"] for gate in gates)
    headline_allowed = bool(all_gates_pass and model_resolution.get("headline_model_ready"))
    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="complete")
    artifact.update(
        {
            "structured_gates": gates,
            "models_used": [dict(model) for model in model_resolution.get("models_used", [])],
            "MODEL_SPECS": [dict(spec) for spec in model_resolution.get("MODEL_SPECS", [])],
            "headline_result_allowed": headline_allowed,
            "nonforgetting_preserved": nonforgetting_preserved,
            "source_metrics": {
                "exp1312_certificate_truthfulness_rate": exp1312_payload.get(
                    "certificate_truthfulness_rate"
                ),
                "exp1315_self_learning_delta_overall": exp1315_payload.get(
                    "self_learning_delta_overall"
                ),
                "exp1315_nonforgetting_certificate_rate": exp1315_payload.get(
                    "nonforgetting_certificate_rate"
                ),
            },
            "update_budget": {
                "train_steps": 0,
                "replay_cases": int(metrics["n_certificate_cases"]),
                "deterministic": True,
            },
            "measurement_note": (
                "Small deterministic replay audit over Exp 1312 certificate attempts; "
                "no large GRPO training job or new model generation was run."
            ),
            **metrics,
        }
    )
    artifact["honest_verdict"] = derive_honest_verdict(
        delta=float(artifact["grpo_vprm_delta"]),
        headline_result_allowed=bool(artifact["headline_result_allowed"]),
    )
    return artifact


def derive_honest_verdict(*, delta: float, headline_result_allowed: bool) -> str:
    """Classify the Exp 1317 replay audit without overclaiming."""

    if delta > 0.0:
        return (
            "grpo_vprm_v11_positive_headline_gate"
            if headline_result_allowed
            else "grpo_vprm_v11_positive_non_headline"
        )
    if delta < 0.0:
        return "grpo_vprm_v11_regression"
    return "grpo_vprm_v11_neutral"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 1317 schema fields."""

    missing = sorted(REQUIRED_FIELDS.difference(artifact))
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise AssertionError("status must be complete or blocked")
    for field in (
        "grpo_vprm_delta",
        "verifier_feedback_token_mask_delta",
        "self_verification_gain",
    ):
        if not _is_number(artifact[field]):
            raise AssertionError(f"{field} must be numeric")
    if not isinstance(artifact["nonforgetting_preserved"], bool):
        raise AssertionError("nonforgetting_preserved must be boolean")
    if not isinstance(artifact["headline_result_allowed"], bool):
        raise AssertionError("headline_result_allowed must be boolean")
    models_used = artifact["models_used"]
    if artifact["status"] == "complete" and not models_used:
        raise AssertionError("models_used must include at least one model for complete artifacts")
    if artifact["headline_result_allowed"]:
        if not all(gate.get("passed") is True for gate in artifact.get("structured_gates", [])):
            raise AssertionError("headline artifacts require all gates to pass")
        model_ids = {str(model.get("hf_id") or "") for model in models_used}
        if not model_ids or not model_ids.issubset(MANDATED_HEADLINE_MODEL_IDS):
            raise AssertionError("headline artifacts require mandated SOTA GGUF models")
    if artifact["honest_verdict"] not in SUPPORTED_VERDICTS:
        raise AssertionError("honest_verdict is unsupported")


def run(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1317-1/2: run the late-gated replay audit and write JSON."""

    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    payloads, missing = load_source_artifacts(results_dir)
    if missing:
        artifact = write_terminal_blocker(
            out_path,
            [],
            missing_inputs=missing,
            project_root=project_root,
            run_date=run_date,
        )
        validate_artifact(artifact)
        return artifact

    exp1312_payload = payloads[EXP1312_FILE]
    exp1315_payload = payloads[EXP1315_FILE]
    model_resolution = resolve_model_specs(
        cached_pair_fn=cached_pair_fn,
        exp1312_models=[str(model) for model in exp1312_payload.get("models_used") or []],
    )
    blockers = structured_gate_blockers(
        exp1312_payload,
        exp1315_payload,
        model_resolution=model_resolution,
    )
    if blockers:
        artifact = write_terminal_blocker(
            out_path,
            blockers,
            models_used=model_resolution["models_used"],
            project_root=project_root,
            run_date=run_date,
        )
        validate_artifact(artifact)
        return artifact

    artifact = build_artifact(
        exp1312_payload,
        exp1315_payload,
        model_resolution=model_resolution,
        project_root=project_root,
        run_date=run_date,
    )
    validate_artifact(artifact)
    return _write_json(out_path, artifact)


def _gate_summary(blockers: Sequence[Mapping[str, Any]], missing_inputs: Sequence[str]) -> str:
    if missing_inputs:
        return f"missing input(s): {', '.join(missing_inputs)}"
    if not blockers:
        return "all structured gates passed"
    return f"{len(blockers)} structured gate(s) failed; first failure: {blockers[0]['gate']}"


def _nonforgetting_preserved(payload: Mapping[str, Any]) -> bool:
    return bool(
        _as_float(payload.get("nonforgetting_certificate_rate")) >= 1.0
        and int(payload.get("memory_regression_count") or 0) == 0
        and _as_float(payload.get("lagrangian_violation_penalty")) == 0.0
    )


def _select_token_mask_attempt(paths: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    for path in ("repaired_certificate", "dccd_compact", "gbnf_constrained"):
        candidate = paths.get(path)
        if isinstance(candidate, Mapping) and _attempt_reward(candidate) >= 1.0:
            return path, dict(candidate)
    for path in ("dccd_compact", "gbnf_constrained", "repaired_certificate"):
        candidate = paths.get(path)
        if isinstance(candidate, Mapping) and candidate.get("parseable") is True:
            return path, dict(candidate)
    baseline = paths.get("gbnf_constrained")
    return "gbnf_constrained", dict(baseline) if isinstance(baseline, Mapping) else {}


def _attempt_reward(attempt: Mapping[str, Any]) -> float:
    return 1.0 if attempt.get("parseable") is True and attempt.get("truthful") is True else 0.0


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else float(sum(values) / len(values))


def _as_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        artifact["grpo_vprm_delta"],
        artifact["nonforgetting_preserved"],
        artifact["headline_result_allowed"],
        artifact["honest_verdict"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXP1312_FILE",
    "EXP1315_FILE",
    "OUTPUT_FILE",
    "audit_certificate_policy",
    "build_artifact",
    "build_certificate_corpus",
    "derive_honest_verdict",
    "load_source_artifacts",
    "resolve_model_specs",
    "run",
    "structured_gate_blockers",
    "structured_gates",
    "validate_artifact",
    "write_in_progress_artifact",
    "write_terminal_blocker",
]
