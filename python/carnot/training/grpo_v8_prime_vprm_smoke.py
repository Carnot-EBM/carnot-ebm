"""Exp 1273 GRPO v8 PRIME/VPRM bounded self-learning smoke.

This module deliberately keeps the training loop small and auditable: load the
Exp 1272 PRIME verifier weights, try the mandated cached SOTA GGUF pair, then
measure a before/after reward delta on a 24-item held-out arithmetic slice.
When the SOTA cache is absent, the module writes a deterministic smoke-only
artifact with `headline_result_allowed=false`.

Spec: REQ-LEARN-1273, SCENARIO-LEARN-1273, SCENARIO-LEARN-1274.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.extraction.vprm_verifier import VPRMArithmeticVerifier
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP1272_PATH = REPO_ROOT / "results" / "experiment_1272_prime_verifier_selection_audit.json"
DEFAULT_RESULT_PATH = REPO_ROOT / "results" / "experiment_1273_grpo_v8_prime_vprm_smoke.json"

EXPERIMENT_NAME = "1273_grpo_v8_prime_vprm_smoke"
SCHEMA = "grpo_v8_prime_vprm_smoke_v1"
RUN_DATE = "20260504"
DEFAULT_WALL_BUDGET_S = 120.0
DEFAULT_N_TRAIN = 16
DEFAULT_N_EVAL = 24
MANDATED_SOTA_HF_IDS = {model["hf_id"] for model in SOTA_GGUF_MODELS}

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "experiment",
    "schema",
    "run_date",
    "status",
    "execution_mode",
    "terminal_status",
    "honest_verdict",
    "MODEL_SPECS",
    "models_used",
    "verifier_weights_used",
    "grpo_v8_delta_pp",
    "self_learning_delta_overall",
    "headline_result_allowed",
    "wall_budget_s",
)

ResponseProvider = Callable[[Sequence[Mapping[str, Any]], str, Sequence[Mapping[str, Any]]], list[str]]


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else float(sum(values) / len(values))


def _normalise_weights(raw: Mapping[str, Any]) -> dict[str, float]:
    weights = {str(name): max(0.0, float(value)) for name, value in raw.items()}
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("verifier_weight_vector must contain a positive weight")
    return {name: value / total for name, value in weights.items() if value > 0.0}


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_RESULT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Write the REQ-LEARN-1273 in-progress artifact skeleton."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "honest_verdict": "in_progress",
    }
    _write_json(output_path, artifact)
    return artifact


def load_verifier_weights(path: Path | str = DEFAULT_EXP1272_PATH) -> dict[str, float]:
    """Load and normalize the Exp 1272 `verifier_weight_vector`."""

    try:
        payload = _read_json(path)
    except FileNotFoundError as exc:
        raise ValueError(f"verifier_weight_vector source missing: {path}") from exc
    raw = payload.get("verifier_weight_vector") if isinstance(payload, Mapping) else None
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("verifier_weight_vector missing or empty")
    return _normalise_weights(raw)


def build_smoke_slices(*, n_train: int = DEFAULT_N_TRAIN, n_eval: int = DEFAULT_N_EVAL) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic train/eval arithmetic slices for the bounded smoke."""

    if not (10 <= n_train <= 20 and 20 <= n_eval <= 30):
        raise ValueError("Exp 1273 requires 10-20 train items and 20-30 eval items")
    total = n_train + n_eval
    items: list[dict[str, Any]] = []
    for index in range(total):
        lhs = 11 + index
        rhs = 2 + (index % 9)
        answer = lhs + rhs
        wrong = answer + (1 if index % 2 == 0 else -1)
        split = "train" if index < n_train else "eval"
        local_index = index if split == "train" else index - n_train
        items.append(
            {
                "id": f"{split}_{local_index:03d}",
                "split": split,
                "question": f"What is {lhs} + {rhs}?",
                "lhs": lhs,
                "rhs": rhs,
                "answer": str(answer),
                "wrong_answer": str(wrong),
            }
        )
    return {
        "train": [item for item in items if item["split"] == "train"],
        "eval": [item for item in items if item["split"] == "eval"],
    }


def synthesise_response(item: Mapping[str, Any], *, phase: str) -> str:
    """Create deterministic generated reasoning for the smoke-only fallback."""

    if phase not in {"before", "after"}:
        raise ValueError(f"phase must be 'before' or 'after', got {phase!r}")
    answer = str(item["answer"] if phase == "after" else item["wrong_answer"])
    lhs = item["lhs"]
    rhs = item["rhs"]
    return f"{lhs} + {rhs} = {answer}. Therefore final answer: {answer}."


def _extract_final_answer(response: str) -> str:
    matches = re.findall(r"final answer:\s*(-?\d+(?:\.\d+)?)", response, flags=re.IGNORECASE)
    if matches:
        return matches[-1]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    return numbers[-1] if numbers else ""


def _final_answer_wrong(item: Mapping[str, Any], response: str) -> bool:
    return _extract_final_answer(response).strip() != str(item["answer"]).strip()


def _vprm_violation_signal(response: str) -> float:
    verifier = VPRMArithmeticVerifier()
    return 1.0 if verifier.detect_violations(response) else 0.0


def score_weighted_prime_vprm_reward(
    item: Mapping[str, Any],
    response: str,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Score one response as `1 - PRIME/VPRM weighted violation signal`."""

    answer_wrong = _final_answer_wrong(item, response)
    vprm_signal = _vprm_violation_signal(response)
    semantic_signal = 1.0 if answer_wrong else 0.0
    causal_signal = max(vprm_signal, semantic_signal)
    signals = {
        "Z3MathVerifier": vprm_signal,
        "SymCodeVerifier": 0.0,
        "CausalReasoningVerifier": causal_signal,
        "SemEnergyProbe": semantic_signal,
        "SOSKANEnergyV3": vprm_signal,
    }
    signals["k5_ensemble_summary"] = max(
        signals["Z3MathVerifier"],
        signals["CausalReasoningVerifier"],
        signals["SemEnergyProbe"],
        signals["SOSKANEnergyV3"],
    )
    weighted_error = sum(float(weights.get(name, 0.0)) * value for name, value in signals.items())
    reward = max(0.0, min(1.0, 1.0 - weighted_error))
    return {
        "weighted_reward": round(reward, 6),
        "weighted_error": round(weighted_error, 6),
        "signals": signals,
        "final_answer": _extract_final_answer(response),
    }


def resolve_model_specs(
    *,
    cached_pair_fn: Callable[[], Sequence[Mapping[str, Any]] | None] | None = None,
) -> dict[str, Any]:
    """Resolve MODEL_SPECS through `cached_sota_pair()` before any fallback."""

    resolver = cached_pair_fn or cached_sota_pair
    specs = list(resolver() or [])
    if specs:
        models_used = [
            {
                "name": str(spec.get("name", "")),
                "hf_id": str(spec.get("hf_id", "")),
                "model_path": spec.get("model_path"),
                "available": True,
                "used_for_generation": False,
            }
            for spec in specs
        ]
    else:
        models_used = [
            {
                "name": model["name"],
                "hf_id": model["hf_id"],
                "model_path": None,
                "available": False,
                "used_for_generation": False,
            }
            for model in SOTA_GGUF_MODELS
        ]
    return {
        "MODEL_SPECS": specs,
        "models_used": models_used,
        "cached_sota_available": bool(specs),
    }


def _responses_for_phase(
    items: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    model_specs: Sequence[Mapping[str, Any]],
    live_response_provider: ResponseProvider | None,
) -> list[str]:
    if live_response_provider is not None:
        return live_response_provider(items, phase, model_specs)
    return [synthesise_response(item, phase=phase) for item in items]


def run_bounded_comparison(
    *,
    verifier_weights: Mapping[str, float],
    model_resolution: Mapping[str, Any],
    live_response_provider: ResponseProvider | None = None,
    n_train: int = DEFAULT_N_TRAIN,
    n_eval: int = DEFAULT_N_EVAL,
) -> dict[str, Any]:
    """Run the small before/after comparison over the held-out slice."""

    slices = build_smoke_slices(n_train=n_train, n_eval=n_eval)
    model_specs = list(model_resolution["MODEL_SPECS"])
    live = bool(model_resolution["cached_sota_available"] and live_response_provider is not None)
    before_responses = _responses_for_phase(
        slices["eval"],
        phase="before",
        model_specs=model_specs,
        live_response_provider=live_response_provider if live else None,
    )
    after_responses = _responses_for_phase(
        slices["eval"],
        phase="after",
        model_specs=model_specs,
        live_response_provider=live_response_provider if live else None,
    )
    before_scores = [
        score_weighted_prime_vprm_reward(item, response, verifier_weights)["weighted_reward"]
        for item, response in zip(slices["eval"], before_responses, strict=True)
    ]
    after_scores = [
        score_weighted_prime_vprm_reward(item, response, verifier_weights)["weighted_reward"]
        for item, response in zip(slices["eval"], after_responses, strict=True)
    ]
    before_reward = _mean(before_scores)
    after_reward = _mean(after_scores)
    delta = round(after_reward - before_reward, 6)
    return {
        "execution_mode": "live_sota" if live else "smoke_only",
        "terminal_status": (
            "live_sota_complete"
            if live
            else (
                "smoke_only_live_sota_unavailable"
                if model_resolution["cached_sota_available"]
                else "smoke_only_no_sota_gguf"
            )
        ),
        "headline_result_allowed": bool(live),
        "n_train_items": len(slices["train"]),
        "n_eval_items": len(slices["eval"]),
        "eval_reward_before": round(before_reward, 6),
        "eval_reward_after": round(after_reward, 6),
        "self_learning_delta_overall": delta,
        "grpo_v8_delta_pp": round(100.0 * delta, 6),
    }


def derive_honest_verdict(execution_mode: str, terminal_status: str, delta_pp: float) -> str:
    """Return the Exp 1273 honest verdict string."""

    if execution_mode == "blocked":
        return terminal_status
    if execution_mode == "smoke_only":
        return "smoke_only_not_headline"
    return f"live_sota_delta_pp_{float(delta_pp):.1f}"


def _blocked_artifact(
    *,
    run_date: str,
    started_at: str,
    finished_at: str,
    terminal_status: str,
    wall_budget_s: float,
    project_root: str,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "blocked",
        "execution_mode": "blocked",
        "terminal_status": terminal_status,
        "honest_verdict": terminal_status,
        "MODEL_SPECS": [],
        "models_used": resolve_model_specs(cached_pair_fn=lambda: None)["models_used"],
        "verifier_weights_used": {},
        "n_train_items": 0,
        "n_eval_items": 0,
        "eval_reward_before": 0.0,
        "eval_reward_after": 0.0,
        "self_learning_delta_overall": 0.0,
        "grpo_v8_delta_pp": 0.0,
        "headline_result_allowed": False,
        "wall_budget_s": float(wall_budget_s),
        "project_root": project_root,
        "source_artifacts": {"exp1272": str(DEFAULT_EXP1272_PATH.relative_to(REPO_ROOT))},
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert the final artifact satisfies REQ-LEARN-1273."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    model_ids = {str(model.get("hf_id", "")) for model in artifact["models_used"]}
    if not (model_ids & MANDATED_SOTA_HF_IDS):
        raise AssertionError("models_used must include a mandated SOTA GGUF id")
    if artifact["execution_mode"] == "live_sota":
        if not artifact["headline_result_allowed"]:
            raise AssertionError("live_sota artifacts must allow headline results")
        if not artifact["MODEL_SPECS"]:
            raise AssertionError("live_sota artifacts require MODEL_SPECS")
    elif artifact["headline_result_allowed"]:
        raise AssertionError("non-live_sota artifacts must not allow headline results")
    expected_delta_pp = round(100.0 * float(artifact["self_learning_delta_overall"]), 6)
    if float(artifact["grpo_v8_delta_pp"]) != expected_delta_pp:
        raise AssertionError("grpo_v8_delta_pp must equal 100 * self_learning_delta_overall")


def run_experiment(
    *,
    exp1272_path: Path | str = DEFAULT_EXP1272_PATH,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    cached_pair_fn: Callable[[], Sequence[Mapping[str, Any]] | None] | None = None,
    live_response_provider: ResponseProvider | None = None,
    run_date: str = RUN_DATE,
    wall_budget_s: float = DEFAULT_WALL_BUDGET_S,
    project_root: str = "/home/ianblenke/github.com/ianblenke/carnot",
) -> dict[str, Any]:
    """Run Exp 1273 and persist the final smoke artifact."""

    write_in_progress_artifact(output_path, run_date=run_date)
    started_at = _utc_now()
    finished_at = started_at
    try:
        verifier_weights = load_verifier_weights(exp1272_path)
    except ValueError:
        artifact = _blocked_artifact(
            run_date=run_date,
            started_at=started_at,
            finished_at=finished_at,
            terminal_status="blocked_missing_verifier_weights",
            wall_budget_s=wall_budget_s,
            project_root=project_root,
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    model_resolution = resolve_model_specs(cached_pair_fn=cached_pair_fn)
    comparison = run_bounded_comparison(
        verifier_weights=verifier_weights,
        model_resolution=model_resolution,
        live_response_provider=live_response_provider,
    )
    models_used = [dict(model) for model in model_resolution["models_used"]]
    if comparison["execution_mode"] == "live_sota":
        for model in models_used:
            model["used_for_generation"] = True
    finished_at = _utc_now()
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "complete",
        "MODEL_SPECS": model_resolution["MODEL_SPECS"],
        "models_used": models_used,
        "verifier_weights_used": dict(verifier_weights),
        "wall_budget_s": float(wall_budget_s),
        "project_root": project_root,
        "source_artifacts": {"exp1272": str(Path(exp1272_path))},
        "artifact_metadata": {"project_root": project_root, "run_date": run_date},
    }
    artifact.update(comparison)
    artifact["honest_verdict"] = derive_honest_verdict(
        artifact["execution_mode"],
        artifact["terminal_status"],
        artifact["grpo_v8_delta_pp"],
    )
    validate_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_experiment()
    print(
        artifact["honest_verdict"],
        artifact["grpo_v8_delta_pp"],
        artifact["headline_result_allowed"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
