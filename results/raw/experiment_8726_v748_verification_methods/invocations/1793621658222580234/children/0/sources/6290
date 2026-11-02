"""Exp 1880 SOTA GGUF ROCE validator-tree evaluation.

Spec: REQ-VERIFY-1880, SCENARIO-VERIFY-1880.

The model output path is intentionally outside the trust boundary.  Local GGUF
models may produce candidate text, but acceptance and false-accept accounting
come only from the Exp 1878 executable ROCE validator tree and the Exp 1879
BEAVER-lite deterministic bound rows.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.pipeline import roce_validator_tree
from carnot.verify import roce_beaver_lite_bounds

JsonDict = dict[str, Any]
GeneratorFn = Callable[[Mapping[str, Any], Mapping[str, Any]], str]
CachedPairFn = Callable[..., list[JsonDict] | None]
ResolverFn = Callable[[str, str], str | None]

RUN_DATE = "20260512"
EXPERIMENT_ID = 1880
EXPERIMENT = "1880_sota_roce_validator_eval"
ARTIFACT_SCHEMA = "carnot.experiment_1880_sota_roce_validator_eval.v1"
SPEC_TRACES = ["REQ-VERIFY-1880", "SCENARIO-VERIFY-1880"]
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[3] / "results" / "experiment_1880_sota_roce_validator_eval.json"
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "honest_verdict",
    "sota_roce_eval_ready",
    "inference_mode",
    "MODEL_SPECS",
    "models_used",
    "zero_false_accepts",
    "false_accept_count",
    "constraint_coverage_rate",
    "tests_run",
)

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "gpu": 0,
        "preferred_quant": "Q4_K_M",
        "role": "flagship_moe_primary",
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "gpu": 1,
        "preferred_quant": "Q4_K_M",
        "role": "flagship_dense_secondary",
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "gpu": 0,
        "preferred_quant": "Q4_K_M",
        "role": "middle_moe_secondary",
    },
)
MANDATED_HF_IDS = frozenset(spec["hf_id"] for spec in MODEL_SPECS)


def default_prompt_cases(count: int = 30) -> list[JsonDict]:
    """Return the 30-case format/arithmetic/lexical/conditional suite."""

    cases: list[JsonDict] = []
    for index in range(count):
        a = index + 2
        b = index + 3
        total = a + b
        token = f"token{index:02d}"
        forbidden = f"blocked{index:02d}"
        audited = f"audited{index:02d}"
        prompt = (
            "Return a single-line JSON object only. "
            'Use strict key order {"answer": ..., "sum": ...} and no other top-level keys. '
            f'Include "{token}". Do not mention "{forbidden}". Keep under 20 words. '
            f"The response must state {a} + {b} = {total}. "
            f'If the response contains "{token}", it must also contain "{audited}".'
        )
        cases.append(
            {
                "case_id": f"roce-1880-{index:02d}",
                "prompt": prompt,
                "constraint_family_tags": ["format", "arithmetic", "lexical", "conditional"],
                "known_good": json.dumps(
                    {"answer": f"{token} {audited}", "sum": f"{a} + {b} = {total}"},
                    separators=(",", ":"),
                ),
                "known_bad": [
                    json.dumps(
                        {"answer": token, "sum": f"{a} + {b} = {total}"},
                        separators=(",", ":"),
                    )
                ],
            }
        )
    return cases


def summarize_prompt_coverage(cases: Iterable[Mapping[str, Any]]) -> JsonDict:
    """Summarize ROCE compiler coverage over the prompt suite."""

    case_list = list(cases)
    total_constraints = 0
    supported_constraints = 0
    unsupported: set[str] = set()
    families: set[str] = set()
    for case in case_list:
        tree = roce_validator_tree.compile_roce_validator_tree(
            str(case.get("prompt") or ""),
            case_id=str(case.get("case_id") or ""),
        )
        total_constraints += tree.total_constraint_count
        supported_constraints += tree.supported_constraint_count
        unsupported.update(tree.unsupported_constraint_types)
        families.update(str(tag) for tag in case.get("constraint_family_tags", []))
    return {
        "prompt_count": len(case_list),
        "total_constraint_count": total_constraints,
        "supported_constraint_count": supported_constraints,
        "constraint_coverage_rate": _rate(supported_constraints, total_constraints),
        "unsupported_constraint_types": sorted(unsupported),
        "constraint_families": sorted(families),
    }


def resolve_mandated_model_specs(
    *,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    resolver_fn: ResolverFn = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve all three mandated GGUF specs, using `cached_sota_pair()` first."""

    pair_paths: dict[str, str] = {}
    for model_indices in ((0, 2), (0, 1)):
        pair = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M", model_indices=model_indices)
        for spec in pair or []:
            if spec.get("hf_id") and spec.get("model_path"):
                pair_paths[str(spec["hf_id"])] = str(spec["model_path"])

    resolved: list[JsonDict] = []
    for spec in MODEL_SPECS:
        row = dict(spec)
        model_path = pair_paths.get(str(spec["hf_id"])) or resolver_fn(
            str(spec["hf_id"]),
            str(spec["preferred_quant"]),
        )
        if model_path:
            row["model_path"] = str(model_path)
        resolved.append(row)
    return resolved


def evaluate_control_cases(cases: Iterable[Mapping[str, Any]]) -> list[JsonDict]:
    """Evaluate known-good and known-bad controls for false-accept accounting."""

    rows: list[JsonDict] = []
    for case in cases:
        tree = roce_validator_tree.compile_roce_validator_tree(
            str(case.get("prompt") or ""),
            case_id=str(case.get("case_id") or ""),
        )
        bounds = roce_beaver_lite_bounds.compute_tree_bounds(tree)
        good = tree.validate(str(case.get("known_good") or ""))
        bad_results = [tree.validate(str(output)) for output in case.get("known_bad", [])]
        false_accept_count = sum(1 for result in bad_results if result.accepted)
        rows.append(
            {
                "case_id": tree.case_id,
                "validator_tree": tree.to_dict(),
                "beaver_lite_bounds": bounds.to_dict(),
                "known_good": good.to_dict(),
                "known_bad": [result.to_dict() for result in bad_results],
                "false_accept_count": false_accept_count,
                "zero_false_accepts": false_accept_count == 0,
            }
        )
    return rows


def evaluate_generations(
    cases: Iterable[Mapping[str, Any]],
    model_specs: Iterable[Mapping[str, Any]],
    *,
    generator_fn: GeneratorFn | None = None,
) -> list[JsonDict]:
    """Generate candidate text and validate it with the executable ROCE tree."""

    rows: list[JsonDict] = []
    for model in model_specs:
        for case in cases:
            output_text, mode = _generate_text(model, case, generator_fn=generator_fn)
            tree = roce_validator_tree.compile_roce_validator_tree(
                str(case.get("prompt") or ""),
                case_id=str(case.get("case_id") or ""),
            )
            validation = tree.validate(output_text)
            bounds = roce_beaver_lite_bounds.compute_tree_bounds(tree)
            rows.append(
                {
                    "case_id": tree.case_id,
                    "model_hf_id": model.get("hf_id"),
                    "model_name": model.get("name") or model.get("hf_id"),
                    "output_text": output_text,
                    "validation": validation.to_dict(),
                    "beaver_lite_bounds": bounds.to_dict(),
                    "provenance": {
                        "mode": mode,
                        "model_path": model.get("model_path"),
                        "hf_id": model.get("hf_id"),
                    },
                }
            )
    return rows


def build_artifact(
    *,
    model_specs: Iterable[Mapping[str, Any]],
    generation_rows: Iterable[Mapping[str, Any]],
    case_results: Iterable[Mapping[str, Any]],
    tests_run: list[str] | None,
    inference_mode: str,
) -> JsonDict:
    """Build the terminal Exp 1880 artifact without writing it."""

    specs = [dict(spec) for spec in model_specs]
    generation_list = [dict(row) for row in generation_rows]
    case_list = [dict(row) for row in case_results]
    coverage = _coverage_from_case_results(case_list)
    false_accept_count = sum(int(row.get("false_accept_count") or 0) for row in case_list)
    known_good_ready = bool(case_list) and all(row["known_good"]["accepted"] for row in case_list)
    bounds_ready = bool(case_list) and all(
        row["beaver_lite_bounds"]["beaver_lite_bounds_ready"] for row in case_list
    )
    zero_false_accepts = bool(case_list) and false_accept_count == 0
    all_models_available = all(bool(spec.get("model_path")) for spec in specs)
    complete_generation = bool(generation_list) and len(generation_list) == len(specs) * len(case_list)
    ready = bool(
        all_models_available
        and complete_generation
        and known_good_ready
        and bounds_ready
        and zero_false_accepts
        and coverage["constraint_coverage_rate"] == 1.0
    )
    status = "complete" if ready else "partial"
    return {
        "status": status,
        "schema": ARTIFACT_SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "timestamp": _timestamp(),
        "spec_traces": list(SPEC_TRACES),
        "honest_verdict": _honest_verdict(status, inference_mode, false_accept_count),
        "sota_roce_eval_ready": ready,
        "headline_accuracy_claimed": ready,
        "inference_mode": inference_mode,
        "MODEL_SPECS": specs,
        "models_used": [str(spec["hf_id"]) for spec in specs if spec.get("model_path")],
        "zero_false_accepts": zero_false_accepts,
        "false_accept_count": false_accept_count,
        "constraint_coverage_rate": coverage["constraint_coverage_rate"],
        "unsupported_constraint_types": coverage["unsupported_constraint_types"],
        "constraint_families": coverage["constraint_families"],
        "prompt_count": len(case_list),
        "output_rows": len(generation_list),
        "case_results": case_list,
        "generation_rows": generation_list,
        "tests_run": list(tests_run or []),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert the Exp 1880 artifact follows the required honest schema."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    coverage = float(artifact["constraint_coverage_rate"])
    assert 0.0 <= coverage <= 1.0, "coverage out of range"
    assert int(artifact["false_accept_count"]) >= 0, "false_accept_count must be nonnegative"
    if artifact["status"] == "complete":
        assert artifact["sota_roce_eval_ready"] is True, "complete requires ready"
        assert artifact["zero_false_accepts"] is True, "complete requires zero false accepts"
        assert artifact["false_accept_count"] == 0, "complete requires false_accept_count=0"
        assert artifact["headline_accuracy_claimed"] is True, "complete requires headline claim"
        assert len(artifact["models_used"]) == len(MODEL_SPECS), "complete requires all models"
        assert artifact["output_rows"] >= 30 * len(MODEL_SPECS), "complete requires 30 prompts/model"
    if artifact["status"] == "blocked":
        assert artifact["sota_roce_eval_ready"] is False, "blocked must not be ready"
        assert artifact["headline_accuracy_claimed"] is False, "blocked must not claim headline accuracy"


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    model_specs: Iterable[Mapping[str, Any]] | None = None,
    prompt_cases: Iterable[Mapping[str, Any]] | None = None,
    generator_fn: GeneratorFn | None = None,
    tests_run: list[str] | None = None,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    resolver_fn: ResolverFn = resolve_cached_gguf,
) -> JsonDict:
    """Run Exp 1880 and write the terminal JSON artifact."""

    specs = (
        [dict(spec) for spec in model_specs]
        if model_specs is not None
        else resolve_mandated_model_specs(cached_pair_fn=cached_pair_fn, resolver_fn=resolver_fn)
    )
    missing = [str(spec["hf_id"]) for spec in specs if not spec.get("model_path")]
    if missing:
        artifact = _blocked_artifact(specs=specs, missing_models=missing, tests_run=tests_run)
        validate_artifact(artifact)
        return _write_json(output_path, artifact)

    cases = list(prompt_cases) if prompt_cases is not None else default_prompt_cases()
    inference_mode = "injected" if generator_fn is not None else "live_gguf"
    case_results = evaluate_control_cases(cases)
    generation_rows = evaluate_generations(cases, specs, generator_fn=generator_fn)
    artifact = build_artifact(
        model_specs=specs,
        generation_rows=generation_rows,
        case_results=case_results,
        tests_run=tests_run,
        inference_mode=inference_mode,
    )
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def _blocked_artifact(
    *,
    specs: list[JsonDict],
    missing_models: list[str],
    tests_run: list[str] | None,
) -> JsonDict:
    models_used = [str(spec["hf_id"]) for spec in specs if spec.get("model_path")]
    return {
        "status": "blocked",
        "schema": ARTIFACT_SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "timestamp": _timestamp(),
        "spec_traces": list(SPEC_TRACES),
        "honest_verdict": (
            "blocked: unavailable mandated SOTA GGUF model(s); no headline "
            f"accuracy claimed; missing={missing_models}"
        ),
        "sota_roce_eval_ready": False,
        "headline_accuracy_claimed": False,
        "inference_mode": "blocked_missing_mandated_gguf",
        "MODEL_SPECS": specs,
        "models_used": models_used,
        "missing_models": list(missing_models),
        "zero_false_accepts": False,
        "false_accept_count": 0,
        "constraint_coverage_rate": 0.0,
        "unsupported_constraint_types": [],
        "constraint_families": [],
        "prompt_count": 0,
        "output_rows": 0,
        "case_results": [],
        "generation_rows": [],
        "tests_run": list(tests_run or []),
    }


def _generate_text(
    model: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    generator_fn: GeneratorFn | None,
) -> tuple[str, str]:
    if generator_fn is not None:
        return str(generator_fn(model, case)), "injected"
    return _generate_live_gguf(model, case), "live_gguf"  # pragma: no cover


def _generate_live_gguf(model: Mapping[str, Any], case: Mapping[str, Any]) -> str:  # pragma: no cover
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model["model_path"]),
        n_gpu_layers=-1,
        n_ctx=1024,
        verbose=False,
    )
    response = llm(
        str(case["prompt"]),
        max_tokens=96,
        temperature=0.0,
        top_p=1.0,
        stop=["\n\n"],
    )
    return str(response["choices"][0]["text"]).strip()


def _coverage_from_case_results(case_results: list[Mapping[str, Any]]) -> JsonDict:
    total = sum(int(row["validator_tree"]["total_constraint_count"]) for row in case_results)
    supported = sum(int(row["validator_tree"]["supported_constraint_count"]) for row in case_results)
    unsupported = sorted(
        {
            str(item)
            for row in case_results
            for item in row["validator_tree"]["unsupported_constraint_types"]
        }
    )
    families = ["format", "arithmetic", "lexical", "conditional"] if case_results else []
    return {
        "constraint_coverage_rate": _rate(supported, total),
        "unsupported_constraint_types": unsupported,
        "constraint_families": families,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _honest_verdict(status: str, inference_mode: str, false_accept_count: int) -> str:
    if status == "complete":
        return (
            "complete: mandated SOTA GGUF outputs gated by ROCE validator trees "
            f"with zero false accepts; inference_mode={inference_mode}"
        )
    return (
        "partial: SOTA ROCE eval did not meet headline gates; "
        f"inference_mode={inference_mode}, false_accept_count={false_accept_count}"
    )


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = dict(payload)
    destination.write_text(
        json.dumps(written, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return written


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_ARTIFACT_PATH))
    args = parser.parse_args(argv)
    artifact = run_experiment(output_path=args.output)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "EXPERIMENT_ID",
    "MANDATED_HF_IDS",
    "MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "default_prompt_cases",
    "evaluate_control_cases",
    "evaluate_generations",
    "resolve_mandated_model_specs",
    "run_experiment",
    "summarize_prompt_coverage",
    "validate_artifact",
]
