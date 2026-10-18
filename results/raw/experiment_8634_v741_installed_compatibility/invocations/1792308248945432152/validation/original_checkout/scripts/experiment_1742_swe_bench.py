#!/usr/bin/env python3
"""Exp 1742: SWE-Bench Lite verify-repair baseline with EqM disabled.

Spec: REQ-BENCH-1742, SCENARIO-BENCH-1742
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.inference.sota_models import cached_sota_pair  # noqa: E402
from carnot.pipeline.swebench_harness import (  # noqa: E402
    PatchEvaluation,
    SweBenchProblem,
    build_results_payload,
    load_swebench_lite_problems,
    run_model_on_problems,
    summarize_model_results,
)

OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1742_swe_bench.json"
EQM_DECODING_ENABLED = False
MAX_REPAIRS = 1
EVALUATOR_BACKEND = "swebench_or_injected"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_model_specs() -> list[dict[str, Any]] | None:
    return cached_sota_pair(model_indices=(0, 2))


def _missing_evaluator(problem: SweBenchProblem, patch: str, model_name: str) -> PatchEvaluation:
    del problem, patch, model_name
    return PatchEvaluation(
        resolved=False,
        status="blocked",
        error_type="missing_swebench_evaluator",
        error_message=(
            "No injected SWE-Bench evaluator was provided. Install/configure the "
            "official SWE-Bench evaluator before citing headline resolve rates."
        ),
    )


def _make_llama_generator(model_specs: list[dict[str, Any]]):
    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError(f"llama_cpp unavailable: {exc}") from exc

    llms = {}
    for spec in model_specs:
        model_path = spec.get("model_path")
        if not model_path:
            raise RuntimeError(f"missing model_path for {spec.get('name', '<unknown>')}")
        llms[str(spec["name"])] = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False,
        )

    def _generate(prompt: str, *, model_name: str, eqm_decoding_enabled: bool) -> str:
        del eqm_decoding_enabled
        output = llms[model_name](prompt, max_tokens=1536, temperature=0.0, echo=False)
        return str(output["choices"][0]["text"]).strip()

    return _generate


def _verdict_for_metrics(metrics: dict[str, Any]) -> str:
    delta = metrics.get("signed_improvement")
    if delta is None:
        return "blocked_no_headline"
    if delta > 0:
        return "verify_repair_improved"
    if delta < 0:
        return "verify_repair_regressed"
    return "baseline_complete"


def run_experiment(
    *,
    output_path: Path = OUTPUT_PATH,
    rows: list[dict[str, Any]] | None = None,
    model_specs_provider=_default_model_specs,
    generator=None,
    evaluator=None,
) -> dict[str, Any]:
    """Run Exp 1742 and write the terminal JSON artifact."""
    started = time.perf_counter()
    blockers: list[str] = []

    try:
        selected = load_swebench_lite_problems(rows=rows, limit=5)
    except Exception as exc:
        selected = []
        blockers.append(f"dataset_fetch_failed:{exc}")

    model_specs = model_specs_provider()
    if not model_specs:
        blockers.append("blocked_no_sota_gguf")
        metrics = summarize_model_results([])
        metrics["n_instances"] = len(selected)
        payload = build_results_payload(
            status="blocked",
            honest_verdict="blocked_no_sota_gguf",
            timestamp=_utc_now(),
            runtime_seconds=time.perf_counter() - started,
            selected_problems=selected,
            model_results=[],
            metrics=metrics,
            blockers=blockers,
            evaluator_backend=EVALUATOR_BACKEND,
            eqm_decoding_enabled=EQM_DECODING_ENABLED,
            max_repairs=MAX_REPAIRS,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    active_generator = generator
    if active_generator is None:
        try:
            active_generator = _make_llama_generator(list(model_specs))
        except RuntimeError as exc:
            blockers.append(f"blocked_model_backend:{exc}")

    active_evaluator = evaluator or _missing_evaluator
    model_results = []
    if selected and active_generator is not None and not blockers:
        for model_spec in model_specs:
            model_results.append(
                run_model_on_problems(
                    selected,
                    model_spec=model_spec,
                    generator=active_generator,
                    evaluator=active_evaluator,
                    max_repairs=MAX_REPAIRS,
                    eqm_decoding_enabled=EQM_DECODING_ENABLED,
                )
            )
    elif not selected:
        blockers.append("blocked_no_selected_instances")

    metrics = summarize_model_results(model_results)
    status = "complete" if metrics["headline_resolve_rates_available"] and not blockers else "blocked"
    honest_verdict = _verdict_for_metrics(metrics) if status == "complete" else "blocked_no_headline"
    payload = build_results_payload(
        status=status,
        honest_verdict=honest_verdict,
        timestamp=_utc_now(),
        runtime_seconds=time.perf_counter() - started,
        selected_problems=selected,
        model_results=model_results,
        metrics=metrics,
        blockers=blockers,
        evaluator_backend=EVALUATOR_BACKEND,
        eqm_decoding_enabled=EQM_DECODING_ENABLED,
        max_repairs=MAX_REPAIRS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    payload = run_experiment()
    print(json.dumps({"status": payload["status"], "honest_verdict": payload["honest_verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
