"""DRAFT (operator-facing): boosted multi-model LLM-judge efficiency panel.

WHY THIS EXISTS
---------------
Exp 3917 (the .361 efficiency head-to-head) reported the energy verifier as
~11,512x cheaper than "an LLM judge" -- but that judge scored AUROC 0.442, which
is ~chance.  The root cause is NOT a weak model: exp3917 ran the judge at
``max_tokens=96`` / ``n_ctx=1024`` -- the SAME under-resourced budget that exp3916
labelled the *weak* reasoner (also 0.442).  The *strong* budget
(``max_tokens=160`` / ``n_ctx=2048``), with the identical judge prompt and the
identical model, reaches 0.663 in exp3916.  So a cost claim made against the
96-token judge is a claim against a deliberately hobbled baseline -- exactly what
an adversarial reviewer would reject.

This draft closes that caveat two ways, per the operator decision (2026-06-07):

  1. BOOST the judge to the strong budget (160 tok / 2048 ctx) so the comparator
     runs at its real accuracy.
  2. Run a PANEL of local SOTA judges instead of one, so the claim becomes
     "the energy verifier cost-dominates EVERY SOTA local judge at
     matched-or-better accuracy", not "cheaper than one throttled judge".

Panel (all local open-weight GGUFs -- decentralization rules 1/2; closed frontier
models are intentionally excluded from the headline):

  - gemma-4-26B-A4B-it   (the exp3917 model, now at the strong budget)
  - Qwen3.6-35B-A3B      (flagship MoE)
  - gemma-4-31B-it       (flagship dense)
  - gemma-4-12B-it       (operator-added 2026-06-07; lightweight SOTA, released
                          2026-06-05 -- may need a download, gated in PRECONDITIONS)

The energy verifier is model-independent, so each panel model re-measures it
back-to-back with the judge: that keeps every cost ratio apples-to-apples on the
same machine state rather than dividing by a single global energy timing.

WHAT THIS IS / IS NOT
---------------------
This is a DRAFT for operator review and a *paused-conductor* internal-GPU run.
It is NOT wired into the conductor roadmap.  It reuses the tested
``carnot.eval.efficiency_head_to_head_3917`` module wholesale -- the only new code
is the panel loop, the per-model GGUF cache precondition, and the aggregation
verdict.  Run it only with the conductor paused (it loads 4 GGUFs sequentially on
the internal GPU).

  # fast sanity check (cheapest model, 24 items, no full corpus):
  .venv/bin/python scripts/experiments/verifier_efficiency_panel_draft.py --smoke
  # full panel:
  .venv/bin/python scripts/experiments/verifier_efficiency_panel_draft.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.eval import efficiency_head_to_head_3917 as base
from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.cost_instrumented_verification import (
    measure_verification_cost,
    model_params_for_path,
    run_energy_verifier,
)
from carnot.verify.gguf_inference import load_gguf_generator
from carnot.verify.reasoner_self_verification import build_judge_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_REL_PATH = Path("results/experiment_efficiency_panel_boosted_draft.json")

# The operator-approved local SOTA panel (model_name keys -> unsloth/<name>-GGUF).
PANEL_MODELS: tuple[str, ...] = (
    "gemma-4-26B-A4B-it",
    "Qwen3.6-35B-A3B",
    "gemma-4-31B-it",
    "gemma-4-12B-it",  # operator-added 2026-06-07; newly released, verify cache
)
# Lightest panel member first so --smoke is fast and a partial cache still runs.
SMOKE_MODEL = "gemma-4-12B-it"

# The strong budget that reaches 0.663 in exp3916 (vs exp3917's hobbled 96/1024).
BOOSTED_MAX_TOKENS = 160
BOOSTED_N_CTX = 2048
DEFAULT_RANDOM_SEED = 3917
DURATION_FLOOR_S = 60.0
COST_DOMINANCE_RATIO = 10.0  # "Nx cheaper" headline floor (matches exp3917)
INFERENCE_SUBSTRATE = (
    "live_llm_inference:boosted_judge_panel_strong_budget_160tok_2048ctx_"
    "plus_exp3905_cost_harness_energy_verifier"
)


def _hf_id(model_name: str) -> str:
    return f"unsloth/{model_name}-GGUF"


def _checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PanelModelResult:
    """One judge model's accuracy + measured cost vs the energy verifier."""

    model_name: str
    cached: bool
    detail: str
    llm_judge_auroc: float | None = None
    llm_judge_ci95: dict[str, float] | None = None
    energy_auroc: float | None = None
    cost_ratio_walltime: float | None = None
    cost_ratio_flops: float | None = None
    llm_per_item_ms: float | None = None
    energy_per_item_ms: float | None = None
    parameter_count: int | None = None
    gguf_path: str | None = None
    energy_within_judge_ci: bool | None = None
    energy_accuracy_gte_judge: bool | None = None

    def as_dict(self) -> dict[str, object]:
        return {k: v for k, v in self.__dict__.items()}


def _precondition_models(models: Sequence[str]) -> list[PanelModelResult]:
    """Resolve each panel model's cached GGUF; record (not skip) the absent ones."""

    out: list[PanelModelResult] = []
    for name in models:
        try:
            path = resolve_cached_gguf(_hf_id(name))
        except Exception as exc:  # pragma: no cover - cache lookup is environment-specific
            out.append(PanelModelResult(name, False, f"resolve_error: {exc!r}"))
            continue
        if path and Path(path).is_file() and Path(path).stat().st_size > 0:
            out.append(PanelModelResult(name, True, f"cached: {path}", gguf_path=path))
        else:
            note = "not_cached"
            if name == "gemma-4-12B-it":
                note += (
                    " (newly released 2026-06-05; download with "
                    "`huggingface-cli download unsloth/gemma-4-12B-it-GGUF` before running)"
                )
            out.append(PanelModelResult(name, False, note))
    return out


def _load_judge_generator(model_name: str, gguf_path: str) -> tuple[Any, dict[str, object]]:
    """Load a judge generator, falling back to a direct llama.cpp load.

    The shared ``load_gguf_generator`` smoke gate emits a 1-token generic prompt
    and rejects any model whose first token is whitespace.  gemma-4-12B-it emits a
    leading newline, so it fails that gate even though it generates valid verdict
    JSON on a real prompt.  Rather than weaken the shared core loader (which the
    conductor depends on), this fallback does a *real-prompt* smoke: it builds the
    actual judge prompt, generates a few tokens, and accepts the model iff the
    output is non-empty after stripping.  Same fair bar, judge-appropriate prompt.
    """

    try:
        return load_gguf_generator(
            prefer_order=[model_name], n_ctx=BOOSTED_N_CTX, max_n_gpu_layers=-1
        )
    except RuntimeError as exc:
        from llama_cpp import Llama

        llm = Llama(
            model_path=gguf_path, n_ctx=BOOSTED_N_CTX, n_gpu_layers=-1, verbose=False
        )
        probe = llm(build_judge_prompt("47 + 28 = 75."), max_tokens=16, temperature=0.0)
        text = str(probe["choices"][0]["text"]).strip()
        if not text:
            raise RuntimeError(f"{model_name} real-prompt smoke empty after fallback: {exc!r}") from exc
        meta: dict[str, object] = {
            "gguf_path": gguf_path,
            "model_used": model_name,
            "loader": "direct_llama_fallback_real_prompt_smoke",
            "fallback_reason": f"shared_loader_smoke_gate_rejected: {exc!r}"[:240],
            "real_prompt_smoke_tokens": int(probe["usage"]["completion_tokens"]),
        }
        return llm, meta


def _score_one_model(
    model_name: str,
    gguf_path: str,
    *,
    bundle: base.CorpusBundle,
    seed: int,
    bootstrap_resamples: int,
) -> PanelModelResult:
    """Load one judge at the STRONG budget, score it + energy back-to-back."""

    generator, meta = _load_judge_generator(model_name, gguf_path)
    try:
        params = model_params_for_path(str(meta.get("gguf_path") or gguf_path))
        model_specs = {
            **dict(meta),
            "model_used": model_name,
            "n_ctx": BOOSTED_N_CTX,
            "max_tokens": BOOSTED_MAX_TOKENS,
            "parameter_count_for_flop_estimate": params,
        }
        # Reuse the tested exp3917 coupled measurement: energy + judge, same state.
        measured = base.measure_head_to_head_costs(
            bundle.items,
            generator=generator,
            model_specs=model_specs,
            max_tokens=BOOSTED_MAX_TOKENS,
        )
    finally:
        del generator
        gc.collect()

    labels = bundle.labels
    llm_auroc = float(measured.llm_cost["auroc"])
    energy_auroc = float(measured.energy_cost["auroc"])
    llm_ci95 = base.bootstrap_ci95(
        labels, measured.llm_scores, seed=seed + 17, resamples=bootstrap_resamples
    )
    energy_ms = float(measured.energy_cost["per_item_wall_ms"])
    llm_ms = float(measured.llm_cost["per_item_wall_ms"])
    ratio_wall = (llm_ms / energy_ms) if energy_ms > 0 else None
    ratio_flops = None
    e_flops = float(measured.energy_cost.get("est_flops") or 0.0)
    l_flops = float(measured.llm_cost.get("est_flops") or 0.0)
    if e_flops > 0:
        ratio_flops = l_flops / e_flops
    return PanelModelResult(
        model_name=model_name,
        cached=True,
        detail=f"scored n={bundle.n_items} at {BOOSTED_MAX_TOKENS}tok/{BOOSTED_N_CTX}ctx",
        llm_judge_auroc=llm_auroc,
        llm_judge_ci95=llm_ci95,
        energy_auroc=energy_auroc,
        cost_ratio_walltime=ratio_wall,
        cost_ratio_flops=ratio_flops,
        llm_per_item_ms=llm_ms,
        energy_per_item_ms=energy_ms,
        parameter_count=params,
        gguf_path=str(meta.get("gguf_path") or gguf_path),
        energy_within_judge_ci=bool(llm_ci95["low"] <= energy_auroc <= llm_ci95["high"]),
        energy_accuracy_gte_judge=bool(energy_auroc >= llm_auroc),
    )


def _panel_verdict(scored: Sequence[PanelModelResult]) -> str:
    """Terminal-prefixed verdict over the scored judges (>=1 required)."""

    real = [r for r in scored if r.llm_judge_auroc is not None]
    if not real:
        return "blocked_no_panel_model_scored"
    energy = real[0].energy_auroc or 0.0
    max_judge = max(float(r.llm_judge_auroc) for r in real)
    worst_ratio = min(float(r.cost_ratio_walltime or 0.0) for r in real)
    all_cheaper = all((r.cost_ratio_walltime or 0.0) > COST_DOMINANCE_RATIO for r in real)
    energy_beats_all = all((r.energy_accuracy_gte_judge for r in real))
    energy_parity_all = all((r.energy_within_judge_ci for r in real))
    n = len(real)
    tag = f"_energy{energy:.4f}_maxjudge{max_judge:.4f}_mincheaper{worst_ratio:.1f}x_n{n}"
    if all_cheaper and energy_beats_all:
        return "complete: efficiency_panel_ENERGY_DOMINATES_all_cheaper_and_more_accurate" + tag
    if all_cheaper and energy_parity_all:
        return "complete: efficiency_panel_ENERGY_PARITY_all_cheaper_within_each_ci" + tag
    if all_cheaper:
        return "complete: efficiency_panel_CHEAPER_all_but_a_judge_more_accurate_honest_partial" + tag
    return "complete: efficiency_panel_NOT_UNIFORMLY_CHEAPER_honest_partial" + tag


def run_panel(
    *,
    models: Sequence[str] = PANEL_MODELS,
    smoke: bool = False,
    seed: int = DEFAULT_RANDOM_SEED,
    write: bool = True,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Run the boosted judge panel (or write a blocked artifact on failed gates)."""

    started = time.time()
    config = base.ExperimentConfig(
        repo_root=REPO_ROOT,
        random_seed=seed,
        max_tokens=BOOSTED_MAX_TOKENS,
        n_ctx=BOOSTED_N_CTX,
    )
    out_path = output_path or (REPO_ROOT / OUTPUT_REL_PATH)
    resamples = 200 if smoke else base.DEFAULT_BOOTSTRAP_RESAMPLES

    # --- PRECONDITIONS (check BEFORE any GGUF load) -------------------------
    cuda = base._probe_cuda_with_venv(config)
    model_pre = _precondition_models([SMOKE_MODEL] if smoke else models)
    cached = [m for m in model_pre if m.cached]
    preconditions = [
        {"resource": "cuda_available", "available": cuda.available, "detail": cuda.detail},
        *[
            {"resource": f"gguf_cached:{m.model_name}", "available": m.cached, "detail": m.detail}
            for m in model_pre
        ],
    ]
    try:
        bundle = base.load_labeled_corpora(REPO_ROOT, random_seed=seed)
        preconditions.append(
            {"resource": "labeled_corpora", "available": True, "detail": f"n={bundle.n_items}"}
        )
    except Exception as exc:
        bundle = None
        preconditions.append(
            {"resource": "labeled_corpora", "available": False, "detail": repr(exc)}
        )

    blocked_reason = None
    if not cuda.available:
        blocked_reason = "blocked_no_cuda"
    elif bundle is None:
        blocked_reason = "blocked_labeled_corpora_not_ready"
    elif not cached:
        blocked_reason = "blocked_no_panel_model_cached"

    if blocked_reason is not None or bundle is None:
        artifact = {
            "experiment": "efficiency_panel_boosted_draft",
            "title": "verifier_efficiency_panel_draft",
            "honest_verdict": blocked_reason or "blocked_labeled_corpora_not_ready",
            "status": blocked_reason or "blocked_labeled_corpora_not_ready",
            "panel_results": [m.as_dict() for m in model_pre],
            "preconditions_checked": preconditions,
            "n_items": 0,
            "random_seed": seed,
            "duration_s": time.time() - started,
            "inference_substrate": "none_blocked_preflight",
            "reproducibility_checksum": _checksum(
                {"reason": blocked_reason, "preconditions": preconditions}
            ),
        }
        if write:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
        return artifact

    if smoke:
        bundle = base.CorpusBundle(
            items=bundle.items[:24],
            labels=bundle.labels[:24],
            corpus_sources=bundle.corpus_sources,
            checksum=bundle.checksum + ":smoke24",
        )

    # --- SCORE EACH CACHED JUDGE AT THE STRONG BUDGET ----------------------
    scored: list[PanelModelResult] = []
    for m in model_pre:
        if not m.cached:
            scored.append(m)
            continue
        scored.append(
            _score_one_model(
                m.model_name,
                str(m.gguf_path),
                bundle=bundle,
                seed=seed,
                bootstrap_resamples=resamples,
            )
        )

    duration_s = time.time() - started
    verdict = _panel_verdict(scored)
    if not smoke and duration_s < DURATION_FLOOR_S and verdict.startswith("complete:"):
        verdict = "blocked_panel_duration_below_floor"

    real = [r for r in scored if r.llm_judge_auroc is not None]
    artifact = {
        "experiment": "efficiency_panel_boosted_draft",
        "title": "verifier_efficiency_panel_draft",
        "run_date": datetime.fromtimestamp(time.time(), tz=UTC).strftime("%Y%m%d"),
        "honest_verdict": verdict,
        "status": verdict,
        "smoke": smoke,
        "energy_auroc": real[0].energy_auroc if real else None,
        "max_judge_auroc": max((float(r.llm_judge_auroc) for r in real), default=None),
        "min_cost_ratio_walltime": min(
            (float(r.cost_ratio_walltime or 0.0) for r in real), default=None
        ),
        "boosted_max_tokens": BOOSTED_MAX_TOKENS,
        "boosted_n_ctx": BOOSTED_N_CTX,
        "panel_results": [r.as_dict() for r in scored],
        "n_models_scored": len(real),
        "n_items": bundle.n_items,
        "preconditions_checked": preconditions,
        "random_seed": seed,
        "random_seeds_used": {"fover_slice": seed, "bootstrap_llm_judge": seed + 17},
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": _checksum(
            {
                "bundle": bundle.checksum,
                "panel": [(r.model_name, r.llm_judge_auroc, r.cost_ratio_walltime) for r in real],
                "boosted": [BOOSTED_MAX_TOKENS, BOOSTED_N_CTX],
                "seed": seed,
            }
        ),
        "field_principles": {
            "energy_auroc": "Model-independent energy verifier accuracy on the shared labels.",
            "max_judge_auroc": "Strongest judge in the panel; energy must match-or-beat to dominate.",
            "min_cost_ratio_walltime": "Worst-case 'Nx cheaper' across the panel; >10x = cost dominance.",
            "boosted_max_tokens": "160 tok = exp3916 STRONG budget (0.663-capable), not exp3917's hobbled 96.",
            "energy_accuracy_gte_judge": "Per model: does the energy verifier match-or-beat this judge.",
        },
        "caveat": (
            "DRAFT. Closes the exp3917 weak-baseline caveat by running each judge at the "
            "strong (160 tok / 2048 ctx) budget across a local SOTA panel. Not conductor-wired."
        ),
    }
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    return artifact


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="cheapest model, 24 items")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--models", nargs="*", default=None, help="override panel model_name keys"
    )
    args = parser.parse_args(argv)
    artifact = run_panel(
        models=tuple(args.models) if args.models else PANEL_MODELS,
        smoke=args.smoke,
        seed=args.seed,
        output_path=args.output_path,
        write=True,
    )
    print(f"{OUTPUT_REL_PATH.name}: {artifact['honest_verdict']}")
    return 0 if str(artifact["honest_verdict"]).startswith("complete:") else 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
