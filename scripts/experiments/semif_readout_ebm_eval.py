#!/usr/bin/env python3
"""A1 driver: readout energy and product of experts (REQ-VERIFY-7750).

Runs the pre-registered A1 experiment from
`docs/research-notes/semif-ebm-arc-experiment-plan-2026-09-20.md` section
"A1. Readout energy and product of experts" end to end: precondition checks,
verifier-only Gibbs training, degenerate-case checks, the synthetic positive
control, the real single-pass readout over the FoVer corpus, grouped
out-of-fold cross-validation, paired group bootstrap intervals, the headroom
check, and the pre-registered pass/fail gate.

CONCRETE STEPS:
  0. PRECONDITIONS (checked before any subsequent step; any failure writes
     honest_verdict `blocked_<resource>` and exits -- no fabrication):
     a. `data/fover_corpus_v4.json` exists and parses.
     b. The mandated GGUF (`unsloth/Qwen3.8-27B-GGUF`) is cached locally.
     c. `llama_cpp` was built with CUDA support
        (`llama_supports_gpu_offload() is True`).
  1. Print a progress line, then load the corpus rows + PCIB features + the
     `calibrated_decision_benchmark` train/held-out split.
  2. Print a progress line, then train the verifier-only Gibbs energy on the
     train split via NCE; refuse to proceed if it is degenerate.
  3. Print a progress line, then run the three degenerate-case checks on
     synthetic fixtures (real production code paths, not fake test-only
     behaviour).
  4. Print a progress line, then run the synthetic positive control (no
     model calls, never mixed with the real fit).
  5. Print a progress line, then load the GGUF model on GPU 1 and resolve
     the option-letter token ids.
  6. Print a progress line before and after; read the option logits for
     every corpus row (cached to disk so a rerun costs zero further model
     calls); print a progress line every 250 rows inside the loop.
  7. Print a progress line, then compute the verifier-only energy for every
     row and assemble the held-out `ScoredRow` list, excluding any row whose
     readout degenerated to `escalate`.
  8. Print a progress line, then run grouped out-of-fold cross-validation
     and the paired group bootstrap.
  9. Print a progress line, then compute the headroom check and the
     pre-registered pass/fail gate.
 10. Print a progress line, then write the results artifact.

Spec: REQ-VERIFY-7750
"""

from __future__ import annotations

import os

# CUDA_VISIBLE_DEVICES must be set before llama_cpp is imported -- the
# installed build enumerates CUDA devices at import time (ggml_cuda_init),
# not at Llama() construction time. GPU 1 is the outer loop's dedicated
# card; GPU 0 is reserved for the live conductor's own generator (CLAUDE.md
# "ARC-AGI-3 Submission Sprint Forcing Function" GPU-allocation note).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault(
    "JAX_PLATFORMS", "cpu"
)  # this run's JAX use (Gibbs NCE) is CPU-only by design

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.inference.sota_models import cached_current_model  # noqa: E402
from carnot.paths import repo_path, results_path  # noqa: E402
from carnot.verify import semif_readout_energy as sre  # noqa: E402

RANDOM_SEED = 7750
KFOLD_K = 5
N_BOOT = 2000
PROGRESS_EVERY_N_ROWS = 250
CORPUS_PATH = repo_path("data", "fover_corpus_v4.json")
RESULT_PATH = results_path("experiment_semif_readout_ebm_eval_a1.json")
CACHE_PATH = repo_path("results", "raw", "semif_readout_ebm_eval", "readout_logits_cache.json")
INFERENCE_SUBSTRATE = "live_llm_single_token_logit_readout"
# `inference_substrate_class` (REQ-SUBSTRATE-CLASS-1, mandatory for any artifact
# dated on or after the 2026-09-07 cutover): the closed-enum vocabulary
# `adversarial_verify.py` actually gates on. `inference_substrate` above is
# free-text prose; this is the machine-checked claim. `model_load_no_generation`
# is the exact match: a real GGUF load, one forward pass per row, no
# autoregressive decoding -- the same class `live_llm_embedding_extraction`
# artifacts declare, just reading output logits instead of hidden states.
INFERENCE_SUBSTRATE_CLASS = "model_load_no_generation"
EXECUTION_VENUE = "host"
INFERENCE_SUBSTRATE_NOTE = (
    "Not yet a registered value in CLAUDE.md's Inference-Substrate Declaration "
    "table. Closest registered analog is `live_llm_embedding_extraction` (a "
    "real GGUF load, one prefill pass per row, no autoregressive decoding) -- "
    "this substrate differs only in WHAT is read from that one pass (the "
    "final-position output logits for three declared option tokens, not a "
    "hidden-state embedding vector). Proposed honestly rather than "
    "mis-declared as either `live_llm_inference` (implies full generation, "
    "which this never does) or `verifier_ensemble_against_cached_candidates` "
    "(implies no model load at all, which is false here)."
)


def _progress(message: str) -> None:
    elapsed = time.monotonic() - _START
    print(f"[semif_readout_ebm_eval] t+{elapsed:6.1f}s  {message}", flush=True)


_START = time.monotonic()


def _honest_block(reason: str, preconditions_checked: list[dict]) -> int:
    artifact = {
        "schema": "carnot.semif_readout_ebm_eval.a1.v1",
        "experiment": "semif_readout_ebm_eval_a1",
        "requirement": "REQ-VERIFY-7750",
        "run_date": datetime.now(UTC).strftime("%Y%m%d"),
        "honest_verdict": f"complete: blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "preconditions_checked": preconditions_checked,
        "duration_s": time.monotonic() - _START,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _reproducibility_checksum(),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"BLOCKED: {reason}. Wrote {RESULT_PATH}", flush=True)
    return 0


def _reproducibility_checksum() -> str:
    hasher = hashlib.sha256()
    if CORPUS_PATH.exists():
        hasher.update(CORPUS_PATH.read_bytes())
    hasher.update(str(RANDOM_SEED).encode())
    hasher.update(str(sre.DEFAULT_WEIGHT_GRID).encode())
    hasher.update(str(KFOLD_K).encode())
    return "sha256:" + hasher.hexdigest()


def _check_preconditions() -> tuple[list[dict], dict | None, bool]:
    """Returns (preconditions_checked, model_spec_or_None, llama_cpp_gpu_ok)."""
    checked: list[dict] = []

    corpus_ok = CORPUS_PATH.exists()
    checked.append({"resource": "fover_corpus_v4_json", "available": corpus_ok})

    model_spec = cached_current_model(gpu_index=1) if corpus_ok else None
    checked.append({"resource": "qwen3_8_27b_gguf_cached", "available": model_spec is not None})

    llama_cpp_gpu_ok = False
    try:
        from llama_cpp import llama_cpp as _llama_cpp_binding

        llama_cpp_gpu_ok = bool(_llama_cpp_binding.llama_supports_gpu_offload())
    except Exception:  # noqa: BLE001 -- optional live dependency; absence is a precondition failure
        llama_cpp_gpu_ok = False
    checked.append({"resource": "llama_cpp_cuda_gpu_offload", "available": llama_cpp_gpu_ok})

    return checked, model_spec, llama_cpp_gpu_ok


def _synthetic_degenerate_checks() -> dict:
    """SCENARIO-VERIFY-7750-DEGENERATE, all three cases, exercised as real
    calls into the production functions (not asserted only by unit tests)."""
    rng = np.random.default_rng(RANDOM_SEED)

    # Case (a): a readout that returns the identical energy for every row.
    e_verifier_real_shaped = rng.normal(size=500)
    case_a_corr = sre.check_uniform_readout_preserves_verifier_ranking(e_verifier_real_shaped)

    # Case (b): a degenerate (all-equal) verifier energy must be rejected
    # before any PoE fit is attempted.
    n = 200
    e_readout_ok = rng.normal(size=n)
    e_verifier_degenerate = np.zeros(n)
    labels = rng.integers(0, 2, size=n)
    case_b_fit = sre.fit_poe_with_guards(e_readout_ok, e_verifier_degenerate, labels)

    # Case (c): a missing or non-finite option logit forces `escalate`.
    case_c_missing = sre.readout_from_logits({"accept": 1.0, "reject": None, "escalate": 0.0})
    case_c_nan = sre.readout_from_logits({"accept": float("nan"), "reject": 0.0, "escalate": 0.0})

    return {
        "case_a_uniform_readout_spearman_corr_vs_verifier": case_a_corr,
        "case_a_passes": bool(abs(case_a_corr - 1.0) < 1e-9),
        "case_b_degenerate_verifier_fit_result": case_b_fit,
        "case_b_passes": case_b_fit is None,
        "case_c_missing_logit_decision": case_c_missing.decision,
        "case_c_nan_logit_decision": case_c_nan.decision,
        "case_c_passes": case_c_missing.decision == "escalate"
        and case_c_nan.decision == "escalate",
    }


def _headroom_check() -> dict:
    """SCENARIO-VERIFY-7750-HEADROOM: this corpus has no multi-candidate
    selection structure (one row per decision, not N candidates per
    question), and a prior static-decision experiment already found
    near-universal accept behaviour with 22 conflicting cells among 3,950
    rows (`ops/verifier_gaps.md` GAP-DECISION-PCIB-OBSERVABLES-7396). So a
    top-one SELECTION-lift claim from this corpus alone would repeat exactly
    the failure mode CLAUDE.md's Adversarial Artifact Verification rule
    calls FALSE_NEGATIVE_RISK. We report the honest headroom numbers that
    DO make sense on this corpus (an always-accept baseline versus a
    perfect oracle) and mark the selection-lift claim not-applicable rather
    than force-fitting a selection experiment this corpus was never built
    to support.
    """
    rows = sre.load_corpus_rows_with_features()
    n = len(rows)
    n_incorrect = sum(1 for r in rows if r.label == 1)
    always_accept_accuracy = 1.0 - (n_incorrect / n) if n else 0.0
    return {
        "applicable_as_a_selection_experiment": False,
        "reason": (
            "data/fover_corpus_v4.json has one row per decision, not multiple "
            "candidates per question -- there is no top-k pool to select "
            "over. Per the plan's own headroom rule, a saturated single-row "
            "corpus cannot support a selection-lift claim; see "
            "ops/verifier_gaps.md GAP-DECISION-PCIB-OBSERVABLES-7396 (22 "
            "conflicting cells among 3,950 rows, near-universal accept)."
        ),
        "always_accept_baseline_accuracy": always_accept_accuracy,
        "oracle_accuracy": 1.0,
        "oracle_minus_incumbent_accuracy_gap": 1.0 - always_accept_accuracy,
        "selection_claim_status": "FALSE_NEGATIVE_RISK_not_applicable",
    }


def main() -> int:
    _progress("step 0: checking preconditions")
    preconditions_checked, model_spec, llama_cpp_gpu_ok = _check_preconditions()
    if not preconditions_checked[0]["available"]:
        return _honest_block("corpus_missing", preconditions_checked)
    if model_spec is None:
        return _honest_block("model_not_cached_qwen3_8_27b", preconditions_checked)
    if not llama_cpp_gpu_ok:
        return _honest_block("llama_cpp_no_gpu_offload", preconditions_checked)
    _progress(f"preconditions OK; model_path={model_spec['model_path']}")

    _progress("step 1: loading corpus rows + PCIB features + train/held-out split")
    rows = sre.load_corpus_rows_with_features()
    train_rows = [r for r in rows if r.split == "train"]
    held_out_rows = [r for r in rows if r.split == "held_out"]
    _progress(f"loaded {len(rows)} rows ({len(train_rows)} train / {len(held_out_rows)} held-out)")

    _progress("step 2: training the verifier-only Gibbs energy via NCE")
    train_correct = [[r.entity_uptake, r.falsifiability] for r in train_rows if r.label == 0]
    train_incorrect = [[r.entity_uptake, r.falsifiability] for r in train_rows if r.label == 1]
    verifier_model = sre.train_gibbs_verifier(train_correct, train_incorrect, seed=RANDOM_SEED)
    held_out_features = [[r.entity_uptake, r.falsifiability] for r in held_out_rows]
    verifier_energies_held_out = sre.gibbs_energy_batch(verifier_model, held_out_features)
    verifier_degenerate = sre.is_degenerate_energy(verifier_energies_held_out)
    _progress(f"verifier trained; held-out energy degenerate={verifier_degenerate}")

    _progress("step 3: running the three degenerate-case checks")
    degenerate_checks = _synthetic_degenerate_checks()
    _progress(
        f"degenerate checks: a={degenerate_checks['case_a_passes']} "
        f"b={degenerate_checks['case_b_passes']} c={degenerate_checks['case_c_passes']}"
    )

    _progress("step 4: running the synthetic positive control")
    positive_control = sre.run_positive_control(seed=RANDOM_SEED)
    _progress(f"positive control passed={positive_control['passed']}")

    if verifier_degenerate:
        preconditions_checked.append(
            {"resource": "verifier_only_energy_non_degenerate", "available": False}
        )
        return _honest_block("verifier_energy_degenerate", preconditions_checked)

    _progress("step 5: loading the GGUF model on GPU 1 and resolving option token ids")
    from llama_cpp import Llama

    # logits_all=True is REQUIRED for `read_option_logits` to see real
    # per-token logits at all -- measured directly in this experiment
    # (2026-09-28): without it, the installed llama-cpp-python's `eval()`
    # never fills `scores` (see `read_option_logits`'s docstring for detail
    # and the exact evidence). n_ctx=2048 gives headroom over the corpus's
    # longest step_text (2,289 characters, roughly 600-700 tokens) plus the
    # prompt's own instruction text.
    llama = Llama(
        model_path=model_spec["model_path"],
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
        seed=RANDOM_SEED,
        logits_all=True,
    )
    option_token_ids = sre.resolve_option_token_ids(llama)
    if option_token_ids is None:
        preconditions_checked.append(
            {"resource": "option_letters_single_token", "available": False}
        )
        return _honest_block("option_labels_not_single_token", preconditions_checked)
    _progress(f"option token ids resolved: {option_token_ids}")

    _progress(f"step 6: reading option logits for {len(rows)} rows (cache={CACHE_PATH})")
    cache = sre.load_readout_cache(CACHE_PATH)
    readout_by_key: dict[str, sre.ReadoutResult] = {}
    cache_hits = 0
    cache_misses = 0
    for i, row in enumerate(rows):
        key = sre.readout_cache_key(row.question_id, row.step_text)
        cached_logits = cache.get(key)
        if cached_logits is not None:
            cache_hits += 1
        else:
            prompt = sre.build_readout_prompt(row.step_text)
            cached_logits = sre.read_option_logits(llama, prompt, option_token_ids)
            cache[key] = cached_logits
            cache_misses += 1
        readout_by_key[key] = sre.readout_from_logits(cached_logits)
        if (i + 1) % PROGRESS_EVERY_N_ROWS == 0:
            sre.save_readout_cache(CACHE_PATH, cache)
            _progress(
                f"  readout {i + 1}/{len(rows)} rows (cache hits={cache_hits} misses={cache_misses})"
            )
    sre.save_readout_cache(CACHE_PATH, cache)
    _progress(f"readout complete: {cache_hits} cache hits, {cache_misses} new model calls")

    escalate_count = sum(1 for r in readout_by_key.values() if r.decision == "escalate")
    _progress(
        f"step 7: assembling held-out ScoredRow list ({escalate_count} rows forced escalate, excluded)"
    )

    scored_rows: list[sre.ScoredRow] = []
    verifier_energy_by_qid: dict[str, float] = {}
    for row, energy in zip(held_out_rows, verifier_energies_held_out):
        verifier_energy_by_qid[row.question_id] = float(energy)
    for row in held_out_rows:
        key = sre.readout_cache_key(row.question_id, row.step_text)
        readout = readout_by_key[key]
        if readout.energy_accept is None:
            continue  # degenerate readout row -- excluded, per SCENARIO-VERIFY-7750-DEGENERATE case (c)
        scored_rows.append(
            sre.ScoredRow(
                question_id=row.question_id,
                label=row.label,
                e_readout=readout.energy_accept,
                e_verifier=verifier_energy_by_qid[row.question_id],
            )
        )
    _progress(
        f"assembled {len(scored_rows)} of {len(held_out_rows)} held-out rows for the real fit"
    )

    # Defense-in-depth: SCENARIO-VERIFY-7750-DEGENERATE case (a) is proven on
    # a synthetic fixture in step 3, but a real readout degrading to a
    # constant (e.g. a future llama.cpp regression reintroducing the
    # scores[-1] bug this experiment found) must not silently reach the fit.
    real_readout_energies = np.asarray([r.e_readout for r in scored_rows], dtype=np.float64)
    if sre.is_degenerate_energy(real_readout_energies):
        preconditions_checked.append(
            {"resource": "readout_energy_non_degenerate", "available": False}
        )
        return _honest_block("readout_energy_degenerate", preconditions_checked)

    _progress("step 8: running grouped out-of-fold cross-validation + paired group bootstrap")
    oof_result = sre.run_oof_cross_validation(scored_rows, k=KFOLD_K)
    bootstrap = sre.paired_group_bootstrap_brier_delta(
        oof_result["pooled"], n_boot=N_BOOT, seed=RANDOM_SEED
    )
    _progress("cross-validation and bootstrap complete")

    _progress("step 9: headroom check and pass/fail gate")
    headroom = _headroom_check()

    ok_folds = [f for f in oof_result["per_fold"] if f["status"] == "ok"]
    ece_delta_ok = bootstrap["ece_delta_vs_better_single_expert"]["point"] <= 0.01
    brier_gate_ok = (
        bootstrap["brier_delta_vs_readout_only"]["ci95"][1] < 0.0
        and bootstrap["brier_delta_vs_verifier_only"]["ci95"][1] < 0.0
    )
    gate_passed = bool(
        brier_gate_ok
        and ece_delta_ok
        and positive_control["passed"]
        and degenerate_checks["case_a_passes"]
        and degenerate_checks["case_b_passes"]
        and degenerate_checks["case_c_passes"]
        and len(ok_folds) > 0
    )

    if gate_passed:
        verdict_text = "a1_readout_poe_passed_pre_registered_gate"
    else:
        verdict_text = "a1_readout_poe_failed_pre_registered_gate"
    honest_verdict = f"complete: {verdict_text}"

    _progress(f"step 10: writing results artifact ({honest_verdict})")

    duration_s = time.monotonic() - _START
    artifact = {
        "schema": "carnot.semif_readout_ebm_eval.a1.v1",
        "experiment": "semif_readout_ebm_eval_a1",
        "requirement": "REQ-VERIFY-7750",
        "run_date": datetime.now(UTC).strftime("%Y%m%d"),
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "inference_substrate_note": INFERENCE_SUBSTRATE_NOTE,
        "model_specs": [model_spec],
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _reproducibility_checksum(),
        "duration_s": duration_s,
        "preconditions_checked": preconditions_checked,
        "field_provenance": {
            "duration_s": {
                "principle": "Real compute takes wall-clock time; a fabricated result would show an implausibly short duration for 6,548 real forward passes.",
                "satisfied_by": "wall_clock measurement across the full driver run, including model load and every row's forward pass",
            },
            "random_seed": {
                "principle": "Determinism is the precondition for reproducibility; a fixed seed lets a third party re-run the Gibbs training, positive control, and bootstrap and get the same numbers.",
                "satisfied_by": f"random_seed={RANDOM_SEED} threaded through Gibbs init, the positive control, and the bootstrap",
            },
            "reproducibility_checksum": {
                "principle": "A content-addressed hash of the corpus file and key config catches silent corpus or config drift between this artifact and any future replication attempt.",
                "satisfied_by": "sha256 over the corpus bytes, the random seed, the weight grid, and k",
            },
        },
        "corpus": {
            "path": "data/fover_corpus_v4.json",
            "n_rows_total": len(rows),
            "n_train": len(train_rows),
            "n_held_out": len(held_out_rows),
            "n_scored_for_real_fit": len(scored_rows),
            "n_readout_forced_escalate_and_excluded": escalate_count,
            "readout_cache_hits": cache_hits,
            "readout_cache_misses": cache_misses,
        },
        "degenerate_case_checks": degenerate_checks,
        "positive_control": positive_control,
        "oof_cross_validation": {
            "k": KFOLD_K,
            "per_fold": oof_result["per_fold"],
            "n_ok_folds": len(ok_folds),
        },
        "paired_group_bootstrap": bootstrap,
        "headroom_check": headroom,
        "acceptance_gates": {
            "poe_brier_delta_vs_readout_only_ci95_upper_below_zero": {
                "value": bootstrap["brier_delta_vs_readout_only"]["ci95"][1],
                "passed": bootstrap["brier_delta_vs_readout_only"]["ci95"][1] < 0.0,
                "principle": "A PoE win over the readout alone must survive a 95 percent paired group bootstrap, not just a point estimate, before it counts as a real calibration gain.",
            },
            "poe_brier_delta_vs_verifier_only_ci95_upper_below_zero": {
                "value": bootstrap["brier_delta_vs_verifier_only"]["ci95"][1],
                "passed": bootstrap["brier_delta_vs_verifier_only"]["ci95"][1] < 0.0,
                "principle": "Same bound against the verifier-only expert -- the PoE must beat BOTH single experts, not just the weaker one.",
            },
            "ece_does_not_worsen_by_more_than_0_01": {
                "value": bootstrap["ece_delta_vs_better_single_expert"]["point"],
                "passed": ece_delta_ok,
                "principle": "A calibration win in Brier that comes with a large ECE regression is not a usable calibration improvement.",
            },
            "positive_control_passed": {
                "value": positive_control["passed"],
                "passed": positive_control["passed"],
                "principle": "If the PoE cannot even detect a noisy copy of the gold label, its fitting machinery cannot be trusted on the real, much weaker signal.",
            },
            "degenerate_case_a_passed": degenerate_checks["case_a_passes"],
            "degenerate_case_b_passed": degenerate_checks["case_b_passes"],
            "degenerate_case_c_passed": degenerate_checks["case_c_passes"],
        },
        "gate_passed": gate_passed,
        "methodology_note": (
            "Verifier-only energy is a real 2-4-1 GibbsModel trained via NCE "
            "on the fixed calibrated_decision_benchmark train split (30 "
            "percent of question groups). Readout energy is a genuine single "
            "forward pass per row over unsloth/Qwen3.8-27B-GGUF on GPU 1, "
            "reading the three declared option logits -- no text is "
            "generated. alpha and beta are fit only on the k-1 training "
            "folds of each out-of-fold split, never on the evaluation fold."
        ),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    _progress(f"wrote {RESULT_PATH} (gate_passed={gate_passed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
