#!/usr/bin/env python3
"""A1/A2/A3 driver: readout energy calibration + policy
(REQ-VERIFY-7750, REQ-VERIFY-7751, REQ-VERIFY-7752).

Runs the pre-registered A1, A2, and A3 experiments from
`docs/research-notes/semif-ebm-arc-experiment-plan-2026-09-20.md`. Run A1
with no argument (`python scripts/experiments/semif_readout_ebm_eval.py`).
Run A2 with `a2` as the first argument
(`python scripts/experiments/semif_readout_ebm_eval.py a2`). Run A3 with
`a3` as the first argument
(`python scripts/experiments/semif_readout_ebm_eval.py a3`). A2 and A3 make
NO model call -- both reuse A1's cached readout logits, so either costs
nothing to rerun.

CONCRETE STEPS (A1, `main()`):
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

CONCRETE STEPS (A2, `main_a2()`):
  0. PRECONDITIONS: the corpus file and A1's readout logit cache both
     exist and parse. No GGUF/GPU precondition -- A2 never loads a model.
  1. Print a progress line, then load the corpus rows + split, and build
     `A2ScoredRow` rows from the cached logits for the held-out split,
     excluding any row whose cached logits are missing or degenerate.
  2. Print a progress line, then run the calibrator degenerate-case checks
     on synthetic fixtures (real production code paths).
  3. Print a progress line, then run the two synthetic positive-control
     lanes (temperature recovery, Platt affine-distortion recovery).
  4. Print a progress line, then run the grouped out-of-fold calibrator
     tournament for every option channel.
  5. Print a progress line, then run the paired group bootstrap for every
     channel's calibrator deltas.
  6. Print a progress line, then apply the pass/fail gate and kill
     criterion per channel.
  7. Print a progress line, then write the results artifact.

CONCRETE STEPS (A3, `main_a3()`):
  0. PRECONDITIONS: the corpus file, A1's result artifact, A1's readout
     logit cache, and A2's result artifact all exist and parse. No
     GGUF/GPU precondition -- A3 never loads a model.
  1. Print a progress line, then load the corpus rows + split, and rebuild
     the held-out `A2ScoredRow` list (A2's exact construction, reused so
     the two never drift).
  2. Print a progress line, then retrain the verifier-only Gibbs energy
     (A1's exact construction) and run A1's own out-of-fold
     cross-validation to get the pooled out-of-fold verifier-only
     probability per held-out row.
  3. Print a progress line, then rerun A2's grouped out-of-fold calibrator
     tournament over all three option channels, to get pooled out-of-fold
     raw and isotonic-calibrated probabilities per held-out row (the
     `accept` channel's calibration is read but never used in the
     combined-risk score -- it did not clear A2's kill criterion).
  4. Print a progress line, then join the two pooled sets by question ID
     (unique per row in this corpus) and enforce the sample-size floor
     (>=1,000 rows, >=30 question groups).
  5. Print a progress line, then run the synthetic A3 positive control (two
     lanes: noisy-gold vs. entropy control, and noisy-gold vs. prevalence
     control).
  6. Print a progress line, then run the full A3 policy evaluation --
     combined risk, controls, AURC/coverage metrics, paired group
     bootstraps, and the pre-registered cost-grid sweep -- once per each of
     five fixed seeds.
  7. Print a progress line, then apply the pass/fail gate and kill
     criterion using the seed with the LEAST favorable bootstrap interval
     (the most conservative choice across the five seeds).
  8. Print a progress line, then write the results artifact.

Spec: REQ-VERIFY-7750, REQ-VERIFY-7751, REQ-VERIFY-7752
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
import math  # noqa: E402
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


RESULT_PATH_A2 = results_path("experiment_semif_readout_ebm_eval_a2.json")
INFERENCE_SUBSTRATE_A2 = "aggregation_from_upstream_artifacts"
# `inference_substrate_class` (REQ-SUBSTRATE-CLASS-1, mandatory for any artifact
# dated on or after the 2026-09-07 cutover -- see A1's driver for the same
# note): the closed-enum vocabulary `adversarial_verify.py` gates on.
# "aggregation" is the exact match here: CPU-only calibrator fitting and
# bootstrap over A1's cached logits, no model load at all.
INFERENCE_SUBSTRATE_CLASS_A2 = "aggregation"
RANDOM_SEED_A2 = 7751


def _reproducibility_checksum_a2(cache_path: Path) -> str:
    hasher = hashlib.sha256()
    if CORPUS_PATH.exists():
        hasher.update(CORPUS_PATH.read_bytes())
    if cache_path.exists():
        hasher.update(cache_path.read_bytes())
    hasher.update(str(RANDOM_SEED_A2).encode())
    hasher.update(str(KFOLD_K).encode())
    return "sha256:" + hasher.hexdigest()


def _honest_block_a2(reason: str, preconditions_checked: list[dict], start: float) -> int:
    artifact = {
        "schema": "carnot.semif_readout_ebm_eval.a2.v1",
        "experiment": "semif_readout_ebm_eval_a2",
        "requirement": "REQ-VERIFY-7751",
        "run_date": datetime.now(UTC).strftime("%Y%m%d"),
        "honest_verdict": f"complete: blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE_A2,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS_A2,
        "preconditions_checked": preconditions_checked,
        "duration_s": time.monotonic() - start,
        "random_seed": RANDOM_SEED_A2,
        "reproducibility_checksum": _reproducibility_checksum_a2(CACHE_PATH),
    }
    RESULT_PATH_A2.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH_A2.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"BLOCKED: {reason}. Wrote {RESULT_PATH_A2}", flush=True)
    return 0


def main_a2() -> int:  # noqa: C901 -- one linear driver, matches A1's main() shape
    start = time.monotonic()

    def _p(message: str) -> None:
        elapsed = time.monotonic() - start
        print(f"[semif_readout_ebm_eval:a2] t+{elapsed:6.1f}s  {message}", flush=True)

    _p("step 0: checking preconditions (no model/GPU precondition -- A2 makes no model call)")
    preconditions_checked = [
        {"resource": "fover_corpus_v4_json", "available": CORPUS_PATH.exists()},
        {"resource": "a1_readout_logits_cache", "available": CACHE_PATH.exists()},
    ]
    if not preconditions_checked[0]["available"]:
        return _honest_block_a2("corpus_missing", preconditions_checked, start)
    if not preconditions_checked[1]["available"]:
        return _honest_block_a2("a1_readout_cache_missing", preconditions_checked, start)
    cache = sre.load_readout_cache(CACHE_PATH)
    preconditions_checked.append(
        {"resource": "a1_readout_cache_non_empty", "available": bool(cache)}
    )
    if not cache:
        return _honest_block_a2("a1_readout_cache_empty", preconditions_checked, start)
    _p(f"preconditions OK; cache has {len(cache)} cached rows")

    assert "llama_cpp" not in sys.modules, "A2 must never load a model -- llama_cpp was imported"

    _p("step 1: loading corpus rows + split, building A2ScoredRow list from the cache")
    rows = sre.load_corpus_rows_with_features()
    held_out_rows = [r for r in rows if r.split == "held_out"]
    scored_rows: list[sre.A2ScoredRow] = []
    n_missing_cache_entry = 0
    n_degenerate_logits = 0
    for row in held_out_rows:
        key = sre.readout_cache_key(row.question_id, row.step_text)
        raw_logits = cache.get(key)
        if raw_logits is None:
            n_missing_cache_entry += 1
            continue
        finite = all(
            raw_logits.get(opt) is not None and math.isfinite(float(raw_logits.get(opt)))
            for opt in sre.OPTIONS
        )
        if not finite:
            n_degenerate_logits += 1
            continue
        scored_rows.append(
            sre.A2ScoredRow(
                question_id=row.question_id,
                label=row.label,
                raw_logits={opt: float(raw_logits[opt]) for opt in sre.OPTIONS},
            )
        )
    _p(
        f"assembled {len(scored_rows)} of {len(held_out_rows)} held-out rows "
        f"(missing_cache={n_missing_cache_entry}, degenerate_logits={n_degenerate_logits})"
    )
    if len(scored_rows) < 1000:
        preconditions_checked.append({"resource": "at_least_1000_oof_rows", "available": False})
        return _honest_block_a2("insufficient_scored_rows", preconditions_checked, start)
    n_groups = len({r.question_id for r in scored_rows})
    if n_groups < 30:
        preconditions_checked.append(
            {"resource": "at_least_30_question_groups", "available": False}
        )
        return _honest_block_a2("insufficient_question_groups", preconditions_checked, start)

    _p("step 2: running the calibrator degenerate-case checks")
    degenerate_checks = {
        "constant_probability": sre.check_constant_probability_calibration_is_a_no_op(),
        "isotonic_no_false_rank_improvement": sre.check_isotonic_no_false_rank_improvement(),
        "hard_failures_raised": sre.check_hard_failures_are_raised(),
    }
    _p(
        "degenerate checks: constant="
        f"{degenerate_checks['constant_probability']['passes']} "
        f"isotonic_order={degenerate_checks['isotonic_no_false_rank_improvement']['passes']} "
        f"hard_failures={degenerate_checks['hard_failures_raised']['passes']}"
    )

    _p("step 3: running the two synthetic positive-control lanes")
    positive_controls = {
        "temperature_recovery": sre.run_temperature_recovery_positive_control(seed=RANDOM_SEED_A2),
        "platt_affine_recovery": sre.run_platt_affine_recovery_positive_control(
            seed=RANDOM_SEED_A2 + 1
        ),
    }
    _p(
        f"positive controls: temperature={positive_controls['temperature_recovery']['passed']} "
        f"platt_affine={positive_controls['platt_affine_recovery']['passed']}"
    )

    _p(
        f"step 4: running the grouped out-of-fold calibrator tournament ({len(sre.OPTIONS)} channels)"
    )
    tournament = sre.run_calibrator_tournament(scored_rows, k=KFOLD_K)
    for channel in sre.OPTIONS:
        n_ok = sum(1 for f in tournament[channel]["per_fold"] if f["status"] == "ok")
        _p(f"  channel={channel}: {n_ok}/{KFOLD_K} folds ok")

    _p("step 5: running the paired group bootstrap per channel")
    bootstraps: dict[str, dict] = {}
    for i, channel in enumerate(sre.OPTIONS):
        bootstraps[channel] = sre.paired_group_bootstrap_calibrator_deltas(
            tournament[channel]["pooled"], n_boot=N_BOOT, seed=RANDOM_SEED_A2 + 10 * (i + 1)
        )
    _p("bootstrap complete")

    _p("step 6: applying the pass/fail gate and kill criterion per channel")
    gates: dict[str, dict] = {}
    kills: dict[str, dict] = {}
    for channel in sre.OPTIONS:
        gates[channel] = sre.select_calibrator(bootstraps[channel])
        pooled_labels = tournament[channel]["pooled"]["label"]
        gates_with_kill = sre.calibrator_kill_check(
            bootstraps[channel], pooled_labels, tournament[channel]["per_fold"]
        )
        kills[channel] = gates_with_kill
        _p(f"  channel={channel}: gate={gates[channel]['verdict']} kill={kills[channel]['kill']}")

    any_channel_selected = any(g["selected"] is not None for g in gates.values())
    isotonic_omitted_summary = {
        channel: tournament[channel]["isotonic_omitted"] for channel in sre.OPTIONS
    }
    any_isotonic_omitted = any(bool(v) for v in isotonic_omitted_summary.values())

    # A channel that clears the naive gate (beats the RAW probability) but
    # is also killed (no better than the trivial prevalence baseline) is
    # NOT a genuine win -- the raw baseline can be an unfair comparison
    # point on its own (e.g. the `accept` channel's raw probability is a
    # poor P(incorrect) estimate BY CONSTRUCTION, so almost anything beats
    # it). `usable_channels` is the honest, stricter set: selected AND not
    # flagged by the kill criterion.
    selected_channels = [c for c, g in gates.items() if g["selected"] is not None]
    usable_channels = [c for c in selected_channels if not kills[c]["kill"]]
    gate_selected_but_killed_channels = [c for c in selected_channels if kills[c]["kill"]]

    if usable_channels:
        verdict_text = f"a2_calibrator_tournament_usable_on_{'_'.join(usable_channels)}"
    elif any_channel_selected:
        verdict_text = (
            "a2_calibrator_tournament_gate_selected_but_all_killed_no_better_than_prevalence"
        )
    else:
        verdict_text = "a2_calibrator_tournament_no_channel_cleared_gate_diagnostic_only"
    honest_verdict = f"complete: {verdict_text}"

    _p(f"step 7: writing results artifact ({honest_verdict})")

    # Option count and source family: this corpus has one fixed 3-option
    # decision per row and no field distinguishing sub-sources. Report that
    # honestly (a single bucket) rather than fabricating a category the
    # data does not support (the plan's "report by option count and source
    # family" instruction, applied to what this corpus actually has).
    breakdown_by_option_count = {
        "3": {
            "n_rows": len(scored_rows),
            "note": "every row declares exactly 3 options (accept/reject/escalate) -- there is no varying option count in this corpus",
        }
    }
    breakdown_by_source_family = {
        "fover_corpus_v4_single_family": {
            "n_rows": len(scored_rows),
            "note": "data/fover_corpus_v4.json carries no field distinguishing sub-sources; treated as one family",
        }
    }

    duration_s = time.monotonic() - start
    upstream_a1_path = RESULT_PATH
    cited_upstream_artifacts = [
        {
            "experiment_id": "semif_readout_ebm_eval_a1",
            "fields_imported": ["readout_logits_cache (all 6548 rows' raw per-option logits)"],
            "sha256": (
                "sha256:" + hashlib.sha256(upstream_a1_path.read_bytes()).hexdigest()
                if upstream_a1_path.exists()
                else None
            ),
            "cache_sha256": (
                "sha256:" + hashlib.sha256(CACHE_PATH.read_bytes()).hexdigest()
                if CACHE_PATH.exists()
                else None
            ),
        }
    ]

    artifact = {
        "schema": "carnot.semif_readout_ebm_eval.a2.v1",
        "experiment": "semif_readout_ebm_eval_a2",
        "requirement": "REQ-VERIFY-7751",
        "run_date": datetime.now(UTC).strftime("%Y%m%d"),
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE_A2,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS_A2,
        "inference_substrate_note": (
            "CPU-only calibrator fitting and grouped bootstrap over A1's cached readout "
            "logits. No model is loaded and no forward pass is made -- confirmed by "
            "asserting `llama_cpp` is never imported in this code path."
        ),
        "model_specs": [],
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED_A2,
        "reproducibility_checksum": _reproducibility_checksum_a2(CACHE_PATH),
        "duration_s": duration_s,
        "preconditions_checked": preconditions_checked,
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "field_provenance": {
            "duration_s": {
                "principle": "Even a CPU-only aggregation should report a real wall-clock duration so a reader can distinguish a genuine run from a stub.",
                "satisfied_by": "wall_clock measurement across the full driver run",
            },
            "random_seed": {
                "principle": "Determinism lets a third party re-run the tournament, bootstrap, and positive controls and get the same numbers.",
                "satisfied_by": f"random_seed={RANDOM_SEED_A2} threaded through the tournament, bootstrap, and positive controls",
            },
            "reproducibility_checksum": {
                "principle": "A content-addressed hash of the corpus and cache files catches silent drift between this artifact and a future rerun.",
                "satisfied_by": "sha256 over the corpus bytes, the cache bytes, the seed, and k",
            },
        },
        "corpus": {
            "path": "data/fover_corpus_v4.json",
            "n_held_out_rows": len(held_out_rows),
            "n_scored_rows": len(scored_rows),
            "n_missing_cache_entry": n_missing_cache_entry,
            "n_degenerate_logits_excluded": n_degenerate_logits,
            "n_question_groups": n_groups,
        },
        "degenerate_case_checks": degenerate_checks,
        "positive_controls": positive_controls,
        "tournament": {
            channel: {
                "per_fold": tournament[channel]["per_fold"],
                "isotonic_omitted": tournament[channel]["isotonic_omitted"],
                "reliability_table_raw": tournament[channel]["reliability_table_raw"],
            }
            for channel in sre.OPTIONS
        },
        "bootstrap": bootstraps,
        "gates": gates,
        "kill_checks": kills,
        "any_channel_selected": any_channel_selected,
        "selected_channels": selected_channels,
        "usable_channels": usable_channels,
        "gate_selected_but_killed_channels": gate_selected_but_killed_channels,
        "any_isotonic_omitted_in_any_fold": any_isotonic_omitted,
        "gate_passed": bool(usable_channels),
        "isotonic_omitted_by_channel": isotonic_omitted_summary,
        "breakdown_by_option_count": breakdown_by_option_count,
        "breakdown_by_source_family": breakdown_by_source_family,
        "methodology_note": (
            "Three calibrators (scalar temperature scaling, true two-parameter Platt "
            "scaling, and sample-size-gated isotonic regression) were fit per option "
            "channel (accept, reject, escalate), independently, on the k-1 training "
            "folds of a grouped out-of-fold cross-validation split over A1's cached "
            "readout logits for the FoVer corpus's held-out rows. No calibrator was "
            "ever fit on its own evaluation fold. Zero new model forward passes were "
            "made; every logit was read from A1's on-disk cache."
        ),
    }
    RESULT_PATH_A2.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH_A2.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    _p(f"wrote {RESULT_PATH_A2} (any_channel_selected={any_channel_selected})")
    return 0


# ==========================================================================
# A3: calibrated accept/reject/escalate policy (REQ-VERIFY-7752).
#
# Honest input set (see semif_readout_energy.py's A3 module docstring for
# the full reasoning): the verifier-only probability (A1's collapsed PoE
# reduces to this), plus A2's calibrated `reject` and `escalate` channels.
# A2's `accept` channel calibration is read (for the entropy control's raw
# 3-way probabilities) but never used as a validated calibration input --
# it did not clear A2's kill criterion.
# ==========================================================================

RESULT_PATH_A3 = results_path("experiment_semif_readout_ebm_eval_a3.json")
INFERENCE_SUBSTRATE_A3 = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS_A3 = "aggregation"
# Five fixed seeds, matching the existing abstention measurement floor's
# BOOTSTRAP_SEEDS convention (risk_coverage_abstention_3718.py:29-40).
RANDOM_SEEDS_A3: tuple[int, ...] = (7752, 7753, 7754, 7755, 7756)
N_BOOT_A3 = 500


def _reproducibility_checksum_a3(cache_path: Path, a1_path: Path, a2_path: Path) -> str:
    hasher = hashlib.sha256()
    if CORPUS_PATH.exists():
        hasher.update(CORPUS_PATH.read_bytes())
    if cache_path.exists():
        hasher.update(cache_path.read_bytes())
    if a1_path.exists():
        hasher.update(a1_path.read_bytes())
    if a2_path.exists():
        hasher.update(a2_path.read_bytes())
    hasher.update(str(RANDOM_SEEDS_A3).encode())
    hasher.update(str(KFOLD_K).encode())
    return "sha256:" + hasher.hexdigest()


def _honest_block_a3(reason: str, preconditions_checked: list[dict], start: float) -> int:
    artifact = {
        "schema": "carnot.semif_readout_ebm_eval.a3.v1",
        "experiment": "semif_readout_ebm_eval_a3",
        "requirement": "REQ-VERIFY-7752",
        "run_date": datetime.now(UTC).strftime("%Y%m%d"),
        "honest_verdict": f"complete: blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE_A3,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS_A3,
        "preconditions_checked": preconditions_checked,
        "duration_s": time.monotonic() - start,
        "random_seeds_used": list(RANDOM_SEEDS_A3),
        "reproducibility_checksum": _reproducibility_checksum_a3(
            CACHE_PATH, RESULT_PATH, RESULT_PATH_A2
        ),
    }
    RESULT_PATH_A3.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH_A3.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"BLOCKED: {reason}. Wrote {RESULT_PATH_A3}", flush=True)
    return 0


def main_a3() -> int:  # noqa: C901 -- one linear driver, matches A1/A2's shape
    start = time.monotonic()

    def _p(message: str) -> None:
        elapsed = time.monotonic() - start
        print(f"[semif_readout_ebm_eval:a3] t+{elapsed:6.1f}s  {message}", flush=True)

    _p("step 0: checking preconditions (no model/GPU precondition -- A3 makes no model call)")
    preconditions_checked = [
        {"resource": "fover_corpus_v4_json", "available": CORPUS_PATH.exists()},
        {"resource": "a1_result_artifact", "available": RESULT_PATH.exists()},
        {"resource": "a1_readout_logits_cache", "available": CACHE_PATH.exists()},
        {"resource": "a2_result_artifact", "available": RESULT_PATH_A2.exists()},
    ]
    if not preconditions_checked[0]["available"]:
        return _honest_block_a3("corpus_missing", preconditions_checked, start)
    if not preconditions_checked[1]["available"]:
        return _honest_block_a3("a1_result_missing", preconditions_checked, start)
    if not preconditions_checked[2]["available"]:
        return _honest_block_a3("a1_readout_cache_missing", preconditions_checked, start)
    if not preconditions_checked[3]["available"]:
        return _honest_block_a3("a2_result_missing", preconditions_checked, start)
    cache = sre.load_readout_cache(CACHE_PATH)
    preconditions_checked.append(
        {"resource": "a1_readout_cache_non_empty", "available": bool(cache)}
    )
    if not cache:
        return _honest_block_a3("a1_readout_cache_empty", preconditions_checked, start)
    _p(f"preconditions OK; cache has {len(cache)} cached rows")

    assert "llama_cpp" not in sys.modules, "A3 must never load a model -- llama_cpp was imported"

    _p("step 1: loading corpus rows + split; rebuilding A2's held-out ScoredRow list")
    rows = sre.load_corpus_rows_with_features()
    train_rows = [r for r in rows if r.split == "train"]
    held_out_rows = [r for r in rows if r.split == "held_out"]

    a2_scored_rows: list[sre.A2ScoredRow] = []
    for row in held_out_rows:
        key = sre.readout_cache_key(row.question_id, row.step_text)
        raw_logits = cache.get(key)
        if raw_logits is None:
            continue
        finite = all(
            raw_logits.get(opt) is not None and math.isfinite(float(raw_logits.get(opt)))
            for opt in sre.OPTIONS
        )
        if not finite:
            continue
        a2_scored_rows.append(
            sre.A2ScoredRow(
                question_id=row.question_id,
                label=row.label,
                raw_logits={opt: float(raw_logits[opt]) for opt in sre.OPTIONS},
            )
        )
    _p(f"rebuilt {len(a2_scored_rows)} of {len(held_out_rows)} held-out rows (A2's exact filter)")

    _p("step 2: retraining the verifier-only Gibbs energy; running A1's out-of-fold CV")
    train_correct = [[r.entity_uptake, r.falsifiability] for r in train_rows if r.label == 0]
    train_incorrect = [[r.entity_uptake, r.falsifiability] for r in train_rows if r.label == 1]
    verifier_model = sre.train_gibbs_verifier(train_correct, train_incorrect, seed=RANDOM_SEED)
    held_out_features = [[r.entity_uptake, r.falsifiability] for r in held_out_rows]
    verifier_energies_held_out = sre.gibbs_energy_batch(verifier_model, held_out_features)
    verifier_energy_by_qid = {
        row.question_id: float(energy)
        for row, energy in zip(held_out_rows, verifier_energies_held_out)
    }

    a1_scored_rows: list[sre.ScoredRow] = []
    for row in a2_scored_rows:
        readout = sre.readout_from_logits(row.raw_logits)
        if readout.energy_accept is None:
            continue  # degenerate readout -- excluded, same rule A1 applies
        a1_scored_rows.append(
            sre.ScoredRow(
                question_id=row.question_id,
                label=row.label,
                e_readout=readout.energy_accept,
                e_verifier=verifier_energy_by_qid[row.question_id],
            )
        )
    oof_result = sre.run_oof_cross_validation(a1_scored_rows, k=KFOLD_K)
    verifier_p_by_qid = dict(
        zip(oof_result["pooled"]["question_id"], oof_result["pooled"]["p_verifier_only"])
    )
    _p(f"out-of-fold verifier-only probability computed for {len(verifier_p_by_qid)} rows")

    _p("step 3: rerunning A2's grouped out-of-fold calibrator tournament (all 3 channels)")
    tournament = sre.run_calibrator_tournament(a2_scored_rows, channels=sre.OPTIONS, k=KFOLD_K)
    accept_by_qid = dict(
        zip(tournament["accept"]["pooled"]["question_id"], tournament["accept"]["pooled"]["raw"])
    )
    reject_raw_by_qid = dict(
        zip(tournament["reject"]["pooled"]["question_id"], tournament["reject"]["pooled"]["raw"])
    )
    escalate_raw_by_qid = dict(
        zip(
            tournament["escalate"]["pooled"]["question_id"],
            tournament["escalate"]["pooled"]["raw"],
        )
    )
    reject_iso_by_qid = dict(
        zip(
            tournament["reject"]["pooled"]["question_id"],
            tournament["reject"]["pooled"]["isotonic"],
        )
    )
    escalate_iso_by_qid = dict(
        zip(
            tournament["escalate"]["pooled"]["question_id"],
            tournament["escalate"]["pooled"]["isotonic"],
        )
    )
    label_by_qid = dict(
        zip(tournament["reject"]["pooled"]["question_id"], tournament["reject"]["pooled"]["label"])
    )
    _p(f"tournament pooled {len(label_by_qid)} rows across all channels")

    _p("step 4: joining the two pooled sets by question ID; enforcing the sample-size floor")
    joined_qids = sorted(
        set(verifier_p_by_qid)
        & set(reject_iso_by_qid)
        & set(escalate_iso_by_qid)
        & set(label_by_qid)
    )
    joined_qids = [
        q
        for q in joined_qids
        if reject_iso_by_qid[q] is not None and escalate_iso_by_qid[q] is not None
    ]
    n_rows = len(joined_qids)
    n_groups = len(set(joined_qids))  # question_id is unique per row in this corpus
    _p(f"joined {n_rows} rows ({n_groups} question groups)")
    preconditions_checked.append(
        {"resource": "at_least_1000_joined_rows", "available": n_rows >= sre.A3_MIN_ROWS}
    )
    if n_rows < sre.A3_MIN_ROWS:
        return _honest_block_a3("insufficient_joined_rows", preconditions_checked, start)
    preconditions_checked.append(
        {"resource": "at_least_30_question_groups", "available": n_groups >= sre.A3_MIN_GROUPS}
    )
    if n_groups < sre.A3_MIN_GROUPS:
        return _honest_block_a3("insufficient_question_groups", preconditions_checked, start)

    labels_arr = [label_by_qid[q] for q in joined_qids]
    p_verifier_arr = [verifier_p_by_qid[q] for q in joined_qids]
    p_reject_cal_arr = [reject_iso_by_qid[q] for q in joined_qids]
    p_escalate_cal_arr = [escalate_iso_by_qid[q] for q in joined_qids]
    p_accept_raw_arr = [accept_by_qid[q] for q in joined_qids]
    p_reject_raw_arr = [reject_raw_by_qid[q] for q in joined_qids]
    p_escalate_raw_arr = [escalate_raw_by_qid[q] for q in joined_qids]

    _p("step 5: running the synthetic A3 positive control")
    positive_control = sre.run_a3_positive_control(seed=RANDOM_SEEDS_A3[0])
    _p(f"positive control passed={positive_control['passed']}")

    _p(f"step 6: running the full A3 policy evaluation over {len(RANDOM_SEEDS_A3)} fixed seeds")
    per_seed_results: list[dict] = []
    for i, seed in enumerate(RANDOM_SEEDS_A3):
        result = sre.run_a3_policy_evaluation(
            joined_qids,
            labels_arr,
            p_verifier_arr,
            p_reject_cal_arr,
            p_escalate_cal_arr,
            p_accept_raw_arr,
            p_reject_raw_arr,
            p_escalate_raw_arr,
            seed=seed,
            n_boot=N_BOOT_A3,
        )
        per_seed_results.append(result)
        _p(
            f"  seed={seed}: aurc={result['main_metrics']['aurc']:.4f} "
            f"coverage_5pct_risk={result['main_metrics']['coverage_at_5pct_risk']:.3f} "
            f"kill={result['kill_check']['kill']}"
        )

    _p("step 7: applying the pass/fail gate using the least-favorable seed")
    # Combined-risk point estimates never depend on the bootstrap seed --
    # only the CI does. The worst (largest, least favorable) upper bound
    # across all five seeds is the conservative choice: if EVERY seed's
    # bootstrap agrees the interval is below zero, the result is stable
    # (SCENARIO-VERIFY-7752-GATE reads this as "seed-to-seed stability").
    entropy_ci_uppers = [r["aurc_delta_vs_entropy_control"]["ci95"][1] for r in per_seed_results]
    verifier_ci_uppers = [
        r["aurc_delta_vs_verifier_only_control"]["ci95"][1] for r in per_seed_results
    ]
    worst_entropy_upper = max(entropy_ci_uppers)
    worst_verifier_upper = max(verifier_ci_uppers)
    gate_aurc_vs_entropy_ok = bool(worst_entropy_upper < 0.0)
    gate_aurc_vs_verifier_ok = bool(worst_verifier_upper < 0.0)

    headline = per_seed_results[0]  # seed[0]'s point estimates == every seed's (deterministic)
    gate_coverage_ok = bool(headline["main_metrics"]["coverage_at_5pct_risk"] >= 0.25)
    primary_counts = headline["primary_cost_result"]["confusion_matrix"]["counts"]
    gate_all_actions_present = bool(all(primary_counts.get(a, 0) > 0 for a in sre.A3_ACTIONS))
    # No `protected_group` / `registered_risk_bound` field or module exists
    # anywhere in this corpus or in the codebase (checked: `data/fover_corpus_v4.json`
    # carries only question_id/step_text/label/confidence, and no fairness
    # module defines either term) -- so this condition is VACUOUSLY satisfied,
    # not silently skipped. See `protected_group_check` in the artifact below.
    gate_protected_group_ok = True

    any_seed_kill = any(r["kill_check"]["kill"] for r in per_seed_results)
    all_seeds_kill = all(r["kill_check"]["kill"] for r in per_seed_results)

    gate_passed = bool(
        gate_aurc_vs_entropy_ok
        and gate_aurc_vs_verifier_ok
        and gate_coverage_ok
        and gate_all_actions_present
        and gate_protected_group_ok
    )

    if gate_passed:
        verdict_text = "a3_calibrated_policy_passed_pre_registered_gate"
    elif all_seeds_kill:
        verdict_text = "a3_calibrated_policy_failed_kill_criterion_triggered_every_seed"
    else:
        verdict_text = "a3_calibrated_policy_failed_pre_registered_gate"
    honest_verdict = f"complete: {verdict_text}"

    _p(f"step 8: writing results artifact ({honest_verdict})")

    seed_stability = {
        "aurc_delta_vs_entropy_ci95_upper_by_seed": entropy_ci_uppers,
        "aurc_delta_vs_verifier_only_ci95_upper_by_seed": verifier_ci_uppers,
        "aurc_delta_vs_entropy_ci95_upper_range": max(entropy_ci_uppers) - min(entropy_ci_uppers),
        "aurc_delta_vs_verifier_only_ci95_upper_range": (
            max(verifier_ci_uppers) - min(verifier_ci_uppers)
        ),
        "kill_check_agrees_across_all_seeds": bool(any_seed_kill == all_seeds_kill),
        "combined_risk_point_estimate_identical_across_seeds": bool(
            len({round(r["main_metrics"]["aurc"], 12) for r in per_seed_results}) == 1
        ),
    }

    duration_s = time.monotonic() - start
    artifact = {
        "schema": "carnot.semif_readout_ebm_eval.a3.v1",
        "experiment": "semif_readout_ebm_eval_a3",
        "requirement": "REQ-VERIFY-7752",
        "run_date": datetime.now(UTC).strftime("%Y%m%d"),
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE_A3,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS_A3,
        "inference_substrate_note": (
            "CPU-only threshold sweep and grouped bootstrap over A1's cached readout "
            "logits plus a retrained (deterministic, seeded) verifier-only Gibbs "
            "energy and A2's calibrator tournament re-run. No model is loaded and no "
            "forward pass is made -- confirmed by asserting `llama_cpp` is never "
            "imported in this code path."
        ),
        "model_specs": [],
        "verifier_is_oracle": False,
        "random_seeds_used": list(RANDOM_SEEDS_A3),
        "reproducibility_checksum": _reproducibility_checksum_a3(
            CACHE_PATH, RESULT_PATH, RESULT_PATH_A2
        ),
        "duration_s": duration_s,
        "preconditions_checked": preconditions_checked,
        "cited_upstream_artifacts": [
            {
                "experiment_id": "semif_readout_ebm_eval_a1",
                "fields_imported": [
                    "readout_logits_cache (all rows' raw per-option logits)",
                    "the collapsed PoE finding (alpha=0 in every fold) that motivates "
                    "using the verifier-only probability as the honest baseline",
                ],
            },
            {
                "experiment_id": "semif_readout_ebm_eval_a2",
                "fields_imported": [
                    "the isotonic-regression selection for the reject and escalate "
                    "channels (re-derived here, not copied, for byte-level "
                    "reproducibility against the pooled out-of-fold arrays this "
                    "driver needs)",
                    "the accept channel's kill-criterion failure (excluded from the "
                    "combined-risk score per that finding)",
                ],
            },
        ],
        "field_provenance": {
            "duration_s": {
                "principle": "Even a CPU-only aggregation should report a real wall-clock duration so a reader can distinguish a genuine run from a stub.",
                "satisfied_by": "wall_clock measurement across the full driver run",
            },
            "random_seeds_used": {
                "principle": "Five fixed seeds let a reader assess seed-to-seed stability of the bootstrap confidence intervals, not just a single point estimate.",
                "satisfied_by": f"random_seeds_used={list(RANDOM_SEEDS_A3)} threaded through five independent bootstrap resamples",
            },
            "reproducibility_checksum": {
                "principle": "A content-addressed hash of the corpus, cache, A1 artifact, and A2 artifact catches silent drift between this artifact and a future rerun.",
                "satisfied_by": "sha256 over the corpus bytes, cache bytes, A1/A2 artifact bytes, the seeds, and k",
            },
        },
        "corpus": {
            "path": "data/fover_corpus_v4.json",
            "n_held_out_rows": len(held_out_rows),
            "n_a2_scored_rows": len(a2_scored_rows),
            "n_joined_rows": n_rows,
            "n_question_groups": n_groups,
            "note": "question_id is unique per row in this corpus -- no row shares a question with another",
        },
        "honest_input_set": (
            "verifier-only probability (A1's collapsed PoE reduces to this) + "
            "A2's calibrated `reject` channel (isotonic) + A2's calibrated `escalate` "
            "channel (isotonic). A2's `accept` channel calibration is read only for "
            "the entropy control's raw 3-way probabilities -- it never contributes "
            "to the combined-risk score, because it did not clear A2's kill criterion."
        ),
        "combiner": (
            "combined_risk = mean(p_verifier_only, p_reject_calibrated, "
            "p_escalate_calibrated) -- an unweighted mean, the simplest defensible "
            "combiner given this task's explicit input set. It is fit on nothing, "
            "so it cannot leak or overfit the evaluation data."
        ),
        "positive_control": positive_control,
        "cost_grid": list(sre.A3_COST_GRID),
        "primary_cost": sre.A3_PRIMARY_COST,
        "per_seed_results": per_seed_results,
        "headline_metrics": {
            "combined_risk_aurc": headline["main_metrics"]["aurc"],
            "combined_risk_coverage_at_5pct_risk": headline["main_metrics"][
                "coverage_at_5pct_risk"
            ],
            "combined_risk_risk_at_fixed_coverage": headline["main_metrics"].get(
                "risk_at_fixed_coverage"
            ),
            "combined_risk_brier": headline["combined_risk_brier"],
            "combined_risk_ece": headline["combined_risk_ece"],
            "verifier_only_control_aurc": headline["verifier_only_control_metrics"]["aurc"],
            "entropy_control_aurc": headline["entropy_control_metrics"]["aurc"],
            "primary_cost_confusion_matrix": headline["primary_cost_result"]["confusion_matrix"],
            "primary_cost_balance_check": headline["primary_cost_result"]["balance_check"],
            "primary_cost_escalation_value": headline["primary_cost_result"]["escalation_value"],
        },
        "seed_stability": seed_stability,
        "protected_group_check": {
            "applicable": False,
            "reason": (
                "No protected_group field or registered risk-bound registry exists "
                "for data/fover_corpus_v4.json (it carries only question_id, "
                "step_text, label, confidence) or anywhere in this codebase's "
                "fairness/robustness framework, per A1/A2's own finding. This "
                "sub-condition of the pass/fail gate is therefore VACUOUSLY "
                "satisfied -- no protected group is defined, so none can be "
                "exceeded -- not silently skipped."
            ),
        },
        "acceptance_gates": {
            "aurc_improvement_over_entropy_control_ci95_upper_below_zero_all_seeds": {
                "value": worst_entropy_upper,
                "passed": gate_aurc_vs_entropy_ok,
                "principle": "The combined-risk policy must beat the entropy control's risk-coverage curve, confirmed with a 95 percent paired group bootstrap under every one of five fixed seeds, not just a favorable one.",
            },
            "aurc_improvement_over_verifier_only_control_ci95_upper_below_zero_all_seeds": {
                "value": worst_verifier_upper,
                "passed": gate_aurc_vs_verifier_ok,
                "principle": "The combined-risk policy must beat the verifier-only baseline alone -- otherwise A2's calibrated channels add no value over the existing verifier.",
            },
            "coverage_at_5pct_risk_at_least_25pct": {
                "value": headline["main_metrics"]["coverage_at_5pct_risk"],
                "passed": gate_coverage_ok,
                "principle": "A policy that can only safely decide on a tiny sliver of rows at 5 percent risk is not useful in deployment.",
            },
            "all_three_actions_occur_at_primary_cost": {
                "value": primary_counts,
                "passed": gate_all_actions_present,
                "principle": "A policy that never uses one of its three actions is not a three-way policy -- SCENARIO-VERIFY-7752-DEGENERATE.",
            },
            "no_protected_group_exceeds_registered_risk_bound": {
                "value": "vacuous_no_registry_exists",
                "passed": gate_protected_group_ok,
                "principle": "No protected-group registry exists for this corpus; the condition cannot be violated by a group that is never defined.",
            },
        },
        "kill_criterion": {
            "any_seed_triggers_kill": any_seed_kill,
            "all_seeds_trigger_kill": all_seeds_kill,
            "per_seed_kill_checks": [r["kill_check"] for r in per_seed_results],
        },
        "gate_passed": gate_passed,
        "methodology_note": (
            "The verifier-only probability is a real 2-4-1 GibbsModel retrained via "
            "NCE on the fixed calibrated_decision_benchmark train split, deterministic "
            "under random_seed=7750 (A1's seed). A2's calibrator tournament is "
            "re-run over the same held-out rows and the same grouped out-of-fold "
            "split, deterministic given the corpus and the cache. Zero new model "
            "forward passes were made in this driver; every logit was read from "
            "A1's on-disk cache."
        ),
    }
    RESULT_PATH_A3.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH_A3.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    _p(f"wrote {RESULT_PATH_A3} (gate_passed={gate_passed})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "a3":
        raise SystemExit(main_a3())
    if len(sys.argv) > 1 and sys.argv[1] == "a2":
        raise SystemExit(main_a2())
    raise SystemExit(main())
