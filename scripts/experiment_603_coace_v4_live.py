#!/usr/bin/env python3
"""Experiment 603 — CoACEExtractorV4 Data-Driven (GenPRM-style).

**Context (RETRO-068):**
    Three versions of hand-engineered patterns (V1/V2/V3) have achieved recall of
    0%, 5.9%, and 4% respectively on live IT-model outputs.  Root cause confirmed:
    the offline/live distribution gap cannot be bridged with more regex patterns.

    The test set (first 25 is_correct=False from live_pairs_578.json) is dominated
    by placeholder responses ("The answer is 42", 16/25 = 64%) with no reasoning
    to extract.  Of the 9 real responses, most contain LOGIC errors with arithmetically
    correct steps — no arithmetic extractor can catch these without knowing the ground
    truth.  The minority with actual arithmetic errors is what V3 already catches.

    V4 uses GenPRM-style LLM-as-extractor (arXiv 2504.00891):
    - LLM mode (CARNOT_FORCE_LIVE=1 + transformers available): Qwen3.5-0.8B CPU
      identifies arithmetic claims in JSON; eval() verifies them.
    - CI stub mode (default): regex fallback covering LaTeX and Unicode operators
      not handled by V3.  Deterministic, no LLM call.

**Error type catalog (from manual analysis of live_pairs_578.json, 25 test entries):**
    placeholder_no_arithmetic:        16  — "The answer is 42", no CoT shown
    logic_error_correct_arithmetic:    5  — correct math, wrong premise/setup
    latex_format_unparseable_by_v3:    4  — \times, \frac, display blocks
    arithmetic_error_in_prose:         2  — 7*1.5=10 and 90/7=12 (CAUGHT by V3)
    ---------------------------------------------------------------------------
    Total incorrect:                  27  (some q_idx appear twice — different models)
    Note: 2 TP already found by V3 were from a single response (q_idx=12, Carlos).

Spec: REQ-EXTRACT-045, REQ-EXTRACT-046,
      SCENARIO-EXTRACT-080, SCENARIO-EXTRACT-081, SCENARIO-EXTRACT-082,
      SCENARIO-EXTRACT-083
"""

from __future__ import annotations

# apply_env_autofix MUST be first — injects CARNOT_FORCE_LIVE when GPU is present.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402

assert_live_or_ci_skip()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.extraction import CoACEExtractorV3, CoACEExtractorV4  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_RESULT_PATH = "results/experiment_603_coace_v4_live.json"
_LIVE_PAIRS_PATH = "results/live_pairs_578.json"
_N_TEST = 25
_N_CORRECT_FP_TEST = 10  # correct responses to check for false positives

_watchdog = ExperimentTimeoutWatchdog(603, timeout_minutes=40)

tmpl = ExperimentTemplate(
    exp_id=603,
    title="CoACEExtractorV4 Data-Driven",
    deliverable=_RESULT_PATH,
    requires_gpu=False,
)
tmpl.setup()


# ---------------------------------------------------------------------------
# Error type catalog — derived from manual analysis of live_pairs_578.json
# These counts describe the 25-entry test set (first 25 is_correct=False entries).
# ---------------------------------------------------------------------------

ERROR_TYPE_CATALOG: dict[str, int] = {
    # 16/25: responses consist entirely of "The answer is 42" — no CoT, nothing to extract.
    # GenPRM and all regex-based approaches are helpless here.
    "placeholder_no_arithmetic": 16,
    # 5/25: the model applies correct arithmetic to a wrong premise.
    # Example: Josh house-flip uses total cost ($130k) as the % base instead of
    # purchase price ($80k).  Every stated equation is arithmetically true;
    # the error is conceptual.  Cannot be caught without knowing the correct setup.
    "logic_error_correct_arithmetic": 5,
    # 4/25: responses use LaTeX notation (\times, \frac, display blocks).
    # V3's plain-text regexes skip these.  V4 adds LaTeX patterns, but the
    # arithmetic inside the LaTeX is still mostly correct (logic errors remain).
    "latex_format_unparseable_by_v3": 4,
    # 2/25: one response (Carlos, q_idx=12) contains two actual arithmetic errors:
    #   7 * $1.5 = $10 (actual: $10.50) and $90/$7 = 12 (actual: 12.86).
    # V3 already catches both; the response is counted as 1 TP.
    "arithmetic_error_in_prose_caught_v3": 2,
}


# ---------------------------------------------------------------------------
# Optional LLM caller setup
# ---------------------------------------------------------------------------


def _build_llm_caller() -> tuple[Optional[Any], str]:
    """Try to build a Qwen3.5-0.8B CPU inference callable.

    Returns (callable, llm_mode_str).  If setup fails, returns (None, 'ci_stub_regex_only').

    Why Qwen3.5-0.8B: smallest model in the SOTA tier; 0.8B parameters fit in CPU
    memory for a single extraction call per response, keeping experiment runtime
    under the 40-minute watchdog even without a GPU.
    """
    force_live = os.environ.get("CARNOT_FORCE_LIVE", "").strip() == "1"
    if not force_live:
        return None, "ci_stub_regex_only"

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        hf_id = "Qwen/Qwen3.5-0.8B"
        print(f"[Exp603] Loading {hf_id} for LLM extraction (CPU mode)...")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
        model.eval()

        def _call(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            tokens = out[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(tokens, skip_special_tokens=True)

        return _call, f"qwen3_5_0_8b_cpu"

    except Exception as exc:
        print(f"[Exp603] LLM setup failed: {exc} — using CI stub")
        return None, "ci_stub_regex_only"


# ---------------------------------------------------------------------------
# Test set loading
# ---------------------------------------------------------------------------


def _load_test_set() -> tuple[list[dict], list[dict]]:
    """Load the 25 incorrect and 10 correct live responses.

    Returns (incorrect_25, correct_10).

    Why 25 incorrect: matches the Exp 591 test set for apples-to-apples comparison.
    Why 10 correct: used to measure false-positive rate (v4 should NOT flag these).
    """
    pairs = json.loads(Path(_LIVE_PAIRS_PATH).read_text())
    incorrect = [p for p in pairs if not p["is_correct"]][:_N_TEST]
    correct = [p for p in pairs if p["is_correct"]][:_N_CORRECT_FP_TEST]
    return incorrect, correct


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def main() -> None:
    llm_caller, llm_mode = _build_llm_caller()
    print(f"[Exp603] llm_mode={llm_mode}")

    incorrect_25, correct_10 = _load_test_set()

    v3_ext = CoACEExtractorV3()
    v4_ext = CoACEExtractorV4(llm_caller=llm_caller)

    # --- Evaluate on 25 incorrect responses ---
    v3_tp = 0
    v4_tp = 0
    per_q_flags = []

    for item in incorrect_25:
        response = item["response"]
        v3_res = v3_ext.extract(response)
        v4_res = v4_ext.extract(response)

        v3_found = v3_res.n_violations > 0
        v4_found = v4_res.n_violations > 0

        if v3_found:
            v3_tp += 1
        if v4_found:
            v4_tp += 1

        per_q_flags.append({
            "question_index": item.get("question_index"),
            "model": item.get("model"),
            "is_correct": False,
            "v3_violation_found": v3_found,
            "v4_violation_found": v4_found,
        })

    # --- Evaluate on 10 correct responses (false positive test) ---
    v4_fp = 0
    for item in correct_10:
        response = item["response"]
        v4_res = v4_ext.extract(response)
        if v4_res.n_violations > 0:
            v4_fp += 1

    # --- Metrics ---
    n = _N_TEST
    n_fp_test = _N_CORRECT_FP_TEST

    v3_recall = v3_tp / n
    v4_recall = v4_tp / n
    v4_tp_rate = v4_recall
    v4_fp_rate = v4_fp / n_fp_test if n_fp_test > 0 else 0.0
    v4_precision = (
        v4_tp / (v4_tp + v4_fp) if (v4_tp + v4_fp) > 0 else 0.0
    )
    recall_improvement = v4_recall - v3_recall

    gate_open = v4_recall >= 0.20
    retro_068_partial = v4_recall > 0.04
    retro_068_resolved = v4_recall >= 0.20

    if v4_recall >= 0.20:
        honest_verdict = "genprm_breakthrough"
    elif v4_recall > 0.04:
        honest_verdict = "genprm_improved"
    else:
        honest_verdict = "no_improvement"

    # --- Build artifact ---
    artifact = tmpl.build_result(
        {
            "n_responses": n,
            "llm_mode": llm_mode,
            "v3_recall": round(v3_recall, 4),
            "v4_recall": round(v4_recall, 4),
            "recall_improvement": round(recall_improvement, 4),
            "v4_tp_rate": round(v4_tp_rate, 4),
            "v4_fp_rate": round(v4_fp_rate, 4),
            "v4_precision": round(v4_precision, 4),
            "v3_tp": v3_tp,
            "v4_tp": v4_tp,
            "v4_fp": v4_fp,
            "error_type_catalog": ERROR_TYPE_CATALOG,
            "per_question_flags": per_q_flags,
            "gate_open": gate_open,
            "retro_068_partial": retro_068_partial,
            "retro_068_resolved": retro_068_resolved,
            "honest_verdict": honest_verdict,
        },
        status="success",
        schema="carnot.coace_v4.v1",
    )

    out_path = Path(_RESULT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"[Exp603] Written: {_RESULT_PATH}")
    print(
        f"[Exp603] v3_recall={v3_recall:.3f} v4_recall={v4_recall:.3f} "
        f"recall_improvement={recall_improvement:+.3f} "
        f"gate_open={gate_open} verdict={honest_verdict}"
    )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
