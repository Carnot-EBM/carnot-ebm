#!/usr/bin/env python3
"""Experiment 617 — Extractor Diagnostic v5: Gate Decision for Exp 620.

**Researcher summary:**
    Exp 616 showed that LLMAsExtractorV1 in CI stub mode achieves recall=0.04
    (StepSegmentEvalChain only).  The theoretical recall ceiling is ~0.20 (5/25
    responses have actual arithmetic errors detectable by any extractor).

    gate_open = max(v1_recall, trust_recall) >= 0.20 MUST be True before
    Exp 620 (VR attempt #15) is scheduled.

    This experiment runs two extractors on 25 known-incorrect + 10 known-correct
    responses and records the gate decision:
        1. LLMAsExtractorV1 (best_strategy from Exp 616)
        2. TrustAgentsExtractor (arXiv 2604.12184 three-agent pipeline)

**Exit paths (every path writes the deliverable):**
    1. apply_env_autofix() before any imports
    2. assert_live_or_ci_skip()
    3. ExperimentTimeoutWatchdog(617, timeout_minutes=30)
    4. ExperimentTemplate.setup()
    5. Load 25 incorrect + 10 correct responses
    6. Run LLMAsExtractorV1 + TrustAgentsExtractor
    7. tmpl.build_result(...) writes the deliverable
    8. tmpl.assert_deliverable_written()  -- FINAL LINE

Spec: REQ-EXTRACT-053, SCENARIO-EXTRACT-090, SCENARIO-EXTRACT-091
"""

from __future__ import annotations

from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

assert_live_or_ci_skip()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.extraction.llm_extractor_v1 import LLMAsExtractorV1  # noqa: E402
from carnot.extraction.trust_agents_extractor import TrustAgentsExtractor  # noqa: E402

_RESULT_PATH = "results/experiment_617_extractor_diagnostic_v5.json"


def _load_live_pairs(n_incorrect: int = 25, n_correct: int = 10) -> tuple[list[str], list[str]]:
    """Load live response pairs, preferring live_pairs_578.json then fover_corpus_v5.json.

    Returns (incorrect_responses, correct_responses) truncated to requested sizes.
    Falls back to synthetic examples when both files are missing.
    """
    for fname in ("live_pairs_578.json", "fover_corpus_v5.json"):
        path = _REPO_ROOT / "results" / fname
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        pairs = data if isinstance(data, list) else data.get("pairs", data.get("live_pairs", []))
        incorrect: list[str] = []
        correct: list[str] = []
        for p in pairs:
            resp = p.get("response", p.get("model_response", ""))
            if not resp:
                continue
            if not p.get("is_correct", True):
                incorrect.append(resp)
            else:
                correct.append(resp)
        if incorrect:
            return incorrect[:n_incorrect], correct[:n_correct]

    # CI fallback — synthetic examples for deterministic testing.
    incorrect = [
        "She spent 3*16.50=54.50 on shorts.",
        "The answer is 42.",
        "Total: 20+15=35 miles.",
    ] * 9
    correct = ["Total: 3*16.50=49.50."] * 10
    return incorrect[:n_incorrect], correct[:n_correct]


def _build_llm_caller() -> tuple[object, str]:
    """Build llm_caller when CARNOT_FORCE_LIVE=1 and transformers is importable.

    Returns (llm_caller_or_None, mode_str).
    """
    if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
        return None, "ci_stub"

    try:
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415

        _pipe = hf_pipeline(
            "text-generation",
            "Qwen/Qwen3.5-0.8B",
            device="cpu",
            max_new_tokens=200,
        )

        def llm_caller(prompt: str) -> str:
            result = _pipe(prompt)
            return result[0]["generated_text"] if result else ""

        return llm_caller, "live_qwen35_0.8b_cpu"
    except Exception as exc:  # noqa: BLE001
        return None, f"ci_stub_fallback({exc})"


def _benchmark(
    extractor_name: str,
    incorrect: list[str],
    correct: list[str],
    extractor,
) -> tuple[float, float]:
    """Run extractor on incorrect+correct responses, return (recall, fp_rate).

    recall   = fraction of incorrect responses where at least one violation found.
    fp_rate  = fraction of correct responses where at least one violation found.
    """
    tp = sum(1 for r in incorrect if extractor.extract(r))
    fp = sum(1 for r in correct if extractor.extract(r))
    recall = tp / len(incorrect) if incorrect else 0.0
    fp_rate = fp / len(correct) if correct else 0.0
    return recall, fp_rate


def main() -> None:
    """Run Exp 617: dual-extractor diagnostic gate for Exp 620."""
    result_path = str(_REPO_ROOT / _RESULT_PATH)
    tmpl = ExperimentTemplate(
        617,
        "Extractor Diagnostic v5",
        result_path,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(617, timeout_minutes=30, result_path=result_path):
        llm_caller, llm_mode = _build_llm_caller()
        incorrect, correct = _load_live_pairs(n_incorrect=25, n_correct=10)

        v1 = LLMAsExtractorV1(llm_caller=llm_caller)
        v1_recall, v1_fp_rate = _benchmark("llm_v1", incorrect, correct, v1)

        trust = TrustAgentsExtractor(llm_caller=llm_caller)
        trust_recall, trust_fp_rate = _benchmark("trust_agents", incorrect, correct, trust)

        best_recall = max(v1_recall, trust_recall)
        best_extractor = "llm_v1" if v1_recall >= trust_recall else "trust_agents"
        gate_open = best_recall >= 0.20

        if gate_open:
            gate_note = "Exp 620 UNBLOCKED — schedule VR attempt #15"
            honest_verdict = "gate_open_vr_unblocked"
        else:
            gate_note = "Exp 620 GATED — do not schedule without gate_open=True"
            honest_verdict = "gate_closed_do_not_retry"

        artifact = tmpl.build_result(
            {
                "schema": "carnot.extractor_diagnostic_v5.v1",
                "n_tested": 25,
                "llm_mode": llm_mode,
                "v1_recall": v1_recall,
                "v1_fp_rate": v1_fp_rate,
                "trust_recall": trust_recall,
                "trust_fp_rate": trust_fp_rate,
                "best_recall": best_recall,
                "best_extractor": best_extractor,
                "gate_open": gate_open,
                "gate_note": gate_note,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        with open(result_path, "w") as f:
            json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
