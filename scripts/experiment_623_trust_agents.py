#!/usr/bin/env python3
"""Experiment 623 — TRUST Agents Full Comparison on Extended Corpus.

**Researcher summary:**
    Exp 617 ran a 25+10 diagnostic gate and found both extractors have low recall
    in CI-stub mode.  This experiment runs the full statistical comparison on the
    extended corpus (50 incorrect + 20 correct) to determine whether
    TrustAgentsExtractor (arXiv 2604.12184) should replace LLMAsExtractorV1 as
    the default extractor in the verification pipeline.

    The three-agent pipeline decomposes extraction across three specialised agents:
        Agent 1 (NER): find all numeric values in the response.
        Agent 2 (ClaimFormer): given those numbers, form arithmetic claims.
        Agent 3 (Verifier): evaluate each claim with safe_eval().
    This decomposition may find claims that LLMAsExtractorV1's single-pass approach
    misses — particularly when numbers appear far from the arithmetic operator.

**Exit paths (every path writes the deliverable):**
    1. apply_env_autofix() before any imports
    2. assert_live_or_ci_skip()
    3. ExperimentTimeoutWatchdog(623, timeout_minutes=35)
    4. ExperimentTemplate.setup()
    5. Load 50 incorrect + 20 correct responses from fover_corpus_v5.json
    6. Run LLMAsExtractorV1 + TrustAgentsExtractor; compute recall, fp_rate, per-response comparison
    7. tmpl.build_result(...) writes the deliverable
    8. tmpl.assert_deliverable_written()  -- FINAL LINE

Spec: REQ-EXTRACT-054, SCENARIO-EXTRACT-092, SCENARIO-EXTRACT-093
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

_RESULT_PATH = "results/experiment_623_trust_agents.json"


def _load_corpus(n_incorrect: int = 50, n_correct: int = 20) -> tuple[list[str], list[str]]:
    """Load responses from fover_corpus_v5.json, falling back to synthetic CI examples.

    Returns (incorrect_responses, correct_responses) truncated to the requested sizes.
    The corpus is sorted by question_index so results are deterministic across runs.

    Why fover_corpus_v5 is preferred: it is the extended corpus built in Exp 615 with
    350 live pairs, giving enough incorrect responses for statistical confidence at n=50.
    """
    for fname in ("fover_corpus_v5.json", "fover_corpus_v4.json", "live_pairs_578.json"):
        path = _REPO_ROOT / "results" / fname
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        raw_pairs = data if isinstance(data, list) else data.get("pairs", data.get("live_pairs", []))
        incorrect: list[str] = []
        correct: list[str] = []
        for p in raw_pairs:
            resp = p.get("response", p.get("model_response", ""))
            if not resp:
                continue
            if not p.get("is_correct", True):
                incorrect.append(resp)
            else:
                correct.append(resp)
        if len(incorrect) >= n_incorrect:
            return incorrect[:n_incorrect], correct[:n_correct]

    # CI fallback — synthetic examples that cover StepSegmentEvalChain patterns.
    # These allow the comparison logic to be tested without live data.
    incorrect = ["She spent 3*16.50=54.50 on shorts.", "Total: 20+15=36 miles."] * 25
    correct = ["Total: 3*16.50=49.50.", "He earned 5*10=50 dollars."] * 10
    return incorrect[:n_incorrect], correct[:n_correct]


def _build_llm_caller() -> tuple[object, str]:
    """Build llm_caller when CARNOT_FORCE_LIVE=1 and transformers is importable.

    When CARNOT_FORCE_LIVE is not set (CI mode), returns (None, 'ci_stub') so that
    both extractors fall back to StepSegmentEvalChain / empty-list respectively.
    This is intentional — running LLM calls in CI would be slow and non-deterministic.

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


def _run_extractor_on_corpus(
    extractor,
    incorrect: list[str],
    correct: list[str],
) -> tuple[float, float, list[bool], list[bool]]:
    """Run extractor on the full corpus; return (recall, fp_rate, inc_flags, cor_flags).

    inc_flags[i] = True if extractor found at least one violation in incorrect[i].
    cor_flags[i] = True if extractor found at least one violation in correct[i].
    recall   = fraction of incorrect responses with at least one violation.
    fp_rate  = fraction of correct responses with at least one violation.

    Why per-response flags: the per-response comparison (only_v1, only_trust, etc.)
    requires knowing which individual responses each extractor fired on, not just
    aggregate counts.
    """
    inc_flags = [bool(extractor.extract(r)) for r in incorrect]
    cor_flags = [bool(extractor.extract(r)) for r in correct]
    recall = sum(inc_flags) / len(incorrect) if incorrect else 0.0
    fp_rate = sum(cor_flags) / len(correct) if correct else 0.0
    return recall, fp_rate, inc_flags, cor_flags


def compute_per_response_comparison(
    v1_flags: list[bool],
    trust_flags: list[bool],
) -> dict[str, int]:
    """Classify each incorrect response into one of four mutual-exclusive categories.

    Categories (for the n_incorrect responses only — correct responses are not classified):
        only_llm_v1  : v1 fired, trust did not
        only_trust   : trust fired, v1 did not
        both         : both fired
        neither      : neither fired

    Why mutual-exclusive classification: this decomposition reveals whether the
    extractors are complementary (only_v1 + only_trust > 0) or redundant (both >> 0).
    It also confirms that n_only_v1 + n_only_trust + n_both + n_neither == len(v1_flags).

    Args:
        v1_flags    : per-response boolean flags from LLMAsExtractorV1.
        trust_flags : per-response boolean flags from TrustAgentsExtractor.

    Returns dict with keys n_only_v1, n_only_trust, n_both, n_neither.
    """
    n_only_v1 = sum(1 for v, t in zip(v1_flags, trust_flags) if v and not t)
    n_only_trust = sum(1 for v, t in zip(v1_flags, trust_flags) if t and not v)
    n_both = sum(1 for v, t in zip(v1_flags, trust_flags) if v and t)
    n_neither = sum(1 for v, t in zip(v1_flags, trust_flags) if not v and not t)
    return {
        "n_only_v1": n_only_v1,
        "n_only_trust": n_only_trust,
        "n_both": n_both,
        "n_neither": n_neither,
    }


def make_verdict(
    v1_recall: float,
    trust_recall: float,
) -> tuple[str, str, str]:
    """Compute best_extractor, recommendation, and honest_verdict from recall values.

    The 0.05 margin threshold prevents noise-driven adoption: if trust recall is only
    marginally higher than v1 recall, the pipeline should stay with the known-good v1.
    A 5-percentage-point margin is meaningful at n=50 (represents 2-3 extra detections).

    Returns (best_extractor, recommendation, honest_verdict).
    """
    if trust_recall > v1_recall:
        best_extractor = "trust_agents"
    else:
        best_extractor = "llm_v1"

    if trust_recall > v1_recall + 0.05:
        recommendation = "Adopt TRUST Agents as default"
        honest_verdict = "trust_better"
    else:
        recommendation = "Keep LLMAsExtractorV1 (trust not significantly better)"
        honest_verdict = "v1_better_or_equivalent"

    return best_extractor, recommendation, honest_verdict


def main() -> None:
    """Run Exp 623: full TRUST Agents vs LLMAsExtractorV1 comparison on extended corpus."""
    result_path = str(_REPO_ROOT / _RESULT_PATH)
    tmpl = ExperimentTemplate(
        623,
        "TRUST Agents Comparison",
        result_path,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(623, timeout_minutes=35, result_path=result_path):
        llm_caller, llm_mode = _build_llm_caller()
        incorrect, correct = _load_corpus(n_incorrect=50, n_correct=20)

        v1 = LLMAsExtractorV1(llm_caller=llm_caller)
        v1_recall, v1_fp_rate, v1_inc_flags, _v1_cor_flags = _run_extractor_on_corpus(
            v1, incorrect, correct
        )

        trust = TrustAgentsExtractor(llm_caller=llm_caller)
        trust_recall, trust_fp_rate, trust_inc_flags, _trust_cor_flags = _run_extractor_on_corpus(
            trust, incorrect, correct
        )

        comparison = compute_per_response_comparison(v1_inc_flags, trust_inc_flags)
        best_extractor, recommendation, honest_verdict = make_verdict(v1_recall, trust_recall)

        artifact = tmpl.build_result(
            {
                "schema": "carnot.trust_agents_comparison.v1",
                "n_incorrect": len(incorrect),
                "n_correct": len(correct),
                "llm_mode": llm_mode,
                "v1_recall": v1_recall,
                "v1_fp_rate": v1_fp_rate,
                "trust_recall": trust_recall,
                "trust_fp_rate": trust_fp_rate,
                "n_only_v1": comparison["n_only_v1"],
                "n_only_trust": comparison["n_only_trust"],
                "n_both": comparison["n_both"],
                "n_neither": comparison["n_neither"],
                "best_extractor": best_extractor,
                "recommendation": recommendation,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        with open(result_path, "w") as f:
            json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
